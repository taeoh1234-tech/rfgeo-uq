"""
run_conformal.py — PHASE 5: elliptical Conformal Prediction on NIW-DER.

Compares three region geometries at each nominal level:
  * DER-alone  : take the NIW ellipse at its face-value chi-square radius
                 (no calibration) — shows EDL's raw (uncalibrated) coverage.
  * circular CP: isotropic Mahalanobis-free score ||y-pred||/std_mag (ablation,
                 == the old radial CP).
  * elliptical CP: Mahalanobis score under the NIW covariance (MAIN pipeline).

Reports empirical coverage and region SIZE (area, m^2) so the efficiency win of
the ellipse (smaller area at equal coverage) is visible.
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rfgeo.evidential import niw_total_cov, niw_scalar_std
from rfgeo.conformal import (elliptical_calibrate, elliptical_coverage,
                             radial_calibrate, radial_coverage)
from train_der import DERModel
SCALE = 5000.0


def load(split, out):
    d = np.load(os.path.join(out, f"gcc_{split}.npz"))
    return d["feats"].astype(np.float32), d["y"].astype(np.float32)


@torch.no_grad()
def der_predict(model, X):
    """Return point pred (N,2, meters), total covariance (N,2,2, meters^2),
    and scalar std magnitude (N,, meters)."""
    model.eval(); xb = torch.from_numpy(X)
    mu0, L, nu = model(xb)
    cov = niw_total_cov(L, nu)                      # (N,2,2) normalized coords
    smag = niw_scalar_std(cov)                      # (N,) normalized
    pred = mu0.numpy() * SCALE
    cov_m = cov.numpy() * SCALE**2                  # meters^2
    smag_m = smag.numpy() * SCALE                   # meters
    return pred, cov_m, smag_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--der-model", default="output_phase3/der_model.pt")
    ap.add_argument("--res-out", default="output_phase5")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.res_out, exist_ok=True)

    ck = torch.load(args.der_model, weights_only=False)
    mean, std_n = ck["mean"], ck["std"]; r = ck.get("r", 1.0)
    model = DERModel(n_pairs=6, r=r); model.load_state_dict(ck["model"])

    Xca, yca = load("calib", args.out)
    Xte, yte = load("test", args.out)
    Xca = (Xca - mean) / std_n; Xte = (Xte - mean) / std_n
    meta_te = json.load(open(os.path.join(args.out, "meta_test.json")))
    snr_te = np.array([m["snr_db"] for m in meta_te])

    pred_ca, cov_ca, smag_ca = der_predict(model, Xca)
    pred_te, cov_te, smag_te = der_predict(model, Xte)
    err_te = np.sqrt(((pred_te - yte) ** 2).sum(1))

    levels = [0.80, 0.90, 0.95]
    results = {"levels": levels, "elliptical": {}, "circular": {}, "der_alone": {},
               "coupling_r": r}

    for p in levels:
        alpha = 1 - p
        # --- DER alone: NIW ellipse at chi-square(2) radius, no calibration ---
        # squared Mahalanobis of a 2D Gaussian ~ chi2(2); radius^2 = chi2.ppf(p,2)
        Q_chi = np.sqrt(chi2.ppf(p, df=2))
        cov_da, area_da = elliptical_coverage(yte, pred_te, cov_te, Q_chi)
        results["der_alone"][f"{p:.2f}"] = {
            "empirical_coverage": float(cov_da.mean()),
            "mean_area_m2": float(area_da.mean())}

        # --- circular CP (ablation) ---
        Qc = radial_calibrate(yca, pred_ca, smag_ca, alpha)
        covc, radc = radial_coverage(yte, pred_te, smag_te, Qc)
        results["circular"][f"{p:.2f}"] = {
            "Q": float(Qc),
            "empirical_coverage": float(covc.mean()),
            "mean_area_m2": float((np.pi * radc**2).mean())}

        # --- elliptical CP (MAIN) ---
        Qe = elliptical_calibrate(yca, pred_ca, cov_ca, alpha)
        cove, areae = elliptical_coverage(yte, pred_te, cov_te, Qe)
        results["elliptical"][f"{p:.2f}"] = {
            "Q": float(Qe),
            "empirical_coverage": float(cove.mean()),
            "mean_area_m2": float(areae.mean()),
            "median_area_m2": float(np.median(areae))}

    # --- SNR-binned coverage at 90% (elliptical vs circular) ---
    alpha = 0.10
    Qe = elliptical_calibrate(yca, pred_ca, cov_ca, alpha)
    cove, areae = elliptical_coverage(yte, pred_te, cov_te, Qe)
    Qc = radial_calibrate(yca, pred_ca, smag_ca, alpha)
    covc, radc = radial_coverage(yte, pred_te, smag_te, Qc)
    areac = np.pi * radc**2
    bins=[(-10,-5),(-5,0),(0,5),(5,10),(10,15),(15,20)]; tab=[]
    for lo,hi in bins:
        m=(snr_te>=lo)&(snr_te<hi)
        if m.sum()==0: continue
        tab.append({"snr_bin":f"[{lo},{hi})","n":int(m.sum()),
                    "ell_coverage":float(cove[m].mean()),
                    "ell_mean_area_m2":float(areae[m].mean()),
                    "circ_coverage":float(covc[m].mean()),
                    "circ_mean_area_m2":float(areac[m].mean())})
    results["by_snr"] = tab
    results["cp_Q_ell_90"] = float(Qe); results["cp_Q_circ_90"] = float(Qc)
    # area-reduction headline (ellipse vs circle), averaged over test set
    results["area_reduction_pct_90"] = float(100*(1 - areae.mean()/max(areac.mean(),1e-9)))

    json.dump(results, open(os.path.join(args.res_out,"conformal.json"),"w"), indent=2)
    np.savez(os.path.join(args.res_out,"cp_pred.npz"),
             pred_te=pred_te, true_te=yte, cov_te=cov_te, smag_te=smag_te,
             err_te=err_te, snr_te=snr_te, ell_area=areae, ell_cov=cove,
             circ_area=areac, circ_cov=covc, Q_ell=Qe, Q_circ=Qc)

    print("=== coverage & area: DER-alone vs circular CP vs elliptical CP ===")
    print(f'{"target":>7}{"DERalone":>10}{"circCP":>9}{"ellCP":>8}'
          f'{"circ area":>12}{"ell area":>12}')
    for p in levels:
        da=results["der_alone"][f"{p:.2f}"]; cc=results["circular"][f"{p:.2f}"]
        ee=results["elliptical"][f"{p:.2f}"]
        print(f'{p:>7.2f}{da["empirical_coverage"]:>10.3f}{cc["empirical_coverage"]:>9.3f}'
              f'{ee["empirical_coverage"]:>8.3f}{cc["mean_area_m2"]:>12.0f}{ee["mean_area_m2"]:>12.0f}')
    print(f'\nEllipse area reduction vs circle @90%: {results["area_reduction_pct_90"]:.1f}%')
    print("\n=== 90% coverage by SNR (elliptical vs circular) ===")
    for r_ in tab:
        print(f'  {r_["snr_bin"]:9s} n={r_["n"]:4d} ell {r_["ell_coverage"]:.3f} '
              f'(A={r_["ell_mean_area_m2"]:.0f})  circ {r_["circ_coverage"]:.3f} '
              f'(A={r_["circ_mean_area_m2"]:.0f})')

if __name__ == "__main__":
    main()
