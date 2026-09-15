"""
classical_corrected.py — PART II / R5: redo C1 and C4 with the CORRECTED SRP reader.

R2/R3 established that C1's SRP-PHAT was under-tuned: it read the stored (6,257)
GCC curve by linear interpolation, which throws away sub-sample information that the
band-limited curve actually carries. Reading the same array with sinc (FFT x8)
interpolation takes the median error from 27.1 m to ~7.7 m, and direct evaluation
from the cross-spectrum confirms the gain is not a windowing artefact.

Every headline number in §18 and §21 therefore has to be recomputed, because region
area scales with error^2 and the whole C4 comparison is downstream of the point
estimate. Nothing else changes: same input array, same calibration/test split, same
conformal machinery, same alpha.

Also reports the by-SNR breakdown, which is the causal check on R1: with a 25 m
algorithmic floor the estimator looked SNR-insensitive (31.6 -> 24.9 m, 1.27x). Once
the floor is removed the noise term should dominate and SNR sensitivity should
REAPPEAR. If it does not, the floor story is incomplete.
"""
from __future__ import annotations
import os, sys, json, argparse
from itertools import combinations
import numpy as np


def _add_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    for up in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        if os.path.isdir(os.path.join(up, "rfgeo")):
            sys.path.insert(0, up); sys.path.insert(0, os.path.join(up, "scripts"))
            sys.path.insert(0, here)
            return up
    raise RuntimeError("rfgeo not found")


REPO = _add_paths()
from rfgeo.conformal import _finite_sample_quantile as fsq
from classical_baseline import lag_table
from srp_interpolation_floor import upsample, srp_scores_dense

BINS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
UP = 8


def run_srp(feats, rx, pairs, G, LT, OFF, tag=""):
    est = np.zeros((len(feats), 2)); psr = np.zeros(len(feats))
    for i in range(len(feats)):
        dense = upsample(feats[i].astype(np.float64), UP)
        s = srp_scores_dense(dense, LT, UP)
        p0 = G[int(np.argmax(s))]
        Gf = p0[None, :] + OFF
        est[i] = Gf[int(np.argmax(srp_scores_dense(dense, lag_table(Gf, rx, pairs), UP)))]
        med = np.median(s); mad = np.median(np.abs(s - med)) + 1e-12
        psr[i] = (s.max() - med) / mad
        if (i + 1) % 1000 == 0:
            print(f"    {tag}{i+1}/{len(feats)}", flush=True)
    return est, psr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=float, default=100.0)
    ap.add_argument("--fine", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default="artifacts/classical_corrected.json")
    args = ap.parse_args()
    A = args.alpha

    rx = np.load(os.path.join(REPO, "output_final/rx_layout.npz"))["rx_pos"]
    pairs = list(combinations(range(rx.shape[0]), 2))
    R = 8000.0
    ax = np.arange(-R, R + 1e-9, args.coarse); gx, gy = np.meshgrid(ax, ax)
    G = np.c_[gx.ravel(), gy.ravel()]; G = G[np.linalg.norm(G, axis=1) <= R + 1e-9]
    LT = lag_table(G, rx, pairs)
    off = np.arange(-args.coarse, args.coarse + 1e-9, args.fine)
    ox, oy = np.meshgrid(off, off); OFF = np.c_[ox.ravel(), oy.ravel()]

    est, psr, snr, y = {}, {}, {}, {}
    for split in ("calib", "test"):
        d = np.load(os.path.join(REPO, f"output_final/gcc_{split}.npz"))
        f = d["feats"]; y[split] = d["y"].astype(np.float64)
        meta = json.load(open(os.path.join(REPO, f"output_final/meta_{split}.json"),
                              encoding="utf-8"))
        snr[split] = np.array([m["snr_db"] for m in meta])
        print(f"SRP (FFT x{UP}) on {split} (n={len(f)}) ...", flush=True)
        est[split], psr[split] = run_srp(f, rx, pairs, G, LT, OFF, f"{split} ")

    e_cal = np.linalg.norm(est["calib"] - y["calib"], axis=1)
    e_te = np.linalg.norm(est["test"] - y["test"], axis=1)
    pred = np.load(os.path.join(REPO, "output_final_der/predictions.npz"))
    e_net = pred["err_m"].astype(np.float64)

    out = {"n_test": int(len(e_te)),
           "srp_corrected": {"median": float(np.median(e_te)),
                             "p90": float(np.percentile(e_te, 90)),
                             "mean": float(np.mean(e_te)),
                             "rmse": float(np.sqrt(np.mean(e_te ** 2)))},
           "ours_der": {"median": float(np.median(e_net)),
                        "p90": float(np.percentile(e_net, 90))},
           "gap_median": float(np.median(e_net) / np.median(e_te))}
    print("\n" + "=" * 88)
    print("C1 CORRECTED — accuracy")
    print("=" * 88)
    print(f'  SRP (linear, as published in C1) : 27.08 m')
    print(f'  SRP (band-limited, corrected)    : {np.median(e_te):8.2f} m')
    print(f'  ours (CNN + NIW-DER)             : {np.median(e_net):8.2f} m')
    print(f'  gap                              : {out["gap_median"]:8.2f}x  '
          f'(was 5.55x)')

    print("\n  by SNR — does sensitivity reappear once the floor is removed?")
    print(f'  {"bin":>10s}{"n":>6s}{"SRP":>9s}{"ours":>9s}{"ratio":>8s}')
    by = {}
    for lo, hi in BINS:
        m = (snr["test"] >= lo) & (snr["test"] < hi)
        if m.sum() == 0:
            continue
        k = f"[{lo},{hi})"
        by[k] = {"n": int(m.sum()), "srp": float(np.median(e_te[m])),
                 "ours": float(np.median(e_net[m]))}
        print(f'  {k:>10s}{m.sum():>6d}{by[k]["srp"]:>9.2f}{by[k]["ours"]:>9.1f}'
              f'{by[k]["ours"]/by[k]["srp"]:>8.2f}')
    lo_hi = max(v["srp"] for v in by.values()) / min(v["srp"] for v in by.values())
    out["by_snr"] = by
    out["srp_snr_spread"] = float(lo_hi)
    print(f'  SRP low/high-SNR spread: {lo_hi:.2f}x   (linear reader gave 1.27x; '
          f'CRLB moves 20x)')

    # ---------------- C4 corrected ----------------
    Qc = fsq(e_cal, A)
    cov_c = e_te <= Qc
    area_c = np.pi * Qc ** 2

    def grp(s): return np.digitize(s, [-5, 0, 5, 10, 15])
    gc, gt = grp(snr["calib"]), grp(snr["test"])
    Qg = {int(k): fsq(e_cal[gc == k], A) for k in np.unique(gc) if (gc == k).sum() >= 50}
    r_m = np.array([Qg.get(int(k), Qc) for k in gt])
    cov_m = e_te <= r_m
    area_m = np.pi * r_m ** 2

    cp = json.load(open(os.path.join(REPO, "output_final_cp/conformal.json"),
                        encoding="utf-8"))
    e90 = cp["elliptical"][f"{1-A:.2f}"]
    ref = e90["median_area_m2"]
    print("\n" + "=" * 88)
    print(f"C4 CORRECTED — conformal regions at {1-A:.0%}")
    print("=" * 88)
    rows = [("SRP + circle (constant)", cov_c.mean(), area_c, area_c),
            ("SRP + Mondrian circle", cov_m.mean(), area_m.mean(), np.median(area_m)),
            ("ours: DER + elliptical CP", e90["empirical_coverage"],
             e90["mean_area_m2"], ref)]
    print(f'{"method":30s}{"coverage":>10s}{"area median":>16s}{"vs ours":>11s}')
    out["cp"] = {}
    for nm, c, am, amed in rows:
        out["cp"][nm] = {"coverage": float(c), "area_median_m2": float(amed)}
        print(f'{nm:30s}{c:>10.3f}{amed/1e6:>13.5f}km²{ref/amed:>10.1f}x')
    print(f"\n  SRP constant radius = {Qc:.1f} m   (C1-linear gave 56.7 m)")

    # conditional coverage
    devs = {"circle": [], "mond": [], "ours": []}
    ours_bins = {b["snr_bin"]: b["ell_coverage"] for b in cp["by_snr"]}
    for lo, hi in BINS:
        m = (snr["test"] >= lo) & (snr["test"] < hi)
        if m.sum() == 0:
            continue
        devs["circle"].append(abs(cov_c[m].mean() - (1 - A)))
        devs["mond"].append(abs(cov_m[m].mean() - (1 - A)))
        devs["ours"].append(abs(ours_bins.get(f"[{lo},{hi})", np.nan) - (1 - A)))
    out["cond_dev_mean"] = {k: float(np.mean(v)) for k, v in devs.items()}
    print("  |cov-0.9| mean: " + " | ".join(f"{k} {np.mean(v):.3f}"
                                            for k, v in devs.items()))

    # how much of the area gap is pure accuracy?
    acc2 = out["gap_median"] ** 2
    obs = ref / out["cp"]["SRP + circle (constant)"]["area_median_m2"]
    out["area_gap_decomposition"] = {"accuracy_squared": float(acc2),
                                     "observed": float(obs),
                                     "residual": float(obs / acc2)}
    print(f"\n  area gap {obs:.0f}x = accuracy^2 {acc2:.0f}x  x  residual {obs/acc2:.2f}x")
    print("  -> the residual is what the UQ layer contributes; the rest is accuracy.")

    op = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=2)
    np.savez_compressed(op.replace(".json", "_persample.npz"),
                        est_test=est["test"], err_test=e_te, psr_test=psr["test"],
                        est_calib=est["calib"], err_calib=e_cal, psr_calib=psr["calib"])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
