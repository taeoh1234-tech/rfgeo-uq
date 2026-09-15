"""
eval_shift.py — PHASE 6: evaluate NIW-DER + elliptical CP under distribution shift.

For each shift set: measure error, aleatoric/epistemic uncertainty (from the NIW
covariance), and elliptical-CP coverage using the Q calibrated on IN-DISTRIBUTION
calibration data (PHASE 5). Question: does epistemic rise with shift severity,
and does CP coverage degrade (showing exchangeability matters)?
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rfgeo.evidential import niw_uncertainties, niw_total_cov, niw_scalar_std
from rfgeo.conformal import elliptical_calibrate, elliptical_coverage
from train_der import DERModel
SCALE=5000.0

def load_npz(p):
    d=np.load(p); return d["feats"].astype(np.float32), d["y"].astype(np.float32), d["snr"]

@torch.no_grad()
def der_full(model,X):
    """Return pred(N,2 norm), aleatoric cov, epistemic cov, total cov (N,2,2 norm)."""
    model.eval(); xb=torch.from_numpy(X); mu0,L,nu=model(xb)
    al,ep=niw_uncertainties(L,nu)
    return mu0.numpy(), al.numpy(), ep.numpy(), (al+ep).numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--der-model", default="output_phase3/der_model.pt")
    ap.add_argument("--calib", default="output/gcc_calib.npz",
                    help="IN-DIST calib set for CP Q calibration")
    ap.add_argument("--shift-dir", default="output_phase6",
                    help="gen_shift 출력 폴더(shift_*.npz)")
    ap.add_argument("--res-out", default=None,
                    help="결과 json 경로. 기본 <shift-dir>/shift_eval.json")
    ap.add_argument("--threads", type=int, default=4)
    args=ap.parse_args()
    torch.set_num_threads(args.threads)
    ck=torch.load(args.der_model,weights_only=False)
    mean,std_n=ck["mean"],ck["std"]; r=ck.get("r",1.0)
    model=DERModel(n_pairs=6,r=r); model.load_state_dict(ck["model"])

    # calibrate elliptical CP Q on IN-DISTRIBUTION calib set (PHASE 5 setup)
    dc=np.load(args.calib); Xca=(dc["feats"].astype(np.float32)-mean)/std_n; yca=dc["y"].astype(np.float32)
    gc,alc,epc,totc=der_full(model,Xca)
    cov_ca=totc*SCALE**2
    Q90=elliptical_calibrate(yca, gc*SCALE, cov_ca, alpha=0.10)

    groups={
        "delay_spread":["delay_0","delay_1","delay_2","delay_3"],
        "snr_ood":["snr_0","snr_1","snr_2"],
        "kfactor_ood":["kf_0","kf_1","kf_2","kf_3"],
        "cfo_ood":["cfo_0","cfo_1","cfo_2","cfo_3","cfo_4"],
        "sco_ood":["sco_0","sco_1","sco_2","sco_3","sco_4"],
    }
    out={"Q90_indist":float(Q90),"groups":{}}
    for gname, names in groups.items():
        rows=[]
        for nm in names:
            X,y,snr=load_npz(os.path.join(args.shift_dir, f"shift_{nm}.npz"))
            Xn=(X-mean)/std_n
            g,al,ep,tot=der_full(model,Xn)
            pred=g*SCALE
            # scalar magnitudes via trace-based std (meters)
            al_m=(np.sqrt(np.trace(al,axis1=1,axis2=2))*SCALE)
            ep_m=(np.sqrt(np.trace(ep,axis1=1,axis2=2))*SCALE)
            cov_te=tot*SCALE**2
            err=np.sqrt(((pred-y)**2).sum(1))
            covered,area=elliptical_coverage(y,pred,cov_te,Q90)
            rows.append({"name":nm,
                "median_err_m":float(np.median(err)),
                "mean_aleatoric_m":float(al_m.mean()),
                "mean_epistemic_m":float(ep_m.mean()),
                "cp_coverage_90":float(covered.mean()),
                "cp_mean_area_m2":float(area.mean())})
        out["groups"][gname]=rows
    res_out=args.res_out or os.path.join(args.shift_dir,"shift_eval.json")
    json.dump(out, open(res_out,"w"), indent=2)

    print(f"elliptical CP Q90 (in-dist) = {Q90:.3f}\n")
    for gname,rows in out["groups"].items():
        print(f"=== {gname} ===")
        print(f'{"level":9s}{"median_err":>11s}{"aleatoric":>11s}{"epistemic":>11s}{"CP cov@90":>11s}{"CP area":>13s}')
        base_ep=rows[0]["mean_epistemic_m"]
        for r_ in rows:
            ratio=r_["mean_epistemic_m"]/base_ep if base_ep else float("nan")
            print(f'{r_["name"]:9s}{r_["median_err_m"]:>11.0f}{r_["mean_aleatoric_m"]:>11.0f}'
                  f'{r_["mean_epistemic_m"]:>11.0f}{r_["cp_coverage_90"]:>11.3f}{r_["cp_mean_area_m2"]:>13.0f}'
                  f'   (epi x{ratio:.2f})')
        print()

if __name__=="__main__": main()
