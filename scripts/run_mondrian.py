"""run_mondrian.py — PHASE 5b: group-conditional (Mondrian) elliptical CP by SNR bin."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from rfgeo.evidential import niw_total_cov
from rfgeo.conformal import (mondrian_elliptical_calibrate, elliptical_calibrate,
                             elliptical_coverage)
from train_der import DERModel
SCALE=5000.0

def load(split,out):
    d=np.load(os.path.join(out,f"gcc_{split}.npz")); return d["feats"].astype(np.float32),d["y"].astype(np.float32)

def snr_group(snr):
    bins=[-5,0,5,10,15]; return np.digitize(snr,bins)

@torch.no_grad()
def der_predict(model,X):
    model.eval(); xb=torch.from_numpy(X); mu0,L,nu=model(xb)
    cov=niw_total_cov(L,nu)
    return mu0.numpy()*SCALE, cov.numpy()*SCALE**2

def main():
    torch.set_num_threads(4)
    ck=torch.load("output_final_der/der_model.pt",weights_only=False)
    mean,std_n=ck["mean"],ck["std"]; r=ck.get("r",1.0)
    model=DERModel(n_pairs=6,r=r); model.load_state_dict(ck["model"])
    Xca,yca=load("calib","output_final"); Xte,yte=load("test","output_final")
    Xca=(Xca-mean)/std_n; Xte=(Xte-mean)/std_n
    meta_ca=json.load(open("output_final/meta_calib.json")); meta_te=json.load(open("output_final/meta_test.json"))
    snr_ca=np.array([m["snr_db"] for m in meta_ca]); snr_te=np.array([m["snr_db"] for m in meta_te])
    pred_ca,cov_ca=der_predict(model,Xca); pred_te,cov_te=der_predict(model,Xte)
    grp_ca=snr_group(snr_ca); grp_te=snr_group(snr_te)
    alpha=0.10
    Qs=mondrian_elliptical_calibrate(yca,pred_ca,cov_ca,grp_ca,alpha)
    Qmarg=elliptical_calibrate(yca,pred_ca,cov_ca,alpha)
    bins=[(-10,-5),(-5,0),(0,5),(5,10),(10,15),(15,20)]; tab=[]
    for lo,hi in bins:
        m=(snr_te>=lo)&(snr_te<hi)
        if m.sum()==0: continue
        g=grp_te[m][0]
        Qg=Qs.get(g,Qmarg)
        cov_mond,area_mond=elliptical_coverage(yte[m],pred_te[m],cov_te[m],Qg)
        cov_marg,area_marg=elliptical_coverage(yte[m],pred_te[m],cov_te[m],Qmarg)
        tab.append({"snr_bin":f"[{lo},{hi})","n":int(m.sum()),
                    "marginal_cp_cov":float(cov_marg.mean()),"marginal_area":float(area_marg.mean()),
                    "mondrian_cp_cov":float(cov_mond.mean()),"mondrian_area":float(area_mond.mean())})
    out={"target":0.90,"Q_marginal":float(Qmarg),
         "Q_by_group":{int(k):float(v) for k,v in Qs.items()},"by_snr":tab}
    json.dump(out,open("output_final_cp/mondrian.json","w"),indent=2)
    print("=== 90% coverage: marginal vs Mondrian (per-SNR) ELLIPTICAL CP ===")
    print(f'{"SNR":10}{"n":>5}{"marg cov":>10}{"marg A":>11}{"mond cov":>10}{"mond A":>11}')
    for r_ in tab:
        print(f'{r_["snr_bin"]:10}{r_["n"]:>5}{r_["marginal_cp_cov"]:>10.3f}{r_["marginal_area"]:>11.0f}'
              f'{r_["mondrian_cp_cov"]:>10.3f}{r_["mondrian_area"]:>11.0f}')

if __name__=="__main__": main()
