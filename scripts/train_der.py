"""
train_der.py — PHASE 3: Multivariate Deep Evidential Regression (NIW head) on
the GCC-PHAT backbone.

Same GCCNet conv trunk as PHASE 2b (fair comparison); only the output head and
loss change: (x,y) MSE  ->  NIW (mu0, L, nu) + multivariate evidential loss
(Meinert 2021). Produces a point estimate plus a full 2x2 aleatoric/epistemic
covariance (tilted error ellipse), which PHASE 5 elliptical CP then calibrates.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfgeo.evidential import (NIWHead, evidential_loss, niw_uncertainties,
                              niw_total_cov, niw_scalar_std)

SCALE = 5000.0


class GCCBackbone(nn.Module):
    """Identical conv trunk to PHASE 2b GCCNet, minus the regression head."""
    def __init__(self, n_pairs=6, width=32, p_drop=0.1):
        super().__init__()
        def block(ci, co, k=5, s=2):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, stride=s, padding=k//2),
                nn.BatchNorm1d(co), nn.ReLU(inplace=True), nn.Dropout(p_drop))
        self.stem = nn.Conv1d(n_pairs, width, 5, stride=1, padding=2)
        self.blocks = nn.Sequential(
            block(width, width*2), block(width*2, width*4),
            block(width*4, width*4), block(width*4, width*8))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(width*8, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(p_drop))
        self.feat_dim = 128
    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        return self.proj(self.pool(x))


class DERModel(nn.Module):
    """GCC-PHAT backbone + NIW evidential head (n_out=2, coupling r)."""
    def __init__(self, n_pairs=6, p_drop=0.1, n_out=2, r=1.0):
        super().__init__()
        self.backbone = GCCBackbone(n_pairs=n_pairs, p_drop=p_drop)
        self.head = NIWHead(self.backbone.feat_dim, n_out=n_out, r=r)
        self.r = r 
    def forward(self, x):
        return self.head(self.backbone(x))          # (mu0, L, nu)


def load(split, out):
    d = np.load(os.path.join(out, f"gcc_{split}.npz"))
    return d["feats"].astype(np.float32), d["y"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--res-out", default="output_phase3")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--r", type=float, default=1.0, help="nu=r*kappa coupling constant")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="evidence-reg weight; 0 with the nu=r*kappa coupling")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    os.makedirs(args.res_out, exist_ok=True)

    Xtr_all, ytr_all = load("train", args.out)
    Xte, yte = load("test", args.out)
    meta_test = json.load(open(os.path.join(args.out, "meta_test.json")))
    snr_test = np.array([m["snr_db"] for m in meta_test])
    crlb_test = np.array([np.nanmean(m["sigma_tdoa_m"]) if isinstance(m.get("sigma_tdoa_m"), list)
                          else m.get("sigma_tdoa_m", np.nan) for m in meta_test])

    # --- Position-level CRLB reference (metric correction) --------------------
    # The model's aleatoric magnitude is a POSITION-level std, so the correct CRLB
    # reference is position-level: sigma_pos = sigma_tdoa (waveform+SNR) x GDOP
    # (geometry). GDOP comes from the receiver layout + each test emitter position.
    # We compute it here so the eval can report aleatoric-CRLB at BOTH levels.
    from itertools import combinations
    def _gdop(p_xy, rx):
        p = np.array([p_xy[0], p_xy[1], 0.0])
        d = rx - p[None, :]
        u = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-9, None)
        H = np.stack([u[i, :2] - u[j, :2]
                      for i, j in combinations(range(len(rx)), 2)], 0)
        G = H.T @ H
        try:
            cov = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return np.nan
        t = np.trace(cov)
        return np.sqrt(t) if t > 0 else np.nan
    try:
        _rx = np.load(os.path.join(args.out, "rx_layout.npz"))["rx_pos"]
        gdop_test = np.array([_gdop(yte[i], _rx) for i in range(len(yte))])
    except FileNotFoundError:
        gdop_test = np.full(len(yte), np.nan)   # falls back to TDOA-level only
    crlb_pos_test = crlb_test * gdop_test        # position-level CRLB reference


    mean = Xtr_all.mean(axis=(0,2), keepdims=True)
    std = Xtr_all.std(axis=(0,2), keepdims=True).clip(min=1e-6)
    norm = lambda A: (A - mean) / std
    Xtr_all = norm(Xtr_all); Xte = norm(Xte)

    n = len(Xtr_all); rng = np.random.default_rng(args.seed); perm = rng.permutation(n)
    nv = int(n*args.val_frac); vi, ti = perm[:nv], perm[nv:]
    tr = TensorDataset(torch.from_numpy(Xtr_all[ti]), torch.from_numpy(ytr_all[ti]/SCALE))
    va = TensorDataset(torch.from_numpy(Xtr_all[vi]), torch.from_numpy(ytr_all[vi]/SCALE))
    te = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte/SCALE))
    trl = DataLoader(tr, batch_size=args.bs, shuffle=True)
    val = DataLoader(va, batch_size=args.bs); tel = DataLoader(te, batch_size=args.bs)

    model = DERModel(n_pairs=Xtr_all.shape[1], p_drop=args.p_drop, r=args.r)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[phase3-NIW] params={sum(p.numel() for p in model.parameters())/1e3:.1f}k "
          f"train={len(ti)} val={len(vi)} test={len(Xte)} r={args.r} lam={args.lam}", flush=True)

    curves = {"train_loss": [], "val_loss": [], "val_rmse_m": [], "val_nll": []}
    best = float("inf")
    for ep in range(args.epochs):
        t0=time.time(); model.train(); tl=0; nt=0
        for xb, yb in trl:
            opt.zero_grad()
            mu0, L, nu = model(xb)
            loss,_,_ = evidential_loss(yb, mu0, L, nu, r=args.r, lam=args.lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tl+=loss.item()*len(xb); nt+=len(xb)
        tl/=nt; sched.step()
        model.eval(); vl=0; nvv=0; ve=[]; vnll=0
        with torch.no_grad():
            for xb, yb in val:
                mu0, L, nu = model(xb)
                loss,nll,_ = evidential_loss(yb, mu0, L, nu, r=args.r, lam=args.lam)
                vl+=loss.item()*len(xb); vnll+=nll.item()*len(xb); nvv+=len(xb)
                e=(mu0-yb)*SCALE; ve.append(torch.sqrt((e**2).sum(1)))
        vl/=nvv; vnll/=nvv; vr=torch.sqrt((torch.cat(ve)**2).mean()).item()
        curves["train_loss"].append(tl); curves["val_loss"].append(vl)
        curves["val_rmse_m"].append(vr); curves["val_nll"].append(vnll)
        if vl<best:
            best=vl
            torch.save({"model":model.state_dict(),"mean":mean,"std":std,
                        "scale":SCALE,"r":args.r,"args":vars(args)},
                       os.path.join(args.res_out,"der_model.pt"))
        if (ep+1)%10==0 or ep==0:
            print(f"  ep {ep+1:3d}/{args.epochs} train {tl:.4f} val {vl:.4f} "
                  f"val_RMSE {vr:6.0f} m nll {vnll:.3f} ({time.time()-t0:.1f}s)", flush=True)
        json.dump(curves, open(os.path.join(args.res_out,"curves.json"),"w"))

    # ---- eval ----
    ck = torch.load(os.path.join(args.res_out,"der_model.pt"), weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    MU,Ls,NU,Y = [],[],[],[]
    with torch.no_grad():
        for xb, yb in tel:
            mu0, L, nu = model(xb)
            MU.append(mu0); Ls.append(L); NU.append(nu); Y.append(yb)
    mu0=torch.cat(MU); L=torch.cat(Ls); nu=torch.cat(NU); y=torch.cat(Y)
    al_cov, ep_cov = niw_uncertainties(L, nu)            # (N,2,2) each, norm coords
    tot_cov = al_cov + ep_cov
    pred = (mu0*SCALE).numpy(); true=(y*SCALE).numpy()
    al_mag = (niw_scalar_std(al_cov)*SCALE).numpy()      # (N,)
    ep_mag = (niw_scalar_std(ep_cov)*SCALE).numpy()
    err = np.sqrt(((pred-true)**2).sum(1))

    from scipy.stats import spearmanr, pearsonr
    overall={"rmse_m":float(np.sqrt((err**2).mean())),
             "median_err_m":float(np.median(err)),
             "p90_err_m":float(np.percentile(err,90)),
             "mean_aleatoric_std_m":float(al_mag.mean()),
             "mean_epistemic_std_m":float(ep_mag.mean()),
             "err_epistemic_corr":float(np.corrcoef(err,ep_mag)[0,1]),
             "err_epistemic_spearman":float(spearmanr(err,ep_mag).correlation),
             "err_aleatoric_corr":float(np.corrcoef(err,al_mag)[0,1])}

    # aleatoric-CRLB correlation, reported FOUR ways (metric correction).
    #   level:  TDOA-level (waveform+SNR)  vs  position-level (x GDOP)
    #   stat :  Pearson (legacy, linear)   vs  Spearman (rank; EDL is rank-valid,
    #           Jurgens 2024 -> Spearman is the theoretically appropriate stat)
    # PRIMARY metric = aleatoric_crlb_pos_spearman. TDOA-level Pearson is retained
    # under its original key for continuity with earlier (0.67-style) numbers.
    def _corr(fn, a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() <= 10:
            return None
        return float(fn(a[m], b[m])[0] if fn is pearsonr
                     else fn(a[m], b[m]).correlation)
    overall["aleatoric_crlb_corr"] = _corr(pearsonr, al_mag, crlb_test)          # legacy: TDOA-level Pearson
    overall["aleatoric_crlb_tdoa_spearman"] = _corr(spearmanr, al_mag, crlb_test)
    overall["aleatoric_crlb_pos_pearson"] = _corr(pearsonr, al_mag, crlb_pos_test)
    overall["aleatoric_crlb_pos_spearman"] = _corr(spearmanr, al_mag, crlb_pos_test)  # PRIMARY


    bins=[(-10,-5),(-5,0),(0,5),(5,10),(10,15),(15,20)]; tab=[]
    for lo,hi in bins:
        m=(snr_test>=lo)&(snr_test<hi)
        if m.sum()==0: continue
        tab.append({"snr_bin":f"[{lo},{hi})","n":int(m.sum()),
                    "rmse_m":float(np.sqrt((err[m]**2).mean())),
                    "median_err_m":float(np.median(err[m])),
                    "mean_aleatoric_std_m":float(al_mag[m].mean()),
                    "mean_epistemic_std_m":float(ep_mag[m].mean())})
    json.dump({"overall":overall,"by_snr":tab},
              open(os.path.join(args.res_out,"snr_table.json"),"w"),indent=2)
    al_std_m = np.sqrt(np.diagonal(al_cov.numpy(), axis1=-2, axis2=-1))*SCALE
    ep_std_m = np.sqrt(np.diagonal(ep_cov.numpy(), axis1=-2, axis2=-1))*SCALE
    np.savez_compressed(os.path.join(args.res_out,"predictions.npz"),
                        pred_m=pred,true_m=true,al_std_m=al_std_m,ep_std_m=ep_std_m,
                        al_cov=(al_cov.numpy()*SCALE**2),ep_cov=(ep_cov.numpy()*SCALE**2),
                        tot_cov=(tot_cov.numpy()*SCALE**2),
                        err_m=err,snr_db=snr_test,crlb_m=crlb_test,
                        gdop=gdop_test,crlb_pos_m=crlb_pos_test)
    print("\n[phase3-NIW] OVERALL (test):"); print(json.dumps(overall,indent=2))
    print("\n[phase3-NIW] BY SNR:")
    for r in tab:
        print(f"  {r['snr_bin']:9s} n={r['n']:4d} RMSE {r['rmse_m']:7.0f} m "
              f"median {r['median_err_m']:7.0f} m alea {r['mean_aleatoric_std_m']:6.0f} "
              f"epi {r['mean_epistemic_std_m']:6.0f}")

if __name__ == "__main__":
    main()
