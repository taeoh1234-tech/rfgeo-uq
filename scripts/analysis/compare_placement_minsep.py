"""
compare_placement_minsep.py — Does breaking the 90-deg receiver symmetry buy GDOP
diversity without costing CP validity?

The question
------------
CLAUDE.md 6.3 records a structural weakness of Scenario A: with 4 receivers at
0/90/180/270 deg the GDOP is nearly constant (0.50..0.99, median 0.59), so
CRLB_tdoa and CRLB_pos are Spearman 0.982 -- the position-level reference is
indistinguishable from the TDOA-level one -- and the predicted ellipses stay
round (eccentricity median 1.72). That caps how much the elliptical CP story can
show.

  A (baseline) : azimuths 0/90/180/270, evenly spaced
  B_low/mid/high : azimuths drawn at random ONCE per dataset, with every adjacent
                 gap (including wrap-around) >= --min-sep degrees

Why three B arms instead of one
-------------------------------
A geometry-only pre-study over 200 min-40deg draws (no IQ, no training -- GDOP and
the CRLB ellipse depend on the layout alone) showed the draw DOMINATES the result:
the implied CRLB-ellipse eccentricity p90 ranges 2.33..5.22 and the
CRLB_tdoa-vs-CRLB_pos Spearman ranges 0.93..0.65 across draws. A single random
layout would therefore have measured which layout got drawn, not whether asymmetry
helps. So the arms are stratified at the p10/p50/p90 quantiles of that model-free
statistic, turning the study into a dose-response: does UQ quality track array
asymmetry? A monotone trend is far harder to dismiss than one A-vs-B contrast.

Why the layout is fixed per dataset, not per sample
---------------------------------------------------
Per-sample re-drawing was already tested twice (compare_placement_AB.py without
receiver coords, compare_placement_AB_v2.py with a coordinate side-input) and
collapses prediction: GCC-PHAT delays alone do not identify a position unless the
geometry is fixed. See sweep_emitter_radius.py's header. So B keeps ONE layout for
train/calib/test -- the backbone still sees a single geometry, exactly as in A --
and only the *shape* of that geometry changes. No architecture change, so A and B
are trained by identical code and are directly comparable.

Why a minimum separation
------------------------
Plain sort(uniform(0, 2pi)) lets two receivers land within a few degrees of each
other, degenerating a 4-sensor array to an effective 3-sensor one and destroying
the very GDOP the experiment is trying to vary. The min-gap constraint removes
that failure mode. (Draw is exact, not rejection-sampled: see
dataset._minsep_azimuths.)

Pairing / fairness
------------------
For each seed, A and B share the SAME data seed, so the emitter positions, SNR
draws, waveform choices and noise realizations are identical -- receivers are
placed before any sample is drawn, so the emitter RNG stream is untouched. The
only difference is the receiver azimuths. B's layout varies across seeds
(rx_layout_seed = the run seed), so the spread across seeds measures layout
variability, not just noise. Backbone, head, loss, optimizer, CP routines and
epochs are identical to train_der.py.

Everything downstream of placement is the real code:
  dataset.generate_sample, gcc.gcc_phat_features, evidential.*, conformal.*

Pre-registered decision rule (printed with the result; see main())
------------------------------------------------------------------
  GATE (validity; failing it rejects the change outright):
        EVERY arm's mean elliptical CP coverage >= nominal - 0.01
  PRIMARY (does asymmetry buy CP efficiency):
        Spearman(asymmetry rank, ellipse-vs-circle area reduction) >= +0.6
  SANITY (geometry did what the pre-study predicted):
        ecc p90 trends up, CRLB_tdoa-vs-CRLB_pos Spearman trends down
  SECONDARY (does the wider UQ story improve):
        pos-level aleatoric-CRLB Spearman trends up,
        Mondrian conditional-coverage deviation trends down
  GUARD (accuracy is not sacrificed for UQ cosmetics):
        worst arm's median error <= 1.15 x baseline median error

Trends are Spearman over ALL runs (arm index = pre-registered asymmetry rank),
not over arm means, so seed noise widens rather than fakes a trend.

Run:
    python scripts/data/compare_placement_minsep.py                  # 4 arms x 3 seeds
    python scripts/data/compare_placement_minsep.py --seeds 5        # tighter error bars
    python scripts/data/compare_placement_minsep.py --arms A B_high  # quick contrast
"""
from __future__ import annotations
import os, sys, json, time, copy, argparse
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr


def _add_repo_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for up in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        if os.path.isdir(os.path.join(up, "rfgeo")):
            sys.path.insert(0, up)
            return up
    sys.path.insert(0, os.getcwd())
    return os.getcwd()

REPO = _add_repo_to_path()

from rfgeo import dataset as ds
from rfgeo.gcc import gcc_phat_features
from rfgeo.evidential import (NIWHead, evidential_loss, niw_uncertainties,
                              niw_scalar_std)
from rfgeo.conformal import (elliptical_calibrate, elliptical_coverage,
                             radial_calibrate, radial_coverage,
                             mondrian_elliptical_calibrate)

SCALE = 5000.0
SNR_BINS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]


# --------------------------------------------------------------------------
# Backbone + head: verbatim from train_der.py (unchanged by this experiment)
# --------------------------------------------------------------------------
class GCCBackbone(nn.Module):
    def __init__(self, n_pairs=6, width=32, p_drop=0.1):
        super().__init__()
        def block(ci, co, k=5, s=2):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(co), nn.ReLU(inplace=True), nn.Dropout(p_drop))
        self.stem = nn.Conv1d(n_pairs, width, 5, stride=1, padding=2)
        self.blocks = nn.Sequential(
            block(width, width * 2), block(width * 2, width * 4),
            block(width * 4, width * 4), block(width * 4, width * 8))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(width * 8, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(p_drop))
        self.feat_dim = 128
    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        return self.proj(self.pool(x))


class DERModel(nn.Module):
    def __init__(self, n_pairs=6, p_drop=0.1, n_out=2, r=1.0):
        super().__init__()
        self.backbone = GCCBackbone(n_pairs=n_pairs, p_drop=p_drop)
        self.head = NIWHead(self.backbone.feat_dim, n_out=n_out, r=r)
        self.r = r
    def forward(self, x):
        return self.head(self.backbone(x))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def gdop_2d(p_xy, rx):
    """2D GDOP of a TDOA array: sqrt(trace((H^T H)^-1)) over receiver-pair rows."""
    p = np.array([p_xy[0], p_xy[1], 0.0])
    d = rx - p[None, :]
    u = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-9, None)
    H = np.stack([u[i, :2] - u[j, :2] for i, j in combinations(range(len(rx)), 2)], 0)
    G = H.T @ H
    try:
        cov = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return np.nan
    t = np.trace(cov)
    return np.sqrt(t) if t > 0 else np.nan


def build_split(cfg, n, rng, rx, vel):
    Xs, ys, metas = [], [], []
    for _ in range(n):
        s = ds.generate_sample(cfg, rng, rx, vel)
        Xs.append(s["X"]); ys.append(s["y"]); metas.append(s["meta"])
    return np.stack(Xs), np.stack(ys), metas


def gcc_of(X, max_lag):
    return np.stack([gcc_phat_features(X[i], max_lag=max_lag)
                     for i in range(len(X))]).astype(np.float32)


def snr_group(snr):
    """Bin index per SNR bin; out-of-range values fall in the nearest edge bin."""
    g = np.full(len(snr), -1, dtype=int)
    for k, (lo, hi) in enumerate(SNR_BINS):
        g[(snr >= lo) & (snr < hi)] = k
    g[g < 0] = 0 if len(g) == 0 else np.clip(g[g < 0], 0, len(SNR_BINS) - 1)
    return g


def make_data(cfg, max_lag):
    """Mirror dataset.generate_dataset's RNG spawn so placement and the three
    splits consume exactly the streams the real pipeline gives them."""
    ss = np.random.SeedSequence(cfg.seed)
    rng_layout, rng_tr, rng_cal, rng_te = [
        np.random.default_rng(s) for s in ss.spawn(4)]
    rx, vel = ds._place_receivers(cfg, rng_layout)     # <- the only thing that differs

    Xtr, ytr, _ = build_split(cfg, cfg.n_train, rng_tr, rx, vel)
    Xca, yca, mca = build_split(cfg, cfg.n_calib, rng_cal, rx, vel)
    Xte, yte, mte = build_split(cfg, cfg.n_test, rng_te, rx, vel)

    gdop_te = np.array([gdop_2d(yte[i], rx) for i in range(len(yte))])
    az = np.sort(np.rad2deg(np.arctan2(rx[:, 1], rx[:, 0])) % 360.0)
    gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
    return dict(
        ftr=gcc_of(Xtr, max_lag), ytr=ytr,
        fca=gcc_of(Xca, max_lag), yca=yca,
        fte=gcc_of(Xte, max_lag), yte=yte,
        snr_ca=np.array([m["snr_db"] for m in mca]),
        snr_te=np.array([m["snr_db"] for m in mte]),
        crlb_tdoa_te=np.array([np.nanmean(m["sigma_tdoa_m"]) for m in mte]),
        kind_te=np.array([m["waveform"] for m in mte]),   # arm 층화용 (가산)
        gdop_te=gdop_te, rx=rx, az_deg=az, gaps_deg=gaps)


# --------------------------------------------------------------------------
# Train / predict (identical recipe for both arms)
# --------------------------------------------------------------------------
def train_der(d, args, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = d["ftr"], d["ytr"]
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    Xtr_n = (Xtr - mean) / std
    n = len(Xtr_n); rng = np.random.default_rng(seed); perm = rng.permutation(n)
    nv = max(1, int(n * args.val_frac)); vi, ti = perm[:nv], perm[nv:]
    trl = DataLoader(TensorDataset(torch.from_numpy(Xtr_n[ti]),
                                   torch.from_numpy(ytr[ti] / SCALE)),
                     batch_size=args.bs, shuffle=True)
    val = DataLoader(TensorDataset(torch.from_numpy(Xtr_n[vi]),
                                   torch.from_numpy(ytr[vi] / SCALE)),
                     batch_size=args.bs)
    model = DERModel(n_pairs=Xtr.shape[1], p_drop=args.p_drop, r=args.r)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = float("inf"); best_state = None
    for ep in range(args.epochs):
        model.train()
        for xb, yb in trl:
            opt.zero_grad()
            mu0, L, nu = model(xb)
            loss, _, _ = evidential_loss(yb, mu0, L, nu, r=args.r, lam=args.lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
        sched.step()
        model.eval(); vl = 0.0; nvv = 0
        with torch.no_grad():
            for xb, yb in val:
                mu0, L, nu = model(xb)
                loss, _, _ = evidential_loss(yb, mu0, L, nu, r=args.r, lam=args.lam)
                vl += loss.item() * len(xb); nvv += len(xb)
        vl /= nvv
        if vl < best:
            best = vl; best_state = copy.deepcopy(model.state_dict())
        if args.verbose and ((ep + 1) % 20 == 0 or ep == 0):
            print(f"      ep {ep+1:3d}/{args.epochs}  val {vl:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, mean, std


@torch.no_grad()
def der_predict(model, X, mean, std):
    model.eval()
    mu0, L, nu = model(torch.from_numpy((X - mean) / std))
    al_cov, ep_cov = niw_uncertainties(L, nu)
    tot = al_cov + ep_cov
    return (mu0.numpy() * SCALE, tot.numpy() * SCALE**2,
            (niw_scalar_std(al_cov) * SCALE).numpy(),
            (niw_scalar_std(ep_cov) * SCALE).numpy(),
            (niw_scalar_std(tot) * SCALE).numpy())


def _sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).correlation) if m.sum() > 10 else float("nan")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate(d, model, mean, std, alpha):
    pt, tot_t, alm_t, epm_t, smag_t = der_predict(model, d["fte"], mean, std)
    pc, tot_c, alm_c, epm_c, smag_c = der_predict(model, d["fca"], mean, std)
    yte, yca = d["yte"], d["yca"]
    err = np.sqrt(((pt - yte) ** 2).sum(1))
    crlb_pos = d["crlb_tdoa_te"] * d["gdop_te"]
    target = 1.0 - alpha

    # --- elliptical vs circular CP (efficiency) ---------------------------
    Qe = elliptical_calibrate(yca, pc, tot_c, alpha)
    cove, areae = elliptical_coverage(yte, pt, tot_t, Qe)
    Qc = radial_calibrate(yca, pc, smag_c, alpha)
    covc, radc = radial_coverage(yte, pt, smag_t, Qc)
    areac = np.pi * radc**2

    # --- DER alone (uncalibrated chi2 ellipse) => the two-force figure -----
    # chi2 with df=2: P(X<=x) = 1-exp(-x/2)  ->  x = -2 ln(alpha)
    Q_chi2 = float(np.sqrt(-2.0 * np.log(alpha)))
    cov_der, area_der = elliptical_coverage(yte, pt, tot_t, Q_chi2)

    # --- conditional coverage: marginal Q vs Mondrian per-SNR-bin Q --------
    g_ca, g_te = snr_group(d["snr_ca"]), snr_group(d["snr_te"])
    Qs = mondrian_elliptical_calibrate(yca, pc, tot_c, g_ca, alpha)
    cov_m = np.zeros(len(yte), dtype=bool); area_m = np.zeros(len(yte))
    for g in np.unique(g_te):
        m = g_te == g
        Qg = Qs.get(g, Qe)
        cg, ag = elliptical_coverage(yte[m], pt[m], tot_t[m], Qg)
        cov_m[m] = cg; area_m[m] = ag
    dev_marg, dev_mond = [], []
    for g in np.unique(g_te):
        m = g_te == g
        if m.sum() < 10:
            continue
        dev_marg.append(abs(cove[m].mean() - target))
        dev_mond.append(abs(cov_m[m].mean() - target))

    # --- ellipse shape ----------------------------------------------------
    w = np.clip(np.linalg.eigvalsh(tot_t), 1e-12, None)
    ecc = np.sqrt(w[:, 1] / w[:, 0])

    gd = d["gdop_te"][np.isfinite(d["gdop_te"])]
    return dict(
        # accuracy (guard, not the objective)
        median_err_m=float(np.median(err)), p90_err_m=float(np.percentile(err, 90)),
        # geometry diversity = the thing this change targets
        gdop_median=float(np.median(gd)),
        gdop_p10=float(np.percentile(gd, 10)), gdop_p90=float(np.percentile(gd, 90)),
        gdop_iqr=float(np.percentile(gd, 75) - np.percentile(gd, 25)),
        crlb_tdoa_pos_spearman=_sp(d["crlb_tdoa_te"], crlb_pos),
        ecc_median=float(np.median(ecc)), ecc_p90=float(np.percentile(ecc, 90)),
        # UQ correlations
        al_crlb_pos_spearman=_sp(alm_t, crlb_pos),
        al_err_spearman=_sp(alm_t, err),
        epistemic_err_spearman=_sp(epm_t, err),
        al_ep_spearman=_sp(alm_t, epm_t),
        # CP validity + efficiency (primary UQ metrics)
        # areas in km^2 to match the units used throughout CLAUDE.md 6.3
        cp_cov_ell=float(cove.mean()), cp_cov_cir=float(covc.mean()),
        cp_area_ell_mean=float(areae.mean() / 1e6),
        cp_area_ell_median=float(np.median(areae) / 1e6),
        cp_area_reduction_pct=float(100 * (1 - areae.mean() / max(areac.mean(), 1e-9))),
        # two-force: DER alone vs DER+CP
        der_alone_cov=float(cov_der.mean()),
        der_alone_area_mean=float(area_der.mean() / 1e6),
        two_force_area_reduction_pct=float(
            100 * (1 - areae.mean() / max(area_der.mean(), 1e-9))),
        # conditional coverage
        cond_dev_marginal_mean=float(np.mean(dev_marg)) if dev_marg else float("nan"),
        cond_dev_marginal_max=float(np.max(dev_marg)) if dev_marg else float("nan"),
        cond_dev_mondrian_mean=float(np.mean(dev_mond)) if dev_mond else float("nan"),
        cond_dev_mondrian_max=float(np.max(dev_mond)) if dev_mond else float("nan"),
        cp_Q_ell=float(Qe))


# ---------------------------------------------------------------------------
# Pre-registered stratified arms.
#
# Chosen BEFORE any training run, on geometry alone (no model, no data): 200
# min-40deg draws were ranked by the CRLB-ellipse eccentricity p90 that their
# geometry implies -- the model-free ceiling on how elongated the learned
# ellipses could honestly be -- and the layouts at the p10/p50/p90 quantiles were
# taken. Selecting on a model-free statistic is what stops this from being a
# post-hoc pick of a flattering layout. The seed reproduces the azimuths exactly
# through dataset._place_receivers.
#
#   arm      layout_seed  gaps (deg)              ecc_p90  GDOP IQR  Spearman(T,P)
#   A (base)  --          90/90/90/90              2.33     0.179     0.934
#   B_low     10050       93/74/69/124             2.48     0.225     0.911
#   B_mid     10105       53/54/101/153            2.97     0.385     0.853
#   B_high    10183       206/64/40/50             4.06     0.865     0.709
#
# The three B arms form a monotone asymmetry gradient, so the study measures a
# dose-response ("does UQ quality track array asymmetry?") rather than a single
# fixed-vs-random contrast that one unlucky draw could decide.
# ---------------------------------------------------------------------------
PREREG_ARMS = [
    ("A",      None,  "uniform 90deg (Scenario A baseline)"),
    ("B_low",  10050, "min-sep random, low asymmetry  (ecc_p90 x1.06)"),
    ("B_mid",  10105, "min-sep random, mid asymmetry  (ecc_p90 x1.27)"),
    ("B_high", 10183, "min-sep random, high asymmetry (ecc_p90 x1.74)"),
]


def run_one(layout_seed, seed, args):
    """layout_seed=None -> uniform baseline; otherwise a fixed min-sep layout.

    `seed` drives the DATA and the TRAINING only. Because rx_layout_seed is a
    separate field, a B arm keeps one layout while emitters/noise/init vary, so
    the seed spread measures run-to-run noise, not layout luck.
    """
    kw = dict(rx_speed_mps=0.0, seed=int(seed),
              channel_mode=args.channel_mode, tdl_profile=args.tdl_profile,
              tdl_delay_spread_ns=args.tdl_delay_spread_ns)
    for k, v in [("n_train", args.n_train), ("n_calib", args.n_calib),
                 ("n_test", args.n_test)]:
        if v is not None:                 # smoke-test overrides; None = real sizes
            kw[k] = int(v)
    if layout_seed is not None:
        kw.update(rx_layout_mode="minsep_random", rx_min_sep_deg=args.min_sep,
                  rx_layout_seed=int(layout_seed))
    cfg = ds.GenConfig(**kw)
    t0 = time.time(); d = make_data(cfg, args.max_lag); t_gen = time.time() - t0
    t0 = time.time(); model, mean, std = train_der(d, args, seed)
    t_train = time.time() - t0
    res = evaluate(d, model, mean, std, alpha=1 - args.cp_level)
    res["_t_gen_s"] = round(t_gen, 1)
    res["_t_train_s"] = round(t_train, 1)
    res["_s_per_epoch"] = round(t_train / max(args.epochs, 1), 2)
    res["layout_az_deg"] = [round(float(a), 1) for a in d["az_deg"]]
    res["layout_gaps_deg"] = [round(float(g), 1) for g in d["gaps_deg"]]
    return res


def agg(vals):
    a = np.array(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser(
        description="Stratified receiver-layout study: uniform 90-deg baseline vs "
                    "min-separation random layouts of increasing asymmetry.")
    ap.add_argument("--seeds", type=int, default=3,
                    help="data/training seeds per arm (the layout is fixed per arm)")
    ap.add_argument("--seed0", type=int, default=20260628)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="subset of %s (default: all)"
                         % [a[0] for a in PREREG_ARMS])
    ap.add_argument("--min-sep", type=float, default=40.0,
                    help="minimum adjacent azimuth gap (deg) for the B arms")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-lag", type=int, default=128)
    ap.add_argument("--cp-level", type=float, default=0.90)
    # baseline convention (CLAUDE.md 6.3): TDL-D + Gold PN, sps=2
    ap.add_argument("--channel-mode", default="tdl", choices=["random", "tdl"])
    ap.add_argument("--tdl-profile", default="TDL-D")
    ap.add_argument("--tdl-delay-spread-ns", type=float, default=100.0)
    ap.add_argument("--std-pn", default="gold", choices=["gold", "mseq", "none"])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--verbose", action="store_true")
    # smoke-test overrides; leave unset for the real 4000/1000/1000 splits
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-calib", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--out", default="output_placement_minsep")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    if args.std_pn != "none":
        from rfgeo.waveform_std import register_std_dsss
        register_std_dsss(args.std_pn)
        print(f"[std-pn] DSSS replaced with standard {args.std_pn} code", flush=True)

    seeds = [args.seed0 + 1000 * k for k in range(args.seeds)]
    arms = [a for a in PREREG_ARMS if not args.arms or a[0] in args.arms]
    print(f"repo={REPO}")
    print(f"Stratified receiver-layout study | min gap {args.min_sep:.0f} deg | "
          f"one FIXED layout per arm, asymmetry increasing down the list")
    for n, ls, desc in arms:
        print(f"   {n:7s} layout_seed={str(ls):>5s}  {desc}")
    print(f"seeds={seeds} | epochs={args.epochs} | channel={args.channel_mode}/"
          f"{args.tdl_profile} | std_pn={args.std_pn} | threads={args.threads}")
    n_runs = len(arms) * len(seeds)
    print(f"runs total = {n_runs}  (~{15*n_runs} min at ~15 min/run)\n")

    raw = {n: {"per_seed": []} for n, _, _ in arms}
    t_all = time.time()
    for si, sd in enumerate(seeds):
        for name, lseed, _ in arms:
            t0 = time.time()
            res = run_one(lseed, sd, args)
            res["seed"] = sd
            res["layout_seed"] = lseed
            raw[name]["per_seed"].append(res)
            lay = ("uniform" if lseed is None
                   else "az=" + ",".join(f"{a:.0f}" for a in res["layout_az_deg"]))
            print(f"  {name:7s} seed {si+1}/{len(seeds)} (sd={sd}) {lay}\n"
                  f"      medErr={res['median_err_m']:6.0f}m  cov={res['cp_cov_ell']:.3f}  "
                  f"ecc_p90={res['ecc_p90']:5.2f}  gdopIQR={res['gdop_iqr']:.3f}  "
                  f"crlbT-P={res['crlb_tdoa_pos_spearman']:.3f}  "
                  f"areaRed={res['cp_area_reduction_pct']:5.1f}%  "
                  f"({time.time()-t0:.0f}s = gen {res['_t_gen_s']:.0f}s + "
                  f"train {res['_t_train_s']:.0f}s @ {res['_s_per_epoch']:.1f}s/ep)",
                  flush=True)

    METS = ["median_err_m", "p90_err_m", "gdop_median", "gdop_p10", "gdop_p90",
            "gdop_iqr", "crlb_tdoa_pos_spearman", "ecc_median", "ecc_p90",
            "al_crlb_pos_spearman", "al_err_spearman", "epistemic_err_spearman",
            "al_ep_spearman", "cp_cov_ell", "cp_cov_cir", "cp_area_ell_mean",
            "cp_area_ell_median", "cp_area_reduction_pct", "der_alone_cov",
            "der_alone_area_mean", "two_force_area_reduction_pct",
            "cond_dev_marginal_mean", "cond_dev_marginal_max",
            "cond_dev_mondrian_mean", "cond_dev_mondrian_max", "cp_Q_ell"]
    for k in raw:
        ps = raw[k]["per_seed"]
        raw[k]["mean"] = {m: agg([x[m] for x in ps])[0] for m in METS}
        raw[k]["std"] = {m: agg([x[m] for x in ps])[1] for m in METS}
    json.dump({"config": vars(args), "seeds": seeds, "results": raw},
              open(os.path.join(args.out, "placement_minsep_result.json"), "w"),
              indent=2)

    names = [n for n, _, _ in arms]
    # dose-response: arm index 0..k-1 IS the pre-registered asymmetry ordering,
    # repeated once per seed, so the trend uses every run rather than the means.
    dose = np.concatenate([np.full(len(raw[n]["per_seed"]), i)
                           for i, n in enumerate(names)])

    def trend(m):
        """Spearman(asymmetry rank, metric) over all runs. nan if degenerate."""
        v = np.concatenate([[x[m] for x in raw[n]["per_seed"]] for n in names])
        ok = np.isfinite(v)
        # a constant metric (e.g. DER-alone coverage pinned at 1.0) has no
        # defined rank correlation -- report n/a rather than a scipy warning
        if (ok.sum() < 4 or len(np.unique(dose[ok])) < 2
                or np.ptp(v[ok]) == 0):
            return float("nan")
        return float(spearmanr(dose[ok], v[ok]).correlation)

    W = 17
    print(f"\n\nSTRATIFIED LAYOUT STUDY (mean +/- std over {len(seeds)} seeds)")
    hdr = f"{'metric':32s}" + "".join(f"{n:>{W}s}" for n in names) + f"{'trend':>8s}"
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))

    def row(lab, m, f, higher_better=None):
        cells = [f"{f.format(raw[n]['mean'][m])}+/-{f.format(raw[n]['std'][m])}"
                 for n in names]
        tr = trend(m)
        tag = f"{tr:+.2f}" if np.isfinite(tr) else "  n/a"
        if higher_better is not None and np.isfinite(tr) and abs(tr) >= 0.6:
            tag += "+" if (tr > 0) == higher_better else "-"
        print(f"{lab:32s}" + "".join(f"{c:>{W}s}" for c in cells) + f"{tag:>8s}")

    print("--- geometry diversity (the target) ---")
    row("GDOP median", "gdop_median", "{:.3f}")
    row("GDOP p10", "gdop_p10", "{:.3f}")
    row("GDOP p90", "gdop_p90", "{:.3f}")
    row("GDOP IQR", "gdop_iqr", "{:.3f}", True)
    row("CRLB tdoa-vs-pos Spearman", "crlb_tdoa_pos_spearman", "{:.3f}", False)
    row("ellipse ecc median", "ecc_median", "{:.2f}", True)
    row("ellipse ecc p90", "ecc_p90", "{:.2f}", True)
    print("--- CP validity + efficiency (primary UQ) ---")
    row("CP coverage elliptical", "cp_cov_ell", "{:.3f}")
    row("CP coverage circular", "cp_cov_cir", "{:.3f}")
    row("ellipse area mean (km^2)", "cp_area_ell_mean", "{:.2f}", False)
    row("ellipse-vs-circle reduction %", "cp_area_reduction_pct", "{:.1f}", True)
    row("DER-alone coverage", "der_alone_cov", "{:.3f}")
    row("DER-alone area mean (km^2)", "der_alone_area_mean", "{:.2f}", False)
    row("two-force area reduction %", "two_force_area_reduction_pct", "{:.1f}", True)
    print("--- conditional coverage ---")
    row("cond dev marginal (mean)", "cond_dev_marginal_mean", "{:.3f}", False)
    row("cond dev marginal (max)", "cond_dev_marginal_max", "{:.3f}", False)
    row("cond dev Mondrian (mean)", "cond_dev_mondrian_mean", "{:.3f}", False)
    row("cond dev Mondrian (max)", "cond_dev_mondrian_max", "{:.3f}", False)
    print("--- UQ correlations ---")
    row("aleatoric-CRLB pos Spearman", "al_crlb_pos_spearman", "{:.3f}", True)
    row("aleatoric-error Spearman", "al_err_spearman", "{:.3f}", True)
    row("epistemic-error Spearman", "epistemic_err_spearman", "{:.3f}", True)
    row("aleatoric-epistemic Spearman", "al_ep_spearman", "{:.3f}", False)
    print("--- accuracy (guard) ---")
    row("median error (m)", "median_err_m", "{:.0f}", False)
    row("p90 error (m)", "p90_err_m", "{:.0f}", False)
    print("-" * len(hdr))
    print("trend = Spearman(asymmetry rank, metric) over all runs; "
          "+/- appended when |trend| >= 0.6 (+ = direction favours asymmetry)")

    # ---------------- pre-registered decision ----------------
    nominal = args.cp_level
    A = raw[names[0]]
    print("\n" + "=" * 62)
    print("PRE-REGISTERED DECISION")
    print("=" * 62)

    bad_cov = [n for n in names if raw[n]["mean"]["cp_cov_ell"] < nominal - 0.01]
    gate = not bad_cov
    print(f"GATE   every arm's CP coverage >= {nominal-0.01:.3f}: "
          f"{'PASS' if gate else 'FAIL -> ' + ','.join(bad_cov)}")
    for n in names:
        print(f"          {n:7s} {raw[n]['mean']['cp_cov_ell']:.3f}")

    worst = max(raw[n]["mean"]["median_err_m"] for n in names[1:]) if len(names) > 1 \
        else A["mean"]["median_err_m"]
    guard = worst <= 1.15 * A["mean"]["median_err_m"]
    print(f"GUARD  worst arm median err <= 1.15x baseline "
          f"({1.15*A['mean']['median_err_m']:.0f} m): {worst:.0f} m -> "
          f"{'PASS' if guard else 'FAIL'}")

    t_area = trend("cp_area_reduction_pct")
    t_ecc = trend("ecc_p90")
    t_crlb = trend("crlb_tdoa_pos_spearman")
    t_spear = trend("al_crlb_pos_spearman")
    t_mond = trend("cond_dev_mondrian_mean")
    print(f"PRIMARY   ellipse-vs-circle area reduction trend = {t_area:+.2f} "
          f"(want >= +0.6)")
    print(f"SANITY    ecc p90 trend = {t_ecc:+.2f} (want > 0, geometry-driven) | "
          f"CRLB T-vs-P trend = {t_crlb:+.2f} (want < 0)")
    print(f"SECONDARY pos-level alea-CRLB Spearman trend = {t_spear:+.2f} "
          f"(want > 0) | Mondrian cond-dev trend = {t_mond:+.2f} (want <= 0)")

    strong = np.isfinite(t_area) and t_area >= 0.6
    weak = np.isfinite(t_area) and t_area > 0
    if not gate:
        verdict = ("REJECT - CP validity is lost on at least one arm. Array asymmetry "
                   "is not worth an invalid coverage guarantee; keep the uniform "
                   "layout and report this as the reason.")
    elif not guard:
        verdict = ("REJECT for the baseline, KEEP as an ablation - the asymmetric "
                   "layouts cost more position error than the 1.15x guard allows. "
                   "Report the accuracy/UQ trade-off rather than switching.")
    elif strong:
        verdict = ("ADOPT the asymmetric layout - CP efficiency improves monotonically "
                   "with array asymmetry while coverage stays valid and accuracy holds. "
                   "This converts the near-constant-GDOP weakness of Scenario A into a "
                   "controlled design variable.")
    elif weak:
        verdict = ("PARTIAL - efficiency trends the right way but under the +0.6 bar. "
                   "Report as an exploratory dose-response; do NOT replace the "
                   "Scenario A baseline on this evidence. More seeds would sharpen it.")
    else:
        verdict = ("KEEP A - asymmetry does not buy CP efficiency. Valuable negative "
                   "result: it shows the near-constant GDOP of Scenario A is set by the "
                   "circular radius, not by the 90-deg symmetry, so the elliptical-CP "
                   "gain is bounded by the emitter/array scale rather than the layout.")
    print(f"\nVERDICT: {verdict}")
    print(f"\nSaved: {os.path.join(args.out, 'placement_minsep_result.json')}  "
          f"(total {time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
