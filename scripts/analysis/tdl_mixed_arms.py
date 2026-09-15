"""
E26 - mixed LOS/NLOS arms: per-RECEIVER channel assignment.

Motivation.  AERPAW carries a per-sensor blockage label (LOStoLW2..LW5) and the
audit showed that blockage count carries nearly all of the real predictable
share (0.265 from n_los alone vs 0.074 from flight identity).  Our arms so far
are the pure endpoints: every receiver LOS (TDL-D) or every receiver NLOS
(TDL-A/C).  Real deployments are mixtures, so the endpoints bracket reality but
neither IS reality.

Design.  rfgeo.dataset.generate_sample applies the channel inside the
per-receiver loop with one fixed profile, so a mixture only needs the profile to
be chosen per receiver.  Each receiver independently draws NLOS with probability
p_nlos; LOS links get TDL-D, NLOS links get the NLOS profile.

NON-CIRCULARITY.  The mixture rate is NOT tuned to reproduce the measured
predictable share.  We run a dose-response sweep (p_nlos = 0.25 / 0.50 / 0.75)
which, together with E24's p=0 and p=1, gives a five-point curve.  Where the
AERPAW blockage rate falls on that curve is then an observation, not a fit.
For reference, AERPAW's pooled n_los distribution over 984 valid measurements
implies a blockage fraction of about 0.39 (LW5 permanently blocked plus roughly
19% blockage on LW2-4).

GATE-M.  The blockage mask is drawn from a SEPARATE rng, so at p_nlos = 0 the
main stream is consumed exactly as rfgeo.dataset.generate_sample consumes it
with TDL-D.  The features must then be bit-identical.  If that fails, nothing
downstream is read.

    python scripts/data/tdl_mixed_arms.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_placement_minsep as H                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from rfgeo import channel as ch, crlb, geometry as geo, waveform as wf   # noqa: E402

OUTDIR = os.path.join(ROOT, "artifacts")
NPZDIR = os.path.join(OUTDIR, "tdl_mixed")
OUT = os.path.join(OUTDIR, "tdl_mixed_arms.json")
NTILE = 3
AERPAW_BLOCK_FRAC = 0.391      # observed, for reference only - not fitted to


def generate_sample_mixed(cfg, rng, rx_pos, rx_vel, mask_rng, p_nlos, nlos_profile):
    """rfgeo.dataset.generate_sample with a per-receiver channel profile.

    Everything drawn from `rng` happens in the same order and quantity as the
    original when every receiver is LOS, which is what GATE-M checks.
    """
    p_tx = H.ds._sample_emitter(cfg, rng)
    kind = rng.choice(cfg.waveforms)
    snr_db = rng.uniform(*cfg.snr_db_range)
    fading = rng.choice(cfg.fading_choices)
    k_db = rng.uniform(*cfg.k_db_range)

    src = wf.generate_waveform(str(kind), cfg.n_samples, cfg.fs, rng)
    tdoa, fdoa, ranges, vrad = geo.compute_tdoa_fdoa(
        p_tx, rx_pos, rx_vel, cfg.carrier_hz, ref=0)

    M = cfg.n_receivers
    is_nlos = mask_rng.random(M) < p_nlos          # separate stream on purpose
    X = np.empty((M, 2, cfg.n_samples), dtype=np.float32)
    pl_db = ch.freespace_path_loss_db(cfg.carrier_hz, ranges)
    for i in range(M):
        rx_iq = geo.render_receiver_iq(src, tdoa[i], fdoa[i], cfg.fs)
        prof = nlos_profile if is_nlos[i] else "TDL-D"
        rx_iq = ch.apply_tdl(rx_iq, rng, profile=prof,
                             delay_spread_ns=cfg.tdl_delay_spread_ns, fs=cfg.fs)
        rx_iq = ch.add_awgn(rx_iq, snr_db, rng)
        X[i, 0] = rx_iq.real.astype(np.float32)
        X[i, 1] = rx_iq.imag.astype(np.float32)

    cr = crlb.summarize_crlb(src, cfg.fs, snr_db, cfg.carrier_hz)
    meta = {"waveform": str(kind), "snr_db": float(snr_db),
            "sigma_tdoa_m": cr["sigma_tdoa_m"],
            "n_los": int(M - is_nlos.sum()), "n_nlos": int(is_nlos.sum())}
    return {"X": X, "p_tx": p_tx[:2].astype(np.float32), "meta": meta}


def build_split_mixed(cfg, n, rng, rx, vel, mask_rng, p_nlos, nlos_profile):
    Xs, ys, metas = [], [], []
    for _ in range(n):
        s = generate_sample_mixed(cfg, rng, rx, vel, mask_rng, p_nlos, nlos_profile)
        Xs.append(s["X"]); ys.append(s["p_tx"]); metas.append(s["meta"])
    return np.stack(Xs), np.stack(ys), metas


def make_data_mixed(cfg, max_lag, p_nlos, nlos_profile, mask_seed):
    ss = np.random.SeedSequence(cfg.seed)
    rng_layout, rng_tr, rng_cal, rng_te = [np.random.default_rng(s) for s in ss.spawn(4)]
    rx, vel = H.ds._place_receivers(cfg, rng_layout)
    ms = np.random.SeedSequence(mask_seed)
    m_tr, m_cal, m_te = [np.random.default_rng(s) for s in ms.spawn(3)]

    Xtr, ytr, _ = build_split_mixed(cfg, cfg.n_train, rng_tr, rx, vel, m_tr,
                                    p_nlos, nlos_profile)
    Xca, yca, mca = build_split_mixed(cfg, cfg.n_calib, rng_cal, rx, vel, m_cal,
                                      p_nlos, nlos_profile)
    Xte, yte, mte = build_split_mixed(cfg, cfg.n_test, rng_te, rx, vel, m_te,
                                      p_nlos, nlos_profile)
    gdop_te = np.array([H.gdop_2d(yte[i], rx) for i in range(len(yte))])
    return dict(
        ftr=H.gcc_of(Xtr, max_lag), ytr=ytr,
        fca=H.gcc_of(Xca, max_lag), yca=yca,
        fte=H.gcc_of(Xte, max_lag), yte=yte,
        snr_ca=np.array([m["snr_db"] for m in mca]),
        snr_te=np.array([m["snr_db"] for m in mte]),
        crlb_tdoa_te=np.array([np.nanmean(m["sigma_tdoa_m"]) for m in mte]),
        n_los_te=np.array([m["n_los"] for m in mte]),
        gdop_te=gdop_te, rx=rx)


def gate_m(cfg, max_lag, nlos_profile):
    """p_nlos=0 must reproduce the stock TDL-D generator bit for bit."""
    c = H.ds.GenConfig(**{**cfg.__dict__, "n_train": 1, "n_calib": 1, "n_test": 40})
    d_mix = make_data_mixed(c, max_lag, 0.0, nlos_profile, mask_seed=1)
    ss = np.random.SeedSequence(c.seed)
    _, rng_tr, rng_cal, rng_te = [np.random.default_rng(s) for s in ss.spawn(4)]
    rx, vel = H.ds._place_receivers(c, np.random.default_rng(ss.spawn(4)[0]))
    ss2 = np.random.SeedSequence(c.seed)
    rl, rt, rc, rte = [np.random.default_rng(s) for s in ss2.spawn(4)]
    rx2, vel2 = H.ds._place_receivers(c, rl)
    Xte, yte, _ = H.build_split(c, c.n_test, rte, rx2, vel2)
    # consume train/calib on the stock path in the same order first
    ss3 = np.random.SeedSequence(c.seed)
    rl3, rt3, rc3, rte3 = [np.random.default_rng(s) for s in ss3.spawn(4)]
    rx3, vel3 = H.ds._place_receivers(c, rl3)
    H.build_split(c, c.n_train, rt3, rx3, vel3)
    H.build_split(c, c.n_calib, rc3, rx3, vel3)
    Xte3, yte3, _ = H.build_split(c, c.n_test, rte3, rx3, vel3)
    f_stock = H.gcc_of(Xte3, max_lag)
    return {"max_abs_diff_feats": float(np.max(np.abs(d_mix["fte"] - f_stock))),
            "max_abs_diff_y": float(np.max(np.abs(d_mix["yte"] - yte3))),
            "n_checked": int(len(yte3))}


def pshare(le, cid, min_per_cell=5):
    groups = [le[cid == c] for c in np.unique(cid) if (cid == c).sum() >= min_per_cell]
    k = len(groups)
    if k < 2:
        return None
    n = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ms_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    ms_w = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    return float(max((ms_b - ms_w) / (n / k), 0.0) /
                 (max((ms_b - ms_w) / (n / k), 0.0) + ms_w))


def tertile(x, k=NTILE):
    return np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-nlos", type=float, nargs="+", default=[0.25, 0.50, 0.75])
    ap.add_argument("--nlos-profile", default="TDL-C")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260628, 20261628, 20262628])
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-calib", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-drop", type=float, default=0.1)
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-lag", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--tdl-delay-spread-ns", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    H.torch.set_num_threads(args.threads)
    os.makedirs(NPZDIR, exist_ok=True)
    out = {"_generated_by": "scripts/data/tdl_mixed_arms.py",
           "_env": {"torch": H.torch.__version__, "python": sys.version.split()[0]},
           "_aerpaw_blockage_fraction_reference": AERPAW_BLOCK_FRAC,
           "config": vars(args), "results": {}}

    base_cfg = H.ds.GenConfig(rx_speed_mps=0.0, seed=args.seeds[0], channel_mode="tdl",
                              tdl_profile="TDL-D",
                              tdl_delay_spread_ns=args.tdl_delay_spread_ns)
    if not args.skip_gate:
        g = gate_m(base_cfg, args.max_lag, args.nlos_profile)
        g["pass"] = bool(g["max_abs_diff_feats"] == 0.0 and g["max_abs_diff_y"] == 0.0)
        out["GATE_M_p0_reproduces_stock_TDLD"] = g
        print(f"GATE-M: max|feat diff|={g['max_abs_diff_feats']:.3e} "
              f"max|y diff|={g['max_abs_diff_y']:.3e} -> "
              f"{'PASS' if g['pass'] else 'FAIL'}", flush=True)
        if not g["pass"]:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)
            print("GATE-M FAILED - nothing downstream is read.")
            return

    for p in args.p_nlos:
        key = f"p{p:.2f}"
        out["results"][key] = []
        for seed in args.seeds:
            t0 = time.time()
            cfg = H.ds.GenConfig(rx_speed_mps=0.0, seed=int(seed), channel_mode="tdl",
                                 tdl_profile="TDL-D",
                                 tdl_delay_spread_ns=args.tdl_delay_spread_ns,
                                 n_train=args.n_train, n_calib=args.n_calib,
                                 n_test=args.n_test)
            d = make_data_mixed(cfg, args.max_lag, p, args.nlos_profile,
                                mask_seed=int(seed) + 7919)
            model, mean, std = H.train_der(d, args, seed)
            res = dict(H.evaluate(d, model, mean, std, args.alpha))

            pt, tot, alm, epm, _ = H.der_predict(model, d["fte"], mean, std)
            err = np.sqrt(((pt - d["yte"]) ** 2).sum(1))
            snr, gdop, nlos = d["snr_te"], d["gdop_te"], d["n_los_te"]
            le = np.log(err)
            keep = err <= 200.0
            res.update({
                "p_nlos": p, "seed": int(seed),
                "mean_err_m": float(err.mean()),
                "mean_over_median": float(err.mean() / np.median(err)),
                "p90_over_median": float(np.percentile(err, 90) / np.median(err)),
                "sd_log_err": float(np.std(le)),
                "frac_over_200m": float(1 - keep.mean()),
                "tail_inflation": float(err.mean() / err[keep].mean()) if keep.any() else None,
                "predictable_share_snr_gdop": pshare(le, tertile(snr) * NTILE + tertile(gdop)),
                "predictable_share_with_nlos": pshare(
                    le, (tertile(snr) * NTILE + tertile(gdop)) * 5 + nlos),
                "predictable_share_nlos_only": pshare(le, nlos),
                "mean_n_los": float(nlos.mean()),
                "n_los_hist": {int(k): int((nlos == k).sum()) for k in np.unique(nlos)},
                "_t_s": round(time.time() - t0, 1),
            })
            out["results"][key].append(res)
            np.savez_compressed(os.path.join(NPZDIR, f"mix{int(p*100):03d}_s{seed}.npz"),
                                err=err, snr=snr, gdop=gdop, epi=epm, ale=alm,
                                n_los=nlos, pred=pt, true=d["yte"])
            print(f"[p_nlos={p:.2f} s{seed}] med={res['median_err_m']:7.1f}m "
                  f"cov={res['cp_cov_ell']:.3f} pshare={res['predictable_share_snr_gdop']:.3f} "
                  f"mean/med={res['mean_over_median']:.2f} "
                  f"epi-r={res['epistemic_err_spearman']:.3f} "
                  f"nlos={4-res['mean_n_los']:.2f}/4  ({res['_t_s']:.0f}s)", flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)

    keys = ["median_err_m", "mean_over_median", "p90_over_median", "sd_log_err",
            "tail_inflation", "frac_over_200m", "predictable_share_snr_gdop",
            "predictable_share_with_nlos", "predictable_share_nlos_only",
            "cp_cov_ell", "cp_area_ell_median", "cond_dev_mondrian_mean",
            "epistemic_err_spearman", "al_crlb_pos_spearman",
            "two_force_area_reduction_pct", "cp_Q_ell", "mean_n_los"]
    out["summary"] = {k: {kk: {"mean": float(np.mean([r[kk] for r in v])),
                               "sd": float(np.std([r[kk] for r in v], ddof=1))}
                          for kk in keys if all(r.get(kk) is not None for r in v)}
                      for k, v in out["results"].items()}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwrote", args.out)
    ks = list(out["summary"])
    print(f"\n{'metric':32s}" + "".join(f"{k:>14s}" for k in ks))
    for m in keys:
        if all(m in out["summary"][k] for k in ks):
            print(f"{m:32s}" + "".join(f"{out['summary'][k][m]['mean']:14.4f}" for k in ks))


if __name__ == "__main__":
    main()
