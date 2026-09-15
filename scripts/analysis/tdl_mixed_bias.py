"""
E27 - NLOS excess-delay bias sweep on top of the mixed LOS/NLOS arm.

================================================================================
PRE-REGISTRATION.  Written before any bias run.  Thresholds are not adjusted
after seeing results.
================================================================================

WHY
E26 reproduced the measured predictable share (0.248 +/- 0.032 at p_nlos = 0.25
vs measured 0.244) but not the tail: p90/median tops out at 7.9 against a
measured 16.6.  Two structural terms are absent from the generator, and one of
them is cheap to add:

  (i)  3GPP TDL profiles normalise delays so the FIRST tap sits at delay 0.
       A blocked link therefore suffers fading but no excess path length, while
       real NLOS adds a positive range bias -- the b_i term in eq. (1) of
       arXiv:2502.01771, which that paper also sets to zero for its CRLB.
  (ii) apply_tdl renormalises to unit energy and add_awgn then scales noise to a
       target SNR, so blockage costs no signal power.  NOT addressed here: it
       would require changing the SNR labelling convention that every earlier
       experiment stratifies on.

This script adds (i) ONLY, and as a SENSITIVITY AXIS, not a baseline change.
The baseline channel stays exactly as it was.  The reason it is a sweep rather
than a fixed value is that there is no single standard for NLOS bias: the
reference paper itself cites Gaussian [11], exponential [12] and uniform [13]
models.  Picking one and calling it "the" model would trade a documented gap for
an undocumented tuned parameter.

HYPOTHESIS UNDER TEST
  H-bias: the residual tail gap and the near-zero explanatory power of blockage
          count are both caused by the missing excess-delay term.

PRE-REGISTERED READOUTS
  PRIMARY   tail statistics move monotonically toward the measurement as B rises
            SUPPORTED  if p90/median at the largest B is >= 12.0 (measured 16.58,
                       current best 7.9; 12.0 is the midpoint, fixed in advance)
            PARTIAL    if it rises but stays below 12.0
            REFUTED    if it does not rise with B

  SECONDARY blockage count becomes predictive:
            predictable_share_nlos_only rises from its current ~0.036 toward the
            measured 0.265.  SUPPORTED if it exceeds 0.10 at the largest B.

  GUARDRAIL the UQ conclusions must not depend on B:
            CP coverage stays in [0.87, 0.93], Mondrian conditional deviation
            stays below 0.05, and the predictable-share law still holds.
            If these move with B, the omission was material and must be
            reported as such rather than waved away.

  FALSIFIER if predictable_share_nlos_only is flat in B, the bias hypothesis is
            wrong and the gap has another cause.

CONTAMINATION GATES (this run must not be able to quietly poison anything)
  GATE-B0  B = 0 must reproduce the no-bias generator BIT-EXACTLY.  The bias is
           drawn from its own RNG so the emitter / channel / noise streams are
           untouched, which also makes the sweep PAIRED across B.
  GATE-B1  a known bias must produce the delay it claims: inject b metres into a
           clean impulse and measure the peak shift against b/c*fs.
  GATE-B2  LOS links must receive exactly zero bias.
  Plus: --npz-dir so smoke tests cannot write into the real output directory,
  and a minimum test size recorded per run.

    python scripts/data/tdl_mixed_bias.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_placement_minsep as H                      # noqa: E402
import tdl_mixed_arms as MIX                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from rfgeo import channel as ch, crlb, geometry as geo, waveform as wf   # noqa: E402

OUTDIR = os.path.join(ROOT, "artifacts")
OUT = os.path.join(OUTDIR, "tdl_mixed_bias.json")
C_LIGHT = 299_792_458.0
NTILE = 3

MEASURED = {"p90_over_median": 16.580, "mean_over_median": 5.229,
            "sd_log_err": 1.519, "predictable_share": 0.244,
            "predictable_share_nlos_only": 0.265}
PRIMARY_THRESHOLD_P90 = 12.0        # fixed in advance
SECONDARY_THRESHOLD_NLOS_SHARE = 0.10


def generate_sample_bias(cfg, rng, rx_pos, rx_vel, mask_rng, bias_rng,
                         p_nlos, nlos_profile, bias_mean_m):
    """MIX.generate_sample_mixed plus a positive excess-delay on NLOS links.

    `bias_rng` is separate from `rng`, and the unit exponential is drawn
    unconditionally, so every arm of the sweep sees the SAME emitter, channel
    and noise realisations. At bias_mean_m = 0 no delay is applied at all, which
    is what GATE-B0 checks against the no-bias generator.
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
    is_nlos = mask_rng.random(M) < p_nlos
    unit = bias_rng.exponential(1.0, M)          # drawn always -> paired sweep
    bias_m = unit * float(bias_mean_m) * is_nlos

    X = np.empty((M, 2, cfg.n_samples), dtype=np.float32)
    for i in range(M):
        rx_iq = geo.render_receiver_iq(src, tdoa[i], fdoa[i], cfg.fs)
        if bias_m[i] > 0.0:
            rx_iq = geo.fractional_delay(rx_iq, bias_m[i] / C_LIGHT * cfg.fs)
        prof = nlos_profile if is_nlos[i] else "TDL-D"
        rx_iq = ch.apply_tdl(rx_iq, rng, profile=prof,
                             delay_spread_ns=cfg.tdl_delay_spread_ns, fs=cfg.fs)
        rx_iq = ch.add_awgn(rx_iq, snr_db, rng)
        X[i, 0] = rx_iq.real.astype(np.float32)
        X[i, 1] = rx_iq.imag.astype(np.float32)

    cr = crlb.summarize_crlb(src, cfg.fs, snr_db, cfg.carrier_hz)
    meta = {"snr_db": float(snr_db), "sigma_tdoa_m": cr["sigma_tdoa_m"],
            "n_los": int(M - is_nlos.sum()),
            "bias_m_mean_nlos": float(bias_m[is_nlos].mean()) if is_nlos.any() else 0.0,
            "bias_m_max": float(bias_m.max()),
            "bias_on_los_links": float(np.abs(bias_m[~is_nlos]).max()) if (~is_nlos).any() else 0.0}
    return {"X": X, "p_tx": p_tx[:2].astype(np.float32), "meta": meta}


def build_split_bias(cfg, n, rng, rx, vel, mask_rng, bias_rng, p, prof, B):
    Xs, ys, metas = [], [], []
    for _ in range(n):
        s = generate_sample_bias(cfg, rng, rx, vel, mask_rng, bias_rng, p, prof, B)
        Xs.append(s["X"]); ys.append(s["p_tx"]); metas.append(s["meta"])
    return np.stack(Xs), np.stack(ys), metas


def make_data_bias(cfg, max_lag, p, prof, B, mask_seed, bias_seed):
    ss = np.random.SeedSequence(cfg.seed)
    rng_layout, rng_tr, rng_cal, rng_te = [np.random.default_rng(s) for s in ss.spawn(4)]
    rx, vel = H.ds._place_receivers(cfg, rng_layout)
    m_tr, m_cal, m_te = [np.random.default_rng(s)
                         for s in np.random.SeedSequence(mask_seed).spawn(3)]
    b_tr, b_cal, b_te = [np.random.default_rng(s)
                         for s in np.random.SeedSequence(bias_seed).spawn(3)]

    Xtr, ytr, _ = build_split_bias(cfg, cfg.n_train, rng_tr, rx, vel, m_tr, b_tr, p, prof, B)
    Xca, yca, mca = build_split_bias(cfg, cfg.n_calib, rng_cal, rx, vel, m_cal, b_cal, p, prof, B)
    Xte, yte, mte = build_split_bias(cfg, cfg.n_test, rng_te, rx, vel, m_te, b_te, p, prof, B)
    gdop_te = np.array([H.gdop_2d(yte[i], rx) for i in range(len(yte))])
    return dict(
        ftr=H.gcc_of(Xtr, max_lag), ytr=ytr,
        fca=H.gcc_of(Xca, max_lag), yca=yca,
        fte=H.gcc_of(Xte, max_lag), yte=yte,
        snr_ca=np.array([m["snr_db"] for m in mca]),
        snr_te=np.array([m["snr_db"] for m in mte]),
        crlb_tdoa_te=np.array([np.nanmean(m["sigma_tdoa_m"]) for m in mte]),
        n_los_te=np.array([m["n_los"] for m in mte]),
        bias_te=np.array([m["bias_m_mean_nlos"] for m in mte]),
        los_bias_max=float(max(m["bias_on_los_links"] for m in mte)),
        gdop_te=gdop_te)


# ------------------------------------------------------------------ gates
def gate_b0(cfg, max_lag, p, prof):
    """B=0 must equal the no-bias generator bit for bit."""
    c = H.ds.GenConfig(**{**cfg.__dict__, "n_train": 1, "n_calib": 1, "n_test": 40})
    a = make_data_bias(c, max_lag, p, prof, 0.0, mask_seed=11, bias_seed=22)
    b = MIX.make_data_mixed(c, max_lag, p, prof, mask_seed=11)
    return {"max_abs_diff_feats": float(np.max(np.abs(a["fte"] - b["fte"]))),
            "max_abs_diff_y": float(np.max(np.abs(a["yte"] - b["yte"]))),
            "n_checked": int(len(a["yte"]))}


def _lag_bandlimited(a, b, up=32):
    """Cross-correlation lag between a and b, read on an FFT-upsampled grid.

    A parabolic fit on the raw samples is NOT usable here: the delayed impulse
    is a Dirichlet kernel, and parabolic interpolation on it is biased by up to
    0.23 samples (35 m at fs = 2 MHz) for sub-sample delays, while being exact
    at integer ones. That bias is a property of the readout, not of the delay -
    the same lesson PART III R2 learned about SRP lag interpolation.
    """
    A, B = np.fft.fft(a), np.fft.fft(b)
    X = A * np.conj(B)
    N = len(X)
    Z = np.zeros(N * up, complex)
    Z[:N // 2] = X[:N // 2]
    Z[-(N // 2):] = X[N // 2:]
    c = np.abs(np.fft.ifft(Z))
    k = int(np.argmax(c))
    if k > N * up // 2:
        k -= N * up
    return k / up


def gate_b1(fs=2.0e6, n=4096, up=32, tol_m=5.0):
    """A commanded bias of b metres must move the estimated lag by b metres.

    Measured on a realistic random waveform with the band-limited readout, so
    this tests the delay as the pipeline's own estimator would see it. The
    readout's own resolution floor is about (1/2up) samples ~ 2.3 m, so the
    5 m tolerance is above the floor and far below the smallest bias tested.
    """
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    rows = []
    for b_m in (25.0, 50.0, 100.0, 300.0):
        want = b_m / C_LIGHT * fs
        got = _lag_bandlimited(geo.fractional_delay(sig, want), sig, up)
        rows.append({"bias_m": b_m, "commanded_samples": want,
                     "measured_samples": float(got),
                     "abs_err_samples": float(abs(got - want)),
                     "abs_err_m": float(abs(got - want) / fs * C_LIGHT)})
    return {"per_bias": rows, "tolerance_m": tol_m, "upsample": up,
            "readout_resolution_m": float(C_LIGHT / fs / (2 * up)),
            "max_abs_err_m": float(max(r["abs_err_m"] for r in rows))}


def pshare(le, cid, min_per_cell=5):
    groups = [le[cid == c] for c in np.unique(cid) if (cid == c).sum() >= min_per_cell]
    k = len(groups)
    if k < 2:
        return None
    n = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ms_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    ms_w = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    vb = max((ms_b - ms_w) / (n / k), 0.0)
    return float(vb / (vb + ms_w))


def tertile(x, k=NTILE):
    return np.searchsorted(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1]), x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias-m", type=float, nargs="+", default=[0.0, 25.0, 50.0, 100.0])
    ap.add_argument("--p-nlos", type=float, default=0.25)
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
    ap.add_argument("--npz-dir", default=os.path.join(OUTDIR, "tdl_bias"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    H.torch.set_num_threads(args.threads)
    os.makedirs(args.npz_dir, exist_ok=True)
    out = {"_generated_by": "scripts/data/tdl_mixed_bias.py",
           "_env": {"torch": H.torch.__version__, "python": sys.version.split()[0]},
           "_measured_reference": MEASURED,
           "_prereg": {"primary_threshold_p90_over_median": PRIMARY_THRESHOLD_P90,
                       "secondary_threshold_nlos_share": SECONDARY_THRESHOLD_NLOS_SHARE},
           "config": vars(args), "results": {}}

    base_cfg = H.ds.GenConfig(rx_speed_mps=0.0, seed=args.seeds[0], channel_mode="tdl",
                              tdl_profile="TDL-D",
                              tdl_delay_spread_ns=args.tdl_delay_spread_ns)
    g0 = gate_b0(base_cfg, args.max_lag, args.p_nlos, args.nlos_profile)
    g0["pass"] = bool(g0["max_abs_diff_feats"] == 0.0 and g0["max_abs_diff_y"] == 0.0)
    g1 = gate_b1()
    g1["pass"] = bool(g1["max_abs_err_m"] < g1["tolerance_m"])
    out["GATE_B0_zero_bias_bit_exact"] = g0
    out["GATE_B1_commanded_delay_correct"] = g1
    print(f"GATE-B0 (B=0 == no-bias generator): feats {g0['max_abs_diff_feats']:.3e} "
          f"y {g0['max_abs_diff_y']:.3e} -> {'PASS' if g0['pass'] else 'FAIL'}", flush=True)
    print(f"GATE-B1 (commanded delay): max err {g1['max_abs_err_m']:.4f} m -> "
          f"{'PASS' if g1['pass'] else 'FAIL'}", flush=True)
    if not (g0["pass"] and g1["pass"]):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("GATE FAILED - nothing downstream is run.")
        return

    for B in args.bias_m:
        key = f"B{B:g}"
        out["results"][key] = []
        for seed in args.seeds:
            t0 = time.time()
            cfg = H.ds.GenConfig(rx_speed_mps=0.0, seed=int(seed), channel_mode="tdl",
                                 tdl_profile="TDL-D",
                                 tdl_delay_spread_ns=args.tdl_delay_spread_ns,
                                 n_train=args.n_train, n_calib=args.n_calib,
                                 n_test=args.n_test)
            d = make_data_bias(cfg, args.max_lag, args.p_nlos, args.nlos_profile, B,
                               mask_seed=int(seed) + 7919, bias_seed=int(seed) + 104729)
            # GATE-B2, checked on every run rather than once
            assert d["los_bias_max"] == 0.0, "GATE-B2 FAILED: bias leaked onto a LOS link"

            model, mean, std = H.train_der(d, args, seed)
            res = dict(H.evaluate(d, model, mean, std, args.alpha))
            pt, tot, alm, epm, _ = H.der_predict(model, d["fte"], mean, std)
            err = np.sqrt(((pt - d["yte"]) ** 2).sum(1))
            snr, gdop, nlos = d["snr_te"], d["gdop_te"], d["n_los_te"]
            le = np.log(err)
            res.update({
                "bias_mean_m": B, "seed": int(seed), "n_test": int(len(err)),
                "mean_err_m": float(err.mean()),
                "mean_over_median": float(err.mean() / np.median(err)),
                "p90_over_median": float(np.percentile(err, 90) / np.median(err)),
                "p99_over_median": float(np.percentile(err, 99) / np.median(err)),
                "sd_log_err": float(np.std(le)),
                "iqr_log_err": float(np.subtract(*np.percentile(le, [75, 25]))),
                "predictable_share_snr_gdop": pshare(le, tertile(snr) * NTILE + tertile(gdop)),
                "predictable_share_nlos_only": pshare(le, nlos),
                "mean_n_los": float(nlos.mean()),
                "realised_bias_mean_m": float(d["bias_te"].mean()),
                "los_bias_max": d["los_bias_max"],
                "_t_s": round(time.time() - t0, 1)})
            out["results"][key].append(res)
            np.savez_compressed(os.path.join(args.npz_dir, f"B{B:g}_s{seed}.npz"),
                                err=err, snr=snr, gdop=gdop, epi=epm, ale=alm,
                                n_los=nlos, pred=pt, true=d["yte"])
            print(f"[B={B:g}m s{seed}] med={res['median_err_m']:7.1f} "
                  f"cov={res['cp_cov_ell']:.3f} p90/med={res['p90_over_median']:6.2f} "
                  f"nlos_share={res['predictable_share_nlos_only']:.3f} "
                  f"pshare={res['predictable_share_snr_gdop']:.3f} "
                  f"mondrian={res['cond_dev_mondrian_mean']:.4f}  ({res['_t_s']:.0f}s)",
                  flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=1)

    keys = ["median_err_m", "p90_over_median", "p99_over_median", "mean_over_median",
            "sd_log_err", "iqr_log_err", "predictable_share_snr_gdop",
            "predictable_share_nlos_only", "cp_cov_ell", "cond_dev_mondrian_mean",
            "epistemic_err_spearman", "cp_area_ell_median", "cp_Q_ell",
            "realised_bias_mean_m"]
    out["summary"] = {k: {m: {"mean": float(np.mean([r[m] for r in v])),
                              "sd": float(np.std([r[m] for r in v], ddof=1))}
                          for m in keys if all(r.get(m) is not None for r in v)}
                      for k, v in out["results"].items()}

    ks = sorted(out["summary"], key=lambda s: float(s[1:]))
    top = ks[-1]
    p90_top = out["summary"][top]["p90_over_median"]["mean"]
    nl_top = out["summary"][top]["predictable_share_nlos_only"]["mean"]
    p90_seq = [out["summary"][k]["p90_over_median"]["mean"] for k in ks]
    nl_seq = [out["summary"][k]["predictable_share_nlos_only"]["mean"] for k in ks]
    out["verdict"] = {
        "primary_p90_at_max_B": p90_top,
        "primary": ("SUPPORTED" if p90_top >= PRIMARY_THRESHOLD_P90
                    else "PARTIAL" if p90_seq[-1] > p90_seq[0] else "REFUTED"),
        "secondary_nlos_share_at_max_B": nl_top,
        "secondary": ("SUPPORTED" if nl_top >= SECONDARY_THRESHOLD_NLOS_SHARE
                      else "PARTIAL" if nl_seq[-1] > nl_seq[0] else "REFUTED"),
        "guardrail_coverage_range": [min(out["summary"][k]["cp_cov_ell"]["mean"] for k in ks),
                                     max(out["summary"][k]["cp_cov_ell"]["mean"] for k in ks)],
        "guardrail_mondrian_max": max(out["summary"][k]["cond_dev_mondrian_mean"]["mean"]
                                      for k in ks),
        "guardrail_pass": bool(
            all(0.87 <= out["summary"][k]["cp_cov_ell"]["mean"] <= 0.93 for k in ks)
            and all(out["summary"][k]["cond_dev_mondrian_mean"]["mean"] < 0.05 for k in ks)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwrote", args.out)
    print(f"\n{'metric':32s}" + "".join(f"{k:>13s}" for k in ks) + f"{'measured':>13s}")
    for m in keys:
        row = "".join(f"{out['summary'][k][m]['mean']:13.4f}" if m in out["summary"][k]
                      else f"{'-':>13s}" for k in ks)
        meas = MEASURED.get(m.replace("_snr_gdop", ""))
        print(f"{m:32s}{row}" + (f"{meas:13.3f}" if meas else f"{'-':>13s}"))
    print("\nVERDICT:", json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
