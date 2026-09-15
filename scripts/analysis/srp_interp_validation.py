"""
srp_interp_validation.py — PART II / R3: is the sinc-interpolation gain REAL?

R2 found that reading the stored (6,257) GCC curve with band-limited (FFT)
interpolation instead of linear interpolation takes SRP from 27.8 m to 7.7 m. Before
that number is allowed anywhere near a conclusion it has to survive one objection:

    The stored curve is a 257-lag WINDOW cut out of a length-N correlation. FFT
    upsampling treats that window as one period of a periodic signal, so spectral
    leakage from the truncation could in principle create a spurious sharpening.

The decisive control is to bypass the window entirely: evaluate the GCC-PHAT cross
correlation DIRECTLY at real-valued lags from the cross-spectrum,

    R_ij(tau) = sum_f  [ X_i(f) X_j*(f) / |X_i(f) X_j*(f)| ] exp(+j 2 pi f tau / N)

which needs no interpolation of any kind. If direct evaluation agrees with FFT
upsampling, the gain is real and the window is harmless. If direct evaluation stays
near 27 m, the gain was a truncation artefact and R2 must be withdrawn.

Arms (all on the SAME regenerated samples, paired):
    stored_linear   stored 257-lag curve, linear interpolation   (= the C1 baseline)
    stored_fft8     stored 257-lag curve, FFT x8                  (= R2's claim)
    direct_exact    cross-spectrum evaluated at the candidate lags, no window at all
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
from rfgeo.dataset import GenConfig, _sample_emitter
from rfgeo import waveform as wf, geometry as geo, channel as ch
from rfgeo.gcc import gcc_phat_features
from rfgeo.waveform_std import register_std_dsss
from classical_baseline import C, FS, MAX_LAG, lag_table
from srp_interpolation_floor import upsample, srp_scores_dense


def gen_one(cfg, k, seed, rx, rx_vel):
    r = np.random.default_rng([seed, 0, k])
    p_tx = _sample_emitter(cfg, r)
    kind = str(r.choice(cfg.waveforms))
    snr = float(r.uniform(*cfg.snr_db_range))
    src = wf.generate_waveform(kind, cfg.n_samples, cfg.fs, r)
    rng_ch = np.random.default_rng([seed, 1, k])
    rng_n = np.random.default_rng([seed, 2, k])
    tdoa, fdoa, *_ = geo.compute_tdoa_fdoa(p_tx, rx, rx_vel, cfg.carrier_hz, ref=0)
    X = np.empty((cfg.n_receivers, 2, cfg.n_samples), dtype=np.float32)
    iq = []
    for i in range(cfg.n_receivers):
        s = geo.render_receiver_iq(src, tdoa[i], fdoa[i], cfg.fs)
        s = ch.apply_tdl(s, rng_ch, profile=cfg.tdl_profile,
                         delay_spread_ns=cfg.tdl_delay_spread_ns, fs=cfg.fs)
        s = ch.add_awgn(s, snr_db=snr, rng=rng_n)
        iq.append(s)
        X[i, 0] = s.real; X[i, 1] = s.imag
    return p_tx[:2], X, np.stack(iq)


def whitened_cross_spectra(iq, pairs):
    """PHAT-whitened cross-spectrum per pair; the exact object gcc_phat_features
    inverse-transforms. Evaluating it at arbitrary tau needs no interpolation."""
    F = np.fft.fft(iq, axis=1)
    out = []
    for i, j in pairs:
        R = F[i] * np.conj(F[j])
        out.append(R / (np.abs(R) + 1e-12))
    return np.stack(out)                      # (P, N) complex


def srp_direct(Rw, freqs, lt):
    """SRP score for candidate lags `lt` (G,P) straight from the cross-spectra."""
    # corr(tau) = sum_f Rw[f] exp(+j 2 pi f tau / N)  -> real part
    ph = np.exp(2j * np.pi * np.outer(lt.ravel(), freqs))          # (G*P, N)
    v = np.real(ph @ Rw.reshape(-1, Rw.shape[-1]).T)               # (G*P, P)
    v = v.reshape(lt.shape[0], lt.shape[1], -1)
    v = np.einsum("gpp->gp", v) if v.shape[1] == v.shape[2] else v[:, np.arange(lt.shape[1]), np.arange(lt.shape[1])]
    return v.sum(axis=1) / Rw.shape[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--coarse", type=float, default=100.0)
    ap.add_argument("--fine", type=float, default=6.0)
    ap.add_argument("--out", default="artifacts/srp_interp_validation.json")
    args = ap.parse_args()
    register_std_dsss("gold")

    cfg = GenConfig(channel_mode="tdl", tdl_profile="TDL-D", tdl_delay_spread_ns=100.0)
    lay = np.load(os.path.join(REPO, "output_final/rx_layout.npz"))
    rx, rx_vel = lay["rx_pos"], lay["rx_vel"]
    pairs = list(combinations(range(cfg.n_receivers), 2))
    N = cfg.n_samples
    freqs = np.fft.fftfreq(N)                                       # cycles / sample

    R = 8000.0
    ax = np.arange(-R, R + 1e-9, args.coarse); gx, gy = np.meshgrid(ax, ax)
    G = np.c_[gx.ravel(), gy.ravel()]; G = G[np.linalg.norm(G, axis=1) <= R + 1e-9]
    LT = lag_table(G, rx, pairs)
    off = np.arange(-args.coarse, args.coarse + 1e-9, args.fine)
    ox, oy = np.meshgrid(off, off); OFF = np.c_[ox.ravel(), oy.ravel()]

    err = {k: [] for k in ("stored_linear", "stored_fft8", "direct_exact")}
    print(f"n={args.n} paired samples | grid {len(G)} + {len(OFF)} fine\n")
    for k in range(args.n):
        p_tx, X, iq = gen_one(cfg, k, args.seed, rx, rx_vel)
        feats = gcc_phat_features(X, max_lag=MAX_LAG).astype(np.float64)
        Rw = whitened_cross_spectra(iq, pairs)

        for arm in ("stored_linear", "stored_fft8", "direct_exact"):
            if arm == "direct_exact":
                s = srp_direct(Rw, freqs, LT)
                p0 = G[int(np.argmax(s))]
                Gf = p0[None, :] + OFF
                est = Gf[int(np.argmax(srp_direct(Rw, freqs, lag_table(Gf, rx, pairs))))]
            else:
                up = 1 if arm == "stored_linear" else 8
                dense = upsample(feats, up)
                s = srp_scores_dense(dense, LT, up)
                p0 = G[int(np.argmax(s))]
                Gf = p0[None, :] + OFF
                est = Gf[int(np.argmax(srp_scores_dense(dense, lag_table(Gf, rx, pairs), up)))]
            err[arm].append(np.linalg.norm(est - p_tx))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{args.n}", flush=True)

    print()
    print("=" * 84)
    print(f'{"arm":16s}{"median":>10s}{"p90":>10s}{"mean":>10s}   what it proves')
    print("=" * 84)
    desc = {"stored_linear": "C1 baseline (linear read of the window)",
            "stored_fft8": "R2 claim (band-limited read of the window)",
            "direct_exact": "no window, no interpolation -> ground truth"}
    res = {}
    for a in ("stored_linear", "stored_fft8", "direct_exact"):
        e = np.array(err[a])
        res[a] = {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
                  "mean": float(np.mean(e)), "n": len(e)}
        print(f'{a:16s}{np.median(e):>10.2f}{np.percentile(e,90):>10.1f}'
              f'{np.mean(e):>10.1f}   {desc[a]}')

    d, f = res["direct_exact"]["median"], res["stored_fft8"]["median"]
    rel = abs(f - d) / d
    print("\n" + "=" * 84); print("VERDICT"); print("=" * 84)
    print(f"  FFT x8 vs direct evaluation: {f:.2f} vs {d:.2f} m  (relative diff {rel*100:.1f}%)")
    if rel < 0.20:
        v = ("VALIDATED — the truncation window is harmless; the sinc-interpolation "
             "gain is real and the C1 SRP baseline was genuinely under-tuned")
    else:
        v = ("NOT VALIDATED — FFT upsampling disagrees with direct evaluation; "
             "R2's gain is a truncation artefact and must be withdrawn")
    print(f"  -> {v}")
    res["verdict"] = {"relative_diff": float(rel), "validated": bool(rel < 0.20),
                      "reading": v}

    op = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
