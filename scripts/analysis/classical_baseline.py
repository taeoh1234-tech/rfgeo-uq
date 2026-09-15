"""
classical_baseline.py — §15-12: classical TDOA localization on the SAME inputs the
CNN sees, so "why learn at all?" has a measured answer.

Fairness design
---------------
Both classical estimators consume the EXACT GCC-PHAT curves stored in
output_final/gcc_test.npz — the same array the trained network is fed. Nothing is
re-generated, no extra information is given to either side. Receiver geometry is
known to the classical methods (as it must be) and is also implicitly available to
the network through training.

  SRP-PHAT       grid search over emitter positions; for each candidate the expected
                 pairwise lags are read off the correlation curves (linearly
                 interpolated) and summed. Uses the FULL correlation surface, which
                 is the fairest counterpart to a CNN that also sees the whole curve.
  peak + Chan-Ho argmax lag per pair with parabolic sub-sample interpolation, then
                 the closed-form two-stage WLS of Chan & Ho (1994).

Sign convention (verified against rfgeo.geometry):
  tdoa[i] = (r_i - r_ref)/c and render_receiver_iq delays receiver i by tdoa[i]*fs,
  so the GCC-PHAT peak of pair (i,j) sits at lag = (r_i - r_j)/c * fs samples.

GATE (implementation validity): the SRP score evaluated at the TRUE emitter position
must sit in the extreme upper tail of the grid score distribution. If the sign or the
index mapping were wrong this fails immediately.
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
            return up
    raise RuntimeError("rfgeo not found")

REPO = _add_paths()
C = 299792458.0
FS = 2e6
MAX_LAG = 128


def lag_table(grid, rx, pairs):
    """Expected GCC lag (in samples, real-valued) for every grid point x pair."""
    d = np.linalg.norm(grid[:, None, :] - rx[None, :, :2], axis=2)      # (G, M)
    return np.stack([(d[:, i] - d[:, j]) / C * FS for i, j in pairs], axis=1)


def srp_scores(feats, lt):
    """feats (P, T) for ONE sample; lt (G, P) expected lags -> (G,) SRP score."""
    idx = lt + MAX_LAG
    i0 = np.floor(idx).astype(np.int64)
    w = idx - i0
    ok = (i0 >= 0) & (i0 < feats.shape[1] - 1)
    i0 = np.clip(i0, 0, feats.shape[1] - 2)
    p = np.arange(lt.shape[1])[None, :]
    v = feats[p, i0] * (1.0 - w) + feats[p, i0 + 1] * w
    return np.where(ok, v, 0.0).sum(axis=1)


def parabolic_peak(curve):
    """Integer argmax + parabolic sub-sample refinement -> lag in samples."""
    k = int(np.argmax(curve))
    if 0 < k < len(curve) - 1:
        ym, y0, yp = curve[k - 1], curve[k], curve[k + 1]
        den = ym - 2 * y0 + yp
        d = 0.5 * (ym - yp) / den if abs(den) > 1e-12 else 0.0
        d = float(np.clip(d, -1, 1))
    else:
        d = 0.0
    return (k + d) - MAX_LAG


def chan_ho(rd, rx, ref=0):
    """
    Chan & Ho (1994) two-stage closed form, 2-D.
    rd[i] = r_i - r_ref for i != ref (metres). Returns (x, y) or None.
    """
    others = [i for i in range(len(rx)) if i != ref]
    x0 = rx[ref, :2]
    S = rx[np.array(others), :2] - x0                      # (3,2) shifted so ref at origin
    r = np.asarray([rd[i] for i in others], float)         # (3,)
    K = (S ** 2).sum(1)
    Ga = -np.c_[S, r]                                      # (3,3)
    h = 0.5 * (r ** 2 - K)
    try:
        th = np.linalg.solve(Ga, h)                        # exactly determined for M=4
    except np.linalg.LinAlgError:
        return None
    # stage 2: impose r_ref^2 = x^2 + y^2
    if th[2] <= 0:
        return th[:2] + x0
    B2 = np.diag([2 * th[0], 2 * th[1], 2 * th[2]])
    try:
        Ga2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        h2 = np.array([th[0] ** 2, th[1] ** 2, th[2] ** 2])
        W2 = np.linalg.inv(B2 @ B2)
        th2 = np.linalg.solve(Ga2.T @ W2 @ Ga2, Ga2.T @ W2 @ h2)
        xy = np.sign(th[:2]) * np.sqrt(np.abs(th2))
        if not np.all(np.isfinite(xy)):
            return th[:2] + x0
        # keep stage-2 only if it does not move the estimate absurdly far
        if np.linalg.norm(xy - th[:2]) > 3 * np.linalg.norm(th[:2]) + 1e3:
            return th[:2] + x0
        return xy + x0
    except np.linalg.LinAlgError:
        return th[:2] + x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/classical_baseline.json")
    ap.add_argument("--coarse", type=float, default=100.0, help="coarse grid step (m)")
    ap.add_argument("--fine", type=float, default=6.0, help="fine grid step (m)")
    ap.add_argument("--n", type=int, default=0, help="limit samples (0 = all)")
    args = ap.parse_args()

    rx = np.load(os.path.join(REPO, "output_final/rx_layout.npz"))["rx_pos"]
    d = np.load(os.path.join(REPO, "output_final/gcc_test.npz"))
    feats = d["feats"].astype(np.float64); y = d["y"].astype(np.float64)
    meta = json.load(open(os.path.join(REPO, "output_final/meta_test.json")))
    snr = np.array([m["snr_db"] for m in meta])
    if args.n:
        feats, y, snr = feats[:args.n], y[:args.n], snr[:args.n]
    pairs = list(combinations(range(rx.shape[0]), 2))
    N = len(feats)
    print(f"receivers {rx.shape} | pairs {pairs} | test n={N}")

    # ---- coarse grid over the emitter disk ----
    R = 8000.0
    ax = np.arange(-R, R + 1e-9, args.coarse)
    gx, gy = np.meshgrid(ax, ax)
    G = np.c_[gx.ravel(), gy.ravel()]
    G = G[np.linalg.norm(G, axis=1) <= R + 1e-9]
    LT = lag_table(G, rx, pairs)
    print(f"coarse grid {len(G)} pts @ {args.coarse:g} m")

    # ---- GATE: SRP at the TRUE position must be extreme ----
    pct = []
    for i in range(min(200, N)):
        s = srp_scores(feats[i], LT)
        st = srp_scores(feats[i], lag_table(y[i][None, :], rx, pairs))[0]
        pct.append((s < st).mean())
    pct = np.array(pct)
    print(f"[GATE] SRP score at TRUE position: median percentile {np.median(pct)*100:.2f}%"
          f" | >99% in {np.mean(pct>0.99)*100:.0f}% of samples"
          f" -> {'PASS' if np.median(pct) > 0.99 else 'FAIL (check sign/index mapping)'}")
    if np.median(pct) <= 0.99:
        print("  aborting: the geometry->lag mapping is wrong, results would be meaningless")
        sys.exit(1)

    # ---- SRP-PHAT (coarse then local refine) + peak/Chan-Ho ----
    srp = np.zeros((N, 2)); ch = np.zeros((N, 2)); ch_ok = np.zeros(N, bool)
    off = np.arange(-args.coarse, args.coarse + 1e-9, args.fine)
    ox, oy = np.meshgrid(off, off); OFF = np.c_[ox.ravel(), oy.ravel()]
    for i in range(N):
        s = srp_scores(feats[i], LT)
        p0 = G[int(np.argmax(s))]
        Gf = p0[None, :] + OFF
        sf = srp_scores(feats[i], lag_table(Gf, rx, pairs))
        srp[i] = Gf[int(np.argmax(sf))]
        lags = np.array([parabolic_peak(feats[i, p]) for p in range(len(pairs))])
        rd = {}
        for k, (a, b) in enumerate(pairs):
            if a == 0:
                rd[b] = -lags[k] * C / FS          # r_b - r_0
        est = chan_ho(rd, rx, ref=0)
        if est is not None and np.all(np.isfinite(est)):
            ch[i] = est; ch_ok[i] = True
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{N}", flush=True)

    e_srp = np.linalg.norm(srp - y, axis=1)
    e_ch = np.linalg.norm(ch - y, axis=1)
    ours = np.load(os.path.join(REPO, "output_final_der/predictions.npz"))
    e_ours = ours["err_m"][:N]

    def stat(e, m=None):
        e = e if m is None else e[m]
        return dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                    mean=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())))
    res = {"n": N, "chan_ho_valid": float(ch_ok.mean()),
           "gate_true_pos_percentile_median": float(np.median(pct)),
           "overall": {"srp_phat": stat(e_srp), "chan_ho": stat(e_ch[ch_ok]),
                       "ours_der": stat(e_ours)}, "by_snr": {}}

    print("\n" + "=" * 84)
    print("ACCURACY — same GCC-PHAT input, no training for the classical methods")
    print("=" * 84)
    print(f'{"method":22s}{"median":>10s}{"p90":>10s}{"mean":>10s}{"RMSE":>10s}')
    for nm, e in [("SRP-PHAT (grid)", e_srp), ("peak + Chan-Ho", e_ch[ch_ok]),
                  ("ours: CNN+NIW-DER", e_ours)]:
        s = stat(e)
        print(f'{nm:22s}{s["median"]:>10.1f}{s["p90"]:>10.1f}{s["mean"]:>10.1f}{s["rmse"]:>10.1f}')
    print(f'  (Chan-Ho produced a finite solution on {ch_ok.mean()*100:.1f}% of samples)')

    print("\n" + "=" * 84); print("BY SNR (median error, m)"); print("=" * 84)
    print(f'{"bin":>10s}{"n":>6s}{"SRP-PHAT":>11s}{"Chan-Ho":>10s}{"ours":>9s}{"ours/SRP":>10s}')
    for lo, hi in [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]:
        m = (snr >= lo) & (snr < hi)
        if m.sum() == 0:
            continue
        a = np.median(e_srp[m]); b = np.median(e_ch[m & ch_ok]); c = np.median(e_ours[m])
        res["by_snr"][f"[{lo},{hi})"] = {"n": int(m.sum()), "srp": float(a),
                                        "chan_ho": float(b), "ours": float(c)}
        print(f'{f"[{lo},{hi})":>10s}{m.sum():>6d}{a:>11.1f}{b:>10.1f}{c:>9.1f}{c/a:>10.2f}')

    op = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    np.savez_compressed(op.replace(".json", "_persample.npz"),
                        srp=srp, chan_ho=ch, chan_ho_ok=ch_ok, truth=y, snr=snr,
                        err_srp=e_srp, err_chan_ho=e_ch, err_ours=e_ours)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
