"""
srp_interpolation_floor.py — PART II / R2: is SRP's ~25 m floor an interpolation
artefact of MY implementation rather than a property of the problem?

R1 (srp_floor_decomposition.py) showed the floor survives with multipath switched off
and at 40 dB SNR, so it is algorithmic. Two algorithmic suspects remain:

    (a) the POSITION grid          -- already ruled out by C1's convergence check
                                      (200/100/50 m -> 28.4/27.4/26.4 m)
    (b) the LAG interpolation      -- srp_scores() reads the stored (6,257) curve at
                                      real-valued lags by LINEAR interpolation

(b) is the untested one, and it matters in a direction that works against this
project: if linear interpolation is the bottleneck, then my SRP baseline is
*under-tuned* and the true classical/learned gap is LARGER than the 5.5x reported in
C1. Honesty requires measuring it.

The correlation curve is band-limited, so the exact continuous curve is recoverable
from its integer-lag samples by sinc (FFT) interpolation. Arms:

    linear_x1     current implementation (baseline)
    fft_x4/x8/x16 curve zero-padded in the frequency domain to 4/8/16x density,
                  then linear interpolation on the dense grid ~ sinc interpolation
    fine_grid     linear_x1 but with a 4x finer position refinement (isolates (a))

Same input array as C1 (`output_final/gcc_test.npz`) — no regeneration, no training.
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
from classical_baseline import C, FS, MAX_LAG, lag_table


def upsample(feats, k):
    """Band-limited (sinc) interpolation of the correlation curves by factor k.

    The GCC curve is the IFFT of a band-limited cross-spectrum, so zero-padding in
    the frequency domain and inverse-transforming recovers the continuous curve at
    k-times denser lags without inventing information.
    """
    if k == 1:
        return feats
    P, T = feats.shape
    Fh = np.fft.rfft(feats, axis=1)
    out = np.fft.irfft(Fh, n=T * k, axis=1) * k
    return out


def srp_scores_dense(dense, lt, k):
    """SRP score using a k-times upsampled curve. `dense` is (P, T*k)."""
    idx = (lt + MAX_LAG) * k                       # lag -> dense sample index
    i0 = np.floor(idx).astype(np.int64)
    w = idx - i0
    ok = (i0 >= 0) & (i0 < dense.shape[1] - 1)
    i0 = np.clip(i0, 0, dense.shape[1] - 2)
    p = np.arange(lt.shape[1])[None, :]
    v = dense[p, i0] * (1.0 - w) + dense[p, i0 + 1] * w
    return np.where(ok, v, 0.0).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--coarse", type=float, default=100.0)
    ap.add_argument("--fine", type=float, default=6.0)
    ap.add_argument("--out", default="artifacts/srp_interp_floor.json")
    args = ap.parse_args()

    rx = np.load(os.path.join(REPO, "output_final/rx_layout.npz"))["rx_pos"]
    d = np.load(os.path.join(REPO, "output_final/gcc_test.npz"))
    feats = d["feats"][:args.n].astype(np.float64)
    y = d["y"][:args.n].astype(np.float64)
    pairs = list(combinations(range(rx.shape[0]), 2))

    R = 8000.0
    ax = np.arange(-R, R + 1e-9, args.coarse); gx, gy = np.meshgrid(ax, ax)
    G = np.c_[gx.ravel(), gy.ravel()]; G = G[np.linalg.norm(G, axis=1) <= R + 1e-9]
    LT = lag_table(G, rx, pairs)

    def run(k, fine):
        off = np.arange(-args.coarse, args.coarse + 1e-9, fine)
        ox, oy = np.meshgrid(off, off); OFF = np.c_[ox.ravel(), oy.ravel()]
        est = np.zeros((len(feats), 2))
        for i in range(len(feats)):
            dense = upsample(feats[i], k)
            s = srp_scores_dense(dense, LT, k)
            p0 = G[int(np.argmax(s))]
            Gf = p0[None, :] + OFF
            est[i] = Gf[int(np.argmax(srp_scores_dense(dense, lag_table(Gf, rx, pairs), k)))]
        return np.linalg.norm(est - y, axis=1)

    ARMS = [("linear_x1", 1, args.fine), ("fft_x4", 4, args.fine),
            ("fft_x8", 8, args.fine), ("fft_x16", 16, args.fine),
            ("fine_grid_x1", 1, args.fine / 4)]
    res = {}
    print(f"n={len(feats)} (same array as C1)   grid {len(G)} coarse pts\n")
    print(f'{"arm":14s}{"lag interp":>12s}{"fine grid":>11s}{"median":>10s}{"p90":>10s}{"mean":>10s}')
    for nm, k, fine in ARMS:
        e = run(k, fine)
        res[nm] = {"upsample": k, "fine_m": fine, "median": float(np.median(e)),
                   "p90": float(np.percentile(e, 90)), "mean": float(np.mean(e)),
                   "n": len(e)}
        print(f'{nm:14s}{("x%d" % k):>12s}{fine:>10.1f}m{np.median(e):>10.2f}'
              f'{np.percentile(e,90):>10.1f}{np.mean(e):>10.1f}', flush=True)

    base = res["linear_x1"]["median"]
    best = min(res[a]["median"] for a in ("fft_x4", "fft_x8", "fft_x16"))
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"  linear (current C1 implementation) : {base:.2f} m")
    print(f"  best band-limited interpolation    : {best:.2f} m   "
          f"({(1-best/base)*100:+.1f}%)")
    print(f"  4x finer position grid only        : {res['fine_grid_x1']['median']:.2f} m")
    if best < base * 0.7:
        v = ("INTERPOLATION-LIMITED — the C1 SRP baseline was under-tuned; "
             "the true classical/learned gap is LARGER than reported")
    elif best < base * 0.95:
        v = "partially interpolation-limited (modest gain)"
    else:
        v = ("NOT interpolation-limited — the floor is intrinsic to the stored GCC "
             "representation, not to how it is read")
    print(f"  -> {v}")
    res["verdict"] = {"linear": base, "best_bandlimited": best,
                      "gain_pct": float((1 - best / base) * 100), "reading": v}

    op = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
