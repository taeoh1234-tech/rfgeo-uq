"""
srp_silent_failure.py — PART III / V2: does SRP+CP fail SILENTLY under forward-model
error, the way our DER pipeline fails silently under channel shift?

`SRP_CP_구조적한계_검증계획_2026-08.md` §2 (L2) makes a sharp, falsifiable claim:

    "수신기 위치가 틀리면 봉우리는 옮겨질 뿐 뭉개지지 않는다"
      -> PSR (peak sharpness)  : unchanged  => no alarm
      -> level-set / region    : unchanged  => region stays small
      -> centre                : biased     => coverage silently collapses

    사전 예측: coverage 급락 + AUROC(-PSR) ~ 0.5
    반증조건 : AUROC >= 0.7 이면 PSR이 경보를 울리므로 L2 거짓

This is exactly the structure of E9/E14/E15 ("confidently wrong"), so if it holds,
the project's diagnostic layer is estimator-agnostic — it is needed whichever point
estimator ships. That is worth knowing before any direction decision.

Design (deployment realism, per the plan)
    perturbation  receiver survey error delta, drawn ONCE per deployment and fixed
    data          generated with the TRUE (displaced) array
    SRP           nominal (wrong) geometry, band-limited reader (R2/R3 corrected)
    CP            Q calibrated at delta = 0 and then FROZEN -- a deployed system
                  cannot recalibrate against ground truth it does not have
    measured      median error / coverage under the frozen Q / PSR / AUROC(-PSR)

Contrast arm: the same samples scored by the classical CONSISTENCY residual, i.e.
how well the estimated position reproduces the observed pairwise lags. That is the
SRP-side analogue of our disagreement flag D, and it is the natural candidate for a
detector if PSR turns out to be blind.
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
from rfgeo.conformal import _finite_sample_quantile as fsq
from classical_baseline import C, FS, MAX_LAG, lag_table, parabolic_peak
from srp_interpolation_floor import upsample, srp_scores_dense

UP = 8


def auroc(pos, neg):
    a = np.concatenate([pos, neg]); r = np.argsort(np.argsort(a)) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gen(cfg, k, seed, rx_true, rx_vel):
    r = np.random.default_rng([seed, 0, k])
    p_tx = _sample_emitter(cfg, r)
    kind = str(r.choice(cfg.waveforms))
    snr = float(r.uniform(*cfg.snr_db_range))
    src = wf.generate_waveform(kind, cfg.n_samples, cfg.fs, r)
    rc, rn = np.random.default_rng([seed, 1, k]), np.random.default_rng([seed, 2, k])
    tdoa, fdoa, *_ = geo.compute_tdoa_fdoa(p_tx, rx_true, rx_vel, cfg.carrier_hz, ref=0)
    X = np.empty((cfg.n_receivers, 2, cfg.n_samples), dtype=np.float32)
    for i in range(cfg.n_receivers):
        s = geo.render_receiver_iq(src, tdoa[i], fdoa[i], cfg.fs)
        s = ch.apply_tdl(s, rc, profile=cfg.tdl_profile,
                         delay_spread_ns=cfg.tdl_delay_spread_ns, fs=cfg.fs)
        s = ch.add_awgn(s, snr, rn)
        X[i, 0] = s.real; X[i, 1] = s.imag
    return gcc_phat_features(X, max_lag=MAX_LAG).astype(np.float64), p_tx[:2]


def solve(F, rx_nom, pairs, G, LT, OFF):
    """SRP estimate + peak-to-sidelobe + lag-consistency residual."""
    dense = upsample(F, UP)
    s = srp_scores_dense(dense, LT, UP)
    p0 = G[int(np.argmax(s))]
    Gf = p0[None, :] + OFF
    ltf = lag_table(Gf, rx_nom, pairs)
    sf = srp_scores_dense(dense, ltf, UP)
    est = Gf[int(np.argmax(sf))]
    med = np.median(s); mad = np.median(np.abs(s - med)) + 1e-12
    psr = (s.max() - med) / mad
    # consistency: observed pairwise peak lags vs those implied by the estimate
    obs = np.array([parabolic_peak(F[p]) for p in range(F.shape[0])])
    imp = lag_table(est[None, :], rx_nom, pairs)[0]
    resid = float(np.sqrt(np.mean((obs - imp) ** 2)))
    return est, float(psr), resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.0, 5.0, 20.0, 50.0, 100.0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--deployments", type=int, default=2)
    ap.add_argument("--seed", type=int, default=45001)
    ap.add_argument("--coarse", type=float, default=100.0)
    ap.add_argument("--fine", type=float, default=6.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default="artifacts/srp_silent_failure.json")
    args = ap.parse_args()
    register_std_dsss("gold")

    cfg = GenConfig(channel_mode="tdl", tdl_profile="TDL-D", tdl_delay_spread_ns=100.0)
    lay = np.load(os.path.join(REPO, "output_final/rx_layout.npz"))
    rx_nom, rx_vel = lay["rx_pos"], lay["rx_vel"]
    pairs = list(combinations(range(cfg.n_receivers), 2))
    R = 8000.0
    ax = np.arange(-R, R + 1e-9, args.coarse); gx, gy = np.meshgrid(ax, ax)
    G = np.c_[gx.ravel(), gy.ravel()]; G = G[np.linalg.norm(G, axis=1) <= R + 1e-9]
    LT = lag_table(G, rx_nom, pairs)
    off = np.arange(-args.coarse, args.coarse + 1e-9, args.fine)
    ox, oy = np.meshgrid(off, off); OFF = np.c_[ox.ravel(), oy.ravel()]

    # Q frozen from the corrected-reader calibration set (deployment realism)
    Q = float(fsq(np.load(os.path.join(REPO, "artifacts/"
                                       "classical_corrected_persample.npz"))["err_calib"],
                  args.alpha))
    print(f"frozen conformal radius Q(alpha={args.alpha}) = {Q:.2f} m "
          f"(calibrated at delta=0, never updated)\n")

    store, res = {}, {}
    print(f'{"delta":>7s}{"med err":>10s}{"coverage":>10s}{"PSR mean":>10s}'
          f'{"resid":>9s}{"AUROC(-PSR)":>13s}{"AUROC(resid)":>14s}')
    for dl in args.deltas:
        E, P, RS = [], [], []
        for dep in range(args.deployments):
            r = np.random.default_rng([args.seed, 777, int(dl), dep])
            rx_true = rx_nom.copy()
            if dl > 0:
                rx_true[:, :2] = rx_nom[:, :2] + r.normal(0, dl, (cfg.n_receivers, 2))
            for k in range(args.n):
                F, y = gen(cfg, dep * 100000 + k, args.seed, rx_true, rx_vel)
                est, psr, resid = solve(F, rx_nom, pairs, G, LT, OFF)
                E.append(np.linalg.norm(est - y)); P.append(psr); RS.append(resid)
        E, P, RS = np.array(E), np.array(P), np.array(RS)
        store[dl] = (E, P, RS)
        a_psr = auroc(-P, -store[0.0][1]) if dl > 0 else float("nan")
        a_res = auroc(RS, store[0.0][2]) if dl > 0 else float("nan")
        res[str(dl)] = {"delta_m": dl, "median_err": float(np.median(E)),
                        "coverage_frozen_Q": float(np.mean(E <= Q)),
                        "psr_mean": float(P.mean()), "psr_median": float(np.median(P)),
                        "resid_median": float(np.median(RS)),
                        "auroc_neg_psr": a_psr, "auroc_resid": a_res, "n": int(len(E))}
        d = res[str(dl)]
        print(f'{dl:>7.0f}{d["median_err"]:>10.1f}{d["coverage_frozen_Q"]:>10.3f}'
              f'{d["psr_mean"]:>10.2f}{d["resid_median"]:>9.3f}'
              f'{a_psr:>13.3f}{a_res:>14.3f}', flush=True)

    print("\n" + "=" * 96)
    print("사전등록 판정 (L2: coverage 급락 + AUROC(-PSR) ~ 0.5 이면 지지 / >=0.7 이면 기각)")
    print("=" * 96)
    worst = max(args.deltas)
    cov0, covw = res["0.0"]["coverage_frozen_Q"], res[str(worst)]["coverage_frozen_Q"]
    a_psr_w = res[str(worst)]["auroc_neg_psr"]
    a_res_w = res[str(worst)]["auroc_resid"]
    print(f"  coverage {cov0:.3f} (delta=0)  ->  {covw:.3f} (delta={worst:.0f} m)")
    print(f"  AUROC(-PSR)              = {a_psr_w:.3f}")
    print(f"  AUROC(consistency resid) = {a_res_w:.3f}   <- the SRP-side analogue of D")
    if covw < cov0 - 0.10 and a_psr_w < 0.65:
        v = ("L2 지지 — 커버리지는 무너지는데 PSR은 조용하다. "
             "SRP+CP도 조용한 실패를 하며, 진단 계층은 추정기 무관하게 필요하다")
    elif a_psr_w >= 0.7:
        v = "L2 기각 — PSR이 모델 오차를 탐지한다"
    else:
        v = "중간 — 커버리지 붕괴가 약하거나 PSR이 부분적으로 반응"
    print(f"  -> {v}")
    if a_res_w >= 0.7 > a_psr_w:
        print("  🎯 그리고 일관성 잔차가 PSR이 놓친 것을 잡는다 "
              "= SRP 쪽에도 '두 관점의 불일치' 탐지기가 성립한다")

    # the single-delta verdict above can hide the structure: report it delta-resolved
    # with the SAME pre-registered thresholds, since the plan's prediction was about a
    # PATTERN (coverage collapses while PSR stays ~0.5), not about one level.
    print("\n  delta별 판정 (동일 임계, 사후 조정 없음):")
    print(f'  {"delta":>7s}{"coverage":>10s}{"AUROC(-PSR)":>13s}{"AUROC(resid)":>14s}   판정')
    per = {}
    for dl in args.deltas:
        if dl == 0:
            continue
        d = res[str(dl)]
        blind = d["auroc_neg_psr"] < 0.65
        broke = d["coverage_frozen_Q"] < cov0 - 0.10
        lab = ("L2 지지 (조용한 실패)" if broke and blind else
               "PSR 탐지" if d["auroc_neg_psr"] >= 0.7 else
               "커버리지 유지" if not broke else "중간")
        per[str(dl)] = lab
        print(f'  {dl:>7.0f}{d["coverage_frozen_Q"]:>10.3f}{d["auroc_neg_psr"]:>13.3f}'
              f'{d["auroc_resid"]:>14.3f}   {lab}')
    res["verdict"] = {"coverage_drop": float(cov0 - covw), "auroc_neg_psr": a_psr_w,
                      "auroc_resid": a_res_w, "frozen_Q_m": Q, "reading": v,
                      "per_delta": per}

    op = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(res, open(op, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
