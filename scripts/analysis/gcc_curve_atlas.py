r"""
gcc_curve_atlas.py — GCC 곡선에서 잴 수 있는 것 전수 조사 + 기하 한계 + 정직도 분해.

세 부분
-------
PART A  기하 한계 : fs=2 MHz, 반경 8 km 배치에서 쌍별 lag 이 물리적으로 어디까지 갈 수 있나.
                   그리고 쌍 사이의 lag 간격이 CNN 수용영역(65 샘플)을 넘는가 (CLAUDE.md §15-35).
PART B  곡선 통계 확장 : 기존 3종(featRMS·pk/far·반치폭) 외에 9종을 더 재고,
                   (b1) 조건별 표본단위 Spearman(통계, 오차)
                   (b2) 학습분포 회귀에 넣었을 때의 증분 R^2
PART C  정직도 분해 : 실제 epi배 / 오차배  =  (규칙 준수도) x (규칙 타당도)
                   -> "역전"과 "과신"이 왜 다른 것을 재는지.

모든 통계는 **라벨 불필요** (closure 포함). lag_err 만 진단용으로 기하를 쓴다.

    python scripts/data/gcc_curve_atlas.py
"""
from __future__ import annotations
import functools, io, json, os, sys
from itertools import combinations
import numpy as np
import torch
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)
from rfgeo import dataset as ds, waveform as wf                    # noqa: E402
from rfgeo.evidential import niw_uncertainties                     # noqa: E402
from train_der import DERModel                                     # noqa: E402
from peak_decomposition import upsample, true_lags, from_npz, gen, SCALE, UP, ML, BAR  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "gcc_curve_atlas.json")
PAIRS = list(combinations(range(4), 2))
C_LIGHT = 299_792_458.0
A = np.zeros((len(PAIRS), 3))
for _r, (_i, _j) in enumerate(PAIRS):
    if _i > 0:
        A[_r, _i - 1] = 1.0
    if _j > 0:
        A[_r, _j - 1] = -1.0
A_PINV = np.linalg.pinv(A)

# 기존 3종 + 확장 9종
BASE = ("feat_rms", "pk_far", "halfwidth")
EXT = ("curvature", "entropy", "energy_conc", "skew", "psr",
       "n_sub", "sub_h", "closure", "pair_cv")


# ================= PART A =================
def geometry_limits(rx, fs, cfg, n=20000):
    seps = {}
    for (i, j) in PAIRS:
        seps[(i, j)] = float(np.linalg.norm(rx[i] - rx[j]))
    max_sep = max(seps.values())
    theo = max_sep / C_LIGHT * fs
    rng = np.random.default_rng(1)
    r = cfg.area_radius_m * np.sqrt(rng.uniform(size=n))
    th = rng.uniform(0, 2 * np.pi, size=n)
    P = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(n)], axis=1)
    R = np.linalg.norm(rx[None, :, :] - P[:, None, :], axis=2)
    L = np.stack([(R[:, i] - R[:, j]) / C_LIGHT * fs for (i, j) in PAIRS], axis=1)
    span = L.max(axis=1) - L.min(axis=1)          # 한 표본 안에서 쌍 간 lag 간격
    return dict(pair_separations_m={"%d-%d" % k: v for k, v in seps.items()},
                max_separation_m=max_sep,
                theoretical_max_lag_samples=float(theo),
                theoretical_max_lag_m=float(max_sep),
                empirical_abs_lag_p100=float(np.abs(L).max()),
                empirical_abs_lag_p99=float(np.percentile(np.abs(L), 99)),
                empirical_abs_lag_median=float(np.median(np.abs(L))),
                pair_span_median=float(np.median(span)),
                pair_span_p90=float(np.percentile(span, 90)),
                pair_span_max=float(span.max()),
                frac_span_gt_65=float((span > 65).mean()),
                window_half_lag=ML)


# ================= PART B =================
def curve_stats_ext(feats):
    P, L = feats.shape
    acc = {k: [] for k in ("pk_far", "halfwidth") + EXT[:-2]}
    lags, pkh = [], []
    for p in range(P):
        cu = upsample(feats[p])
        n = len(cu)
        k = int(np.argmax(cu))
        pk = float(cu[k])
        lags.append(k / UP - ML)
        pkh.append(pk)
        # 반치폭
        h = pk / 2.0
        a, b = k, k
        while a > 0 and cu[a] > h:
            a -= 1
        while b < n - 1 and cu[b] > h:
            b += 1
        acc["halfwidth"].append((b - a) / UP)
        # pk/far
        lo, hi = max(0, k - 8 * UP), min(n, k + 8 * UP + 1)
        far = np.concatenate([cu[:lo], cu[hi:]])
        rms = float(np.sqrt(np.mean(far ** 2))) if far.size else 1e-12
        acc["pk_far"].append(pk / max(rms, 1e-12))
        acc["psr"].append(pk / (float(np.max(np.abs(far))) + 1e-12) if far.size else np.nan)
        # 곡률: -c''(k)/c(k)  (업샘플 격자 -> 원 샘플 단위로 환산)
        if 0 < k < n - 1:
            cur = -(cu[k - 1] - 2 * cu[k] + cu[k + 1]) * (UP ** 2) / max(pk, 1e-12)
        else:
            cur = np.nan
        acc["curvature"].append(cur)
        # 엔트로피 (곡선을 확률분포로)
        q = np.maximum(cu, 0.0) ** 2
        s = q.sum()
        if s > 0:
            q = q / s
            acc["entropy"].append(float(-(q[q > 0] * np.log(q[q > 0])).sum()))
        else:
            acc["entropy"].append(np.nan)
        # 에너지 집중도: 봉우리 +-4 샘플 / 창 전체
        l2, h2 = max(0, k - 4 * UP), min(n, k + 4 * UP + 1)
        tot = float(np.sum(cu ** 2))
        acc["energy_conc"].append(float(np.sum(cu[l2:h2] ** 2)) / tot if tot > 0 else np.nan)
        # 좌우 비대칭 (봉우리 +-16 샘플)
        w = 16 * UP
        lft = cu[max(0, k - w):k]
        rgt = cu[k + 1:min(n, k + w + 1)]
        m = min(len(lft), len(rgt))
        if m > 0:
            lf, rg = lft[-m:][::-1], rgt[:m]
            den = float(np.sum(np.abs(lf) + np.abs(rg)))
            acc["skew"].append(float(np.sum(rg - lf)) / den if den > 0 else np.nan)
        else:
            acc["skew"].append(np.nan)
        # 부봉우리
        g = 2 * UP
        mask = np.ones(n, dtype=bool)
        mask[max(0, k - g):min(n, k + g + 1)] = False
        o = np.where(mask, cu, -np.inf)
        loc = (o[1:-1] > o[:-2]) & (o[1:-1] > o[2:]) & (o[1:-1] > 0.5 * pk)
        acc.setdefault("n_sub", []).append(int(loc.sum()))
        rest = cu[mask]
        acc.setdefault("sub_h", []).append(float(np.max(rest)) / pk if rest.size else np.nan)
    lags = np.asarray(lags)
    th = A_PINV @ lags
    clo = float(np.linalg.norm(A @ th - lags) / np.sqrt(3))
    pkh = np.asarray(pkh)
    out = {k: float(np.nanmedian(v)) for k, v in acc.items()}
    out["closure"] = clo
    out["pair_cv"] = float(np.std(pkh) / max(np.mean(pkh), 1e-12))
    out["_lags"] = lags
    return out


def batch_ext(F, Y, rx, fs):
    n = len(F)
    keys = ("pk_far", "halfwidth") + EXT
    o = {k: np.empty(n) for k in keys}
    o["lag_err"] = np.empty(n)
    for i in range(n):
        s = curve_stats_ext(F[i])
        for k in keys:
            o[k][i] = s[k]
        o["lag_err"][i] = float(np.median(np.abs(s["_lags"] - true_lags(Y[i], rx, fs))))
    o["feat_rms"] = np.sqrt(np.mean(F.astype(np.float64) ** 2, axis=(1, 2)))
    return o


CASES = [
    ("ref (학습분포)", None),
    ("sco_4 (NULL)", lambda: from_npz("sco_4")),
    ("delay_3", lambda: from_npz("delay_3")),
    ("occ 0.05", lambda: gen("_at_occ05", functools.partial(wf.gen_ofdm, used_frac=0.05))),
    ("TDL-C", lambda: gen(prof="TDL-C")),
    ("kf_3", lambda: from_npz("kf_3")),
    ("snr_1", lambda: from_npz("snr_1")),
]


def main():
    cfg = ds.GenConfig()
    rx, _ = ds._place_receivers(cfg, np.random.default_rng(0))
    fs = cfg.fs

    print(BAR)
    print("PART A — 기하 한계 (fs=%.1f MHz, 1 샘플 = %.1f m, 배치 반경 %.0f m)"
          % (fs / 1e6, C_LIGHT / fs, cfg.rx_layout_radius_m))
    print(BAR)
    G = geometry_limits(rx, fs, cfg)
    for k, v in G["pair_separations_m"].items():
        print("  쌍 %s 수신기 간격 %8.0f m = %7.2f 샘플" % (k, v, v / C_LIGHT * fs))
    print("  ---")
    print("  이론 최대 |lag|      : %8.2f 샘플  (= 최대 수신기 간격 %.0f m / c x fs)"
          % (G["theoretical_max_lag_samples"], G["max_separation_m"]))
    print("  실측 최대 |lag|      : %8.2f 샘플  (방사원 2만개 추첨)"
          % G["empirical_abs_lag_p100"])
    print("  실측 |lag| p99 / 중앙: %8.2f / %.2f 샘플"
          % (G["empirical_abs_lag_p99"], G["empirical_abs_lag_median"]))
    print("  GCC 창 반폭          : %8d 샘플  -> 여유 %.1f%%"
          % (G["window_half_lag"], 100 * (ML / G["theoretical_max_lag_samples"] - 1)))
    print("  ---")
    print("  한 표본 안 쌍 간 lag 간격  중앙 %.1f / p90 %.1f / 최대 %.1f 샘플"
          % (G["pair_span_median"], G["pair_span_p90"], G["pair_span_max"]))
    print("  그 간격이 CNN 최종층 수용영역(65 샘플)을 넘는 비율: %.1f%%"
          % (100 * G["frac_span_gt_65"]))

    ck = torch.load(os.path.join(ROOT, "output_final_der", "der_model.pt"), weights_only=False)
    mean, std = ck["mean"], ck["std"]
    model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
    model.load_state_dict(ck["model"])
    model.eval()

    @torch.no_grad()
    def pred(Z):
        mu, L, nu = model(torch.from_numpy(Z))
        al, ep = niw_uncertainties(L, nu)
        return mu.numpy(), ep.numpy()

    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    REF = (dt["feats"].astype(np.float32), dt["y"].astype(np.float32))

    D = {}
    for nm, fn in CASES:
        F, Y = REF if fn is None else fn()
        S = batch_ext(F, Y, rx, fs)
        p, ep = pred((F - mean) / std)
        yn = Y / SCALE
        S["err"] = np.sqrt(((p - yn) ** 2).sum(1)) * SCALE
        S["epi"] = np.sqrt(np.trace(ep, axis1=1, axis2=2)) * SCALE
        D[nm] = S
        print("  수집 %-16s n=%d" % (nm, len(F)))

    ALL = BASE + EXT
    print("\n" + BAR)
    print("PART B-0 — 각 통계의 중앙값 (기준선 대비 배율)")
    print(BAR)
    print("  %-16s" % "조건" + "".join("%12s" % s for s in ALL))
    r0 = {s: float(np.nanmedian(D["ref (학습분포)"][s])) for s in ALL}
    MED = {}
    for nm, _ in CASES:
        MED[nm] = {s: float(np.nanmedian(D[nm][s])) for s in ALL}
        print("  %-16s" % nm + "".join("%12.3f" % (MED[nm][s] / r0[s] if abs(r0[s]) > 1e-12
                                                   else np.nan) for s in ALL))

    print("\n" + BAR)
    print("PART B-1 — 표본단위 Spearman( 통계, 실제 오차 )   조건 내부")
    print(BAR)
    print("  %-16s" % "조건" + "".join("%12s" % s for s in ALL))
    SP = {}
    for nm, _ in CASES:
        d = D[nm]
        SP[nm] = {}
        for s in ALL:
            m = np.isfinite(d[s]) & np.isfinite(d["err"])
            SP[nm][s] = float(sps.spearmanr(d[s][m], d["err"][m]).statistic)
        print("  %-16s" % nm + "".join("%12.3f" % SP[nm][s] for s in ALL))

    print("\n" + BAR)
    print("PART B-2 — 학습분포에서 log(epi) 회귀: 기존 3종에 하나씩 더했을 때 증분 R^2")
    print(BAR)
    ref = D["ref (학습분포)"]
    y = np.log(ref["epi"])

    def fit_r2(cols):
        X = np.column_stack([np.ones(len(y))] +
                            [np.log(np.maximum(ref[c], 1e-12)) if np.nanmin(ref[c]) > 0
                             else np.asarray(ref[c], dtype=float) for c in cols])
        m = np.all(np.isfinite(X), axis=1)
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        return float(1.0 - np.var(y[m] - X[m] @ beta) / np.var(y[m])), int(m.sum())

    base_r2, nb = fit_r2(list(BASE))
    print("  기준 (featRMS, pk/far, 반치폭)          R^2 = %.4f   (n=%d)" % (base_r2, nb))
    INC = {}
    for s in EXT:
        r2, nn = fit_r2(list(BASE) + [s])
        INC[s] = dict(r2=r2, delta=r2 - base_r2, n=nn)
        print("    + %-14s  R^2 = %.4f   증분 %+.4f" % (s, r2, r2 - base_r2))
    r2_all, na = fit_r2(list(ALL))
    print("  전부 (12종)                              R^2 = %.4f   증분 %+.4f"
          % (r2_all, r2_all - base_r2))

    print("\n" + BAR)
    print("PART C — 정직도 분해   실제epi배/오차배 = (규칙 준수도) x (규칙 타당도)")
    print(BAR)
    MT = json.load(io.open(os.path.join(ROOT, "artifacts",
                                        "epi_mapping_transfer.json"), encoding="utf-8"))["result"]
    key = "ref (학습분포)"
    e0, p0, r0e = MT[key]["actual"], MT[key]["pred"], MT[key]["err"]
    print("  %-16s %9s %9s %9s | %11s %11s %11s"
          % ("조건", "오차배", "예측epi배", "실제epi배", "규칙준수도", "규칙타당도", "정직도"))
    DEC = {}
    for nm in MT:
        er = MT[nm]["err"] / r0e
        pr = MT[nm]["pred"] / p0
        ar = MT[nm]["actual"] / e0
        DEC[nm] = dict(err_ratio=er, pred_ratio=pr, act_ratio=ar,
                       compliance=ar / pr, validity=pr / er, honesty=ar / er)
        t = DEC[nm]
        print("  %-16s %9.2f %9.2f %9.2f | %11.3f %11.3f %11.3f"
              % (nm, er, pr, ar, t["compliance"], t["validity"], t["honesty"]))

    print("\n  [읽는 법] 규칙준수도<1 = head 가 자기 규칙보다 낮게 말함")
    print("            규칙타당도<1 = 규칙 자체가 오차 증가를 못 따라감")
    print("            정직도<1     = 불확실성이 오차 증가를 못 따라감 (둘의 곱)")
    print("\n  역전 여부(실제epi배<1) vs 정직도 순위:")
    for nm in sorted(DEC, key=lambda k: DEC[k]["honesty"]):
        t = DEC[nm]
        tag = "역전" if t["act_ratio"] < 1.0 else "상승"
        print("    %-16s 정직도 %6.3f   실제epi배 %5.2f (%s)   준수 %5.3f / 타당 %5.3f"
              % (nm, t["honesty"], t["act_ratio"], tag, t["compliance"], t["validity"]))

    out = dict(_note="GCC 곡선 통계 전수 + 기하 한계 + 정직도 분해.",
               geometry=G, median=MED, ref_median=r0,
               spearman_vs_err=SP, base_r2=base_r2, incremental=INC, r2_all=r2_all,
               decomposition=DEC)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
