r"""
peak_decomposition.py — 봉우리 **위치** 와 **모양** 을 분리해 재고, 어느 쪽이 epistemic 을 움직이나.

사전등록: `prereg/PEAK_DECOMP_PREREG_2026-08-26.md` (실행 전 기록)

배경
----
featRMS / 분해능 / Brms 세 설명이 전부 실패했다. 남은 가설은 연구자 제안 —
"오차는 봉우리 **위치**가, epistemic 은 봉우리 **모양**이 결정한다".

주의: 기존 `curve_stats` 의 halfwidth 는 정수 샘플이라 서브샘플 확대를 못 본다.
여기서는 **대역제한 FFT x8 업샘플** 판독으로 교체한다.

    python scripts/data/peak_decomposition.py
"""
from __future__ import annotations
import functools, io, json, os, sys
from itertools import combinations
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)
from rfgeo import dataset as ds, waveform as wf                    # noqa: E402
from rfgeo.gcc import gcc_phat_features                            # noqa: E402
from rfgeo.evidential import niw_uncertainties                     # noqa: E402
from train_der import DERModel                                     # noqa: E402

ML, SCALE, ALPHA, UP = 128, 5000.0, 0.10, 8
SEEDS = (777, 20260628, 20261628)
N_GEN = 200
OOD = os.path.join(ROOT, "output_final_ood")
OUT = os.path.join(ROOT, "artifacts", "peak_decomposition.json")
PAIRS = list(combinations(range(4), 2))
RNG = np.random.default_rng(0)
BAR = "=" * 118


# ---------------- 대역제한 업샘플 판독 ----------------
def upsample(c, U=UP):
    """실수 대역제한 곡선을 U배 FFT 업샘플. 원 표본은 정확히 보존된다."""
    L = len(c)
    C = np.fft.rfft(c)
    Cp = np.zeros(U * L // 2 + 1, dtype=complex)
    Cp[:len(C)] = C
    if L % 2 == 0:
        Cp[len(C) - 1] *= 0.5
    return np.fft.irfft(Cp, U * L) * U


def peak_stats_one(feats):
    """(P,L) 한 표본 -> 쌍별 lag / 서브샘플 반치폭 / pk-far / pk-near"""
    P, L = feats.shape
    lags, hw, ptf, ptn = [], [], [], []
    for p in range(P):
        cu = upsample(feats[p])
        n = len(cu)
        k = int(np.argmax(cu))
        pk = float(cu[k])
        lags.append(k / UP - ML)
        h = pk / 2.0
        a = k
        while a > 0 and cu[a] > h:
            a -= 1
        b = k
        while b < n - 1 and cu[b] > h:
            b += 1
        hw.append((b - a) / UP)
        lo, hi = max(0, k - 8 * UP), min(n, k + 8 * UP + 1)
        far = np.concatenate([cu[:lo], cu[hi:]])
        rms = float(np.sqrt(np.mean(far ** 2))) if far.size else 1e-12
        ptf.append(pk / max(rms, 1e-12))
        n1 = cu[max(0, k - 8 * UP):max(0, k - 2 * UP)]
        n2 = cu[min(n, k + 2 * UP):min(n, k + 8 * UP)]
        nb = np.concatenate([n1, n2])
        ptn.append(pk / (float(np.max(nb)) + 1e-12) if nb.size else np.nan)
    return (np.asarray(lags), float(np.median(hw)),
            float(np.median(ptf)), float(np.nanmedian(ptn)))


def true_lags(y, rx, fs):
    p3 = np.array([y[0], y[1], 0.0])
    r = np.linalg.norm(rx - p3[None, :], axis=1)
    toa = r / 299_792_458.0
    return np.array([(toa[i] - toa[j]) * fs for (i, j) in PAIRS])


def batch_peak(F, Y, rx, fs):
    n = len(F)
    out = {k: np.empty(n) for k in ("lag_err", "halfwidth_sub", "pk_far", "pk_near")}
    for i in range(n):
        lg, h, f_, nr = peak_stats_one(F[i])
        out["lag_err"][i] = float(np.median(np.abs(lg - true_lags(Y[i], rx, fs))))
        out["halfwidth_sub"][i] = h
        out["pk_far"][i] = f_
        out["pk_near"][i] = nr
    return out


# ---------------- 조건 ----------------
def from_npz(name):
    d = np.load(os.path.join(OOD, "shift_%s.npz" % name))
    return d["feats"].astype(np.float32), d["y"].astype(np.float32)


def gen(gname=None, genfn=None, prof="TDL-D"):
    F, Y = [], []
    for sd in SEEDS:
        kw = {}
        if gname is not None:
            wf.WAVEFORM_GENERATORS[gname] = genfn
            kw["waveforms"] = (gname,)
        cfg = ds.GenConfig(n_train=1, n_calib=1, n_test=N_GEN, seed=sd,
                           channel_mode="tdl", tdl_profile=prof, **kw)
        X, y, _ = ds.generate_dataset(cfg)["test"]
        F.append(np.stack([gcc_phat_features(X[i], max_lag=ML) for i in range(len(X))]))
        Y.append(y)
    return np.concatenate(F).astype(np.float32), np.concatenate(Y).astype(np.float32)


CASES = [
    ("ref (학습분포)", None),
    ("sco_4 (NULL)", lambda: from_npz("sco_4")),
    ("delay_1", lambda: from_npz("delay_1")),
    ("delay_2", lambda: from_npz("delay_2")),
    ("delay_3", lambda: from_npz("delay_3")),
    ("occ 0.20", lambda: gen("_pd_occ20", functools.partial(wf.gen_ofdm, used_frac=0.20))),
    ("occ 0.05", lambda: gen("_pd_occ05", functools.partial(wf.gen_ofdm, used_frac=0.05))),
    ("TDL-A", lambda: gen(prof="TDL-A")),
    ("TDL-C", lambda: gen(prof="TDL-C")),
    ("kf_3", lambda: from_npz("kf_3")),
    ("snr_1", lambda: from_npz("snr_1")),
]


def boot(v, B=2000):
    idx = RNG.integers(0, len(v), (B, len(v)))
    return float(np.std(np.median(v[idx], axis=1)))


def main():
    cfg0 = ds.GenConfig()
    rx, _ = ds._place_receivers(cfg0, np.random.default_rng(0))
    fs = cfg0.fs
    print("수신기 배치 (uniform, 시드 불변): fs=%.1f MHz" % (fs / 1e6))
    for i, r in enumerate(rx):
        print("   rx%d: (%9.1f, %9.1f, %5.1f)" % (i, r[0], r[1], r[2]))

    ck = torch.load(os.path.join(ROOT, "output_final_der", "der_model.pt"), weights_only=False)
    mean, std = ck["mean"], ck["std"]
    model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
    model.load_state_dict(ck["model"])
    model.eval()

    @torch.no_grad()
    def pred(Z):
        mu, L, nu = model(torch.from_numpy(Z))
        al, ep = niw_uncertainties(L, nu)
        return mu.numpy(), ep.numpy(), (al + ep).numpy()

    def maha(y, p, c):
        d = (y - p)[:, :, None]
        return np.sqrt(np.maximum((np.linalg.solve(c, d) * d).sum((1, 2)), 0.0))

    dc = np.load(os.path.join(ROOT, "output_final", "gcc_calib.npz"))
    pc, _, tc = pred((dc["feats"].astype(np.float32) - mean) / std)
    sc = maha(dc["y"].astype(np.float32) / SCALE, pc, tc)
    nq = len(sc)
    Q = float(np.sort(sc)[min(int(np.ceil((nq + 1) * (1 - ALPHA))), nq) - 1])

    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    REF = (dt["feats"].astype(np.float32), dt["y"].astype(np.float32))

    R, RAW = {}, {}
    print("\n" + BAR)
    print("  %-16s %5s | %9s %9s %9s %9s | %9s %9s %6s"
          % ("조건", "n", "lag오차", "반치폭", "pk/far", "pk/near", "오차(m)", "epi(m)", "cov"))
    print(BAR)
    for nm, fn in CASES:
        F, Y = REF if fn is None else fn()
        S = batch_peak(F, Y, rx, fs)
        p, ep, tot = pred((F - mean) / std)
        yn = Y / SCALE
        e = np.sqrt(((p - yn) ** 2).sum(1)) * SCALE
        epv = np.sqrt(np.trace(ep, axis1=1, axis2=2)) * SCALE
        cov = float((maha(yn, p, tot) <= Q).mean())
        allv = dict(S, err=e, epi=epv)
        RAW[nm] = allv
        R[nm] = {k: (float(np.median(v)), boot(v)) for k, v in allv.items()}
        R[nm]["cov"] = (cov, float("nan"))
        print("  %-16s %5d | %9.4f %9.4f %9.3f %9.3f | %9.1f %9.1f %6.3f"
              % (nm, len(F), R[nm]["lag_err"][0], R[nm]["halfwidth_sub"][0],
                 R[nm]["pk_far"][0], R[nm]["pk_near"][0],
                 R[nm]["err"][0], R[nm]["epi"][0], cov))

    ref = R["ref (학습분포)"]
    print("\n" + BAR)
    print("기준선(학습분포) 대비 배율")
    print(BAR)
    print("  %-16s | %10s %10s %10s %10s | %8s %8s"
          % ("조건", "lag오차비", "반치폭비", "pk/far비", "pk/near비", "오차배", "epi배"))
    RAT = {}
    keys = ("lag_err", "halfwidth_sub", "pk_far", "pk_near", "err", "epi")
    for nm, _ in CASES:
        RAT[nm] = {k: R[nm][k][0] / ref[k][0] for k in keys}
        t = RAT[nm]
        print("  %-16s | %10.3f %10.3f %10.3f %10.3f | %8.2f %8.3f"
              % (nm, t["lag_err"], t["halfwidth_sub"], t["pk_far"], t["pk_near"],
                 t["err"], t["epi"]))

    print("\n" + BAR)
    print("관문")
    print(BAR)
    g1 = ref["lag_err"][0] < 1.0
    print("  G1 판독기 타당성  학습분포 median lag오차 = %.4f 샘플 (<1.0)  %s"
          % (ref["lag_err"][0], "PASS" if g1 else "FAIL"))
    nl = RAT["sco_4 (NULL)"]
    fz = (0.8 <= nl["lag_err"] <= 1.25) and (0.8 <= nl["halfwidth_sub"] <= 1.25)
    print("  FALSIFIER  sco_4 lag %.3f / 반치폭 %.3f (둘 다 0.8~1.25)  %s"
          % (nl["lag_err"], nl["halfwidth_sub"], "PASS" if fz else "FAIL"))
    for nm, lo, hi in (("delay_3", 7.9, 9.9), ("snr_1", 22.9, 28.9), ("kf_3", 6.8, 8.8)):
        v = RAT[nm]["err"]
        print("  G2 조작점검  %-8s 오차배 %6.2f (기대 %.1f~%.1f)  %s"
              % (nm, v, lo, hi, "PASS" if lo <= v <= hi else "FAIL"))

    print("\n" + BAR)
    print("사전 수치 예측 채점")
    print(BAR)
    preds = [("P1 delay_3 lag오차비 >= 3.0", RAT["delay_3"]["lag_err"], 3.0, "ge"),
             ("P2 occ0.05 반치폭비 >= 3.0", RAT["occ 0.05"]["halfwidth_sub"], 3.0, "ge"),
             ("P3 delay_3 반치폭비 <= 1.5", RAT["delay_3"]["halfwidth_sub"], 1.5, "le"),
             ("P4a TDL-C lag오차비 >= 2.0", RAT["TDL-C"]["lag_err"], 2.0, "ge"),
             ("P4b TDL-C 반치폭비 >= 2.0", RAT["TDL-C"]["halfwidth_sub"], 2.0, "ge")]
    hit = 0
    scored = []
    for lab, v, thr, mode in preds:
        good = (v >= thr) if mode == "ge" else (v <= thr)
        hit += int(good)
        scored.append(dict(label=lab, value=v, threshold=thr, mode=mode, hit=bool(good)))
        print("  %-32s 실측 %8.3f   %s" % (lab, v, "적중" if good else "빗나감"))
    print("\n  적중 %d/5" % hit)

    print("\n" + BAR)
    print("P-KEY — 표본을 lag오차 중앙값으로 이분")
    print(BAR)
    PK = {}
    for nm in ("delay_3", "TDL-C", "occ 0.05"):
        d = RAW[nm]
        m = float(np.median(d["lag_err"]))
        hi_m = d["lag_err"] > m
        lo_m = ~hi_m
        print("\n  [%s]  분할 임계 lag오차 = %.4f 샘플  (大 %d / 小 %d)"
              % (nm, m, int(hi_m.sum()), int(lo_m.sum())))
        print("    %-14s %11s %11s %7s %8s" % ("지표", "bias-小", "bias-大", "배율", "z"))
        PK[nm] = {}
        for k, lab in (("err", "오차(m)"), ("epi", "epi(m)"), ("halfwidth_sub", "반치폭"),
                       ("pk_far", "pk/far"), ("pk_near", "pk/near")):
            a, b = d[k][lo_m], d[k][hi_m]
            ma, mb = float(np.median(a)), float(np.median(b))
            z = float((mb - ma) / np.sqrt(boot(a) ** 2 + boot(b) ** 2))
            PK[nm][k] = dict(lo=ma, hi=mb, ratio=mb / ma, z=z)
            print("    %-14s %11.3f %11.3f %7.3f %+8.2f %s"
                  % (lab, ma, mb, mb / ma, z, "<<<" if abs(z) > 3 else ""))

    d3 = PK["delay_3"]
    cond_a = d3["err"]["z"] > 3
    cond_b = (d3["halfwidth_sub"]["z"] < 3) and (d3["pk_far"]["z"] > -3)
    cond_c = d3["epi"]["z"] < 3
    if cond_a and cond_b and cond_c:
        verdict = "지지"
    elif d3["halfwidth_sub"]["z"] > 3 or d3["pk_far"]["z"] < -3:
        verdict = "반증"
    else:
        verdict = "부분"
    print("\n  (a) bias-大 오차 유의 증가   z=%+.2f   %s" % (d3["err"]["z"], cond_a))
    print("  (b) 모양 동등/우수           반치폭 z=%+.2f / pk/far z=%+.2f   %s"
          % (d3["halfwidth_sub"]["z"], d3["pk_far"]["z"], cond_b))
    print("  (c) epistemic 같거나 낮음    z=%+.2f   %s" % (d3["epi"]["z"], cond_c))
    print("\n  ==> P-KEY 판정: %s" % verdict)

    out = dict(_note="봉우리 위치/모양 분해. 사전등록 PEAK_DECOMP_PREREG_2026-08-26.md",
               config=dict(seeds=list(SEEDS), n_gen=N_GEN, up=UP, alpha=ALPHA, Q=Q, fs=fs),
               rx=rx.tolist(),
               median={k: {kk: vv[0] for kk, vv in v.items()} for k, v in R.items()},
               boot_se={k: {kk: vv[1] for kk, vv in v.items()} for k, v in R.items()},
               ratio=RAT, gates=dict(g1=bool(g1), falsifier=bool(fz)),
               predictions=scored, prediction_hits=int(hit),
               pkey=PK, pkey_verdict=verdict)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
