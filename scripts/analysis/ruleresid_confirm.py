r"""
ruleresid_confirm.py — 플래그 B(`규칙 잔차`) 탐지기의 3중 독립 확증.

사전등록: `prereg/RULERESID_CONFIRM_PREREG_2026-08-31.md` (실행 전·결과 확인 전 기록)

선별(SELECT, 이미 끝남)
  모델 frozen(= seed 0) · 생성시드 777 · 조건 11개
확증(CONFIRM)
  모델 4 (§72) x 생성시드 5 (output_ood_s0~s4) = 20 셀 · held-out 레벨 9종

점수 (선별에서 고정)
  MAP 에서  log(epi) ~ b0 + b1 log(featRMS) + b2 log(pk/far) + b3 log(반치폭)  적합
  s = (beta . x) - log(epi)      (규칙보다 덜 불확실하다고 말할수록 큼)
  conformal p-value, 1차 alpha = 0.05

곡선 통계는 **모델 무관**이므로 한 번만 계산해 npz 로 캐시한다(가장 비싼 단계).

    python scripts/data/ruleresid_confirm.py
"""
from __future__ import annotations
import io, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)
from rfgeo import dataset as ds                                     # noqa: E402
from rfgeo.evidential import niw_uncertainties                      # noqa: E402
from train_der import DERModel                                      # noqa: E402
from peak_decomposition import batch_peak, SCALE, BAR               # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "ruleresid_confirm.json")
STATS_CACHE = os.path.join(ROOT, "artifacts", "ruleresid_curvestats.npz")
ARCH = os.path.join(ROOT, "이전 파일", "outputs_ood")
FEATS = ("feat_rms", "pk_far", "halfwidth_sub")
ALPHAS = (0.01, 0.05, 0.10, 0.20)
ALPHA1 = 0.05
N_REP = 200

CONFIRM_MODELS = [
    ("seed 20260628", os.path.join(ROOT, "output_seed20260628_clean", "der_model.pt")),
    ("seed 20261628", os.path.join(ROOT, "output_seed20261628_clean", "der_model.pt")),
    ("seed 20262628", os.path.join(ROOT, "output_seed20262628_clean", "der_model.pt")),
    ("seed 20263628", os.path.join(ROOT, "output_seed20263628_clean", "der_model.pt")),
]
GEN_SEEDS = [("s%d" % i, os.path.join(ARCH, "output_ood_s%d" % i)) for i in range(5)]
SEL_LEVELS = ["delay_1", "delay_2", "delay_3", "kf_3", "sco_4", "snr_1"]
HOLD_LEVELS = ["delay_0", "kf_1", "kf_2", "cfo_1", "cfo_2", "cfo_3", "cfo_4",
               "sco_2", "snr_2"]
LEVELS = SEL_LEVELS + HOLD_LEVELS


def build_stats():
    """곡선 통계 (모델 무관) — in-dist + 5 생성시드 x 15 레벨."""
    if os.path.exists(STATS_CACHE):
        z = np.load(STATS_CACHE, allow_pickle=True)
        print("  [cache] %s" % STATS_CACHE)
        return {k: z[k].item() for k in z.files}
    cfg = ds.GenConfig()
    rx, _ = ds._place_receivers(cfg, np.random.default_rng(0))
    fs = cfg.fs
    D = {}
    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    F, Y = dt["feats"].astype(np.float32), dt["y"].astype(np.float32)
    S = batch_peak(F, Y, rx, fs)
    S["feat_rms"] = np.sqrt(np.mean(F.astype(np.float64) ** 2, axis=(1, 2)))
    D["__indist__"] = S
    print("  곡선통계 __indist__ n=%d" % len(F))
    for gname, gdir in GEN_SEEDS:
        for lv in LEVELS:
            p = os.path.join(gdir, "shift_%s.npz" % lv)
            if not os.path.exists(p):
                continue
            d = np.load(p)
            F, Y = d["feats"].astype(np.float32), d["y"].astype(np.float32)
            S = batch_peak(F, Y, rx, fs)
            S["feat_rms"] = np.sqrt(np.mean(F.astype(np.float64) ** 2, axis=(1, 2)))
            D["%s|%s" % (gname, lv)] = S
        print("  곡선통계 %s 완료 (%d 레벨)" % (gname, len(LEVELS)))
    np.savez_compressed(STATS_CACHE, **{k: np.array(v, dtype=object) for k, v in D.items()})
    print("  [saved cache] %s" % STATS_CACHE)
    return D


def main():
    CS = build_stats()
    keys = [k for k in CS if k != "__indist__"]

    # 모델별 epi 를 붙인다
    RES, BETA = {}, {}
    for label, path in CONFIRM_MODELS:
        ck = torch.load(path, weights_only=False)
        mean, std = ck["mean"], ck["std"]
        model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
        model.load_state_dict(ck["model"]); model.eval()

        @torch.no_grad()
        def epi_of(F):
            mu, L, nu = model(torch.from_numpy((F - mean) / std))
            al, ep = niw_uncertainties(L, nu)
            return np.sqrt(np.trace(ep.numpy(), axis1=1, axis2=2)) * SCALE

        dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
        EPI = {"__indist__": epi_of(dt["feats"].astype(np.float32))}
        for gname, gdir in GEN_SEEDS:
            for lv in LEVELS:
                k = "%s|%s" % (gname, lv)
                if k not in CS:
                    continue
                EPI[k] = epi_of(np.load(os.path.join(gdir, "shift_%s.npz" % lv))
                                ["feats"].astype(np.float32))

        def design(st, idx=None):
            n = len(st["feat_rms"]) if idx is None else len(idx)
            cols = [np.ones(n)]
            for f in FEATS:
                v = st[f] if idx is None else st[f][idx]
                cols.append(np.log(np.maximum(v, 1e-12)))
            return np.column_stack(cols)

        ind, ei = CS["__indist__"], EPI["__indist__"]
        n = len(ei)
        rng = np.random.default_rng(0)
        names = ["rule_resid", "epi_low", "psr_only"]
        pw = {a: {s: {k: [] for k in keys + ["NEG"]} for s in names} for a in ALPHAS}
        bl = []
        for _ in range(N_REP):
            idx = rng.permutation(n)
            t = n // 3
            i_map, i_cal, i_neg = idx[:t], idx[t:2 * t], idx[2 * t:]
            beta, *_ = np.linalg.lstsq(design(ind, i_map), np.log(ei[i_map]), rcond=None)
            bl.append(beta)

            def sc(st, ep, ii=None):
                X = design(st, ii)
                e = ep if ii is None else ep[ii]
                pf = st["pk_far"] if ii is None else st["pk_far"][ii]
                return dict(rule_resid=X @ beta - np.log(e),
                            epi_low=-np.log(e),
                            psr_only=-np.log(np.maximum(pf, 1e-12)))

            s_cal = sc(ind, ei, i_cal)
            s_neg = sc(ind, ei, i_neg)
            for nmn in names:
                cal = np.sort(s_cal[nmn]); nc = len(cal)

                def pval(v):
                    return (1 + (nc - np.searchsorted(cal, v, side="left"))) / (nc + 1.0)

                pn = pval(s_neg[nmn])
                for a in ALPHAS:
                    pw[a][nmn]["NEG"].append(float((pn <= a).mean()))
                for k in keys:
                    pk = pval(sc(CS[k], EPI[k])[nmn])
                    for a in ALPHAS:
                        pw[a][nmn][k].append(float((pk <= a).mean()))
        B = np.array(bl)
        BETA[label] = dict(mean=B.mean(axis=0).tolist(), std=B.std(axis=0).tolist())
        RES[label] = {str(a): {s: {k: float(np.mean(v)) for k, v in dd.items()}
                               for s, dd in d.items()} for a, d in pw.items()}
        print("  [측정 완료] %s" % label)

    A1 = str(ALPHA1)
    conf = [L for L, _ in CONFIRM_MODELS]
    gs = [g for g, _ in GEN_SEEDS]
    R = lambda L, lv: RES[L][A1]["rule_resid"]["%s|%s" % (lv[0], lv[1])]

    def grid(lv):
        return [(L, g, RES[L][A1]["rule_resid"]["%s|%s" % (g, lv)]) for L in conf for g in gs]

    # ---------- FALSIFIER ----------
    print("\n" + BAR); print("FALSIFIER"); print(BAR)
    negs = [(L, RES[L][A1]["rule_resid"]["NEG"]) for L in conf]
    bad_neg = [(L, v) for L, v in negs if not (0.03 <= v <= 0.07)]
    for L, v in negs:
        print("  NEG %-16s %.4f  %s" % (L, v, "PASS" if 0.03 <= v <= 0.07 else "FAIL"))
    sc4 = grid("sco_4")
    bad_sco = [(L, g, v) for L, g, v in sc4 if not (0.02 <= v <= 0.10)]
    print("  sco_4(NULL) 20셀: %.4f ~ %.4f  (창 0.02~0.10)  %s"
          % (min(v for _, _, v in sc4), max(v for _, _, v in sc4),
             "PASS" if not bad_sco else "FAIL %d셀" % len(bad_sco)))

    # ---------- 주 격자 ----------
    print("\n" + BAR)
    print("주 격자 — 규칙 잔차, alpha = %.2f 탐지력" % ALPHA1)
    print(BAR)
    SELV = dict(delay_1=0.499, delay_2=0.633, delay_3=0.714, kf_3=0.535,
                sco_4=0.054, snr_1=0.084)
    for lv in ("delay_3", "kf_3", "cfo_4", "snr_1", "delay_1", "delay_2"):
        print("\n  [%s]%s" % (lv, "  (held-out)" if lv in HOLD_LEVELS else ""))
        print("    %-16s" % "모델" + "".join("%9s" % g for g in gs) + "%13s" % "범위")
        for L in conf:
            v = [RES[L][A1]["rule_resid"]["%s|%s" % (g, lv)] for g in gs]
            print("    %-16s" % L + "".join("%9.3f" % x for x in v)
                  + "  %.3f~%.3f" % (min(v), max(v)))
        if lv in SELV:
            print("    %-16s (선별 참조: %.3f)" % ("", SELV[lv]))

    # ---------- 채점 ----------
    print("\n" + BAR); print("사전 예측 채점"); print(BAR)
    SC = {}
    qa = [x for x in grid("delay_3") if x[2] >= 0.40]
    SC["Q-A delay_3>=0.40"] = dict(hit=len(qa), n=20)
    print("  Q-A  delay_3 >= 0.40             %2d/20  %s"
          % (len(qa), "적중" if len(qa) == 20 else "부분/실패"))
    # 사전등록 §8 정정: Q-B 는 kf_3 단독 (TDL-C 는 gen 조건이라 주 격자에 없음)
    qb_cells = [(L, g, RES[L][A1]["rule_resid"]["%s|kf_3" % g]) for L in conf for g in gs]
    qb = [x for x in qb_cells if x[2] >= 0.25]
    SC["Q-B kf_3>=0.25"] = dict(hit=len(qb), n=20)
    print("  Q-B  kf_3 >= 0.25 (광범위성)      %2d/20  %s"
          % (len(qb), "적중" if len(qb) == 20 else "부분/실패"))
    qc = [x for x in grid("snr_1") if x[2] <= 0.15]
    SC["Q-C snr_1<=0.15"] = dict(hit=len(qc), n=20)
    print("  Q-C  snr_1 <= 0.15               %2d/20  %s"
          % (len(qc), "적중" if len(qc) == 20 else "부분/실패"))
    qd = [x for x in grid("cfo_4") if x[2] > 0.15]
    SC["Q-D cfo_4>0.15"] = dict(hit=len(qd), n=20)
    print("  Q-D  cfo_4 > 0.15 (held-out)     %2d/20  %s"
          % (len(qd), "적중" if len(qd) == 20 else "부분/실패"))
    qe = 0
    for L in conf:
        for g in gs:
            a, b, c = (RES[L][A1]["rule_resid"]["%s|delay_%d" % (g, i)] for i in (1, 2, 3))
            qe += int(a < b < c)
    SC["Q-E delay 단조"] = dict(hit=qe, n=20)
    print("  Q-E  delay_1<2<3 단조            %2d/20  %s"
          % (qe, "적중" if qe == 20 else "부분/실패"))
    qf = [L for L in conf if BETA[L]["mean"][3] < 0]
    SC["Q-F 반치폭 계수 음수"] = dict(hit=len(qf), n=len(conf))
    print("  Q-F  반치폭 계수 < 0              %2d/%d  %s"
          % (len(qf), len(conf), "적중" if len(qf) == len(conf) else "실패"))
    print("\n  회귀 계수 (4모델):")
    print("    %-16s %10s %10s %10s %10s" % ("모델", "절편", "featRMS", "pk/far", "반치폭"))
    for L in conf:
        m = BETA[L]["mean"]
        print("    %-16s %10.4f %10.4f %10.4f %10.4f" % (L, m[0], m[1], m[2], m[3]))

    # ---------- 보조: 후보 3종 비교 (판정 미사용) ----------
    print("\n" + BAR)
    print("보조 — 후보 3종 비교 (사전등록 §7-2: 판정에 쓰지 않음). 20셀 평균")
    print(BAR)
    print("  %-12s %14s %14s %14s" % ("조건", "rule_resid", "epi_low", "psr_only"))
    for lv in ("delay_3", "kf_3", "cfo_4", "snr_1", "sco_4"):
        row = []
        for s in ("rule_resid", "epi_low", "psr_only"):
            row.append(float(np.mean([RES[L][A1][s]["%s|%s" % (g, lv)]
                                      for L in conf for g in gs])))
        print("  %-12s %14.3f %14.3f %14.3f" % (lv, row[0], row[1], row[2]))

    # ---------- 판정 ----------
    na, nb = SC["Q-A delay_3>=0.40"]["hit"], SC["Q-B kf_3>=0.25"]["hit"]
    if bad_neg:
        verdict = "무효 — FALSIFIER (NEG 오기권률)"
    elif na == 20 and nb == 20:
        verdict = "승격 — 광범위 조용한실패 플래그로 채택"
    elif na == 20 and nb >= 15:
        verdict = "조건부 승격 — 광범위성 예외 축 명시"
    elif na == 20:
        verdict = "부분 — delay 축 전용으로 범위 축소 (C 와 역할 중복 검토)"
    elif na >= 15:
        verdict = "부분 — 실패 셀 특정 필요"
    else:
        verdict = "기각 — 선별 낙관"
    if bad_sco and not bad_neg:
        verdict += "  [단 sco_4 FALSIFIER 위반 %d셀 -> Q-A/Q-B 무효]" % len(bad_sco)
    print("\n  ==> 판정: %s" % verdict)

    out = dict(_note="규칙 잔차 탐지기 3중 확증. 사전등록 RULERESID_CONFIRM_PREREG_2026-08-31.md",
               config=dict(alpha1=ALPHA1, n_rep=N_REP, confirm_models=conf,
                           gen_seeds=gs, sel_levels=SEL_LEVELS, hold_levels=HOLD_LEVELS),
               power=RES, beta=BETA,
               falsifier=dict(neg=dict(negs), bad_neg=bad_neg,
                              sco_min=min(v for _, _, v in sc4),
                              sco_max=max(v for _, _, v in sc4), bad_sco=bad_sco),
               score=SC, verdict=verdict)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
