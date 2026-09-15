r"""
phinorm_confirm.py — `||phi||` 역전 탐지기의 3중 독립 확증.

사전등록: `prereg/PHINORM_CONFIRM_PREREG_2026-08-31.md` (실행 전·결과 확인 전 기록)

선별(SELECT, 이미 끝남 — 다시 안 건드림)
  모델 seed 0 (= frozen 비트동일) · 생성시드 777 · 조건 9개

확증(CONFIRM, 3중 독립)
  모델   : seed 20260628 / 20261628 / 20262628 / 20263628      (4)
  생성시드: output_ood_s0 ~ s4  (이전 파일/outputs_ood/, 777 과 비트 상이) (5)
  조건   : held-out 레벨 cfo_1~4 · kf_1/2 · delay_0 · sco_2 · snr_2 · TDL-A · occ 0.20
  주 격자 = 4 x 5 = 20 셀 (npz 조건),  보조 arm = 4 모델 (gen() 조건)

점수·절차 (선별에서 고정, 변경 없음)
  s = log ||phi||  (상단 꼬리)
  in-dist(2000) 을 MAP/CAL/NEG 3분할 x 200 순열,  p=(1+#{S_cal>=S})/(n_cal+1),  기권 p<=alpha
  1차 alpha = 0.05

    python scripts/data/phinorm_confirm.py
"""
from __future__ import annotations
import functools, io, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)
from train_der import DERModel                                     # noqa: E402
from peak_decomposition import gen, BAR                            # noqa: E402
from rfgeo import waveform as wf                                    # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "phinorm_confirm.json")
ARCH = os.path.join(ROOT, "이전 파일", "outputs_ood")
ALPHAS = (0.01, 0.05, 0.10, 0.20)
ALPHA1 = 0.05
N_REP = 200

SELECT_MODEL = ("seed 0 (SELECT)", os.path.join(ROOT, "output_seed0", "der_model.pt"))
CONFIRM_MODELS = [
    ("seed 20260628", os.path.join(ROOT, "output_seed20260628_clean", "der_model.pt")),
    ("seed 20261628", os.path.join(ROOT, "output_seed20261628_clean", "der_model.pt")),
    ("seed 20262628", os.path.join(ROOT, "output_seed20262628_clean", "der_model.pt")),
    ("seed 20263628", os.path.join(ROOT, "output_seed20263628_clean", "der_model.pt")),
]
GEN_SEEDS = [("s777 (SELECT)", os.path.join(ROOT, "output_final_ood"))] + \
            [("s%d" % i, os.path.join(ARCH, "output_ood_s%d" % i)) for i in range(5)]

# 선별에 쓴 조건 / held-out 조건
SEL_LEVELS = ["delay_1", "delay_2", "delay_3", "kf_3", "sco_4", "snr_1"]
HOLD_LEVELS = ["delay_0", "kf_1", "kf_2", "cfo_1", "cfo_2", "cfo_3", "cfo_4",
               "sco_2", "snr_2"]
NPZ_LEVELS = SEL_LEVELS + HOLD_LEVELS

GEN_CASES = [
    ("occ 0.05", lambda: gen("_pc_occ05", functools.partial(wf.gen_ofdm, used_frac=0.05))),
    ("occ 0.20", lambda: gen("_pc_occ20", functools.partial(wf.gen_ofdm, used_frac=0.20))),
    ("TDL-A", lambda: gen(prof="TDL-A")),
    ("TDL-C", lambda: gen(prof="TDL-C")),
]


def main():
    # ---------- 입력을 한 번만 만든다 (모델 간 동일 보장) ----------
    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    FEAT = {"__indist__": dt["feats"].astype(np.float32)}
    for gname, gdir in GEN_SEEDS:
        for lv in NPZ_LEVELS:
            p = os.path.join(gdir, "shift_%s.npz" % lv)
            if os.path.exists(p):
                FEAT[(gname, lv)] = np.load(p)["feats"].astype(np.float32)
    for nm, fn in GEN_CASES:
        F, _ = fn()
        FEAT[("gen", nm)] = F
    print("  입력 %d개 로드" % len(FEAT))

    def phi_norm(model, mean, std, F):
        with torch.no_grad():
            Z = torch.from_numpy((F - mean) / std)
            phi = model.backbone(Z)
        return np.linalg.norm(phi.numpy().astype(np.float64), axis=1)

    RES, RATIO = {}, {}
    for label, path in [SELECT_MODEL] + CONFIRM_MODELS:
        ck = torch.load(path, weights_only=False)
        mean, std = ck["mean"], ck["std"]
        model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
        model.load_state_dict(ck["model"]); model.eval()

        S = {k: np.log(phi_norm(model, mean, std, F)) for k, F in FEAT.items()}
        base = float(np.median(np.exp(S["__indist__"])))
        RATIO[label] = {k: float(np.median(np.exp(v))) / base
                        for k, v in S.items() if k != "__indist__"}

        ind = S["__indist__"]
        n = len(ind)
        rng = np.random.default_rng(0)
        keys = [k for k in S if k != "__indist__"]
        pw = {a: {k: [] for k in keys + ["NEG"]} for a in ALPHAS}
        for _ in range(N_REP):
            idx = rng.permutation(n)
            t = n // 3
            cal = np.sort(ind[idx[t:2 * t]])
            neg = ind[idx[2 * t:]]
            nc = len(cal)

            def pval(v):
                return (1 + (nc - np.searchsorted(cal, v, side="left"))) / (nc + 1.0)

            pn = pval(neg)
            for a in ALPHAS:
                pw[a]["NEG"].append(float((pn <= a).mean()))
            for k in keys:
                pk = pval(S[k])
                for a in ALPHAS:
                    pw[a][k].append(float((pk <= a).mean()))
        RES[label] = {str(a): {("|".join(k) if isinstance(k, tuple) else k):
                               float(np.mean(v)) for k, v in d.items()}
                      for a, d in pw.items()}
        print("  [측정 완료] %s" % label)

    A1 = str(ALPHA1)
    conf = [L for L, _ in CONFIRM_MODELS]
    gs = [g for g, _ in GEN_SEEDS if g != "s777 (SELECT)"]

    # ---------------- FALSIFIER ----------------
    print("\n" + BAR); print("FALSIFIER"); print(BAR)
    negs = [(L, RES[L][A1]["NEG"]) for L in RES]
    bad_neg = [(L, v) for L, v in negs if not (0.03 <= v <= 0.07)]
    print("  NEG 오기권률 (창 0.03~0.07):")
    for L, v in negs:
        print("    %-20s %.4f  %s" % (L, v, "PASS" if 0.03 <= v <= 0.07 else "FAIL"))
    sco = [(L, g, RES[L][A1]["%s|sco_4" % g]) for L in conf for g in gs]
    bad_sco = [(L, g, v) for L, g, v in sco if not (0.02 <= v <= 0.10)]
    print("  sco_4(NULL) 탐지력 20셀 범위: %.4f ~ %.4f  (창 0.02~0.10)  %s"
          % (min(v for _, _, v in sco), max(v for _, _, v in sco),
             "PASS" if not bad_sco else "FAIL %d셀" % len(bad_sco)))

    # ---------------- 주 격자 ----------------
    print("\n" + BAR)
    print("주 격자 — 4 모델 x 5 생성시드 = 20 셀   (alpha = %.2f 탐지력)" % ALPHA1)
    print(BAR)
    for lv in ("delay_3", "kf_3", "cfo_4", "delay_1", "delay_2"):
        print("\n  [%s]" % lv)
        print("    %-16s" % "모델" + "".join("%10s" % g for g in gs) + "%12s" % "범위")
        for L in conf:
            v = [RES[L][A1]["%s|%s" % (g, lv)] for g in gs]
            print("    %-16s" % L + "".join("%10.3f" % x for x in v)
                  + "  %.3f~%.3f" % (min(v), max(v)))
        sv = RES[SELECT_MODEL[0]][A1]["s777 (SELECT)|%s" % lv]
        print("    %-16s (선별 참조: seed0 x s777 = %.3f)" % ("", sv))

    # ---------------- 채점 ----------------
    print("\n" + BAR); print("사전 예측 채점"); print(BAR)
    SC = {}

    def grid(lv):
        return [(L, g, RES[L][A1]["%s|%s" % (g, lv)]) for L in conf for g in gs]

    pa = [(L, g, v) for L, g, v in grid("delay_3") if v >= 0.30]
    SC["P-A delay_3 >= 0.30"] = dict(hit=len(pa), n=20)
    print("  P-A  delay_3 탐지력 >= 0.30        %2d/20  %s"
          % (len(pa), "적중" if len(pa) == 20 else "부분/실패"))
    pb = [(L, g, v) for L, g, v in grid("kf_3") if v <= 0.15]
    SC["P-B kf_3 <= 0.15"] = dict(hit=len(pb), n=20)
    print("  P-B  kf_3 탐지력 <= 0.15           %2d/20  %s"
          % (len(pb), "적중" if len(pb) == 20 else "부분/실패"))
    pc = [(L, g, v) for L, g, v in grid("cfo_4") if v <= 0.15]
    SC["P-C cfo_4 <= 0.15"] = dict(hit=len(pc), n=20)
    print("  P-C  cfo_4 탐지력 <= 0.15 (heldout) %2d/20  %s"
          % (len(pc), "적중" if len(pc) == 20 else "부분/실패"))
    pd_ = 0
    for L in conf:
        for g in gs:
            a, b, c = (RES[L][A1]["%s|delay_%d" % (g, i)] for i in (1, 2, 3))
            pd_ += int(a < b < c)
    SC["P-D delay 단조"] = dict(hit=pd_, n=20)
    print("  P-D  delay_1<2<3 단조              %2d/20  %s"
          % (pd_, "적중" if pd_ == 20 else "부분/실패"))

    # P-E : ||phi||비 부호 <-> 탐지력
    ok = tot = 0
    viol = []
    for L in conf:
        for g in gs:
            for lv in NPZ_LEVELS:
                k = "%s|%s" % (g, lv)
                r = RATIO[L].get((g, lv))
                if r is None:
                    continue
                v = RES[L][A1][k]
                if r > 1.02:
                    tot += 1; ok += int(v > 0.15)
                    if v <= 0.15:
                        viol.append((L, g, lv, r, v, ">1.02 인데 침묵"))
                elif r < 0.98:
                    tot += 1; ok += int(v <= 0.15)
                    if v > 0.15:
                        viol.append((L, g, lv, r, v, "<0.98 인데 발화"))
    SC["P-E 부호 일관"] = dict(hit=ok, n=tot, frac=ok / max(tot, 1))
    print("  P-E  ||phi||비 부호 <-> 탐지력      %3d/%3d (%.1f%%)  %s"
          % (ok, tot, 100 * ok / max(tot, 1), "적중" if ok / max(tot, 1) >= 0.90 else "실패"))
    if viol:
        print("       위반 셀 (최대 8개):")
        for L, g, lv, r, v, why in viol[:8]:
            print("         %-16s %-6s %-9s ||phi||비 %.3f 탐지력 %.3f  %s"
                  % (L, g, lv, r, v, why))

    # P-F : 보조 arm
    print("\n  P-F  보조 arm (gen 조건, 모델 축만)")
    print("    %-16s %10s %10s %10s %10s" % ("모델", "occ0.05", "occ0.20", "TDL-A", "TDL-C"))
    pf = 0
    for L in conf:
        o5 = RES[L][A1]["gen|occ 0.05"]; o2 = RES[L][A1]["gen|occ 0.20"]
        ta = RES[L][A1]["gen|TDL-A"]; tc = RES[L][A1]["gen|TDL-C"]
        pf += int(ta <= 0.15 and o2 > 0.05)
        print("    %-16s %10.3f %10.3f %10.3f %10.3f" % (L, o5, o2, ta, tc))
    SC["P-F 보조"] = dict(hit=pf, n=len(conf))
    print("    P-F (TDL-A<=0.15 이고 occ0.20>0.05)  %d/%d" % (pf, len(conf)))

    # ---------------- 판정 (사전등록 6장 조합표) ----------------
    na, nb = SC["P-A delay_3 >= 0.30"]["hit"], SC["P-B kf_3 <= 0.15"]["hit"]
    if bad_neg:
        verdict = "무효 — FALSIFIER (NEG 오기권률)"
    elif na == 20 and nb == 20:
        verdict = "승격 — 역전 전용 플래그로 채택"
    elif na == 20 and nb >= 15:
        verdict = "조건부 승격 — 선택성 예외 축 명시"
    elif na == 20:
        verdict = "부분 — 탐지는 되나 역전 전용이 아니다 (플래그 이름 변경 필요)"
    elif na >= 15:
        verdict = "부분 — 실패 셀 특정 필요"
    else:
        verdict = "기각 — 선별 낙관"
    if bad_sco and not bad_neg:
        verdict += "  [단 sco_4 FALSIFIER 위반 %d셀 -> P-A/P-B 판정 무효]" % len(bad_sco)
    print("\n  ==> 판정: %s" % verdict)

    out = dict(_note="||phi|| 탐지기 3중 독립 확증. 사전등록 PHINORM_CONFIRM_PREREG_2026-08-31.md",
               config=dict(alpha1=ALPHA1, n_rep=N_REP, confirm_models=conf, gen_seeds=gs,
                           sel_levels=SEL_LEVELS, hold_levels=HOLD_LEVELS),
               power=RES,
               phi_ratio={L: {("|".join(k) if isinstance(k, tuple) else k): v
                              for k, v in d.items()} for L, d in RATIO.items()},
               falsifier=dict(neg=dict(negs), bad_neg=bad_neg,
                              sco_min=min(v for _, _, v in sco),
                              sco_max=max(v for _, _, v in sco), bad_sco=bad_sco),
               score=SC, violations=[list(map(str, v)) for v in viol], verdict=verdict)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
