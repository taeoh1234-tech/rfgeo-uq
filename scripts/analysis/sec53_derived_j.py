# -*- coding: utf-8 -*-
r"""sec53_derived_j.py - 논문 5.3절 파생 주장을 **자원 j + 현행 에너지 플래그(LS)** 로 재계산.

배경 (2026-09-14): 기존 sec53_derived.py 는 2026-09-08 산출물로, 내부에서
  "ENERGY" -> MAXLA2 로 매핑하고 탐지는 자원 h, RULE/PHI 복원은 자원 g 를 읽는다.
에너지 플래그를 LS 로 교체(09-12)하고 표2 탐지 블록을 자원 j 로 통일(09-14)한 뒤에는
그 값들이 표2와 어긋나므로 여기서 다시 계산해 영구화한다.

입력 (기존 산출물만, 재학습/재생성 없음)
  detector_j.json            탐지력 @alpha=0.05 (자원 j, 5모델)
  corrector_j_arms.json      RULE/PHI/CLO 교정 (자원 j: 92001/92002 + j1~j3)
  ls040_confirm.json         LS@0.40 교정 + L5 조건부편차 (자원 j)
  mondrian_j_conddev040.json Mondrian 조건부편차 (자원 j, 동일 풀링)

규약
  복원 커버리지 = (모델 x 세트) 셀 평균,  영역 = 셀 중앙값 / 탐색원반(201.06 km^2)
  R = clip((cov - base)/(0.9 - base), 0, 1),  영역 > 20% 이면 R = 0 (게이트)

    python scripts/data/sec53_derived_j.py
"""
from __future__ import annotations
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
J = os.path.join(ROOT, "artifacts")
CONDS = ["delay_1", "delay_2", "delay_3", "kf_3", "cfo_4",
         "snr_1", "snr_2", "sco_4", "occ 0.05", "TDL-A", "TDL-C"]
KAP = {"RULE": "0.6", "PHI": "1.5"}
CTRL = "sco_4"


def load(n):
    with io.open(os.path.join(J, n), encoding="utf-8") as f:
        return json.load(f)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    DJ = load("detector_j.json")["cells"]
    CJ = load("corrector_j_arms.json")["cells"]
    LS = load("ls040_confirm.json")
    MO = load("mondrian_j_conddev040.json")["scoring"]["MONDRIAN_J"]["per_model"]

    det = {}
    for sig, key in (("RULE", "RULE"), ("PHI", "PHI"), ("ENERGY", "LS")):
        for c in CONDS:
            v = [DJ[k] for k in DJ
                 if k.split("|")[0] == key and k.split("|")[2:3] == [c]]
            if v:
                det[(sig, c)] = float(np.mean(v))

    def arms(arm, kap, c):
        v = [CJ[k] for k in CJ for p in [k.split("|")]
             if len(p) >= 5 and p[0] == "PRIMARY" and p[1] == arm
             and p[2] == kap and p[4] == c]
        return (float(np.mean([x[0] for x in v])),
                float(np.median([x[1] for x in v]))) if v else None

    def lsarm(pref, c):
        C = LS["cells"]
        v = [C[k] for k in C for p in [k.split("|")]
             if p[0] == pref and len(p) >= 3 and p[2] == c
             and isinstance(C[k], list) and len(C[k]) == 2]
        return (float(np.mean([x[0] for x in v])),
                float(np.median([x[1] for x in v]))) if v else None

    pairs = {}
    for c in CONDS:
        for sig in ("RULE", "PHI", "ENERGY"):
            if sig == "ENERGY":
                a, b = lsarm("LS@0.40", c), lsarm("LS@0.00", c)
            else:
                a, b = arms(sig, KAP[sig], c), arms(sig, "0.0", c)
            if not a or not b:
                continue
            cov, dsc, base = a[0], a[1], b[0]
            R = 0.0 if dsc > 0.20 else float(
                np.clip((cov - base) / (0.9 - base), 0, 1))
            pairs["%s|%s" % (sig, c)] = {
                "det": det.get((sig, c)), "base": round(base, 4),
                "cov": round(cov, 4), "dcov": round(cov - base, 4),
                "disc_med": round(dsc, 4), "R": round(R, 4)}

    out = {"_note": "5.3절 파생 주장 재계산 (자원 j + LS). 규약은 소스 docstring.",
           "pairs": pairs}

    # (a) 분포내 커버리지 (정규화기 적용)
    ind = {}
    for arm in ("RULE", "PHI"):
        v = [CJ[k][0] for k in CJ for p in [k.split("|")]
             if len(p) >= 5 and p[0] == "PRIMARY" and p[1] == arm
             and p[2] == KAP[arm] and p[4] == "IND"]
        ind["%s@%s" % (arm, KAP[arm])] = [round(min(v), 4), round(max(v), 4)]
    C = LS["cells"]
    v = [C[k][0] for k in C if k.split("|")[0] == "LS@0.40" and "IND-D" in k
         and isinstance(C[k], list)]
    ind["LS@0.40 (IND-D j1~j3)"] = [round(min(v), 4), round(max(v), 4)]
    lo = min(x[0] for x in ind.values())
    hi = max(x[1] for x in ind.values())
    out["a_indist_cov"] = {"per_arm": ind, "union": [round(lo, 4), round(hi, 4)]}

    # (b) silent 쌍
    sil = {k: v for k, v in pairs.items()
           if v["det"] is not None and v["det"] <= 0.10
           and not k.endswith("|" + CTRL)}
    out["b_silent_pairs_exclctrl"] = {
        "criterion": "det<=0.10, sco_4 제외", "n": len(sil),
        "max_abs_dcov": round(max(abs(v["dcov"]) for v in sil.values()), 4),
        "list": sorted(sil)}

    # (c) 발화하나 복원 없음
    fire = {k: v for k, v in pairs.items()
            if v["det"] is not None and v["det"] >= 0.50 and v["R"] == 0.0}
    out["c_fire_no_repair"] = {
        "criterion": "det>=0.50 and R==0", "n": len(fire),
        "det_range": [round(min(v["det"] for v in fire.values()), 4),
                      round(max(v["det"] for v in fire.values()), 4)],
        "list": {k: round(v["det"], 4) for k, v in sorted(fire.items())}}

    # (d) Spearman(det, dcov) — 게이트 안, 대조군 제외
    sp = {}
    for sig in ("RULE", "PHI", "ENERGY"):
        xs = [(v["det"], v["dcov"]) for k, v in pairs.items()
              if k.startswith(sig + "|") and v["det"] is not None
              and v["disc_med"] <= 0.20 and not k.endswith("|" + CTRL)]
        if len(xs) >= 3:
            sp[sig] = {"rho": round(spearman(np.array([a for a, _ in xs]),
                                             np.array([b for _, b in xs])), 4),
                       "n": len(xs)}
    out["d_spearman_det_vs_dcov"] = {
        "in_gate_exclctrl": sp,
        "range": [round(min(x["rho"] for x in sp.values()), 4),
                  round(max(x["rho"] for x in sp.values()), 4)]}

    # (e) 조건부 편차: LS vs Mondrian (둘 다 자원 j, 동일 풀링)
    l5 = LS["scoring"]["L5"]["per_model"]
    out["e_conddev_j"] = {
        "LS@0.40": {m: round(x, 4) for m, x in l5.items()},
        "mondrian": {m: round(x, 4) for m, x in MO.items()},
        "LS_range": [round(min(l5.values()), 4), round(max(l5.values()), 4)],
        "mondrian_range": [round(min(MO.values()), 4), round(max(MO.values()), 4)],
        "n_models_LS_better": sum(1 for m in l5 if l5[m] < MO[m])}

    # (f) 게이트 분리 (9 조건)
    gate = {"multipath": pairs["RULE|delay_3"]["disc_med"],
            "bandwidth": pairs["PHI|occ 0.05"]["disc_med"],
            "fading": pairs["ENERGY|kf_3"]["disc_med"],
            "TDL-A": pairs["ENERGY|TDL-A"]["disc_med"],
            "TDL-C": pairs["ENERGY|TDL-C"]["disc_med"],
            "benign sco_4": pairs["ENERGY|sco_4"]["disc_med"],
            "cfo_4": pairs["ENERGY|cfo_4"]["disc_med"],
            "snr_2": pairs["ENERGY|snr_2"]["disc_med"]}
    rp = max(x for k, x in gate.items() if k not in ("cfo_4", "snr_2"))
    ab = min(gate["cfo_4"], gate["snr_2"])
    out["f_gate_separation"] = {
        "per_condition_disc": {k: round(x, 4) for k, x in gate.items()},
        "repair_max": round(rp, 4), "abstain_min": round(ab, 4),
        "ratio": round(ab / rp, 2)}

    fp = os.path.join(J, "sec53_derived_j.json")
    with io.open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[saved]", fp)
    print("(a) in-dist union      %.4f--%.4f" % (lo, hi))
    print("(b) silent ssang       n=%d max|dcov|=%.4f"
          % (len(sil), out["b_silent_pairs_exclctrl"]["max_abs_dcov"]))
    print("(c) fire-no-repair     n=%d  det %.3f--%.3f"
          % (len(fire), *out["c_fire_no_repair"]["det_range"]))
    print("(d) Spearman           %.3f--%.3f"
          % tuple(out["d_spearman_det_vs_dcov"]["range"]))
    print("(e) LS %.4f--%.4f  vs  Mondrian %.4f--%.4f (LS better %d/5)"
          % (*out["e_conddev_j"]["LS_range"],
             *out["e_conddev_j"]["mondrian_range"],
             out["e_conddev_j"]["n_models_LS_better"]))
    print("(f) gate: repair max %.1f%% vs abstain min %.1f%% (%.1fx)"
          % (rp * 100, ab * 100, ab / rp))


if __name__ == "__main__":
    main()
