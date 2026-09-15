# -*- coding: utf-8 -*-
r"""sec51_mondrian_j.py - 5.1절 "0.180->0.674 / 0.127->0.184" 문장의 자원 j 대체값 영구화.

배경 (2026-09-14, A7): 현행 tex 5.1절의 수치는 구자원(E17/E18: 단일 모델 seed0 +
구 생성시드) 산이다. 표2·5.3절이 자원 j 로 통일된 뒤 이 문장만 구자원으로 남아
있어, corrector_j_arms.json 의 mondrian(그룹조건부) / kappa=0(marginal) 셀에서
같은 양을 재계산한다.

규약: 표2와 동일 - 커버리지 = (모델 x 이동세트) 셀 평균. 5모델 x 92001/92002.
  marginal  = PRIMARY|CLO|0.0|{m}|{lv}|{gs}   (kappa=0 이면 arm 무관, GATE-1 확인됨)
  mondrian  = PRIMARY|mondrian|0.0|{m}|{lv}|{gs}

    python scripts/data/sec51_mondrian_j.py
"""
from __future__ import annotations
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "artifacts")
OUT = os.path.join(SRC, "sec51_mondrian_j.json")
MODELS = ["m0", "m1", "m2", "m3", "seed0"]
SETS = ["92001", "92002"]


def main():
    with io.open(os.path.join(SRC, "corrector_j_arms.json"), encoding="utf-8") as f:
        C = json.load(f)["cells"]

    out = {"_note": "5.1절 SNR-이동 vs multipath 의 marginal/mondrian 커버리지 (자원 j, "
                    "표2 규약: 셀 평균 + [min,max])", "levels": {}}
    for lv in ("snr_2", "delay_3"):
        row = {}
        for name, key in (("marginal", "PRIMARY|CLO|0.0|%s|%s|%s"),
                          ("mondrian", "PRIMARY|mondrian|0.0|%s|%s|%s")):
            v = [C[key % (m, lv, g)][0] for m in MODELS for g in SETS]
            row[name] = {"mean": round(float(np.mean(v)), 4),
                         "min": round(float(np.min(v)), 4),
                         "max": round(float(np.max(v)), 4), "n_cells": len(v)}
        out["levels"][lv] = row

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for lv, r in out["levels"].items():
        print("%-8s marginal %.4f [%.4f-%.4f]  ->  mondrian %.4f [%.4f-%.4f]"
              % (lv, r["marginal"]["mean"], r["marginal"]["min"], r["marginal"]["max"],
                 r["mondrian"]["mean"], r["mondrian"]["min"], r["mondrian"]["max"]))
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
