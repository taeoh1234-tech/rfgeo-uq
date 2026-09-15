r"""
ceiling_pairing.py - 고정점수 천장(1-beta*)과 정규화기 복원 커버리지의 모델별 짝지음.

배경(2026-09-09, 외부 AI 검토 #4 검증): 논문 5.3절의 옛 문구 "the support ceiling in
action" 은 delay_3 복원이 0.69 에서 멈춘 원인을 지지집합 천장에 귀속시켰다. 그러나
식 (4) 정규화기는 점수 자체를 바꾸므로 고정점수 천장이 적용되지 않는다.
검증 규율(§14): 시드 범위 [0.53,0.72] 와 풀링 평균 0.697 의 비교는 판정력이 없다 -
반드시 같은 모델끼리 짝지어야 한다.

입력: betastar_table3.json (모델별 beta*, cal_max)
      normalizer_reconfirm.json (RULE+60 = kappa 0.6, delay_3 모델별 복원 커버리지)
출력: ceiling_pairing.json

    python scripts/data/ceiling_pairing.py
"""
from __future__ import annotations
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "artifacts")
OUT = os.path.join(SRC, "ceiling_pairing.json")


def load(n):
    with io.open(os.path.join(SRC, n), encoding="utf-8") as f:
        return json.load(f)


def main():
    bs = load("betastar_table3.json")
    rc = load("normalizer_reconfirm.json")["cells"]

    # 모델 키 매핑: reconfirm 은 'seed0*', betastar 는 'seed0' (cal_max 5.218 동일)
    name = {"m0": "m0", "m1": "m1", "m2": "m2", "m3": "m3", "seed0*": "seed0"}
    rows, exceed = [], 0
    for rk, bk in name.items():
        cov = [rc[k]["cov"] for k in rc
               if k.startswith("RULE+60|%s|delay_3|" % rk)]
        c = float(np.mean(cov))
        b = bs[bk]["delay_3"]
        over = c > 1 - b
        exceed += over
        rows.append({"model": bk, "beta_star": round(b, 4),
                     "ceiling": round(1 - b, 4), "repaired_cov": round(c, 4),
                     "margin": round(c - (1 - b), 4), "exceeds": bool(over),
                     "n_gensets": len(cov)})

    # kappa 용량-반응 (포화 없음 = 게이트가 구속의 실증)
    sweep = {}
    for arm, kap in (("RULE+", 0.45), ("RULE+60", 0.6)):
        cov = [v["cov"] for k, v in rc.items()
               if k.split("|")[0] == arm and "delay_3" in k]
        dsc = [v["disc"] for k, v in rc.items()
               if k.split("|")[0] == arm and "delay_3" in k]
        sweep[str(kap)] = {"cov_mean": round(float(np.mean(cov)), 4),
                           "disc_med": round(float(np.median(dsc)), 4)}

    out = {
        "_note": ("정규화기(RULE kappa=0.6)의 delay_3 복원 커버리지 vs 각 모델 자신의 "
                  "고정점수 천장 1-beta*. 짝지은 비교. 5.3절 'past the fixed-score "
                  "ceiling' 의 근거. kappa 스윕은 포화 부재(=게이트 구속)의 근거."),
        "rows": rows, "n_exceed": exceed, "n_models": len(rows),
        "pooled": {"beta_star_mean": round(float(np.mean(
            [r["beta_star"] for r in rows])), 4),
            "cov_mean": round(float(np.mean(
                [r["repaired_cov"] for r in rows])), 4)},
        "kappa_sweep_delay3": sweep,
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("모델별 짝지음 (delay_3, RULE kappa=0.6):")
    for r in rows:
        print("  %-6s beta*=%.4f  천장=%.4f  복원=%.4f  %s"
              % (r["model"], r["beta_star"], r["ceiling"], r["repaired_cov"],
                 "초과 +%.3f" % r["margin"] if r["exceeds"]
                 else "이내 %.3f" % r["margin"]))
    print("=> %d/%d 모델이 자기 천장 초과. kappa 스윕: %s"
          % (exceed, len(rows), out["kappa_sweep_delay3"]))
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
