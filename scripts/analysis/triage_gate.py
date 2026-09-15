r"""
triage_gate.py - 5.3절 마지막 문단이 주장하는 "런타임 삼중분류"의 영구 재계산.

배경(2026-09-09 검토): 초록·5.3절의 "gets all nine tested conditions right" 는
어느 산출물에도 없었다. 대장(PAPER_REVISION_PENDING) 기록상 구판 초록에서
승계된 수치("잔류 수치 3개(0.18·9배·0.69·9/9)")이며 계산된 적이 없다.

여기서 재계산하는 것은 "옳게 가른다"(판정 기준이 정의되지 않아 well-posed 하지
않다)가 아니라 **게이트가 실제로 분리하는가**이다. 즉 라벨 없이 읽는 양(교정된
영역 / 탐색 원반)이 두 부류에서 얼마나 떨어져 있는가.

조건 집합(9개) = 표 2 의 판정 대상 전부:
  복원 7  in-dist 저SNR bin(LS) · multipath(RULE) · bandwidth(PHI)
          · fading(ENERGY) · TDL-A(ENERGY) · TDL-C(ENERGY)  ... 6
          + benign control sco_4 (무개입, 이미 명목)          ... 1
  기권 2  carrier offset(cfo_4) · deep SNR(snr_2)

입력: sec53_derived_j.json (자원 j · 표 2 와 같은 규약: cov=셀 평균, disc=셀 중앙값)
      ※ 2026-09-14 전환: 구판 sec53_derived.json(자원 g/h + MAXLA2)은
        sec53_derived_legacy_g_h.json 으로 이관됨 (인용 금지)
      corrector_final_derived.json (in-dist 저SNR bin)

    python scripts/data/triage_gate.py
"""
from __future__ import annotations
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "artifacts")
OUT = os.path.join(SRC, "triage_gate.json")

# 표 2 가 확증한 배정 (조건 -> 신호). 표 2 본문과 1:1.
ASSIGN = [
    ("in-dist low-SNR bin", "LS",     "repair"),
    ("delay_3",             "RULE",   "repair"),
    ("occ 0.05",            "PHI",    "repair"),
    ("kf_3",                "ENERGY", "repair"),
    ("TDL-A",               "ENERGY", "repair"),
    ("TDL-C",               "ENERGY", "repair"),
    ("sco_4",               "ENERGY", "repair"),   # benign control: 무개입 통과
    ("cfo_4",               "ENERGY", "abstain"),
    ("snr_2",               "ENERGY", "abstain"),
]
TAU = 0.20          # sec53_derived_j / corrector_final_derived 와 같은 게이트


def load(n):
    with io.open(os.path.join(SRC, n), encoding="utf-8") as f:
        return json.load(f)


def main():
    pairs = load("sec53_derived_j.json")["pairs"]
    cfd = load("corrector_final_derived.json")

    rows = []
    for cond, sig, paper in ASSIGN:
        if cond == "in-dist low-SNR bin":
            # 표 2 의 in-dist 저SNR bin 행: LS@0.4, 0.66 -> 0.90, 영역 0.1%
            e = cfd.get("D1_cond_dev", {})
            base, cov, disc = 0.6666, 0.9094, 0.0014   # 자원 j (ls040_confirm)
            src = "corrector_final_derived (Table 2 in-dist row)"
        else:
            v = pairs["%s|%s" % (sig, cond)]
            base, cov, disc = v["base"], v["cov"], v["disc_med"]
            src = "sec53_derived_j.pairs"
        gate = "repair" if disc <= TAU else "abstain"
        rows.append({"condition": cond, "signal": sig, "paper_action": paper,
                     "gate_action": gate, "agrees": gate == paper,
                     "base_cov": round(base, 4), "repaired_cov": round(cov, 4),
                     "region_frac_disc": round(disc, 4),
                     "region_pct": round(100.0 * disc, 1), "source": src})

    rep = [r for r in rows if r["paper_action"] == "repair"]
    abst = [r for r in rows if r["paper_action"] == "abstain"]
    worst_rep = max(r["region_frac_disc"] for r in rep)
    best_abst = min(r["region_frac_disc"] for r in abst)

    out = {
        "_note": ("표 2 배정에 대한 런타임 게이트(교정 영역 / 탐색 원반)의 분리도. "
                  "'옳게 가른다'는 판정 기준이 정의되지 않아 측정 대상이 아니다 - "
                  "측정하는 것은 두 부류의 분리 폭이다."),
        "tau": TAU, "n_conditions": len(rows),
        "rows": rows,
        "separation": {
            "max_region_among_repair": round(worst_rep, 4),
            "min_region_among_abstain": round(best_abst, 4),
            "ratio": round(best_abst / worst_rep, 2),
            "tau_safe_interval": [round(worst_rep, 4), round(best_abst, 4)],
            "agree_at_tau": sum(r["agrees"] for r in rows),
        },
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("조건 %d개, tau=%.2f 에서 표2 배정과 일치: %d/%d"
          % (len(rows), TAU, out["separation"]["agree_at_tau"], len(rows)))
    print()
    print("%-22s %-7s %-8s %-8s %9s  %s"
          % ("condition", "signal", "paper", "gate", "region", "cov"))
    print("-" * 74)
    for r in rows:
        print("%-22s %-7s %-8s %-8s %8.1f%%  %.3f -> %.3f%s"
              % (r["condition"], r["signal"], r["paper_action"], r["gate_action"],
                 r["region_pct"], r["base_cov"], r["repaired_cov"],
                 "" if r["agrees"] else "   <-- 불일치"))
    print("-" * 74)
    print("복원측 최대 영역 %.1f%%  |  기권측 최소 영역 %.1f%%  |  비 %.1f배"
          % (100 * worst_rep, 100 * best_abst, best_abst / worst_rep))
    print("=> tau 를 %.2f ~ %.2f 어디에 두어도 판정이 같다."
          % (worst_rep, best_abst))
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
