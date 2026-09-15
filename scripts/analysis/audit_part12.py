r"""
audit_part12.py — CLAUDE.md PART XII 에 등재한 수치를 산출물 JSON 과 기계 대조한다.

CLAUDE.md §14 반복 패턴 ③("산출물에 없는 수치를 인용", 3회 재발) 방지 장치.
파형 축 세션에서 도입한 MD<->JSON 감사와 같은 목적.

    python scripts/data/audit_part12.py
"""
from __future__ import annotations
import io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
J = os.path.join(ROOT, "artifacts")
MD = os.path.join(ROOT, "CLAUDE.md")


def load(name):
    with io.open(os.path.join(J, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def part12(md):
    i = md.index("# PART XII")
    return md[i:]


def main():
    md = part12(io.open(MD, encoding="utf-8").read())
    EMT = load("epi_mapping_transfer")["result"]
    ATL = load("gcc_curve_atlas")
    RRD = load("rule_residual_detector")
    MAB = load("mechanism_ABC")
    MND = load("mechanism_norm_vs_direction")
    PND = load("phi_norm_detector")
    EBF = load("energy_budget_full_lag")["result"]
    AVL = load("ambiguity_vs_local_error")
    PKD = load("peak_decomposition")
    PKF = load("peak_decomposition_followup")
    DRV = load("part12_derived")
    SRM = load("seed_replication_mechanism")
    PNC = load("phinorm_confirm")
    RRC = load("ruleresid_confirm")

    C = []          # (라벨, 문서 문자열, 실제값, 허용오차)

    def chk(lab, txt, val, tol=0.0):
        C.append((lab, txt, val, tol))

    # --- §69.2 규칙 이전 ---
    for nm, key in (("ref", "ref (학습분포)"), ("sco_4", "sco_4 (NULL)"),
                    ("occ0.20", "occ 0.20"), ("occ0.05", "occ 0.05"),
                    ("kf_3", "kf_3"), ("TDL-A", "TDL-A"), ("TDL-C", "TDL-C"),
                    ("delay_2", "delay_2"), ("delay_3", "delay_3"), ("snr_1", "snr_1")):
        chk("EMT ratio %s" % nm, "%.3f" % EMT[key]["ratio"], EMT[key]["ratio"])
        chk("EMT pred %s" % nm, "%.1f" % EMT[key]["pred"], EMT[key]["pred"])
        chk("EMT act %s" % nm, "%.1f" % EMT[key]["actual"], EMT[key]["actual"])

    # --- §69.3 정직도 분해 ---
    DEC = ATL["decomposition"]
    for nm in ("delay_3", "delay_2", "TDL-C", "TDL-A", "kf_3", "snr_1",
               "occ 0.05", "occ 0.20", "sco_4 (NULL)", "delay_1"):
        d = DEC[nm]
        chk("DEC 준수도 %s" % nm, "%.3f" % d["compliance"], d["compliance"])
        chk("DEC 타당도 %s" % nm, "%.3f" % d["validity"], d["validity"])
        chk("DEC 정직도 %s" % nm, "%.3f" % d["honesty"], d["honesty"])

    # --- §69.6(a) 기하 ---
    G = ATL["geometry"]
    chk("geo 이론최대", "%.2f" % G["theoretical_max_lag_samples"], G["theoretical_max_lag_samples"])
    chk("geo 실측최대", "%.2f" % G["empirical_abs_lag_p100"], G["empirical_abs_lag_p100"])
    chk("geo p99", "%.2f" % G["empirical_abs_lag_p99"], G["empirical_abs_lag_p99"])
    chk("geo 중앙", "%.2f" % G["empirical_abs_lag_median"], G["empirical_abs_lag_median"])
    chk("geo span med", "%.2f" % G["pair_span_median"], G["pair_span_median"])
    chk("geo span max", "%.2f" % G["pair_span_max"], G["pair_span_max"])
    chk("geo >65 비율", "%.1f" % (100 * G["frac_span_gt_65"]), 100 * G["frac_span_gt_65"])

    # --- §69.6(b) 증분 R^2 ---
    chk("atlas base R2", "%.4f" % ATL["base_r2"], ATL["base_r2"])
    chk("atlas n_sub 증분", "+%.4f" % ATL["incremental"]["n_sub"]["delta"],
        ATL["incremental"]["n_sub"]["delta"])
    chk("atlas 전체 증분", "+%.4f" % (ATL["r2_all"] - ATL["base_r2"]), ATL["r2_all"] - ATL["base_r2"])

    # --- §69.4 탐지력 ---
    for nm in ("NEG", "sco_4 (NULL)", "delay_1", "delay_2", "delay_3",
               "occ 0.20", "occ 0.05", "TDL-A", "TDL-C", "kf_3", "snr_1"):
        for k in ("rule_resid", "epi_low", "psr_only", "D_psr"):
            v = RRD["power"][nm]["0.05"][k]
            chk("RRD a05 %s/%s" % (nm, k), "%.3f" % v, v)

    # --- §70 관문 ---
    chk("GATE1", "%.3f" % (MAB["gates"]["g1_rel"] * 1e7), MAB["gates"]["g1_rel"] * 1e7)
    chk("GATE2", "%.3f" % (MAB["gates"]["g2_rel"] * 1e7), MAB["gates"]["g2_rel"] * 1e7)
    chk("PCA k", "%d" % MAB["gates"]["pca_k"], MAB["gates"]["pca_k"])
    chk("죽은유닛", "%d" % MAB["gates"]["n_dead"], MAB["gates"]["n_dead"])

    # --- §70.3 A3/B1/B3/C3 ---
    for nm in ("ref (학습분포)", "delay_3", "occ 0.05", "TDL-C", "kf_3", "snr_1",
               "sco_4 (NULL)", "delay_1", "delay_2"):
        chk("A3 주봉z %s" % nm, "%.2f" % MAB["A3"][nm]["peak_z"], MAB["A3"][nm]["peak_z"])
        chk("A3 부봉z %s" % nm, "%.3f" % MAB["A3"][nm]["sub_z"], MAB["A3"][nm]["sub_z"])
        chk("A3 비 %s" % nm, "%.3f" % MAB["A3"][nm]["ratio"], MAB["A3"][nm]["ratio"])
        chk("B1 norm %s" % nm, "%.3f" % MAB["B1"][nm]["ratio"], MAB["B1"][nm]["ratio"])
        chk("B3 perp %s" % nm, "%.4f" % MAB["B3"][nm]["perp_frac"], MAB["B3"][nm]["perp_frac"])
        chk("B3 dl_par %s" % nm, "%.4f" % MAB["B3"][nm]["dl_par"], MAB["B3"][nm]["dl_par"])
        chk("B3 dl_perp %s" % nm, "%.4f" % MAB["B3"][nm]["dl_perp"], MAB["B3"][nm]["dl_perp"])
        chk("B3 회복 %s" % nm, "%.3f" % MAB["B3"][nm]["epi_par_only_ratio"],
            MAB["B3"][nm]["epi_par_only_ratio"])
        chk("C3 tr %s" % nm, "%.4f" % MAB["C3"][nm]["tr_ratio"], MAB["C3"][nm]["tr_ratio"])
        chk("C3 ell몫 %s" % nm, "%.1f" % (100 * MAB["C3"][nm]["ell_share"]),
            100 * MAB["C3"][nm]["ell_share"])

    # --- §70.4 사슬 · 크기/방향 ---
    chk("w_eff mean", "%.5f" % MND["w_eff_mean"], MND["w_eff_mean"])
    chk("w_eff neg", "%.3f" % MND["w_eff_neg_frac"], MND["w_eff_neg_frac"])
    for nm in ("ref (학습분포)", "delay_1", "delay_2", "delay_3", "occ 0.05",
               "TDL-C", "kf_3", "snr_1", "sco_4 (NULL)"):
        ch = MND["chain"][nm]; dc = MND["decomposition"][nm]
        chk("chain map %s" % nm, "%.3f" % ch["map_mean"], ch["map_mean"])
        chk("chain norm %s" % nm, "%.3f" % ch["norm"], ch["norm"])
        chk("chain dell %s" % nm, "%.4f" % ch["d_ell"], ch["d_ell"])
        chk("chain tr %s" % nm, "%.3f" % ch["tr_ratio"], ch["tr_ratio"])
        chk("chain epi %s" % nm, "%.3f" % ch["epi"], ch["epi"])
        chk("dir 방향비 %s" % nm, "%.3f" % dc["proj_ratio"], dc["proj_ratio"])
        chk("dir dsize %s" % nm, "%.3f" % dc["d_size"], dc["d_size"])
        chk("act 활성비율 %s" % nm, "%.3f" % MND["activation"][nm]["active_ratio"],
            MND["activation"][nm]["active_ratio"])

    # --- §70.5 phi_norm 탐지기 ---
    for nm in ("NEG", "sco_4 (NULL)", "delay_1", "delay_2", "delay_3",
               "occ 0.05", "TDL-C", "kf_3", "snr_1"):
        for k in ("phi_norm", "rule_resid", "OR"):
            v = PND["power"][nm]["0.05"][k]
            chk("PND a05 %s/%s" % (nm, k), "%.3f" % v, v)

    # --- §69.6(c) 에너지 예산 ---
    for nm in ("ref (학습분포)", "delay_1", "delay_3", "occ 0.05", "occ 0.20",
               "TDL-C", "kf_3", "snr_1"):
        for f in ("peak", "near", "mid", "out", "in_window"):
            chk("EBF %s/%s" % (nm, f), "%.4f" % EBF[nm][f], EBF[nm][f])

    # --- §70.2 H-D2 / H-E1 ---
    chk("HD2 r2 epi", "%.4f" % AVL["hd2"]["r2_epi"], AVL["hd2"]["r2_epi"])
    chk("HD2 r2 err", "%.4f" % AVL["hd2"]["r2_err"], AVL["hd2"]["r2_err"])
    chk("HD2 ratio", "%.2f" % AVL["hd2"]["ratio"], AVL["hd2"]["ratio"])
    for nm, k in (("ref (학습분포)", 3), ("TDL-C", 3), ("kf_3", 3),
                  ("delay_2", 3), ("delay_3", 3)):
        st = AVL["stratified"].get(nm, {})
        if str(k) in st:
            chk("HE1 %s ratio" % nm, "%.2f" % st[str(k)]["ratio"], st[str(k)]["ratio"])
    for nm in ("delay_3", "delay_2", "occ 0.05", "ref (학습분포)"):
        r = AVL["recovery"][nm]
        chk("HE1 sp_clean %s" % nm, "%.3f" % r["sp_clean"], r["sp_clean"])
        chk("HE1 sp_dirty %s" % nm, "%.3f" % r["sp_dirty"], r["sp_dirty"])

    # --- §69.5 봉우리 분해 ---
    for nm in ("ref (학습분포)", "delay_3", "occ 0.05", "TDL-C"):
        chk("PKD lag %s" % nm, "%.3f" % PKD["ratio"][nm]["lag_err"], PKD["ratio"][nm]["lag_err"])
        chk("PKD hw %s" % nm, "%.3f" % PKD["ratio"][nm]["halfwidth_sub"],
            PKD["ratio"][nm]["halfwidth_sub"])
    for nm in ("delay_3", "TDL-C"):
        m = PKF["matched"][nm]
        for f in ("pk_far", "halfwidth_sub", "feat_rms", "err", "epi"):
            chk("PKF %s/%s" % (nm, f), "%.3f" % m[f], m[f])

    # --- 72 시드재현 ---
    SM = SRM["models"]
    for L, r in SM.items():
        chk("SRM proj %s" % L, "%.4f" % r["proj_ref"], r["proj_ref"])
        chk("SRM wmean %s" % L, "%.5f" % r["w_eff_mean"], r["w_eff_mean"])
        chk("SRM rho %s" % L, "%.4f" % r["spearman_norm_epi"], r["spearman_norm_epi"])
        chk("SRM err %s" % L, "%.1f" % r["err_ref"], r["err_ref"])
        for nm in ("delay_3", "TDL-C", "occ 0.05", "sco_4 (NULL)"):
            c = r["chain"][nm]
            for f in ("act_ratio", "norm_ratio", "tr_ratio", "epi_ratio"):
                chk("SRM %s/%s/%s" % (L, nm, f), "%.3f" % c[f], c[f])

    # --- 73 phi_norm 확증 ---
    A1 = str(PNC["config"]["alpha1"])
    for L in PNC["config"]["confirm_models"]:
        for g in PNC["config"]["gen_seeds"]:
            for lv in ("delay_3", "kf_3", "cfo_4"):
                v = PNC["power"][L][A1]["%s|%s" % (g, lv)]
                chk("PNC %s/%s/%s" % (L, g, lv), "%.3f" % v, v)
        for nm in ("occ 0.05", "occ 0.20", "TDL-A", "TDL-C"):
            v = PNC["power"][L][A1]["gen|%s" % nm]
            chk("PNC %s/%s" % (L, nm), "%.3f" % v, v)
    for L, v in PNC["falsifier"]["neg"].items():
        chk("PNC NEG %s" % L, "%.4f" % v, v)
    chk("PNC sco min", "%.4f" % PNC["falsifier"]["sco_min"], PNC["falsifier"]["sco_min"])
    chk("PNC sco max", "%.4f" % PNC["falsifier"]["sco_max"], PNC["falsifier"]["sco_max"])

    # --- 74 규칙잔차 확증 ---
    A2 = str(RRC["config"]["alpha1"])
    for L in RRC["config"]["confirm_models"]:
        for g in RRC["config"]["gen_seeds"]:
            for lv in ("delay_3", "kf_3", "cfo_4", "snr_1", "delay_0", "kf_1", "kf_2",
                       "sco_2", "snr_2", "delay_1", "delay_2"):
                v = RRC["power"][L][A2]["rule_resid"]["%s|%s" % (g, lv)]
                chk("RRC %s/%s/%s" % (L, g, lv), "%.3f" % v, v)
        for i, f in enumerate(("절편", "featRMS", "pk/far", "반치폭")):
            v = RRC["beta"][L]["mean"][i]
            chk("RRC beta %s/%s" % (L, f), "%.4f" % v, v)
    for L, v in RRC["falsifier"]["neg"].items():
        chk("RRC NEG %s" % L, "%.4f" % v, v)
    chk("RRC sco min", "%.4f" % RRC["falsifier"]["sco_min"], RRC["falsifier"]["sco_min"])
    chk("RRC sco max", "%.4f" % RRC["falsifier"]["sco_max"], RRC["falsifier"]["sco_max"])

    # ---------- 대조 ----------
    print("=" * 100)
    print("PART XII  MD <-> JSON 기계 감사")
    print("=" * 100)
    miss, ok = [], 0
    for lab, txt, val, tol in C:
        # 문서에 그 숫자 문자열이 존재하는가 (부호/천단위 콤마 변형 허용)
        cands = {txt, txt.lstrip("+"), txt.replace("-", "−"),
                 txt.lstrip("+").replace("-", "−")}
        try:
            f = float(val)
            if abs(f) >= 1000:
                cands.add("{:,.0f}".format(f))
                cands.add("{:,.1f}".format(f))
        except (TypeError, ValueError):
            pass
        if any(c in md for c in cands):
            ok += 1
        else:
            miss.append((lab, txt, val))
    print("  대조 항목 %d개 중 문서에서 확인 %d개 / 미확인 %d개" % (len(C), ok, len(miss)))
    if miss:
        print("\n  [미확인] — 문서에 안 쓴 수치이거나 반올림이 다른 것")
        for lab, txt, val in miss[:60]:
            print("    %-28s 문서표기 '%s'  실제 %r" % (lab, txt, val))
        if len(miss) > 60:
            print("    ... 외 %d개" % (len(miss) - 60))
    print("\n  ⚠️ '미확인'은 오류가 아니라 **문서에 인용하지 않은 값**일 수 있다.")
    print("     실제 위험은 그 반대 — 문서에 있는데 JSON 에 없는 값이다. 아래에서 본다.")

    # 역방향: 문서의 소수 3자리 수치 중 어떤 JSON 에도 없는 것
    allv = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)):
            for fmt in ("%.1f", "%.2f", "%.3f", "%.4f"):
                allv.add(fmt % o)
                allv.add(fmt % (100 * o))
                allv.add(fmt % (-o))
            # 과학표기 가수부 (GATE 처럼 3.325e-07 를 "3.325" 로 인용하는 경우)
            import math
            if o != 0 and math.isfinite(o):
                m = abs(o) / (10 ** math.floor(math.log10(abs(o))))
                for fmt in ("%.2f", "%.3f", "%.4f"):
                    allv.add(fmt % m)
    for d in (EMT, ATL, RRD, MAB, MND, PND, EBF, AVL, PKD, PKF, DRV, SRM, PNC, RRC):
        walk(d)
    nums = set(re.findall(r"(?<![\w.])\d+\.\d{3}(?![\d])", md))
    orphan = sorted(n for n in nums if n not in allv)
    print("\n  문서의 소수3자리 수치 %d종 중 **어느 JSON 에도 없는 것: %d종**" % (len(nums), len(orphan)))
    if orphan:
        print("    " + " ".join(orphan[:40]))
    print("\n[감사 종료]")


if __name__ == "__main__":
    main()
