r"""
part12_derived.py — PART XII 문서가 인용하는 **파생 수치**를 영구화한다.

왜 필요한가
-----------
`audit_part12.py` 가 문서의 소수 3자리 수치 208종 중 **22종이 어느 JSON 에도 없음**을 적발했다.
대부분은 콘솔에서 한 번 계산한 파생량이었다 — CLAUDE.md §14 반복 패턴 ③
("산출물에 없는 수치를 인용", 이번이 **4번째 재발**).

여기서 계산·저장하는 것
-----------------------
  D1  규칙 기여 분해   각 항이 예측 epi 를 log 로 얼마나 밀었나 (중앙값 대입 근사)
  D2  정직도 <-> 실제 CP 커버리지 Spearman
  D3  ref 의 w.phi_hat  (ell 크기/방향 분해의 기준값)
  D4  delay_3 주봉 에너지 손실의 행방 (수지)
  D5  P-KEY 그룹차를 규칙에 대입한 '7.8배 위반' 산술

    python scripts/data/part12_derived.py
"""
from __future__ import annotations
import io, json, os, sys
import numpy as np
import torch
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)
from train_der import DERModel                                     # noqa: E402

J = os.path.join(ROOT, "artifacts")
OUT = os.path.join(J, "part12_derived.json")
CACHE = os.path.join(J, "peak_decomp_persample.npz")
FEATS = ("feat_rms", "pk_far", "halfwidth_sub")
BAR = "=" * 100


def load(n):
    with io.open(os.path.join(J, n + ".json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    MT = load("epi_mapping_transfer")
    beta = MT["beta"]                       # [절편, featRMS, pk/far, 반치폭]
    ATL = load("gcc_curve_atlas")
    PKD = load("peak_decomposition")
    PKF = load("peak_decomposition_followup")
    MND = load("mechanism_norm_vs_direction")
    EBF = load("energy_budget_full_lag")["result"]

    z = np.load(CACHE, allow_pickle=True)
    D = {k: z[k].item() for k in z.files}
    REF = "ref (학습분포)"
    ref = D[REF]
    rmed = {f: float(np.median(ref[f])) for f in FEATS}

    out = {"_note": "PART XII 문서가 인용하는 파생 수치. audit_part12.py 가 대조한다."}

    # ---------- D1 : 규칙 기여 분해 ----------
    print(BAR); print("D1 — 규칙 기여 분해 (중앙값 대입 근사, log 단위)"); print(BAR)
    print("  회귀식 계수: 절편 %+.4f  featRMS %+.4f  pk/far %+.4f  반치폭 %+.4f" % tuple(beta))
    print("  %-16s %14s %14s %14s %14s"
          % ("조건", "featRMS 기여", "pk/far 기여", "반치폭 기여", "합"))
    D1 = {}
    for nm in D:
        med = {f: float(np.median(D[nm][f])) for f in FEATS}
        c = [beta[i + 1] * np.log(med[f] / rmed[f]) for i, f in enumerate(FEATS)]
        D1[nm] = dict(feat_rms=float(c[0]), pk_far=float(c[1]),
                      halfwidth=float(c[2]), total=float(sum(c)))
        if nm in ("delay_3", "TDL-C", "occ 0.05", "snr_1"):
            print("  %-16s %14.3f %14.3f %14.3f %14.3f"
                  % (nm, c[0], c[1], c[2], sum(c)))
    out["D1_rule_contributions"] = D1

    # ---------- D2 : 정직도 <-> 커버리지 ----------
    print("\n" + BAR); print("D2 — 정직도와 실제 CP 커버리지의 순위 일치"); print(BAR)
    DEC = ATL["decomposition"]; COV = PKD["median"]
    ks = [k for k in DEC if k in COV]
    h = [DEC[k]["honesty"] for k in ks]
    c = [COV[k]["cov"] for k in ks]
    sp = sps.spearmanr(h, c)
    D2 = dict(n=len(ks), spearman=float(sp.statistic), p_value=float(sp.pvalue),
              pairs={k: dict(honesty=DEC[k]["honesty"], coverage=COV[k]["cov"]) for k in ks})
    print("  n=%d  Spearman = %+.4f  (p = %.3e)" % (D2["n"], D2["spearman"], D2["p_value"]))
    out["D2_honesty_vs_coverage"] = D2

    # ---------- D3 : ref 의 w.phi_hat ----------
    print("\n" + BAR); print("D3 — ell 크기/방향 분해의 기준값"); print(BAR)
    ck = torch.load(os.path.join(ROOT, "output_final_der", "der_model.pt"), weights_only=False)
    model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
    model.load_state_dict(ck["model"]); model.eval()
    W = model.head.fc.weight.detach().numpy().astype(np.float64)
    bb = model.head.fc.bias.detach().numpy().astype(np.float64)
    w_eff = W[2] + W[4]; b_eff = float(bb[2] + bb[4])
    dec = MND["decomposition"][REF]
    D3 = dict(w_eff_mean=float(w_eff.mean()), w_eff_neg_frac=float((w_eff < 0).mean()),
              b_eff=b_eff, ref_norm=dec["norm"], ref_proj=dec["proj"], ref_ell=dec["ell"],
              w_ell0_mean=float(W[2].mean()), w_ell0_neg=float((W[2] < 0).mean()),
              w_ell2_mean=float(W[4].mean()), w_ell2_neg=float((W[4] < 0).mean()))
    print("  w_eff (ell0+ell2) 평균 %+.5f  음수비율 %.3f   b_eff %+.4f"
          % (D3["w_eff_mean"], D3["w_eff_neg_frac"], b_eff))
    print("  ref: ||phi|| = %.4f   w.phi_hat = %.4f   ell(합) = %.4f"
          % (D3["ref_norm"], D3["ref_proj"], D3["ref_ell"]))
    print("  개별: ell0 평균 %+.5f (음수 %.3f) / ell2 평균 %+.5f (음수 %.3f)"
          % (D3["w_ell0_mean"], D3["w_ell0_neg"], D3["w_ell2_mean"], D3["w_ell2_neg"]))
    out["D3_weight_reference"] = D3

    # ---------- D4 : delay_3 주봉 에너지 손실의 행방 ----------
    print("\n" + BAR); print("D4 — delay_3 주봉 에너지 손실의 행방 (전 lag 총 에너지 = 1)"); print(BAR)
    r, d = EBF[REF], EBF["delay_3"]
    D4 = dict(peak_loss=r["peak"] - d["peak"],
              to_near=d["near"] - r["near"], to_mid=d["mid"] - r["mid"],
              to_out=d["out"] - r["out"])
    D4["balance"] = D4["to_near"] + D4["to_mid"] + D4["to_out"]
    D4["residual"] = D4["peak_loss"] - D4["balance"]
    print("  주봉 손실       %.4f  (%.4f -> %.4f)" % (D4["peak_loss"], r["peak"], d["peak"]))
    print("    -> 부봉(창안)  +%.4f" % D4["to_near"])
    print("    -> 먼쪽(창안)  +%.4f" % D4["to_mid"])
    print("    -> 창밖        +%.4f" % D4["to_out"])
    print("  수지 합 %.4f   잔차 %.2e  (0 이어야 정상)" % (D4["balance"], D4["residual"]))
    # occ 대조
    o = EBF["occ 0.05"]
    D4["occ_to_near"] = o["near"] - r["near"]; D4["occ_to_out"] = o["out"] - r["out"]
    print("  [대조] occ 0.05: 부봉 +%.4f (거의 0) / 창밖 +%.4f"
          % (D4["occ_to_near"], D4["occ_to_out"]))
    out["D4_energy_balance"] = D4

    # ---------- D5 : 7.8배 위반 산술 ----------
    print("\n" + BAR); print("D5 — P-KEY 그룹차를 규칙에 대입 (delay_3 bias-大 / bias-小)"); print(BAR)
    m = PKF["matched"]["delay_3"]
    terms = dict(feat_rms=beta[1] * np.log(m["feat_rms"]),
                 pk_far=beta[2] * np.log(m["pk_far"]),
                 halfwidth=beta[3] * np.log(m["halfwidth_sub"]))
    tot = float(sum(terms.values()))
    D5 = dict(ratios={k: m[k] for k in ("feat_rms", "pk_far", "halfwidth_sub", "epi", "err")},
              terms={k: float(v) for k, v in terms.items()}, sum_log=tot,
              predicted_epi_ratio=float(np.exp(tot)), observed_epi_ratio=m["epi"],
              violation=float(np.exp(tot) / m["epi"]))
    for k, v in terms.items():
        print("  %-12s 배율 %.3f  ->  기여 %+.4f" % (k, m[k if k != "halfwidth" else "halfwidth_sub"], v))
    print("  합 %+.4f  =>  규칙 예측 epi 배율 %.3f   실측 %.3f   **위반 %.2f 배**"
          % (tot, D5["predicted_epi_ratio"], D5["observed_epi_ratio"], D5["violation"]))
    out["D5_pkey_violation"] = D5

    # ---------- D6 : 문서 표에만 나오는 합성 열 ----------
    print(BAR); print("D6 — 문서 표의 합성 열 (방향 기여 합 · ref 정규화 비)"); print(BAR)
    D6 = {"dir_total": {}, "ref_normalized_ratio": {}}
    for nm, dc in MND["decomposition"].items():
        D6["dir_total"][nm] = float(dc["d_dir"] + dc["d_cross"])
    MTr = MT["result"]
    base = MTr[REF]["ratio"]
    for nm, v in MTr.items():
        D6["ref_normalized_ratio"][nm] = float(v["ratio"] / base)
    print("  %-16s %14s %16s" % ("조건", "Dell 방향(합)", "실제/예측 (ref 정규화)"))
    for nm in ("delay_3", "occ 0.05", "TDL-C", "kf_3", "snr_1"):
        print("  %-16s %14.3f %16.3f"
              % (nm, D6["dir_total"][nm], D6["ref_normalized_ratio"].get(nm, float("nan"))))
    out["D6_table_columns"] = D6

    # ---------- D7 : 74.6 보조표 (후보 3종 x 조건, 20셀 평균) ----------
    print(BAR); print("D7 — 74.6 보조 비교표 (20셀 평균)"); print(BAR)
    RRC = load("ruleresid_confirm")
    cfg = RRC["config"]; A2 = str(cfg["alpha1"])
    D7 = {}
    print("  %-12s %14s %14s %14s" % ("조건", "rule_resid", "epi_low", "psr_only"))
    for lv in ("delay_3", "kf_3", "cfo_4", "snr_1", "sco_4", "delay_1", "delay_2",
               "delay_0", "kf_1", "kf_2", "sco_2", "snr_2"):
        row = {}
        for sname in ("rule_resid", "epi_low", "psr_only"):
            row[sname] = float(np.mean([RRC["power"][L][A2][sname]["%s|%s" % (g, lv)]
                                        for L in cfg["confirm_models"]
                                        for g in cfg["gen_seeds"]]))
        D7[lv] = row
        if lv in ("delay_3", "kf_3", "cfo_4", "snr_1", "sco_4"):
            print("  %-12s %14.3f %14.3f %14.3f"
                  % (lv, row["rule_resid"], row["epi_low"], row["psr_only"]))
    out["D7_aux_candidate_means"] = D7

    # ---------- D8 : 라벨 불필요 양(준수도) vs 커버리지 ----------
    print(BAR); print("D8 - 준수도(라벨 불필요) vs 커버리지"); print(BAR)
    ks8 = [k for k in DEC if k in COV]
    comp = [DEC[k]["compliance"] for k in ks8]
    cv = [COV[k]["cov"] for k in ks8]
    r8 = sps.spearmanr(comp, cv)
    k9 = [k for k in ks8 if k != "snr_1"]
    r9 = sps.spearmanr([DEC[k]["compliance"] for k in k9], [COV[k]["cov"] for k in k9])
    D8 = dict(n=len(ks8), spearman=float(r8.statistic), p_value=float(r8.pvalue),
              n_wo_snr=len(k9), spearman_wo_snr=float(r9.statistic),
              p_wo_snr=float(r9.pvalue))
    print("  준수도(라벨 불필요) vs 커버리지 : rho = %+.4f (p=%.2e, n=%d)"
          % (D8["spearman"], D8["p_value"], D8["n"]))
    print("    snr_1 제외                    : rho = %+.4f (p=%.2e, n=%d)"
          % (D8["spearman_wo_snr"], D8["p_wo_snr"], D8["n_wo_snr"]))
    print("  [대조] 정직도(라벨 **필요**)      : rho = %+.4f" % D2["spearman"])
    out["D8_compliance_vs_coverage"] = D8

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
