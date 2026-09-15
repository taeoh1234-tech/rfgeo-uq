r"""
seed_replication_mechanism.py — §70 메커니즘이 학습 시드에 견고한가.

사전등록: `prereg/SEED_REPLICATION_PREREG_2026-08-31.md` (학습·측정 전 기록)

설계
----
데이터(`output_final`)·OOD 레시피 고정, **학습 시드만** 0 / 20260628 / 20261628.
동결 모델(`output_final_der`, torch 2.5.1)은 **환경 대조**로만 병기하고 P1~P8 판정에서 제외한다.

측정 (모델당)
-------------
  GATE-A  해석적 수식 == niw_uncertainties
  GATE-B  phi -> head.fc == model(x)
  GATE-C  mean/std 가 모델 간 비트 동일 (같은 데이터 확인)
  P1      (w_eff . phi_hat)_ref < 0        <- 크기 경로의 부호를 정하는 양
  P2      mean(w_eff) < 0
  P3      조건 간 Spearman(||phi||비, epi배) < -0.7
  P4      delay_3 epi 배율 < 1.0
  P5      TDL-C  epi 배율 > 1.0
  P6      ||phi||비: delay_3 > 1.0 이고 TDL-C < 1.0
  P7      C3 ell 몫 > 90% (delay_3)
  P8      B3 안(par) 몫 > 50% (delay_3)
  FALSIFIER  sco_4 epi 배율 in [0.85,1.15]  그리고  in-dist median 오차 in [120,260] m

    python scripts/data/seed_replication_mechanism.py
"""
from __future__ import annotations
import functools, io, json, os, sys
import numpy as np
import torch
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)
from rfgeo.evidential import niw_uncertainties                     # noqa: E402
from train_der import DERModel                                     # noqa: E402
from peak_decomposition import from_npz, gen, SCALE, BAR           # noqa: E402
from rfgeo import waveform as wf                                    # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "seed_replication_mechanism.json")
VAR_KEEP = 0.99
REF = "ref (학습분포)"

# frozen 은 seed 0 과 **비트 동일**로 확인됐다(34 텐서 max|diff| = 0). 중복 채점을 피해
# is_new=False 로 두고 seed 0 만 채점에 넣는다. 채점 분모 = 5 시드.
MODELS = [
    ("frozen (=seed0, 비트동일)", os.path.join(ROOT, "output_final_der", "der_model.pt"), False),
    ("seed 0", os.path.join(ROOT, "output_seed0", "der_model.pt"), True),
    ("seed 20260628", os.path.join(ROOT, "output_seed20260628_clean", "der_model.pt"), True),
    ("seed 20261628", os.path.join(ROOT, "output_seed20261628_clean", "der_model.pt"), True),
    ("seed 20262628", os.path.join(ROOT, "output_seed20262628_clean", "der_model.pt"), True),
    ("seed 20263628", os.path.join(ROOT, "output_seed20263628_clean", "der_model.pt"), True),
]

CASES = [
    (REF, None),
    ("sco_4 (NULL)", lambda: from_npz("sco_4")),
    ("delay_1", lambda: from_npz("delay_1")),
    ("delay_2", lambda: from_npz("delay_2")),
    ("delay_3", lambda: from_npz("delay_3")),
    ("occ 0.05", lambda: gen("_sr_occ05", functools.partial(wf.gen_ofdm, used_frac=0.05))),
    ("TDL-C", lambda: gen(prof="TDL-C")),
    ("kf_3", lambda: from_npz("kf_3")),
    ("snr_1", lambda: from_npz("snr_1")),
]


def epi_from_phi(W, b, phi):
    """phi (N,128) -> (log epi_scale, log tr(LL^T), nu). 전부 numpy, 정확 재현."""
    out = phi @ W.T + b
    ell = out[:, 2:5]
    d0 = np.exp(2.0 * np.clip(ell[:, 0], None, 8.0))
    d2 = np.exp(2.0 * np.clip(ell[:, 2], None, 8.0))
    tr = d0 + ell[:, 1] ** 2 + d2
    nu = 8.0 + 5.0 * np.tanh(out[:, 5])
    return 0.5 * np.log(tr) - 0.5 * np.log(np.maximum(nu - 3.0, 1e-3)), np.log(tr), nu


def main():
    # ---------- 입력 특징을 한 번만 만든다 (모델 간 동일 보장) ----------
    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    DATA = {}
    for nm, fn in CASES:
        F, Y = ((dt["feats"].astype(np.float32), dt["y"].astype(np.float32))
                if fn is None else fn())
        DATA[nm] = (F, Y)
        print("  입력 %-16s n=%d" % (nm, len(F)))

    RES = {}
    ms_ref = None
    for label, path, is_new in MODELS:
        if not os.path.exists(path):
            print("  [skip] %s 없음: %s" % (label, path))
            continue
        ck = torch.load(path, weights_only=False)
        mean, std = ck["mean"], ck["std"]
        model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
        model.load_state_dict(ck["model"]); model.eval()
        W = model.head.fc.weight.detach().numpy().astype(np.float64)
        b = model.head.fc.bias.detach().numpy().astype(np.float64)
        w_eff = W[2] + W[4]
        bb = model.backbone

        @torch.no_grad()
        def fwd(F):
            Z = torch.from_numpy((F - mean) / std)
            x = torch.relu(bb.stem(Z))
            amap = bb.blocks(x)
            phi = bb.proj(bb.pool(amap))
            mu, L, nu = model.head(phi)
            al, ep = niw_uncertainties(L, nu)
            return (amap.numpy().astype(np.float64), phi.numpy().astype(np.float64),
                    mu.numpy(), np.sqrt(np.trace(ep.numpy(), axis1=1, axis2=2)) * SCALE)

        P = {}
        for nm, _ in CASES:
            F, Y = DATA[nm]
            amap, phi, mu, epi = fwd(F)
            P[nm] = dict(amap=amap, phi=phi, epi=epi,
                         err=np.sqrt(((mu - Y / SCALE) ** 2).sum(1)) * SCALE)
        ref = P[REF]

        # ----- GATE -----
        lg, _, nu_m = epi_from_phi(W, b, ref["phi"])
        gA = float(np.max(np.abs(np.exp(lg) * SCALE - ref["epi"]) / np.maximum(ref["epi"], 1e-12)))
        lg2, _, _ = epi_from_phi(W, b, P["delay_3"]["phi"])
        gB = float(np.max(np.abs(np.exp(lg2) * SCALE - P["delay_3"]["epi"])
                          / np.maximum(P["delay_3"]["epi"], 1e-12)))
        ms = (mean.astype(np.float64).ravel().tolist(), std.astype(np.float64).ravel().tolist())
        if ms_ref is None:
            ms_ref = ms
        gC = bool(np.array_equal(np.array(ms[0]), np.array(ms_ref[0]))
                  and np.array_equal(np.array(ms[1]), np.array(ms_ref[1])))

        # ----- 기준선 양 -----
        nrm = np.linalg.norm(ref["phi"], axis=1)
        proj_ref = float(np.median((ref["phi"] / np.maximum(nrm[:, None], 1e-30)) @ w_eff))
        nrm_ref = float(np.median(nrm))
        epi_ref = float(np.median(ref["epi"]))
        err_ref = float(np.median(ref["err"]))
        thr = float(np.percentile(ref["amap"], 90))
        b_eff = float(b[2] + b[4])
        ell_ref_m = float(np.median(ref["phi"] @ w_eff + b_eff))
        _, lt_ref, nu_ref = epi_from_phi(W, b, ref["phi"])
        lt_ref_m = float(np.median(lt_ref)); nu_ref_m = float(np.median(nu_ref))

        # ----- 다양체 (B3) -----
        mu_phi = ref["phi"].mean(axis=0)
        Xc = ref["phi"] - mu_phi
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = S ** 2 / (S ** 2).sum()
        k = int(np.searchsorted(np.cumsum(var), VAR_KEEP) + 1)
        Pk = Vt[:k]
        lg_mu = float(epi_from_phi(W, b, mu_phi[None, :])[0][0])

        chain, extra = {}, {}
        for nm, _ in CASES:
            d = P[nm]
            n_m = float(np.median(np.linalg.norm(d["phi"], axis=1)))
            lg_f, lt, nu_c = epi_from_phi(W, b, d["phi"])
            d_ell = float(np.median(lt)) - lt_ref_m       # = Dlog tr(LL^T) 정확값
            d_ell_logepi = 0.5 * d_ell                   # log(epi) 로의 기여
            d_nu = -0.5 * (np.log(max(float(np.median(nu_c)) - 3.0, 1e-3))
                           - np.log(max(nu_ref_m - 3.0, 1e-3)))
            dd = d["phi"] - mu_phi
            d_par = (dd @ Pk.T) @ Pk
            lg_par, _, _ = epi_from_phi(W, b, mu_phi[None, :] + d_par)
            # mechanism_ABC.py 와 동일하게 **표본별 차이를 먼저 내고** 중앙값을 취한다
            dl_par = float(np.median(lg_par - lg_mu))
            dl_perp = float(np.median(lg_f - lg_par))
            chain[nm] = dict(
                act_ratio=float(np.median((d["amap"] > thr).mean(axis=(1, 2)))
                                / np.median((ref["amap"] > thr).mean(axis=(1, 2)))),
                norm_ratio=n_m / nrm_ref, d_ell=float(d_ell),
                d_l0l2=float(np.median(d["phi"] @ w_eff + b_eff) - ell_ref_m),
                tr_ratio=float(np.exp(d_ell)),
                epi_ratio=float(np.median(d["epi"])) / epi_ref,
                err_m=float(np.median(d["err"])))
            extra[nm] = dict(
                ell_share=float(abs(d_ell_logepi)
                                / max(abs(d_ell_logepi) + abs(d_nu), 1e-30)),
                par_share=float(abs(dl_par) / max(abs(dl_par) + abs(dl_perp), 1e-30)),
                dl_par=dl_par, dl_perp=dl_perp)

        ks = [c[0] for c in CASES]
        rho = float(sps.spearmanr([chain[k2]["norm_ratio"] for k2 in ks],
                                  [chain[k2]["epi_ratio"] for k2 in ks]).statistic)

        RES[label] = dict(
            is_new=is_new, gates=dict(A=gA, B=gB, C=gC, pca_k=k),
            w_eff_mean=float(w_eff.mean()), w_eff_neg=float((w_eff < 0).mean()),
            proj_ref=proj_ref, norm_ref=nrm_ref, epi_ref=epi_ref, err_ref=err_ref,
            spearman_norm_epi=rho, chain=chain, extra=extra)
        print("  [측정 완료] %s" % label)

    # ---------------- 보고 ----------------
    print("\n" + BAR); print("관문"); print(BAR)
    print("  %-26s %12s %12s %8s %8s" % ("모델", "GATE-A", "GATE-B", "GATE-C", "PCA k"))
    for L, r in RES.items():
        g = r["gates"]
        print("  %-26s %12.3e %12.3e %8s %8d"
              % (L, g["A"], g["B"], "PASS" if g["C"] else "FAIL", g["pca_k"]))

    print("\n" + BAR); print("FALSIFIER (정상 학습 확인)"); print(BAR)
    print("  %-26s %14s %14s %10s" % ("모델", "in-dist 오차", "sco_4 epi배", "판정"))
    for L, r in RES.items():
        e = r["err_ref"]; s = r["chain"]["sco_4 (NULL)"]["epi_ratio"]
        ok = (120 <= e <= 260) and (0.85 <= s <= 1.15)
        print("  %-26s %12.1f m %14.3f %10s" % (L, e, s, "PASS" if ok else "FAIL"))

    print("\n" + BAR); print("P1/P2 — 크기 경로의 부호"); print(BAR)
    print("  %-26s %16s %14s %12s" % ("모델", "(w_eff.phi_hat)_ref", "mean(w_eff)", "음수비율"))
    for L, r in RES.items():
        print("  %-26s %16.4f %14.5f %12.3f"
              % (L, r["proj_ref"], r["w_eff_mean"], r["w_eff_neg"]))

    print("\n" + BAR); print("P3~P6 — 사슬 재현"); print(BAR)
    for L, r in RES.items():
        print("\n  [%s]   Spearman(||phi||비, epi배) = %+.4f" % (L, r["spearman_norm_epi"]))
        print("  %-16s %9s %9s %11s %11s %9s %9s"
              % ("조건", "활성비율", "||phi||", "D(l0+l2)", "Dlog tr", "tr(LLT)", "epi"))
        for nm, _ in CASES:
            c = r["chain"][nm]
            print("  %-16s %9.3f %9.3f %+11.4f %+11.4f %9.4f %9.3f"
                  % (nm, c["act_ratio"], c["norm_ratio"], c["d_l0l2"], c["d_ell"],
                     c["tr_ratio"], c["epi_ratio"]))

    print("\n" + BAR); print("P7/P8 — delay_3 의 ell 몫 · 다양체 안 몫"); print(BAR)
    print("  %-26s %12s %12s %12s %12s" % ("모델", "ell 몫%", "안(par) 몫%", "Dlog 안", "Dlog 밖"))
    for L, r in RES.items():
        e = r["extra"]["delay_3"]
        print("  %-26s %11.1f%% %11.1f%% %12.4f %12.4f"
              % (L, 100 * e["ell_share"], 100 * e["par_share"], e["dl_par"], e["dl_perp"]))

    # ---------------- 채점 ----------------
    new = {L: r for L, r in RES.items() if r["is_new"]}
    print("\n" + BAR); print("사전 예측 채점 (신규 시드만 · frozen 은 seed0 과 비트동일이라 제외)"); print(BAR)
    checks = [
        ("P1  (w_eff.phi_hat)_ref < 0", lambda r: r["proj_ref"] < 0),
        ("P2  mean(w_eff) < 0", lambda r: r["w_eff_mean"] < 0),
        ("P3  Spearman < -0.7", lambda r: r["spearman_norm_epi"] < -0.7),
        ("P4  delay_3 epi배 < 1.0", lambda r: r["chain"]["delay_3"]["epi_ratio"] < 1.0),
        ("P5  TDL-C epi배 > 1.0", lambda r: r["chain"]["TDL-C"]["epi_ratio"] > 1.0),
        ("P6  ||phi||: d3>1 & TDL-C<1",
         lambda r: r["chain"]["delay_3"]["norm_ratio"] > 1.0
         and r["chain"]["TDL-C"]["norm_ratio"] < 1.0),
        ("P7  ell 몫 > 90%", lambda r: r["extra"]["delay_3"]["ell_share"] > 0.90),
        ("P8  안(par) 몫 > 50%", lambda r: r["extra"]["delay_3"]["par_share"] > 0.50),
    ]
    score = {}
    for lab, fn in checks:
        hits = [L for L, r in new.items() if fn(r)]
        score[lab] = dict(hit=len(hits), n=len(new), seeds=hits)
        print("  %-30s %d/%d  %s" % (lab, len(hits), len(new),
                                     "적중" if len(hits) == len(new) else "부분/실패"))

    # ---- 사전등록 3장: 메커니즘 정합성 (부호가 바뀌면 현상도 바뀌어야 한다) ----
    print(BAR)
    print("메커니즘 정합성 — 예측 부호 vs 관측 부호 (사전등록 3장)")
    print(BAR)
    print("  예측: sign(Dell) = sign(D||phi||) x sign(w.phi_hat)  ->  sign(D epi) = 같은 부호")
    print("  %-26s %-12s %10s %10s %10s %10s %8s"
          % ("모델", "조건", "w.phi_hat", "||phi||-1", "예측 부호", "관측 부호", "정합"))
    CONS = {}
    for L, r in RES.items():
        sp = np.sign(r["proj_ref"])
        CONS[L] = {}
        for nm in ("delay_3", "TDL-C", "occ 0.05", "kf_3"):
            c = r["chain"][nm]
            dn = np.sign(c["norm_ratio"] - 1.0)
            pred = sp * dn                       # Dell 의 부호 = D epi 의 부호
            obs = np.sign(c["epi_ratio"] - 1.0)
            ok = bool(pred == obs) if dn != 0 else None
            CONS[L][nm] = dict(pred=float(pred), obs=float(obs), ok=ok)
            print("  %-26s %-12s %10.4f %10.3f %10.0f %10.0f %8s"
                  % (L if nm == "delay_3" else "", nm, r["proj_ref"],
                     c["norm_ratio"] - 1.0, pred, obs, "O" if ok else "X"))
    nc = sum(1 for L, r in RES.items() if r["is_new"]
             for nm, v in CONS[L].items() if v["ok"])
    nt = sum(1 for L, r in RES.items() if r["is_new"] for _ in CONS[L])
    print("")
    print("  신규 시드 정합 %d/%d 셀" % (nc, nt))

    core = ["P1  (w_eff.phi_hat)_ref < 0", "P3  Spearman < -0.7",
            "P4  delay_3 epi배 < 1.0", "P5  TDL-C epi배 > 1.0",
            "P6  ||phi||: d3>1 & TDL-C<1"]
    allcore = all(score[c]["hit"] == score[c]["n"] for c in core)
    verdict = "메커니즘 시드 견고 (§15-41 종결)" if allcore else "부분/반증 — 상세 판정 필요"
    print("\n  ==> 판정: %s" % verdict)

    out = dict(_note="§70 메커니즘의 학습시드 재현. 사전등록 SEED_REPLICATION_PREREG_2026-08-31.md",
               models={L: {k: v for k, v in r.items()} for L, r in RES.items()},
               consistency=CONS, consistency_hits=[nc, nt],
               score=score, verdict=verdict)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
