r"""
ls035_confirm.py - LS(kappa=0.35 동결) held-out 확증.

사전등록: prereg/LS035_CONFIRM_PREREG_2026-09-12.md  (실행 전 기록)
maxla2_confirm.py 와 절차·규약이 동일하고 arm 과 채점만 사전등록 L1~L6 으로 바꿨다.

자원 (kappa=0.35 선별 자원 h 는 제외):
  PRIMARY   이동시드 92001/92002 + 세트 j1~j3   (MAXLA2 가 확증받은 자원 = 짝지은 비교)
  SECONDARY 이동시드 90001/90002 + 세트 g3~g5   (--resource secondary)
비교 arm: MAXLA2@0.4 (현직, L3), LS@0.0 (L4 기준), LS@0.4 (이웃, 보고만).

    python scripts/data/ls035_confirm.py [--resource primary|secondary]
"""
from __future__ import annotations
import functools, io, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for q in (ROOT, HERE):
    if q not in sys.path:
        sys.path.insert(0, q)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import rfgeo.dataset as ds                                          # noqa: E402
import rfgeo.waveform as wf                                         # noqa: E402
from rfgeo.gcc import gcc_phat_features                             # noqa: E402
from train_der import DERModel                                      # noqa: E402
from rfgeo.evidential import niw_uncertainties                      # noqa: E402
from peak_decomposition import peak_stats_one, SCALE, ML, ALPHA      # noqa: E402

OUTDIR = os.path.join(ROOT, "artifacts")
_RES = "secondary" if "--resource" in sys.argv and "secondary" in sys.argv else "primary"
# --arm 0.40 : 등록 kappa(0.35) 대신 이웃 kappa 를 같은 검정으로 채점 (사후 적용)
_KAP = "0.40" if "--arm" in sys.argv and "0.40" in sys.argv else "0.35"
OUT = os.path.join(OUTDIR, "ls%s_confirm%s.json"
                   % ("040" if _KAP == "0.40" else "035",
                      "_sec" if _RES == "secondary" else ""))
CACHE = os.path.join(OUTDIR, "cfs_cache")
MODELS = [("m0", "output_seed20260628_clean"), ("m1", "output_seed20261628_clean"),
          ("m2", "output_seed20262628_clean"), ("m3", "output_seed20263628_clean"),
          ("seed0", "output_seed0")]
NEWG = ["90001", "90002"] if _RES == "secondary" else ["92001", "92002"]
SHIFT = ["kf_1", "kf_2", "kf_3", "cfo_2", "cfo_3", "cfo_4", "snr_2", "sco_4", "delay_3"]
NEWSET = ({"g3": (61001, 61002, 61003), "g4": (72011, 72012, 72013),
           "g5": (83021, 83022, 83023)} if _RES == "secondary" else
          {"j1": (84001, 84002, 84003), "j2": (85001, 85002, 85003),
           "j3": (86001, 86002, 86003)})
GENC = ["IND-D", "occ 0.05", "TDL-A", "TDL-C"]
ARMS = [("LS", 0.0), ("LS", 0.35), ("LS", 0.4), ("MAXLA2", 0.4)]
EDGES = [-5.0, 0.0, 5.0, 10.0, 15.0]
DISC = np.pi * 8000.0 ** 2 / 1e6
N_GEN = 200


def gen_seeds(seeds, gname=None, genfn=None, prof="TDL-D"):
    F, Y, S = [], [], []
    for sd in seeds:
        kw = {}
        if gname is not None:
            wf.WAVEFORM_GENERATORS[gname] = genfn
            kw["waveforms"] = (gname,)
        cfg = ds.GenConfig(n_train=1, n_calib=1, n_test=N_GEN, seed=sd,
                           channel_mode="tdl", tdl_profile=prof, **kw)
        X, y, mt = ds.generate_dataset(cfg)["test"]
        F.append(np.stack([gcc_phat_features(X[i], max_lag=ML) for i in range(len(X))]))
        Y.append(y)
        S.append(np.array([m["snr_db"] for m in mt], float))
    return (np.concatenate(F).astype(np.float32), np.concatenate(Y).astype(np.float32),
            np.concatenate(S))


def gen_cached(cond, tag, sds):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "%s_%s.npz" % (cond.replace(" ", "").replace(".", ""), tag))
    if os.path.exists(p):
        z = np.load(p)
        return z["feats"], z["y"], z["snr"]
    if cond.startswith("occ"):
        uf = float(cond.split()[1])
        F, Y, S = gen_seeds(sds, "_mx_occ%s" % uf, functools.partial(wf.gen_ofdm, used_frac=uf))
    elif cond == "IND-D":
        F, Y, S = gen_seeds(sds, prof="TDL-D")
    else:
        F, Y, S = gen_seeds(sds, prof=cond)
    np.savez_compressed(p, feats=F, y=Y, snr=S)
    return F, Y, S


def curve(F):
    n = len(F)
    hw = np.empty(n)
    pf = np.empty(n)
    for i in range(n):
        _, h, f_, _ = peak_stats_one(F[i])
        hw[i] = h
        pf[i] = f_
    rms = np.sqrt(np.mean(F.astype(np.float64) ** 2, axis=(1, 2)))
    return np.column_stack([np.ones(n), np.log(np.maximum(rms, 1e-12)),
                            np.log(np.maximum(pf, 1e-12)), np.log(np.maximum(hw, 1e-12))])


def fsq(v, a):
    n = len(v)
    return float(np.sort(v)[min(int(np.ceil((n + 1) * (1 - a))), n) - 1])


def main():
    RAW, SNR = {}, {}
    dc = np.load(os.path.join(ROOT, "output_final", "gcc_calib.npz"))
    RAW["CALFULL"] = (dc["feats"].astype(np.float32), dc["y"].astype(np.float32))
    for lv in SHIFT:
        F, Y = [], []
        for gs in NEWG:
            d = np.load(os.path.join(ROOT, "output_ood_s%s" % gs, "shift_%s.npz" % lv))
            F.append(d["feats"].astype(np.float32))
            Y.append(d["y"].astype(np.float32))
        RAW[lv] = (np.concatenate(F), np.concatenate(Y))
    for cond in GENC:
        for tag, sds in NEWSET.items():
            F, Y, S = gen_cached(cond, tag, sds)
            RAW["%s|%s" % (cond, tag)] = (F.astype(np.float32), Y.astype(np.float32))
            SNR["%s|%s" % (cond, tag)] = S
        print("생성 %s 완료" % cond)
    print("곡선 통계 (%d 집합) ..." % len(RAW))
    X = {k: curve(v[0]) for k, v in RAW.items()}
    KEYS = [k for k in RAW if k != "CALFULL"]

    res = {"_prereg": "prereg/MAXLA2_CONFIRM_PREREG_2026-09-04.md", "cells": {}}
    for mlab, sub in MODELS:
        ck = torch.load(os.path.join(ROOT, sub, "der_model.pt"), weights_only=False)
        mean, std = ck["mean"], ck["std"]
        mo = DERModel(n_pairs=6, r=ck.get("r", 1.0))
        mo.load_state_dict(ck["model"])
        mo.eval()

        @torch.no_grad()
        def pack(key):
            F, Y = RAW[key]
            Z = torch.from_numpy(((F - mean) / std).astype(np.float32))
            h = torch.relu(mo.backbone.stem(Z))
            h = mo.backbone.blocks(h)
            act = (h > 0).float().mean(dim=(1, 2)).numpy().astype(np.float64)
            phi = mo.backbone.proj(mo.backbone.pool(h))
            mu, L, nu = mo.head(phi)
            al, ep = niw_uncertainties(L, nu)
            tot = (al + ep).numpy().astype(np.float64)
            p = mu.numpy().astype(np.float64)
            yn = (Y / SCALE).astype(np.float64)
            d = (yn - p)[:, :, None]
            return dict(X=X[key], act=act,
                        err=np.sqrt(((p - yn) ** 2).sum(1)) * SCALE,
                        M=np.sqrt(np.maximum((np.linalg.solve(tot, d) * d).sum((1, 2)), 0.0)),
                        sdet=np.sqrt(np.maximum(np.linalg.det(tot), 1e-30)))

        S = {k: pack(k) for k in KEYS}
        CF = pack("CALFULL")
        FIT = {k: v[0:2000] for k, v in CF.items()}
        CAL = {k: v[2000:4000] for k, v in CF.items()}
        b_ls = np.linalg.lstsq(FIT["X"], np.log(np.maximum(FIT["err"], 1e-9)), rcond=None)[0]
        ml_, sl_ = (float(np.mean(FIT["X"] @ b_ls)), float(np.std(FIT["X"] @ b_ls) + 1e-12))
        ma_, sa_ = float(np.mean(FIT["act"])), float(np.std(FIT["act"]) + 1e-12)

        def zarm(s, arm):
            zl = (s["X"] @ b_ls - ml_) / sl_
            za = (s["act"] - ma_) / sa_
            if arm == "LS":
                return zl
            if arm == "ACT":
                return np.maximum(za, 0.0)
            return np.maximum(zl, za)

        Z2 = {}
        for arm in ("LS", "ACT", "MAXLA2"):
            v = zarm(FIT, arm)
            Z2[arm] = (float(np.mean(v)), float(np.std(v) + 1e-12))

        def zh(s, arm):
            z = (zarm(s, arm) - Z2[arm][0]) / Z2[arm][1]
            return np.maximum(z, 0.0) if arm == "ACT" else z

        for arm, kap in ARMS:
            gc = np.exp(kap * zh(CAL, arm))
            Q = fsq(CAL["M"] / gc, ALPHA)
            for k in KEYS:
                gv = np.exp(kap * zh(S[k], arm))
                A = np.pi * (Q * gv) ** 2 * S[k]["sdet"] * SCALE ** 2 / 1e6
                res["cells"]["%s@%.2f|%s|%s" % (arm, kap, mlab, k)] = [
                    float((S[k]["M"] / gv <= Q).mean()), float(np.median(A) / DISC)]
                if k.startswith("IND-D"):
                    g = np.digitize(SNR[k], EDGES)
                    hit = S[k]["M"] / gv <= Q
                    res["cells"]["%s@%.2f|%s|%s|__bins__" % (arm, kap, mlab, k.split("|")[1])] = [
                        float(hit[g == b].mean()) if (g == b).sum() > 0 else None
                        for b in range(6)]
        print("  %s 완료" % mlab)

    # ---------------- 채점 ----------------
    C = res["cells"]
    ML_ = [m for m, _ in MODELS]
    A0, AREF = "LS@%s" % _KAP, "MAXLA2@0.40"

    def agg(aid, cond, tags):
        v = [C["%s|%s|%s" % (aid, m, "%s|%s" % (cond, t) if tags else cond)]
             for m in ML_ for t in (tags or [""])]
        return (float(np.mean([x[0] for x in v])), float(np.median([x[1] for x in v])),
                min(x[0] for x in v))

    def sh(aid, lv):
        v = [C["%s|%s|%s" % (aid, m, lv)] for m in ML_]
        return (float(np.mean([x[0] for x in v])), float(np.median([x[1] for x in v])),
                min(x[0] for x in v))

    P = {}
    # L-F0  FALSIFIER: IND 커버리지 [0.86,0.94] · sco_4 >= 0.85
    viol = []
    for m in ML_:
        iv = np.mean([C["%s|%s|IND-D|%s" % (A0, m, t)][0] for t in NEWSET])
        if not (0.86 <= iv <= 0.94):
            viol.append((m, "ind", round(float(iv), 4)))
        sc = C["%s|%s|sco_4" % (A0, m)][0]
        if sc < 0.85:
            viol.append((m, "sco", round(float(sc), 4)))
    P["L_F0"] = {"violations": viol, "n_models_bad": len({v[0] for v in viol}),
                 "pass": len({v[0] for v in viol}) < 2}
    # L1  에너지 4 지정축 (임계 = 선별값 - 0.06)
    THR = {"kf_3": 0.48, "cfo_3": 0.52, "TDL-A": 0.65, "TDL-C": 0.63}
    l1 = {}
    for lv in ("kf_3", "cfo_3"):
        c, d, mn = sh(A0, lv)
        l1[lv] = {"cov": c, "disc": d, "min": mn, "thr": THR[lv], "pass": c >= THR[lv]}
    for lv in ("TDL-A", "TDL-C"):
        c, d, mn = agg(A0, lv, list(NEWSET))
        l1[lv] = {"cov": c, "disc": d, "min": mn, "thr": THR[lv], "pass": c >= THR[lv]}
    P["L1"] = {"axes": l1, "n_pass": sum(1 for v in l1.values() if v["pass"])}
    # L2  게이트: TDL-A 영역 셀 중앙값 <= 20%
    P["L2"] = {"tdla_disc_med": l1["TDL-A"]["disc"], "thr": 0.20,
               "pass": l1["TDL-A"]["disc"] <= 0.20}
    # L3  현직 대비: LS@0.35 >= MAXLA2@0.4 - 0.02, 4축 중 3축 이상
    l3 = {}
    for lv in ("kf_3", "cfo_3", "TDL-A", "TDL-C"):
        f = sh if lv in ("kf_3", "cfo_3") else (lambda a, l: agg(a, l, list(NEWSET)))
        a, b = f(A0, lv)[0], f(AREF, lv)[0]
        l3[lv] = {"ls": a, "maxla2": b, "delta": a - b, "pass": a >= b - 0.02}
    P["L3"] = {"axes": l3, "n_pass": sum(1 for v in l3.values() if v["pass"])}
    # L4  무해성 (LS@0.0 기준)
    d3b, d30 = sh("LS@0.00", "delay_3"), sh(A0, "delay_3")
    ocb = agg("LS@0.00", "occ 0.05", list(NEWSET))
    oc0 = agg(A0, "occ 0.05", list(NEWSET))
    P["L4"] = {"delay_3": [d3b[0], d30[0], d30[1]], "occ": [ocb[0], oc0[0], oc0[1]],
               "pass": (d30[0] >= d3b[0] - 0.05 and oc0[0] >= ocb[0] - 0.05
                        and d30[1] <= 0.20 and oc0[1] <= 0.20)}
    # L5  분포내 조건부 편차 <= 0.035 (4/5 모델)
    l5 = {}
    for m in ML_:
        cnt = {}
        for t in NEWSET:
            g = np.digitize(SNR["IND-D|%s" % t], EDGES)
            b6 = C["%s|%s|%s|__bins__" % (A0, m, t)]
            for b in range(6):
                nb = int((g == b).sum())
                if b6[b] is not None and nb:
                    s0, n0 = cnt.get(b, (0.0, 0))
                    cnt[b] = (s0 + b6[b] * nb, n0 + nb)
        l5[m] = float(np.mean([abs(v[0] / v[1] - 0.9) for v in cnt.values()]))
    P["L5"] = {"per_model": l5, "n_pass": sum(1 for v in l5.values() if v <= 0.035),
               "pass": sum(1 for v in l5.values() if v <= 0.035) >= 4}
    # 참고 (판정 무관): LS@0.4 이웃
    ref = {}
    for lv in ("kf_3", "cfo_3", "TDL-A", "TDL-C"):
        f = sh if lv in ("kf_3", "cfo_3") else (lambda a, l: agg(a, l, list(NEWSET)))
        ref[lv] = {"LS@0.40": f("LS@0.40", lv)[:2], "MAXLA2@0.40": f(AREF, lv)[:2]}
    P["_ref_neighbour"] = ref
    res["scoring"] = P
    n1, n3 = P["L1"]["n_pass"], P["L3"]["n_pass"]
    if not P["L_F0"]["pass"]:
        verdict = "무효 (FALSIFIER)"
    elif n1 <= 2:
        verdict = "기각"
    elif not P["L2"]["pass"]:
        verdict = "기각 (게이트)"
    elif n3 >= 3:
        verdict = "승격"
    elif n3 == 2:
        verdict = "부분 (현직 유지)"
    else:
        verdict = "기각"
    res["verdict"] = {"L1": n1, "L2": P["L2"]["pass"], "L3": n3,
                      "L_F0": P["L_F0"]["pass"], "L4": P["L4"]["pass"],
                      "L5": P["L5"]["pass"], "verdict": verdict}

    print("\n=== L-F0 :", "PASS" if P["L_F0"]["pass"] else P["L_F0"]["violations"])
    print("L1 (에너지 4 지정축, %s):" % A0)
    for lv, v in l1.items():
        print("   %-7s cov %.3f (min %.3f) 영역 %5.1f%%  thr %.2f -> %s"
              % (lv, v["cov"], v["min"], 100 * v["disc"], v["thr"], v["pass"]))
    print("L2 게이트 TDL-A 영역중앙 %.1f%% <= 20%% -> %s"
          % (100 * P["L2"]["tdla_disc_med"], P["L2"]["pass"]))
    print("L3 현직(MAXLA2@0.40) 대비:")
    for lv, v in l3.items():
        print("   %-7s LS %.3f  vs  MX %.3f   delta %+.3f -> %s"
              % (lv, v["ls"], v["maxla2"], v["delta"], v["pass"]))
    print("L4 무해성 delay_3 %.3f->%.3f  occ %.3f->%.3f -> %s"
          % (d3b[0], d30[0], ocb[0], oc0[0], P["L4"]["pass"]))
    print("L5 조건부편차 %s -> %d/5"
          % ({m: round(v, 4) for m, v in l5.items()}, P["L5"]["n_pass"]))
    print("\n=== 이웃 kappa (보고용, 판정 무관) ===")
    for aid in ("LS@0.00", "LS@0.35", "LS@0.40", "MAXLA2@0.40"):
        row = []
        for lv in ("kf_3", "cfo_3"):
            c, d, _ = sh(aid, lv)
            row.append("%s %.3f(%4.1f%%)" % (lv, c, 100 * d))
        for lv in ("TDL-A", "TDL-C"):
            c, d, _ = agg(aid, lv, list(NEWSET))
            row.append("%s %.3f(%4.1f%%)" % (lv, c, 100 * d))
        print("  %-10s %s" % (aid, "  ".join(row)))
    print("\nVERDICT:", verdict)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
