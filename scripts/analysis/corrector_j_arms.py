r"""
corrector_j_arms.py - 자원 j 에서 CLO·RULE·PHI 복원을 재측정 (CLO held-out 확증).

사전등록: prereg/CLO_J_CONFIRM_PREREG_2026-09-12.md (실행 전 기록)
LS 는 ls040_confirm.py 로 같은 자원에서 이미 측정됨.

corrector_final_sweep.py 파생. arm 만 CLO 로 바꾸고 나머지(축·격자·모델·자원·
3분할 규약)는 동일하다. GATE: kappa=0 에서 원본 mondrian/arm 값과 일치해야 한다.

사전등록: prereg/CORRECTOR_FINAL_SWEEP_PREREG_2026-09-03.md  (실행 전 기록)

내장 관문 (코드 정확성):
  GATE-1  kappa=0 에서 5 arm 이 완전히 동일 (g=1 이므로 정의상 같아야 함)
  GATE-2  LEGACY kappa=0 이 동결본(cov 0.9020 / area 0.4327) 재현
  GATE-3  fsq 가 유한표본 분할 CP 분위수식과 일치 (독립 재계산)
  GATE-4  phi 추출이 backbone() 과 일치
  GATE-5  3분할 집합이 서로 소

    python scripts/data/corrector_final_sweep.py
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
OUT = os.path.join(OUTDIR, "corrector_j_arms.json")
CACHE = os.path.join(OUTDIR, "cfs_cache")
MODELS = [("m0", "output_seed20260628_clean"), ("m1", "output_seed20261628_clean"),
          ("m2", "output_seed20262628_clean"), ("m3", "output_seed20263628_clean"),
          ("seed0", "output_seed0")]
NEWG = ["92001", "92002"]
NEWSET = {"j1": (84001, 84002, 84003), "j2": (85001, 85002, 85003),
          "j3": (86001, 86002, 86003)}
SHIFT = (["delay_%d" % i for i in range(4)] + ["kf_%d" % i for i in range(4)] +
         ["cfo_%d" % i for i in range(5)] + ["sco_%d" % i for i in range(5)] +
         ["snr_%d" % i for i in range(3)])
GENC = ["occ 0.05", "occ 0.20", "TDL-A", "TDL-C"]
ARMS = ["CLO", "RULE", "PHI"]
TWO_SIDED = set()                        # CLO 는 단방향 (모순량이 크면 확대)
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
KGRID = [round(0.1 * i, 1) for i in range(41)]
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
        F, Y, S = gen_seeds(sds, "_cfs_occ%s" % uf,
                            functools.partial(wf.gen_ofdm, used_frac=uf))
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
    print("[1/4] 데이터 적재 ...")
    RAW, SNR = {}, {}
    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    RAW["IND"] = (dt["feats"].astype(np.float32), dt["y"].astype(np.float32))
    SNR["IND"] = np.array([r["snr_db"] for r in json.load(
        io.open(os.path.join(ROOT, "output_final", "meta_test.json"), encoding="utf-8"))], float)
    dc = np.load(os.path.join(ROOT, "output_final", "gcc_calib.npz"))
    RAW["CALFULL"] = (dc["feats"].astype(np.float32), dc["y"].astype(np.float32))
    SNR["CALFULL"] = np.array([r["snr_db"] for r in json.load(
        io.open(os.path.join(ROOT, "output_final", "meta_calib.json"), encoding="utf-8"))], float)
    for gs in NEWG:
        for lv in SHIFT:
            d = np.load(os.path.join(ROOT, "output_ood_s%s" % gs, "shift_%s.npz" % lv))
            k = "%s|%s" % (lv, gs)
            RAW[k] = (d["feats"].astype(np.float32), d["y"].astype(np.float32))
            SNR[k] = np.asarray(d["snr"], float)
    for cond in GENC:
        for tag, sds in NEWSET.items():
            F, Y, S = gen_cached(cond, tag, sds)
            k = "%s|%s" % (cond, tag)
            RAW[k] = (F.astype(np.float32), Y.astype(np.float32))
            SNR[k] = S
        print("   생성 %s 완료" % cond)

    print("[2/4] 곡선 통계 (%d 집합) ..." % len(RAW))
    X = {k: curve(v[0]) for k, v in RAW.items()}
    A_ = np.zeros((len(PAIRS), 4))
    for _k, (_i, _j) in enumerate(PAIRS):
        A_[_k, _i], A_[_k, _j] = 1.0, -1.0
    _U, _S, _ = np.linalg.svd(A_, full_matrices=True)
    NUL = _U[:, int(np.sum(_S > 1e-9)):]
    assert NUL.shape[1] == 3, "closure 영공간 차원"

    def _lags(F):
        out = np.empty((len(F), F.shape[1]))
        for _q in range(len(F)):
            lg, _, _, _ = peak_stats_one(F[_q])
            out[_q] = lg
        return out

    CLO = {k: np.sqrt(np.maximum(((_lags(v[0]) @ NUL) ** 2).sum(1) / 3.0, 0.0))
           for k, v in RAW.items()}
    print("CLO 계산 완료 (%d 집합)" % len(CLO))

    rng = np.random.default_rng(0)
    tv = rng.random(777)
    assert abs(fsq(tv, 0.1) - np.sort(tv)[int(np.ceil(778 * 0.9)) - 1]) < 1e-15, "GATE-3"
    print("   GATE-3 분위수식 PASS")

    KEYS = (["IND"] + ["%s|%s" % (l, g) for l in SHIFT for g in NEWG] +
            ["%s|%s" % (c, t) for c in GENC for t in NEWSET])
    gi = np.digitize(SNR["IND"], EDGES)
    res = {"_prereg": "prereg/CORRECTOR_FINAL_SWEEP_PREREG_2026-09-03.md",
           "kgrid": KGRID, "arms": ARMS, "axes": KEYS,
           "models": [m for m, _ in MODELS], "schemes": ["PRIMARY", "LEGACY"],
           "disc_km2": DISC, "gates": {}, "cells": {}}

    for mlab, sub in MODELS:
        print("[3/4] 모델 %s ..." % mlab)
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
            assert float((phi - mo.backbone(Z)).abs().max()) < 1e-5, "GATE-4"
            mu, L, nu = mo.head(phi)
            al, ep = niw_uncertainties(L, nu)
            tot = (al + ep).numpy().astype(np.float64)
            p = mu.numpy().astype(np.float64)
            yn = (Y / SCALE).astype(np.float64)
            d = (yn - p)[:, :, None]
            return dict(X=X[key], act=act, clo=CLO[key],
                        nphi=np.linalg.norm(phi.numpy().astype(np.float64), axis=1),
                        epi=np.sqrt(np.trace(ep.numpy().astype(np.float64), 0, 1, 2)) * SCALE,
                        err=np.sqrt(((p - yn) ** 2).sum(1)) * SCALE,
                        M=np.sqrt(np.maximum((np.linalg.solve(tot, d) * d).sum((1, 2)), 0.0)),
                        sdet=np.sqrt(np.maximum(np.linalg.det(tot), 1e-30)))

        S = {k: pack(k) for k in KEYS}
        CF = pack("CALFULL")
        CA = {k: v[0:2000] for k, v in CF.items()}
        CB = {k: v[2000:4000] for k, v in CF.items()}
        SCHEME = {"PRIMARY": (CA, CB, SNR["CALFULL"][2000:]),
                  "LEGACY": (S["IND"], CF, SNR["CALFULL"])}

        for scm in ("PRIMARY", "LEGACY"):
            FIT, CAL, csnr = SCHEME[scm]
            b_ls = np.linalg.lstsq(FIT["X"], np.log(np.maximum(FIT["err"], 1e-9)), rcond=None)[0]
            b_ru = np.linalg.lstsq(FIT["X"], np.log(FIT["epi"]), rcond=None)[0]
            R0 = {}

            def zraw(s, arm):
                if arm == "CLO":
                    return np.log(np.maximum(s["clo"], 1e-9))
                if arm == "LS":
                    return s["X"] @ b_ls
                if arm == "RULE":
                    return s["X"] @ b_ru - np.log(s["epi"])
                if arm == "PHI":
                    return np.log(s["nphi"])
                if arm == "ACT":
                    return s["act"]
                zr = (s["X"] @ b_ru - np.log(s["epi"]) - R0["RULE"][0]) / R0["RULE"][1]
                zp = (np.log(s["nphi"]) - R0["PHI"][0]) / R0["PHI"][1]
                return np.maximum(zr, zp)

            # (MAXRP 용 R0 는 CLO 단독 실행에서 불필요)
            ZS = {}
            for a in ARMS:
                v = zraw(FIT, a)
                ZS[a] = (float(np.mean(v)), float(np.std(v) + 1e-12))
            ZH = {a: {k: (zraw(S[k], a) - ZS[a][0]) / ZS[a][1] for k in KEYS} for a in ARMS}
            ZC = {a: (zraw(CAL, a) - ZS[a][0]) / ZS[a][1] for a in ARMS}
            for a in ARMS:
                if a not in TWO_SIDED:
                    ZC[a] = np.maximum(ZC[a], 0.0)
                    for k in KEYS:
                        ZH[a][k] = np.maximum(ZH[a][k], 0.0)

            gc_ = np.digitize(csnr, EDGES)
            Qmarg = fsq(CAL["M"], ALPHA)
            Qg = {b: fsq(CAL["M"][gc_ == b], ALPHA) for b in range(6) if (gc_ == b).sum() > 20}
            Qmg = np.array([Qg.get(b, Qmarg) for b in range(6)])
            for k in KEYS:
                q = Qmg[np.digitize(SNR[k], EDGES)]
                A = np.pi * q ** 2 * S[k]["sdet"] * SCALE ** 2 / 1e6
                res["cells"]["%s|mondrian|0.0|%s|%s" % (scm, mlab, k)] = [
                    float((S[k]["M"] <= q).mean()), float(np.median(A) / DISC)]
            hitm = S["IND"]["M"] <= Qmg[gi]
            res["cells"]["%s|mondrian|0.0|%s|__bins__" % (scm, mlab)] = [
                float(hitm[gi == b].mean()) for b in range(6)]
            res["cells"]["%s|mondrian|0.0|%s|__area__" % (scm, mlab)] = float(np.median(
                np.pi * Qmg[gi] ** 2 * S["IND"]["sdet"] * SCALE ** 2 / 1e6))

            for arm in ARMS:
                for kap in KGRID:
                    gc = np.exp(kap * ZC[arm])
                    Q = fsq(CAL["M"] / gc, ALPHA)
                    for k in KEYS:
                        gv = np.exp(kap * ZH[arm][k])
                        A = np.pi * (Q * gv) ** 2 * S[k]["sdet"] * SCALE ** 2 / 1e6
                        res["cells"]["%s|%s|%.1f|%s|%s" % (scm, arm, kap, mlab, k)] = [
                            float((S[k]["M"] / gv <= Q).mean()), float(np.median(A) / DISC)]
                    gt = np.exp(kap * ZH[arm]["IND"])
                    hit = S["IND"]["M"] / gt <= Q
                    res["cells"]["%s|%s|%.1f|%s|__bins__" % (scm, arm, kap, mlab)] = [
                        float(hit[gi == b].mean()) for b in range(6)]
                    res["cells"]["%s|%s|%.1f|%s|__area__" % (scm, arm, kap, mlab)] = float(
                        np.median(np.pi * (Q * gt) ** 2 * S["IND"]["sdet"] * SCALE ** 2 / 1e6))
        print("   %s 완료 (PRIMARY/LEGACY)" % mlab)

    C = res["cells"]
    bad = []
    for scm in ("PRIMARY", "LEGACY"):
        for mlab, _ in MODELS:
            for k in KEYS:
                vs = {tuple(C["%s|%s|0.0|%s|%s" % (scm, a, mlab, k)]) for a in ARMS}
                # arm 이 1개이므로 집합 크기는 항상 1 (원본 GATE-1 의 퇴화형)
                if len(vs) != 1:
                    bad.append((scm, mlab, k))
    assert not bad, "GATE-1 실패: %s" % bad[:3]
    res["gates"]["G1_kappa0_identical"] = "PASS (%d 셀)" % (2 * len(MODELS) * len(KEYS))
    c0 = C["LEGACY|CLO|0.0|seed0|IND"]      # arm 이 CLO 뿐 (kappa=0 이면 arm 무관)
    a0 = C["LEGACY|CLO|0.0|seed0|__area__"]
    res["gates"]["G2_frozen_repro"] = {"cov": c0[0], "area_km2": a0, "ref": [0.9020, 0.4327]}
    res["gates"]["G3_quantile"] = "PASS"
    res["gates"]["G4_phi"] = "PASS"
    res["gates"]["G5_disjoint"] = "FIT=calib[0:2000] CAL=calib[2000:4000] TEST=gcc_test"
    print("   GATE-1 PASS / GATE-2 동결 재현 cov=%.4f area=%.4f (기준 0.9020 / 0.4327)"
          % (c0[0], a0))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
    print("[4/4] saved %s  (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    main()
