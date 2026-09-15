r"""
detector_raim.py - RAIM 계열 고전 잔차를 우리 탐지기와 같은 작동점에서 잰다 (선별 측정).

질문(2026-09-12): 라벨 없는 고전 자기진단(GNSS RAIM 의 패리티/잔차 검정)을
세 label-free 플래그와 **같은 프로토콜**로 비교할 수 있는가?

신호 (둘 다 라벨 불필요):
  CLO  패리티/closure  : 쌍별 lag 6개 중 독립인 것은 3개 -> 나머지 3차원이 모순량.
                         ||(I - A A^+) d_obs|| / sqrt(3).  **기하·모델 불필요 = 완전 고전**
  FIT  잔차형 RAIM     : ||d~(p_hat) - d_obs|| / sqrt(6).  기하 + 추정 위치 사용
비교 대상: RULE · PHI · **LS(신규 에너지 플래그)** · MAXLA2(구 에너지) · epi · PSR
   ※ detector_raim.py 파생 — LS 를 탐지기로 노출해 CLO 와 **같은 자원(h)** 에서 비교

프로토콜·자원은 detector_unify_confirm.py 와 동일:
  in-dist(gcc_test 2000) -> MAP/CAL/NEG 3분할 x 200 순열, conformal p, alpha=0.05,
  모델 5개(m0~m3+seed0), 이동 생성시드 92001/92002, 세트 j1~j3 (자원 j 통일).

⚠️ 등급: **선별 측정**(사전등록 없음). 채택하려면 새 생성시드 확증이 필요하다.

    python scripts/data/detector_raim.py
"""
from __future__ import annotations
import functools
import io
import json
import os
import sys

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
from peak_decomposition import peak_stats_one, SCALE, ML             # noqa: E402

OUTDIR = os.path.join(ROOT, "artifacts")
OUT = os.path.join(OUTDIR, "detector_j.json")
CACHE = os.path.join(OUTDIR, "cfs_cache")
MODELS = [("m0", "output_seed20260628_clean"), ("m1", "output_seed20261628_clean"),
          ("m2", "output_seed20262628_clean"), ("m3", "output_seed20263628_clean"),
          ("seed0", "output_seed0")]
NEWG = ["92001", "92002"]
SHIFT = ["sco_4", "snr_1", "snr_2", "cfo_2", "cfo_4", "kf_3",
         "delay_1", "delay_2", "delay_3"]
NEWSET = {"j1": (84001, 84002, 84003), "j2": (85001, 85002, 85003),
          "j3": (86001, 86002, 86003)}
GENC = {"IND-D": ("IND-D", ["j1", "j2", "j3"]),
        "occ 0.05": ("occ005", ["j1", "j2", "j3"]),
        "TDL-A": ("TDL-A", ["j1", "j2", "j3"]),
        "TDL-C": ("TDL-C", ["j1", "j2", "j3"])}
DET = ["PSR", "RULE", "PHI", "LS", "MAXLA2", "epi", "CLO", "FIT"]
ALPHA_D = 0.05
NPERM = 200
C0 = 299792458.0
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def gen_seeds(seeds, gname=None, genfn=None, prof="TDL-D"):
    cfg = ds.GenConfig(n_test=600, channel_mode="tdl", tdl_profile=prof,
                       std_pn="gold", seed=int(seeds[0]))
    if genfn is not None:
        wf.register_waveform(gname, genfn)
        cfg.waveforms = (gname,)
    out = ds.generate_dataset(cfg, split="test")
    F = gcc_phat_features(out["iq"], max_lag=cfg.max_lag)
    return F.astype(np.float32), out["p_tx"][:, :2].astype(np.float32), \
        np.asarray(out["snr_db"], float)


def lags_of(F):
    """(N,P,L) -> (N,P) 쌍별 봉우리 lag [샘플]."""
    n = len(F)
    out = np.empty((n, F.shape[1]))
    for i in range(n):
        lg, _, _, _ = peak_stats_one(F[i])
        out[i] = lg
    return out


def curve(F):
    n = len(F)
    hw, pf = np.empty(n), np.empty(n)
    for i in range(n):
        _, h, f_, _ = peak_stats_one(F[i])
        hw[i], pf[i] = h, f_
    rms = np.sqrt(np.mean(F.astype(np.float64) ** 2, axis=(1, 2)))
    return np.column_stack([np.ones(n), np.log(np.maximum(rms, 1e-12)),
                            np.log(np.maximum(pf, 1e-12)),
                            np.log(np.maximum(hw, 1e-12))])


def main():
    cfg0 = ds.GenConfig()
    rx, _ = ds._place_receivers(cfg0, np.random.default_rng(0))
    fs = cfg0.fs
    # --- CLO: 쌍별 lag 의 '차이 구조' 영공간 (기하 불필요) ---
    #   d_ij = t_i - t_j  이므로 A(6x4) 의 열공간에 있어야 한다. 밖의 성분이 모순량.
    A = np.zeros((len(PAIRS), 4))
    for k, (i, j) in enumerate(PAIRS):
        A[k, i], A[k, j] = 1.0, -1.0
    U, S_, _ = np.linalg.svd(A, full_matrices=True)
    NUL = U[:, np.sum(S_ > 1e-9):]              # (6 x 3) 영공간 기저
    print("closure 영공간 차원 %d (쌍 %d - 독립 %d)"
          % (NUL.shape[1], len(PAIRS), int(np.sum(S_ > 1e-9))))

    RAW = {}
    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    RAW["IND"] = (dt["feats"].astype(np.float32), dt["y"].astype(np.float32))
    for lv in SHIFT:
        F, Y = [], []
        for gs in NEWG:
            d = np.load(os.path.join(ROOT, "output_ood_s%s" % gs,
                                     "shift_%s.npz" % lv))
            F.append(d["feats"].astype(np.float32))
            Y.append(d["y"].astype(np.float32))
        RAW[lv] = (np.concatenate(F), np.concatenate(Y))
    for lab, (stem, tags) in GENC.items():
        F, Y = [], []
        for t in tags:
            p = os.path.join(CACHE, "%s_%s.npz" % (stem, t))
            if not os.path.exists(p):
                if lab.startswith("occ"):
                    uf = float(lab.split()[1])
                    a, b, c = gen_seeds(NEWSET[t], "_raim_occ%s" % uf,
                                        functools.partial(wf.gen_ofdm,
                                                          used_frac=uf))
                elif lab == "IND-D":
                    a, b, c = gen_seeds(NEWSET[t], prof="TDL-D")
                else:
                    a, b, c = gen_seeds(NEWSET[t], prof=lab)
                np.savez_compressed(p, feats=a, y=b, snr=c)
                print("   생성 %s %s" % (lab, t))
            z = np.load(p)
            F.append(z["feats"].astype(np.float32))
            Y.append(z["y"].astype(np.float32))
        RAW[lab] = (np.concatenate(F), np.concatenate(Y))
    print("적재 %d 집합, 곡선통계·lag ..." % len(RAW))
    X = {k: curve(v[0]) for k, v in RAW.items()}
    LG = {k: lags_of(v[0]) for k, v in RAW.items()}
    CLO = {k: np.sqrt(np.maximum(((v @ NUL) ** 2).sum(1) / NUL.shape[1], 0.0))
           for k, v in LG.items()}
    CONDS = [k for k in RAW if k != "IND"]

    res = {"_note": ("선별 측정: RAIM 계열(CLO 패리티 / FIT 잔차)을 우리 플래그와 "
                     "같은 프로토콜로 비교. 사전등록 없음."),
           "alpha": ALPHA_D, "cells": {}}
    rng = np.random.default_rng(20260912)
    perms = [rng.permutation(2000) for _ in range(NPERM)]

    for mlab, sub in MODELS:
        ck = torch.load(os.path.join(ROOT, sub, "der_model.pt"),
                        weights_only=False)
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
            p = mu.numpy().astype(np.float64) * SCALE          # [m]
            yn = Y.astype(np.float64)
            # FIT: 추정 위치가 함의하는 쌍별 lag 과 관측 lag 의 잔차
            d = np.linalg.norm(p[:, None, :] - rx[None, :, :2], axis=2)  # (N,4)
            pred = np.stack([(d[:, i] - d[:, j]) / C0 * fs for i, j in PAIRS], 1)
            fit = np.sqrt(((pred - LG[key]) ** 2).mean(axis=1))
            return dict(X=X[key], act=act, clo=CLO[key], fit=fit,
                        nphi=np.linalg.norm(phi.numpy().astype(np.float64),
                                            axis=1),
                        epi=np.sqrt(np.trace(ep.numpy().astype(np.float64),
                                             0, 1, 2)) * SCALE,
                        err=np.sqrt(((p - yn) ** 2).sum(1)))

        S = {k: pack(k) for k in RAW}
        IND = S["IND"]

        def scores(s, det, b_ls, b_ru):
            if det == "PSR":
                return -s["X"][:, 2]
            if det == "RULE":
                return s["X"] @ b_ru - np.log(s["epi"])
            if det == "PHI":
                return np.log(s["nphi"])
            if det == "LS":
                return s["X"] @ b_ls
            if det == "MAXLA2":
                zl = (s["X"] @ b_ls - ZL[0]) / ZL[1]
                za = (s["act"] - ZA[0]) / ZA[1]
                return np.maximum(zl, za)
            if det == "CLO":
                return np.log(np.maximum(s["clo"], 1e-9))
            if det == "FIT":
                return np.log(np.maximum(s["fit"], 1e-9))
            return np.log(s["epi"])

        acc = {(det, c): [] for det in DET for c in CONDS + ["NEG"]}
        for pi in perms:
            mp, cl, ng = pi[:666], pi[666:1333], pi[1333:]
            b_ls = np.linalg.lstsq(IND["X"][mp],
                                   np.log(np.maximum(IND["err"][mp], 1e-9)),
                                   rcond=None)[0]
            b_ru = np.linalg.lstsq(IND["X"][mp], np.log(IND["epi"][mp]),
                                   rcond=None)[0]
            _zl = IND["X"][mp] @ b_ls
            ZL = (float(np.mean(_zl)), float(np.std(_zl) + 1e-12))
            ZA = (float(np.mean(IND["act"][mp])),
                  float(np.std(IND["act"][mp]) + 1e-12))
            for det in DET:
                cal = np.sort(scores({k: v[cl] for k, v in IND.items()},
                                     det, b_ls, b_ru))
                n = len(cal)

                def pow_(sv):
                    pv = (1.0 + n - np.searchsorted(cal, sv, side="left")) \
                        / (n + 1.0)
                    return float((pv <= ALPHA_D).mean())
                acc[(det, "NEG")].append(pow_(scores(
                    {k: v[ng] for k, v in IND.items()}, det, b_ls, b_ru)))
                for c in CONDS:
                    acc[(det, c)].append(pow_(scores(S[c], det, b_ls, b_ru)))
        for (det, c), v in acc.items():
            res["cells"]["%s|%s|%s" % (det, mlab, c)] = float(np.mean(v))
        print("  %s 완료" % mlab)

    C = res["cells"]
    ORD = ["NEG", "IND-D", "sco_4", "snr_1", "snr_2", "cfo_2", "cfo_4", "kf_3",
           "delay_1", "delay_2", "delay_3", "occ 0.05", "TDL-A", "TDL-C"]
    print("\n=== 탐지력 @ alpha=0.05 (5 모델 평균 [최소~최대]) ===")
    print("%-10s | %s" % ("조건", " ".join("%17s" % d for d in DET)))
    for c in ORD:
        row = []
        for det in DET:
            v = [C["%s|%s|%s" % (det, m, c)] for m, _ in MODELS
                 if "%s|%s|%s" % (det, m, c) in C]
            row.append("%.3f[%.2f~%.2f]" % (np.mean(v), min(v), max(v))
                       if v else "—")
        print("%-10s | %s" % (c, " ".join("%17s" % x for x in row)))
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n[saved]", OUT)


if __name__ == "__main__":
    main()
