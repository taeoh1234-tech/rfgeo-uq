# -*- coding: utf-8 -*-
r"""snapshot_gate_j.py - 0B: tau=0.20 동결 게이트의 스냅샷 단위 평가 (family-wise).

명세: prereg/SNAPSHOT_GATE_SPEC_2026-09-14.md  (결과 계산 전 고정)

범위: 사전 배정 arm + 동결 kappa + 설계 상수 tau=0.20 아래에서 스냅샷별
  accept_i = (교정 영역 A_i / 탐색 원반) <= tau
의 acceptance rate / accepted 커버리지 / accepted 영역을 잰다.
**런타임 selector 검증이 아니다** (배정은 조건 라벨로 이뤄진다).

GATE-R: 이 스크립트가 접촉하는 ls040_confirm.json / corrector_j_arms.json 의
모든 셀을 재계산해 상대오차 <=1e-9 로 재현해야 한다. 실패 시 abort.

코드 경로는 ls035_confirm.py(--arm 0.40) / corrector_j_arms.py(PRIMARY) 를
축자 복제했다 (z 표준화·클리핑 규약 포함). 원본 무수정.

    python scripts/data/snapshot_gate_j.py
"""
from __future__ import annotations
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

from rfgeo.evidential import niw_uncertainties                       # noqa: E402
from train_der import DERModel                                       # noqa: E402
from peak_decomposition import peak_stats_one, SCALE, ML, ALPHA      # noqa: E402

OUTDIR = os.path.join(ROOT, "artifacts")
OUT = os.path.join(OUTDIR, "snapshot_gate_j.json")
OUTNPZ = os.path.join(OUTDIR, "snapshot_gate_j_persample.npz")
CACHE = os.path.join(OUTDIR, "cfs_cache")
MODELS = [("m0", "output_seed20260628_clean"), ("m1", "output_seed20261628_clean"),
          ("m2", "output_seed20262628_clean"), ("m3", "output_seed20263628_clean"),
          ("seed0", "output_seed0")]
NEWG = ["92001", "92002"]
TAGS = ["j1", "j2", "j3"]
EDGES = [-5.0, 0.0, 5.0, 10.0, 15.0]
DISC = np.pi * 8000.0 ** 2 / 1e6
TAU = 0.20
TAUGRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
LS_SHIFT = ["kf_3", "cfo_4", "snr_2", "sco_4"]          # ls 하네스 · 풀링 셀
LS_GENC = ["IND-D", "TDL-A", "TDL-C"]                    # ls 하네스 · 태그별 셀
RTOL = 1e-9


def fsq(v, a):
    n = len(v)
    return float(np.sort(v)[min(int(np.ceil((n + 1) * (1 - a))), n) - 1])


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


def load_json(n):
    with io.open(os.path.join(OUTDIR, n), encoding="utf-8") as f:
        return json.load(f)


class GateR:
    def __init__(self):
        self.n = 0
        self.max_rel = 0.0
        self.bad = []

    def check(self, label, got, ref):
        self.n += 1
        for g, r in zip(np.atleast_1d(got), np.atleast_1d(ref)):
            if g is None and r is None:
                continue
            rel = abs(g - r) / max(1.0, abs(r))
            self.max_rel = max(self.max_rel, rel)
            if rel > RTOL:
                self.bad.append((label, float(g), float(r)))


def main():
    # ---------------- 데이터 적재 ----------------
    print("[1/4] 데이터 적재 ...")
    RAW, SNR = {}, {}
    dc = np.load(os.path.join(ROOT, "output_final", "gcc_calib.npz"))
    RAW["CALFULL"] = (dc["feats"].astype(np.float32), dc["y"].astype(np.float32))
    for lv in LS_SHIFT:                                  # ls: 92001+92002 풀링
        F, Y = [], []
        for gs in NEWG:
            d = np.load(os.path.join(ROOT, "output_ood_s%s" % gs, "shift_%s.npz" % lv))
            F.append(d["feats"].astype(np.float32))
            Y.append(d["y"].astype(np.float32))
        RAW[lv] = (np.concatenate(F), np.concatenate(Y))
    for gs in NEWG:                                      # corrector: 세트별
        d = np.load(os.path.join(ROOT, "output_ood_s%s" % gs, "shift_delay_3.npz"))
        RAW["delay_3|%s" % gs] = (d["feats"].astype(np.float32),
                                  d["y"].astype(np.float32))
    for cond in LS_GENC + ["occ 0.05"]:                  # 캐시 (재생성 없음)
        for tag in TAGS:
            p = os.path.join(CACHE, "%s_%s.npz"
                             % (cond.replace(" ", "").replace(".", ""), tag))
            z = np.load(p)
            k = "%s|%s" % (cond, tag)
            RAW[k] = (z["feats"].astype(np.float32), z["y"].astype(np.float32))
            SNR[k] = np.asarray(z["snr"], float)
    print("   집합 %d개" % len(RAW))

    print("[2/4] 곡선 통계 ...")
    X = {k: curve(v[0]) for k, v in RAW.items()}

    LSJ = load_json("ls040_confirm.json")["cells"]
    CRJ = load_json("corrector_j_arms.json")["cells"]
    gr = GateR()

    # 스냅샷 단위 산출: per[(mlab, cond)] = dict(frac, hit, snr?)
    per = {}
    QREC = {}

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
            phi = mo.backbone.proj(mo.backbone.pool(h))
            mu, L, nu = mo.head(phi)
            al, ep = niw_uncertainties(L, nu)
            tot = (al + ep).numpy().astype(np.float64)
            p = mu.numpy().astype(np.float64)
            yn = (Y / SCALE).astype(np.float64)
            d = (yn - p)[:, :, None]
            return dict(
                X=X[key],
                nphi=np.linalg.norm(phi.numpy().astype(np.float64), axis=1),
                epi=np.sqrt(np.trace(ep.numpy().astype(np.float64), 0, 1, 2)) * SCALE,
                err=np.sqrt(((p - yn) ** 2).sum(1)) * SCALE,
                M=np.sqrt(np.maximum((np.linalg.solve(tot, d) * d).sum((1, 2)), 0.0)),
                sdet=np.sqrt(np.maximum(np.linalg.det(tot), 1e-30)))

        S = {k: pack(k) for k in RAW if k != "CALFULL"}
        CF = pack("CALFULL")
        FIT = {k: v[0:2000] for k, v in CF.items()}
        CAL = {k: v[2000:4000] for k, v in CF.items()}

        # ---------- ls 하네스 (LS@0.40) : ls035_confirm.py 축자 ----------
        b_ls = np.linalg.lstsq(FIT["X"], np.log(np.maximum(FIT["err"], 1e-9)),
                               rcond=None)[0]
        ml_, sl_ = (float(np.mean(FIT["X"] @ b_ls)),
                    float(np.std(FIT["X"] @ b_ls) + 1e-12))

        def zl(s):
            return (s["X"] @ b_ls - ml_) / sl_

        v = zl(FIT)
        Z2 = (float(np.mean(v)), float(np.std(v) + 1e-12))

        def zh_ls(s):                                    # LS 는 클리핑 없음
            return (zl(s) - Z2[0]) / Z2[1]

        kap = 0.4
        gc = np.exp(kap * zh_ls(CAL))
        Q_ls = fsq(CAL["M"] / gc, ALPHA)

        def snap_ls(key):
            gv = np.exp(kap * zh_ls(S[key]))
            A = np.pi * (Q_ls * gv) ** 2 * S[key]["sdet"] * SCALE ** 2 / 1e6
            return A / DISC, S[key]["M"] / gv <= Q_ls

        for lv in LS_SHIFT:
            frac, hit = snap_ls(lv)
            gr.check("LS|%s|%s" % (mlab, lv),
                     [float(hit.mean()), float(np.median(frac))],
                     LSJ["LS@0.40|%s|%s" % (mlab, lv)])
            per[(mlab, lv)] = dict(frac=frac, hit=hit)
        for cond in LS_GENC:
            fr_all, hi_all, sn_all = [], [], []
            for tag in TAGS:
                k = "%s|%s" % (cond, tag)
                frac, hit = snap_ls(k)
                gr.check("LS|%s|%s" % (mlab, k),
                         [float(hit.mean()), float(np.median(frac))],
                         LSJ["LS@0.40|%s|%s" % (mlab, k)])
                if cond == "IND-D":                      # __bins__ 재현
                    g = np.digitize(SNR[k], EDGES)
                    ref = LSJ["LS@0.40|%s|%s|__bins__" % (mlab, tag)]
                    got = [float(hit[g == b].mean()) if (g == b).sum() > 0 else None
                           for b in range(6)]
                    gr.check("LSbins|%s|%s" % (mlab, tag), got, ref)
                fr_all.append(frac)
                hi_all.append(hit)
                sn_all.append(SNR[k])
            per[(mlab, cond)] = dict(frac=np.concatenate(fr_all),
                                     hit=np.concatenate(hi_all),
                                     snr=np.concatenate(sn_all))

        # ------- corrector 하네스 (PRIMARY · RULE@0.6 / PHI@1.5) : 축자 -------
        b_ru = np.linalg.lstsq(FIT["X"], np.log(FIT["epi"]), rcond=None)[0]

        def zraw(s, arm):
            if arm == "RULE":
                return s["X"] @ b_ru - np.log(s["epi"])
            return np.log(s["nphi"])                     # PHI

        ZS = {}
        for a in ("RULE", "PHI"):
            v = zraw(FIT, a)
            ZS[a] = (float(np.mean(v)), float(np.std(v) + 1e-12))

        def zh_c(s, arm):                                # 클리핑 있음 (원본과 동일)
            return np.maximum((zraw(s, arm) - ZS[arm][0]) / ZS[arm][1], 0.0)

        Q_c = {}
        for arm, kapc in (("RULE", 0.6), ("PHI", 1.5)):
            gcc_ = np.exp(kapc * zh_c(CAL, arm))
            Q_c[arm] = fsq(CAL["M"] / gcc_, ALPHA)

        def snap_c(key, arm, kapc):
            gv = np.exp(kapc * zh_c(S[key], arm))
            A = np.pi * (Q_c[arm] * gv) ** 2 * S[key]["sdet"] * SCALE ** 2 / 1e6
            return A / DISC, S[key]["M"] / gv <= Q_c[arm]

        fr_all, hi_all = [], []
        for gs in NEWG:
            k = "delay_3|%s" % gs
            frac, hit = snap_c(k, "RULE", 0.6)
            gr.check("RULE|%s|%s" % (mlab, k),
                     [float(hit.mean()), float(np.median(frac))],
                     CRJ["PRIMARY|RULE|0.6|%s|%s" % (mlab, k)])
            fr_all.append(frac)
            hi_all.append(hit)
        per[(mlab, "delay_3")] = dict(frac=np.concatenate(fr_all),
                                      hit=np.concatenate(hi_all))
        fr_all, hi_all = [], []
        for tag in TAGS:
            k = "occ 0.05|%s" % tag
            frac, hit = snap_c(k, "PHI", 1.5)
            gr.check("PHI|%s|%s" % (mlab, k),
                     [float(hit.mean()), float(np.median(frac))],
                     CRJ["PRIMARY|PHI|1.5|%s|%s" % (mlab, k)])
            fr_all.append(frac)
            hi_all.append(hit)
        per[(mlab, "occ 0.05")] = dict(frac=np.concatenate(fr_all),
                                       hit=np.concatenate(hi_all))

        QREC[mlab] = {"LS@0.40": Q_ls, "RULE@0.6": Q_c["RULE"], "PHI@1.5": Q_c["PHI"]}
        print("   %s 완료 (GATE-R 누적 max_rel=%.2e, bad=%d)"
              % (mlab, gr.max_rel, len(gr.bad)))

    # ---------------- GATE-R 판정 ----------------
    if gr.bad:
        print("GATE-R FAIL:", gr.bad[:5])
        raise SystemExit("GATE-R 실패 - 결과를 해석하지 않는다 (spec §5)")
    print("GATE-R PASS: %d개 대조, max_rel=%.2e" % (gr.n, gr.max_rel))

    # ---------------- 조건 행 구성 ----------------
    ROWS = [
        ("in-dist low-SNR bin", "LS@0.40", "repair"),
        ("delay_3",             "RULE@0.6", "repair"),
        ("occ 0.05",            "PHI@1.5", "repair"),
        ("kf_3",                "LS@0.40", "repair"),
        ("TDL-A",               "LS@0.40", "repair"),
        ("TDL-C",               "LS@0.40", "repair"),
        ("sco_4",               "LS@0.40", "repair (benign control)"),
        ("cfo_4",               "LS@0.40", "abstain"),
        ("snr_2",               "LS@0.40", "abstain"),
        ("IND-D (all bins, ref)", "LS@0.40", "reference"),
    ]

    def cond_arrays(mlab, cond):
        if cond == "in-dist low-SNR bin":
            d = per[(mlab, "IND-D")]
            m = np.digitize(d["snr"], EDGES) == 0
            return d["frac"][m], d["hit"][m]
        if cond == "IND-D (all bins, ref)":
            d = per[(mlab, "IND-D")]
            return d["frac"], d["hit"]
        d = per[(mlab, cond)]
        return d["frac"], d["hit"]

    def stats(frac, hit, tau):
        acc = frac <= tau
        na, nr = int(acc.sum()), int((~acc).sum())
        o = {"n": int(len(frac)), "n_accept": na,
             "acc_rate": round(float(acc.mean()), 4),
             "cov_all": round(float(hit.mean()), 4),
             "cov_accept": round(float(hit[acc].mean()), 4) if na else None,
             "cov_reject": round(float(hit[~acc].mean()), 4) if nr else None}
        o["med_frac_accept"] = (round(float(np.median(frac[acc])), 5) if na else None)
        o["p90_frac_accept"] = (round(float(np.percentile(frac[acc], 90)), 5)
                                if na else None)
        o["med_area_km2_accept"] = (round(float(np.median(frac[acc])) * DISC, 3)
                                    if na else None)
        return o

    MLBL = [m for m, _ in MODELS]
    rows = {}
    sweep = {}
    for cond, arm, action in ROWS:
        pm = {}
        for m in MLBL:
            frac, hit = cond_arrays(m, cond)
            pm[m] = stats(frac, hit, TAU)
        vals = lambda key: [pm[m].get(key) for m in MLBL
                            if pm[m].get(key) is not None]

        def rng(key):
            v = vals(key)
            return {"mean": round(float(np.mean(v)), 4),
                    "min": round(float(np.min(v)), 4),
                    "max": round(float(np.max(v)), 4), "n_models": len(v)} if v else None
        rows[cond] = {"arm": arm, "paper_action": action, "per_model": pm,
                      "summary": {k: rng(k) for k in
                                  ("acc_rate", "cov_all", "cov_accept", "cov_reject",
                                   "med_frac_accept", "p90_frac_accept")}}
        sw = {}
        for t in TAUGRID:
            ar, ca = [], []
            for m in MLBL:
                frac, hit = cond_arrays(m, cond)
                a = frac <= t
                ar.append(float(a.mean()))
                if a.sum():
                    ca.append(float(hit[a].mean()))
            sw["%.2f" % t] = {
                "acc_rate": [round(float(np.min(ar)), 4), round(float(np.max(ar)), 4)],
                "acc_rate_mean": round(float(np.mean(ar)), 4),
                "cov_accept": ([round(float(np.min(ca)), 4),
                                round(float(np.max(ca)), 4)] if ca else None),
                "cov_accept_mean": round(float(np.mean(ca)), 4) if ca else None,
                "n_models_with_accepts": len(ca)}
        sweep[cond] = sw

    out = {
        "_spec": "prereg/SNAPSHOT_GATE_SPEC_2026-09-14.md",
        "_scope": ("family-wise gate 평가: 사전 배정 arm + 동결 kappa + tau=0.20. "
                   "런타임 selector 검증이 아니며 사전등록 확증도 아니다. "
                   "accepted 커버리지에는 유한표본 보장이 없다 (데이터 의존 선택)."),
        "tau": TAU, "alpha": ALPHA, "disc_km2": DISC,
        "quantile_convention": "fsq: ceil((n+1)(1-alpha))-th order statistic (확증 계열)",
        "gate_R": {"n_cells": gr.n, "max_rel_err": gr.max_rel, "pass": True},
        "Q": {m: {k: round(v, 6) for k, v in QREC[m].items()} for m in QREC},
        "rows": rows, "tau_sweep": sweep,
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    np.savez_compressed(OUTNPZ, **{
        "%s|%s|%s" % (m, c, a): v[a]
        for (m, c), v in per.items() for a in v})

    # ---------------- 콘솔 요약 ----------------
    print("\n[4/4] tau=%.2f 스냅샷 게이트 (5모델 mean [min-max])" % TAU)
    print("%-22s %-9s %-10s  %-24s %-22s" % ("condition", "arm", "action",
                                             "acc_rate", "cov_accept"))
    print("-" * 92)
    for cond, arm, action in ROWS:
        s = rows[cond]["summary"]
        a, c = s["acc_rate"], s["cov_accept"]
        ca = ("%.3f [%.3f-%.3f] n=%d" % (c["mean"], c["min"], c["max"], c["n_models"])
              if c else "-")
        print("%-22s %-9s %-10s  %.3f [%.3f-%.3f]      %s"
              % (cond[:22], arm, action.split()[0], a["mean"], a["min"], a["max"], ca))
    print("\n[saved]", OUT)
    print("[saved]", OUTNPZ)


if __name__ == "__main__":
    main()
