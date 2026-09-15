r"""
make_fig2_ruletransfer_band.py - Fig. 2 밴드 추가 **변형안** (2026-09-12).

면담 지적: 대각선 하나만 그리면 "선 위 = 정상"으로 읽히나, 규칙은 R^2=0.54~0.60
이라 정상 자체가 산포를 갖는다. 탐지기는 그 산포의 분위수와 비교하므로(conformal)
밴드를 그려야 그림이 방법과 일치한다. 근거 수치는 fig2_band_stats.py.
원본 make_fig2_ruletransfer.py 와 논문 그림은 건드리지 않는다.

배경: 논문에 실린 fig2_ruletransfer.png 은 임시 스크립트로 만들어져 저장소에 생성
코드가 없었다(2026-09-08 확인). 여기서 영구화하면서, 조건별 중앙값을 개별 마커로
분리해 **선택성**(무해 0.96 / 저SNR 1.66 / 유해 0.08~0.71)을 그림 안에서 보이게 한다.

입력 (전부 기존 영구 산출물, 재생성/재학습 없음):
  peak_decomp_persample.npz   조건별 600표본 x {feat_rms, pk_far, halfwidth_sub, epi}
  epi_mapping_transfer.json   회귀계수 beta, 조건별 pred/actual/ratio

GATE: 재계산한 조건별 ratio 가 JSON 의 공개값과 1% 이내로 일치해야 그림을 그린다.

    python scripts/data/make_fig2_ruletransfer.py
"""
from __future__ import annotations
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "artifacts")
CACHE = os.path.join(SRC, "peak_decomp_persample.npz")
JSONF = os.path.join(SRC, "epi_mapping_transfer.json")
OUT = os.path.join(ROOT, "이미지", "fig2_ruletransfer_band.png")
OUT2 = os.path.join(ROOT, "paper", "preview_fig2_band.png")
# --base 로 실행하면 라벨을 뺀 배경 PNG + 라벨 앵커 JSON 을 만든다 (PPT 조립용)
BASE_PNG = os.path.join(ROOT, "이미지", "fig2_ruletransfer_band_base.png")
ANCHOR_JSON = os.path.join(ROOT, "이미지", "fig2_band_anchors.json")
BASE_ONLY = "--base" in sys.argv

# (json 키, 표시 라벨, 그룹)  group: ind / benign / harmful / noisy
COND = [
    ("ref (학습분포)",  "in-dist.",          "ind"),
    ("sco_4 (NULL)",    "benign",            "benign"),
    ("occ 0.20",        None,                "harmful"),
    ("occ 0.05",        "bandwidth",         "harmful"),
    ("kf_3",            "fading",            "harmful"),
    ("TDL-A",           None,                "harmful"),
    ("TDL-C",           "channel family",    "harmful"),
    ("delay_2",         None,                "harmful"),
    ("delay_3",         "strong multipath",  "harmful"),
    ("snr_1",           "low SNR",           "noisy"),
]
STYLE = {  # (색, 마커, 산점 alpha)
    "ind":     ("#2c5f9e", "o", 0.16),
    "benign":  ("#7fb2e5", "s", 0.16),
    "harmful": ("#b03a2e", "v", 0.09),
    "noisy":   ("#1f7a6c", "*", 0.13),
}


def main():
    J = json.load(io.open(JSONF, encoding="utf-8"))
    beta = np.asarray(J["beta"], float)
    res = J["result"]
    D = np.load(CACHE, allow_pickle=True)

    def rule_sigma(o):
        X = np.column_stack([np.ones(len(o["feat_rms"])),
                             np.log(np.maximum(np.asarray(o["feat_rms"], float), 1e-12)),
                             np.log(np.maximum(np.asarray(o["pk_far"], float), 1e-12)),
                             np.log(np.maximum(np.asarray(o["halfwidth_sub"], float), 1e-12))])
        return np.exp(X @ beta)

    P, A, ok = {}, {}, True
    ref_ratio = None
    print("GATE  조건            pred(재계산)  actual   ratio  JSON ratio   차이")
    for key, _lab, _g in COND:
        o = D[key].item()
        pred = rule_sigma(o)
        act = np.asarray(o["epi"], float)
        P[key], A[key] = pred, act
        r = float(np.median(act) / np.median(pred))
        rj = float(res[key]["actual"]) / float(res[key]["pred"])
        d = abs(r - rj) / rj
        if key == "ref (학습분포)":
            ref_ratio = r
        flag = "OK" if d < 0.01 else "FAIL"
        ok &= d < 0.01
        print("  %-16s %9.1f %9.1f %7.3f %8.3f %8.2e %s"
              % (key, np.median(pred), np.median(act), r, rj, d, flag))
    if not ok:
        raise SystemExit("GATE FAIL - 재현 불일치, 그림을 그리지 않는다")
    print("GATE PASS (ref ratio = %.4f, 본문 정규화 기준)" % ref_ratio)

    print("\nref 정규화 배율 (본문 인용값과 대조):")
    for key, lab, _g in COND:
        rn = (np.median(A[key]) / np.median(P[key])) / ref_ratio
        print("  %-16s x%.3f   %s" % (key, rn, "<- " + lab if lab else ""))

    # ---------------- 그림 ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 400,
                         "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.5, 3.05))

    AL = {"ind": 0.20, "benign": 0.12, "harmful": 0.055, "noisy": 0.10}
    for key, _lab, g in COND:
        c, m, _ = STYLE[g]
        ax.scatter(P[key], A[key], s=4.5, marker=m, c=c, alpha=AL[g],
                   linewidths=0, zorder=2, rasterized=True)

    lo, hi = 45.0, 1.4e4          # x/y 동일 범위 -> 대각선이 상자 대각선과 일치

    # ---- 정상분포 잔차 밴드 (2026-09-12 추가) --------------------------------
    # r = log(actual/pred). in-dist 의 중앙 90% 구간을 음영으로, 그 하단 모서리가
    # 곧 단측 alpha=0.05 탐지 경계다(q05(r) = -q95(-r)). 즉 밴드 아래 = 발화.
    rref = np.log(A["ref (학습분포)"]) - np.log(P["ref (학습분포)"])
    q05, q95 = float(np.quantile(rref, 0.05)), float(np.quantile(rref, 0.95))
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs * np.exp(q05), xs * np.exp(q95),
                    color="0.55", alpha=0.13, lw=0, zorder=1)
    ax.plot(xs, xs * np.exp(q95), color="0.55", lw=0.6, ls=":", zorder=3)
    ax.plot(xs, xs * np.exp(q05), color="#8a1c1c", lw=1.0, ls="-", zorder=3)

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, zorder=4)

    # ---- 모든 자유 텍스트는 여기 한 곳에 모은다 (PPT 편집본과 좌표를 공유) ----
    # (텍스트, 축분수 x, y, ha, va, pt, 색, 회전)
    TEXTS = [
        ("rule obeyed", 0.845, 0.905, "center", "center", 6.2, "0.30", 41),
        ("in-dist. spread (90%)", 0.880, 0.800, "center", "center", 5.6,
         "0.42", 41),
        ("flag fires below ($\\alpha$=0.05)", 0.775, 0.635, "center",
         "center", 5.9, "#8a1c1c", 41),
        ("over-alarm", 0.030, 0.660, "left", "bottom", 6.4, "0.50", 0),
        ("(reported > rule)", 0.030, 0.630, "left", "bottom", 5.6, "0.62", 0),
        ("over-confident", 0.985, 0.062, "right", "bottom", 6.4, "0.50", 0),
        ("(reported < rule)", 0.985, 0.028, "right", "bottom", 5.6, "0.62", 0),
    ]
    # 4항목 범례 — 액자·마커는 그림에, **글씨는 TEXTS 로** 빼서 PPT 에서 편집 가능하게.
    RN = {k: (np.median(A[k]) / np.median(P[k])) / ref_ratio for k in P}
    LX0, LY0, LW, LH = 0.028, 0.720, 0.379, 0.262      # 액자 (축 분수, 내용물에 맞춤 2026-09-09)
    LEG = [("ind", "in-distribution"),
           ("benign", "benign shift  ×%.2f" % RN["sco_4 (NULL)"]),
           ("harmful", "shift under test"),
           ("noisy", "low SNR  ×%.2f" % RN["snr_1"])]
    ax.add_patch(plt.Rectangle((LX0, LY0), LW, LH, transform=ax.transAxes,
                               fc="white", ec="0.62", lw=0.6, zorder=9))
    for i, (g, t) in enumerate(LEG):
        ry = LY0 + LH - 0.040 - i * 0.0605
        ax.scatter([LX0 + 0.048], [ry], s=34 if STYLE[g][1] != "*" else 78,
                   marker=STYLE[g][1], c=STYLE[g][0], edgecolors="white",
                   linewidths=0.6, transform=ax.transAxes, zorder=10)
        TEXTS.append((t, LX0 + 0.090, ry, "left", "center", 6.2, "0.10", 0))

    # 라벨 위치는 축 분수 좌표로 직접 지정 (fx, fy, ha, va, 지시선)
    # 범례가 in-dist / benign / low SNR 을 이미 이름 짓는다.
    # 직접 라벨은 하나의 마커를 공유하는 '유해 이동' 4종에만 붙인다.
    LAB = {
        "TDL-C":          (0.985, 0.470, "right", "bottom", True),
        "kf_3":           (0.300, 0.185, "left", "bottom", True),
        "delay_3":        (0.985, 0.150, "right", "bottom", True),
        "occ 0.05":       (0.250, 0.045, "left", "bottom", True),
    }
    # 마커 모양별 시각 크기 보정 (별은 같은 s 에서 작게 보인다)
    SZ = {"o": 88, "s": 84, "v": 92, "*": 210}
    anchors = []
    for key, lab, g in COND:
        if lab is None:            # 라벨 없는 조건은 중앙값 마커를 그리지 않는다
            continue
        c, m, _ = STYLE[g]
        x, y = float(np.median(P[key])), float(np.median(A[key]))
        rn = (y / x) / ref_ratio
        zo = {"harmful": 7.0, "noisy": 7.4, "benign": 7.6, "ind": 8.0}[g]
        ax.scatter([x], [y], s=SZ[m], marker=m, c=c,
                   edgecolors="white", linewidths=1.1, zorder=zo)
        if key not in LAB:          # 범례가 이름을 붙이는 조건은 마커만
            continue
        fx, fy, ha, va, lead = LAB[key]
        txt = lab if key == "ref (학습분포)" else r"%s $\times$%.2f" % (lab, rn)
        if not BASE_ONLY:
            ax.annotate(txt, (x, y), xycoords="data", xytext=(fx, fy),
                        textcoords="axes fraction",
                        ha=ha, va=va, fontsize=6.4, color=c, fontweight="bold",
                        zorder=8,
                        bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.9),
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.6,
                                        shrinkA=2, shrinkB=6) if lead else None)
        anchors.append(dict(key=key, label=lab, ratio=round(rn, 3),
                            text=("%s" % lab) if key == "ref (학습분포)"
                            else ("%s ×%.2f" % (lab, rn)),
                            color=c, data=[x, y], label_axfrac=[fx, fy],
                            ha=ha, va=va, lead=bool(lead)))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # 축 라벨도 PPT 에서 고칠 수 있도록 자유 텍스트로 배치 (축 분수, 축 밖 좌표 허용)
    AXLAB = [(r"rule-implied $\sigma_{epi}$ [m]   (GCC curve only, no labels)",
              "rule-implied sigma_epi [m]   (GCC curve only, no labels)",
              0.50, -0.098, "center", "top", 7.0, 0),
             (r"reported $\sigma_{epi}$ [m]", "reported sigma_epi [m]",
              -0.118, 0.50, "center", "bottom", 7.0, 90)]
    for tex, plain, tx, ty, ha_, va_, sz, rot in AXLAB:
        if not BASE_ONLY:
            ax.text(tx, ty, tex, transform=ax.transAxes, ha=ha_, va=va_,
                    fontsize=sz, color="black", rotation=rot, zorder=11)
        TEXTS.append((plain, tx, ty, ha_, va_, sz, "0.00", rot))
    if not BASE_ONLY:      # 자유 텍스트 렌더 (BASE_ONLY 면 PPT 가 얹는다)
        for t, tx, ty, ha_, va_, sz, col, rot in TEXTS[:11]:
            ax.text(tx, ty, t.replace(">", "$>$").replace("<", "$<$"),
                    transform=ax.transAxes, ha=ha_, va=va_, fontsize=sz,
                    color=col, rotation=rot, zorder=11)

    ax.tick_params(labelsize=6.5, length=2.5, pad=1.5)
    ax.grid(True, which="major", lw=0.4, color="0.91", zorder=0)

    fig.subplots_adjust(left=0.155, right=0.988, top=0.988, bottom=0.155)

    if BASE_ONLY:                                   # PPT 배경용: 라벨 없는 판 + 앵커 좌표
        fig.canvas.draw()
        W, H = fig.get_size_inches()
        bb = ax.get_position()          # 축 상자의 figure 분수 좌표
        for a in anchors:
            px, py = ax.transData.transform(a["data"])
            a["anchor_frac"] = [float(px / (fig.dpi * W)), float(py / (fig.dpi * H))]
            fx, fy = a["label_axfrac"]
            a["label_frac"] = [float(bb.x0 + fx * bb.width),
                               float(bb.y0 + fy * bb.height)]
        free = [dict(text=t, ha=ha_, va=va_, pt=sz, color=col, rot=rot,
                     frac=[float(bb.x0 + tx * bb.width),
                           float(bb.y0 + ty * bb.height)])
                for t, tx, ty, ha_, va_, sz, col, rot in TEXTS]
        fig.savefig(BASE_PNG)
        with io.open(ANCHOR_JSON, "w", encoding="utf-8") as f:
            json.dump({"fig_in": [float(W), float(H)], "anchors": anchors,
                       "texts": free}, f, ensure_ascii=False, indent=1)
        plt.close(fig)
        print("[saved]", BASE_PNG)
        print("[saved]", ANCHOR_JSON)
        return

    for p2 in (OUT, OUT2):
        fig.savefig(p2)
    plt.close(fig)
    print("[saved]", OUT)
    print("[saved]", OUT2)


if __name__ == "__main__":
    main()
