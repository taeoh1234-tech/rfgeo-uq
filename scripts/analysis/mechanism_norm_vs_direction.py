r"""
mechanism_norm_vs_direction.py — 마지막 고리: 활성 '크기'인가 '방향'인가.

`mechanism_ABC.py` 가 밝힌 것
------------------------------
  * head fc 의 ell0/ell2(타원 크기) 가중치 평균이 **음수** (-0.032, -0.049)
  * phi 는 ReLU 출력이라 **모든 성분 >= 0**
  * 조건 간 Spearman(||phi||, epi) = **-0.905** (snr_1 제외 -0.964)
  * 역전은 학습 다양체 **안**에서 일어난다 (밖 성분 제거해도 delay_3 은 0.548 -> 0.640)

남은 질문
---------
  ell = w . phi = ||phi|| * (w . phi_hat)
  (1) ||phi|| 가 커져서 ell 이 내려갔나            <- '크기' 경로
  (2) 방향 phi_hat 이 나쁜 쪽으로 돌아서 내려갔나   <- '방향' 경로

그리고 ||phi|| 는 왜 커지나 — 평균 풀링 직전 활성맵 (256, 17) 에서
**활성 위치 수**가 늘어난 것인지 **위치별 크기**가 커진 것인지 본다.

    python scripts/data/mechanism_norm_vs_direction.py
"""
from __future__ import annotations
import functools, io, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, _p)
from rfgeo.evidential import niw_uncertainties                     # noqa: E402
from train_der import DERModel                                     # noqa: E402
from peak_decomposition import from_npz, gen, SCALE, BAR           # noqa: E402
from rfgeo import waveform as wf                                    # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "mechanism_norm_vs_direction.json")
REF = "ref (학습분포)"

CASES = [
    (REF, None),
    ("sco_4 (NULL)", lambda: from_npz("sco_4")),
    ("delay_1", lambda: from_npz("delay_1")),
    ("delay_2", lambda: from_npz("delay_2")),
    ("delay_3", lambda: from_npz("delay_3")),
    ("occ 0.05", lambda: gen("_nd_occ05", functools.partial(wf.gen_ofdm, used_frac=0.05))),
    ("TDL-C", lambda: gen(prof="TDL-C")),
    ("kf_3", lambda: from_npz("kf_3")),
    ("snr_1", lambda: from_npz("snr_1")),
]


def main():
    ck = torch.load(os.path.join(ROOT, "output_final_der", "der_model.pt"), weights_only=False)
    mean, std = ck["mean"], ck["std"]
    model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
    model.load_state_dict(ck["model"]); model.eval()
    bb = model.backbone
    W = model.head.fc.weight.detach().numpy().astype(np.float64)
    b = model.head.fc.bias.detach().numpy().astype(np.float64)
    w_eff = W[2] + W[4]          # ell0 + ell2 (타원 크기 두 대각)
    b_eff = b[2] + b[4]

    @torch.no_grad()
    def forward(F):
        Z = torch.from_numpy((F - mean) / std)
        x = torch.relu(bb.stem(Z))
        amap = bb.blocks(x)                       # (B, 256, 17)  풀링 직전
        phi = bb.proj(bb.pool(amap))              # (B, 128)
        mu, L, nu = model.head(phi)
        al, ep = niw_uncertainties(L, nu)
        epi = np.sqrt(np.trace(ep.numpy(), axis1=1, axis2=2)) * SCALE
        return amap.numpy().astype(np.float64), phi.numpy().astype(np.float64), mu.numpy(), epi

    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    D = {}
    for nm, fn in CASES:
        F, Y = ((dt["feats"].astype(np.float32), dt["y"].astype(np.float32))
                if fn is None else fn())
        amap, phi, mu, epi = forward(F)
        D[nm] = dict(amap=amap, phi=phi, epi=epi,
                     err=np.sqrt(((mu - Y / SCALE) ** 2).sum(1)) * SCALE)
        print("  수집 %-16s n=%d  활성맵 %s" % (nm, len(F), amap.shape))

    ref = D[REF]
    thr = float(np.percentile(ref["amap"], 90))       # 학습분포 90 백분위를 '활성' 기준으로
    nrm_r = float(np.median(np.linalg.norm(ref["phi"], axis=1)))
    proj_r = float(np.median((ref["phi"] / np.maximum(
        np.linalg.norm(ref["phi"], axis=1, keepdims=True), 1e-30)) @ w_eff))
    ell_r = float(np.median(ref["phi"] @ w_eff + b_eff))

    # ---------- 1. 크기 vs 방향 ----------
    print("\n" + BAR)
    print("1. ell = ||phi|| x (w . phi_hat) 분해   [w = ell0+ell2 가중치, 평균 %.5f]" % w_eff.mean())
    print(BAR)
    print("  %-16s %10s %10s %12s | %11s %11s %10s"
          % ("조건", "||phi||비", "방향비", "ell(합)", "Dell 총", "Dell 크기", "Dell 방향"))
    R1 = {}
    for nm, _ in CASES:
        phi = D[nm]["phi"]
        nrm = np.linalg.norm(phi, axis=1)
        hat = phi / np.maximum(nrm[:, None], 1e-30)
        proj = hat @ w_eff
        ell = phi @ w_eff + b_eff
        n_m, p_m, e_m = float(np.median(nrm)), float(np.median(proj)), float(np.median(ell))
        # 곱 분해: n*p - n_r*p_r = (n-n_r)*p_r + n_r*(p-p_r) + (n-n_r)*(p-p_r)
        d_size = (n_m - nrm_r) * proj_r
        d_dir = nrm_r * (p_m - proj_r)
        d_cross = (n_m - nrm_r) * (p_m - proj_r)
        R1[nm] = dict(norm=n_m, norm_ratio=n_m / nrm_r, proj=p_m, proj_ratio=p_m / proj_r,
                      ell=e_m, d_tot=e_m - ell_r, d_size=d_size, d_dir=d_dir, d_cross=d_cross)
        print("  %-16s %10.3f %10.3f %12.4f | %11.4f %11.4f %10.4f"
              % (nm, n_m / nrm_r, p_m / proj_r, e_m, e_m - ell_r, d_size, d_dir + d_cross))

    # ---------- 2. ||phi|| 는 왜 커지나 ----------
    print("\n" + BAR)
    print("2. 풀링 직전 활성맵 (256 채널 x 17 위치) — 개수가 늘었나 크기가 커졌나")
    print(BAR)
    print("  임계 = 학습분포 90 백분위 = %.4f" % thr)
    print("  %-16s %12s %12s %12s %12s"
          % ("조건", "활성 비율", "활성 비율배", "활성 평균", "맵 전체평균비"))
    R2 = {}
    fr_r = float(np.median((ref["amap"] > thr).mean(axis=(1, 2))))
    mn_r = float(np.median(ref["amap"].mean(axis=(1, 2))))
    for nm, _ in CASES:
        a = D[nm]["amap"]
        fr = float(np.median((a > thr).mean(axis=(1, 2))))
        mn = float(np.median(a.mean(axis=(1, 2))))
        av = a.copy(); av[av <= thr] = np.nan
        amn = float(np.nanmedian(np.nanmean(av, axis=(1, 2))))
        R2[nm] = dict(active_frac=fr, active_ratio=fr / fr_r, active_mean=amn,
                      map_mean_ratio=mn / mn_r)
        print("  %-16s %12.4f %12.3f %12.4f %12.3f"
              % (nm, fr, fr / fr_r, amn, mn / mn_r))

    # ---------- 3. 인과 사슬 요약 ----------
    print("\n" + BAR)
    print("3. 인과 사슬 (기준선 대비 배율)")
    print(BAR)
    print("  %-16s %10s %10s %10s %10s %10s"
          % ("조건", "맵평균", "||phi||", "ell(차)", "tr(LLT)", "epi"))
    epi_r = float(np.median(ref["epi"]))
    R3 = {}
    for nm, _ in CASES:
        d_ell = R1[nm]["d_tot"]
        R3[nm] = dict(map_mean=R2[nm]["map_mean_ratio"], norm=R1[nm]["norm_ratio"],
                      d_ell=d_ell, tr_ratio=float(np.exp(d_ell)),
                      epi=float(np.median(D[nm]["epi"]) / epi_r))
        t = R3[nm]
        print("  %-16s %10.3f %10.3f %+10.4f %10.3f %10.3f"
              % (nm, t["map_mean"], t["norm"], t["d_ell"], t["tr_ratio"], t["epi"]))

    out = dict(_note="ell 을 크기/방향으로 분해 + 풀링 직전 활성맵 진단.",
               w_eff_mean=float(w_eff.mean()), w_eff_neg_frac=float((w_eff < 0).mean()),
               threshold=thr, decomposition=R1, activation=R2, chain=R3)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
