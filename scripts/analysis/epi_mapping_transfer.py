r"""
epi_mapping_transfer.py — 학습분포에서 배운 "입력 품질 -> 불확실성" 매핑이 분포 밖으로 옮겨가는가.

왜 이 질문인가
--------------
`peak_decomposition_followup.py` 가 보인 것:
  delay_3 과 TDL-C 는 입력 통계(pk/far 0.225 vs 0.230, featRMS 0.624 vs 0.571)가
  **거의 완전히 짝맞춰지는데** epistemic 이 0.681배 vs 1.914배로 갈린다.

그런데 조건 **내부** Spearman(featRMS, epi) 은 두 조건 모두 음(-0.586 / -0.659)으로 정상이다.
=> 관계의 **기울기**는 살아 있는데 **절편**이 조건마다 이동한다는 가설이 선다.

검정
----
  1. 학습분포(ref)에서만  log(epi) ~ b0 + b1 log(featRMS) + b2 log(pk_far) + b3 log(halfwidth)
     를 적합한다.  (교차적합으로 과적합 배제)
  2. 그 매핑을 각 이동 조건에 **그대로 적용**해 epi 를 예측한다.
  3. 잔차 median( log 실제 - log 예측 ) 을 본다.
       ~ 0  : 매핑이 옮겨간다 (불확실성이 입력 품질의 안정된 함수)
       < 0  : 실제 epi 가 예측보다 **낮다** = 과신 (조용한 실패)
       > 0  : 실제 epi 가 예측보다 높다 = 과대경보
  4. 조건 내부 Spearman(예측 epi, 실제 epi) 로 기울기 보존 여부를 따로 본다.

    python scripts/data/epi_mapping_transfer.py
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
from rfgeo import dataset as ds, waveform as wf                    # noqa: E402
from rfgeo.evidential import niw_uncertainties                     # noqa: E402
from train_der import DERModel                                     # noqa: E402
from peak_decomposition import batch_peak, from_npz, gen, SCALE, BAR   # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "epi_mapping_transfer.json")
CACHE = os.path.join(ROOT, "artifacts", "peak_decomp_persample.npz")
FEATS = ("feat_rms", "pk_far", "halfwidth_sub")

CASES = [
    ("ref (학습분포)", None),
    ("sco_4 (NULL)", lambda: from_npz("sco_4")),
    ("delay_1", lambda: from_npz("delay_1")),
    ("delay_2", lambda: from_npz("delay_2")),
    ("delay_3", lambda: from_npz("delay_3")),
    ("occ 0.20", lambda: gen("_mt_occ20", functools.partial(wf.gen_ofdm, used_frac=0.20))),
    ("occ 0.05", lambda: gen("_mt_occ05", functools.partial(wf.gen_ofdm, used_frac=0.05))),
    ("TDL-A", lambda: gen(prof="TDL-A")),
    ("TDL-C", lambda: gen(prof="TDL-C")),
    ("kf_3", lambda: from_npz("kf_3")),
    ("snr_1", lambda: from_npz("snr_1")),
]


def collect():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print("  [cache] %s" % CACHE)
        return {k: z[k].item() for k in z.files}
    cfg0 = ds.GenConfig()
    rx, _ = ds._place_receivers(cfg0, np.random.default_rng(0))
    fs = cfg0.fs
    ck = torch.load(os.path.join(ROOT, "output_final_der", "der_model.pt"), weights_only=False)
    mean, std = ck["mean"], ck["std"]
    model = DERModel(n_pairs=6, r=ck.get("r", 1.0))
    model.load_state_dict(ck["model"])
    model.eval()

    @torch.no_grad()
    def pred(Z):
        mu, L, nu = model(torch.from_numpy(Z))
        al, ep = niw_uncertainties(L, nu)
        return mu.numpy(), ep.numpy()

    dt = np.load(os.path.join(ROOT, "output_final", "gcc_test.npz"))
    REF = (dt["feats"].astype(np.float32), dt["y"].astype(np.float32))
    D = {}
    for nm, fn in CASES:
        F, Y = REF if fn is None else fn()
        S = batch_peak(F, Y, rx, fs)
        S["feat_rms"] = np.sqrt(np.mean(F.astype(np.float64) ** 2, axis=(1, 2)))
        p, ep = pred((F - mean) / std)
        yn = Y / SCALE
        S["err"] = np.sqrt(((p - yn) ** 2).sum(1)) * SCALE
        S["epi"] = np.sqrt(np.trace(ep, axis1=1, axis2=2)) * SCALE
        D[nm] = {k: np.asarray(v, dtype=np.float64) for k, v in S.items()}
        print("  수집 %-16s n=%d" % (nm, len(F)))
    np.savez_compressed(CACHE, **{k: np.array(v, dtype=object) for k, v in D.items()})
    print("  [saved cache] %s" % CACHE)
    return D


def design(d):
    return np.column_stack([np.ones(len(d["epi"]))] +
                           [np.log(np.maximum(d[f], 1e-12)) for f in FEATS])


def main():
    D = collect()
    ref = D["ref (학습분포)"]
    Xr, yr = design(ref), np.log(ref["epi"])

    # 2-fold 교차적합으로 in-sample 낙관 배제
    n = len(yr)
    idx = np.random.default_rng(0).permutation(n)
    h = n // 2
    r2s = []
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        beta, *_ = np.linalg.lstsq(Xr[a], yr[a], rcond=None)
        pr = Xr[b] @ beta
        r2s.append(1.0 - np.var(yr[b] - pr) / np.var(yr[b]))
    beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    r2_in = 1.0 - np.var(yr - Xr @ beta) / np.var(yr)
    print("\n" + BAR)
    print("학습분포에서 적합한 매핑  log(epi) ~ featRMS, pk/far, 반치폭 (전부 log)")
    print(BAR)
    print("  절편 %+.4f   b(featRMS) %+.4f   b(pk/far) %+.4f   b(반치폭) %+.4f"
          % (beta[0], beta[1], beta[2], beta[3]))
    print("  R2 in-sample %.4f   교차적합 %.4f / %.4f" % (r2_in, r2s[0], r2s[1]))

    print("\n" + BAR)
    print("그 매핑을 이동 조건에 그대로 적용")
    print(BAR)
    print("  %-16s %11s %11s %10s %11s %11s"
          % ("조건", "예측 epi", "실제 epi", "실제/예측", "log잔차", "sp(예측,실제)"))
    R = {}
    for nm, _ in CASES:
        d = D[nm]
        pr = np.exp(design(d) @ beta)
        ratio = float(np.median(d["epi"]) / np.median(pr))
        resid = float(np.median(np.log(d["epi"]) - np.log(pr)))
        rho = float(sps.spearmanr(pr, d["epi"]).statistic)
        R[nm] = dict(pred=float(np.median(pr)), actual=float(np.median(d["epi"])),
                     ratio=ratio, log_resid=resid, rho_pred_actual=rho,
                     err=float(np.median(d["err"])))
        print("  %-16s %11.1f %11.1f %10.3f %+11.3f %11.3f"
              % (nm, R[nm]["pred"], R[nm]["actual"], ratio, resid, rho))

    print("\n" + BAR)
    print("판독")
    print(BAR)
    print("  실제/예측 < 1  =>  입력 품질이 함의하는 것보다 **덜** 불확실하다고 말함 (과신)")
    order = sorted([k for k, _ in CASES], key=lambda k: R[k]["ratio"])
    for k in order:
        tag = "과신" if R[k]["ratio"] < 0.8 else ("과대경보" if R[k]["ratio"] > 1.25 else "매핑 유지")
        print("  %-16s 실제/예측 %6.3f   오차 %8.1f m   %s" % (k, R[k]["ratio"], R[k]["err"], tag))

    out = dict(_note="학습분포 입력품질->epistemic 매핑의 분포 밖 이전성.",
               features=list(FEATS), beta=beta.tolist(),
               r2_in_sample=float(r2_in), r2_cv=[float(x) for x in r2s], result=R)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
