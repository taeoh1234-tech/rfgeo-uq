"""
gen_data.py — PHASE 1/2b: build the dataset with the ORIGINAL generator, then
compute GCC-PHAT features and write gcc_{split}.npz + meta_{split}.json.

Data generation seed=20260628 (byte-reproducible for a FIXED config).
sps is unified to 2 in waveform.py; --std-pn gold swaps DSSS PN to Gold code
(verified UQ-input-invariant: Brms/GCC-peak |d|<0.1 vs random PN).
GCC-PHAT is the fixed preprocessing (main pipeline). Nothing here changed for
the NIW/elliptical upgrade — the UQ layer sits entirely downstream.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfgeo import dataset as ds
from rfgeo import waveform as wf_mod
from rfgeo.gcc import gcc_phat_features, n_pairs


def build(split_name, split, out, max_lag, beta=1.0):
    X, y, metas = split
    feats = np.stack([gcc_phat_features(X[i], max_lag=max_lag, beta=beta) for i in range(len(X))])
    np.savez_compressed(os.path.join(out, f"gcc_{split_name}.npz"),
                        feats=feats.astype(np.float32), y=y.astype(np.float32))
    json.dump(metas, open(os.path.join(out, f"meta_{split_name}.json"), "w"),
              default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"  {split_name}: feats={feats.shape} y={y.shape}", flush=True)
    return feats.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--max-lag", type=int, default=128)
    ap.add_argument("--channel-mode", default="random", choices=["random", "tdl"],
                    help="'random' (legacy, reproduces original data) | "
                         "'tdl' (3GPP TR 38.901 TDL profile, fractional delay)")
    ap.add_argument("--tdl-profile", default="TDL-D")
    ap.add_argument("--tdl-delay-spread-ns", type=float, default=100.0)
    ap.add_argument("--std-pn", choices=["gold", "mseq"], default=None,
                    help="DSSS PN 을 표준코드로 교체(gold 권장). 미지정시 원본 무작위 PN.")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="GCC whitening exponent (beta-PHAT ablation): "
                         "1.0=PHAT (default, byte-reproducible), 0.0=pure CC, "
                         "0<beta<1=partial whitening. Ablation only; pipeline default is 1.0.")
    ap.add_argument("--waveforms", default=None,
                    help="쉼표 구분 방사원 파형 목록. 미지정이면 GenConfig 기본값"
                         "(random,dsss,fhss,burst) 그대로 → 기존 실행과 바이트 동일. "
                         "재머 arm: jam_cw,jam_am,jam_chirp,jam_pulse,jam_nb "
                         "(JAMMER_ARM_PREREG.md). ⚠️ 두 계열을 섞지 말 것 — "
                         "CLAUDE.md §5.1(sps 통일)이 기본 4종의 Brms 교란 제거를 전제한다.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg_kw = {}
    if args.waveforms is not None:
        wfs = tuple(w.strip() for w in args.waveforms.split(",") if w.strip())
        unknown = [w for w in wfs if w not in wf_mod.WAVEFORM_GENERATORS]
        if unknown:
            raise SystemExit(f"unknown waveform(s): {unknown}; "
                             f"choose from {list(wf_mod.WAVEFORM_GENERATORS)}")
        base = set(wf_mod.BASELINE_WAVEFORMS) & set(wfs)
        jam = set(wf_mod.JAMMER_WAVEFORMS) & set(wfs)
        if base and jam:
            raise SystemExit(
                "baseline 파형과 재머 파형을 한 데이터셋에 섞을 수 없다 "
                "(CLAUDE.md §5.1: sps 통일은 기본 4종의 Brms 교란 제거가 목적이고, "
                "재머 arm 은 Brms 차이가 독립변수다). 별도 arm 으로 생성할 것.")
        cfg_kw["waveforms"] = wfs

    cfg = ds.GenConfig(channel_mode=args.channel_mode,
                       tdl_profile=args.tdl_profile,
                       tdl_delay_spread_ns=args.tdl_delay_spread_ns,
                       **cfg_kw)
    if args.std_pn is not None:
        from rfgeo.waveform_std import register_std_dsss
        register_std_dsss(args.std_pn)   # DSSS -> 표준 PN(gold). sps 통일은 waveform.py 에 이미 반영.
        print(f"[std-pn] DSSS replaced with standard {args.std_pn} code", flush=True)
    print(f"generating dataset (seed={cfg.seed}, channel_mode={cfg.channel_mode}, "
          f"std_pn={args.std_pn}, beta={args.beta}, waveforms={cfg.waveforms}) ...",
          flush=True)
    data = ds.generate_dataset(cfg)
    for nm in ["train", "calib", "test"]:
        build(nm, data[nm], args.out, args.max_lag, beta=args.beta)
    np.savez(os.path.join(args.out, "rx_layout.npz"),
             rx_pos=data["rx_pos"], rx_vel=data["rx_vel"])
    json.dump(data["config"], open(os.path.join(args.out, "gen_config.json"), "w"), indent=2)
    print("n_pairs =", n_pairs(cfg.n_receivers))
    print("DONE")


if __name__ == "__main__":
    main()
