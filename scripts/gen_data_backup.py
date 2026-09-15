"""
gen_data.py — PHASE 1/2b: build the dataset with the ORIGINAL generator, then
compute GCC-PHAT features and write gcc_{split}.npz + meta_{split}.json.

Data generation is untouched (seed=20260628 in GenConfig -> byte-reproducible).
GCC-PHAT is the fixed preprocessing (main pipeline). Nothing here changed for
the NIW/elliptical upgrade — the UQ layer sits entirely downstream.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfgeo import dataset as ds
from rfgeo.gcc import gcc_phat_features, n_pairs


def build(split_name, split, out, max_lag):
    X, y, metas = split
    feats = np.stack([gcc_phat_features(X[i], max_lag=max_lag) for i in range(len(X))])
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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = ds.GenConfig(channel_mode=args.channel_mode,
                       tdl_profile=args.tdl_profile,
                       tdl_delay_spread_ns=args.tdl_delay_spread_ns)
    from rfgeo.waveform_std import register_std_dsss # IQ[2] 신호 파형 단계
    register_std_dsss("gold") # IQ[2] 신호 파형 단계
    print(f"generating dataset (seed={cfg.seed}, channel_mode={cfg.channel_mode}) ...",
          flush=True)
    data = ds.generate_dataset(cfg)
    for nm in ["train", "calib", "test"]:
        build(nm, data[nm], args.out, args.max_lag)
    np.savez(os.path.join(args.out, "rx_layout.npz"),
             rx_pos=data["rx_pos"], rx_vel=data["rx_vel"])
    json.dump(data["config"], open(os.path.join(args.out, "gen_config.json"), "w"), indent=2)
    print("n_pairs =", n_pairs(cfg.n_receivers))
    print("DONE")


if __name__ == "__main__":
    main()
