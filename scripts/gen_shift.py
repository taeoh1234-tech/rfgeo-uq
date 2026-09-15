"""
gen_shift.py — PHASE 6: generate out-of-distribution (OOD) test sets.

Training distribution (PHASE 1):
  SNR in [-10, +20] dB, Rician K in [3,10] dB or Rayleigh, multipath 3 taps,
  max delay spread 4 samples.

Shift axes (each with graded severity 0..K, 0 = in-distribution):
  A. delay_spread : heavier multipath -> more taps, larger max delay.
                    (channel STATISTICS the model never saw)
  B. snr_ood      : push SNR below training floor (-20..-10 dB).
  C. kfactor_ood  : extreme NLOS (very low K), stronger diffuse scattering.
  D. cfo_ood      : receiver LO carrier-frequency offset (ppm). TDOA-only
                    pipeline is largely robust (PHAT), tested as stress axis.
  E. sco_ood      : receiver sampling-clock offset (ppm). Scales the time
                    axis -> directly perturbs TDOA. Independent of CFO.

For each shift level we build GCC-PHAT features (same pipeline) so the trained
DER model can be evaluated directly. Emitter geometry stays in-distribution so
we isolate the CHANNEL shift effect on epistemic uncertainty.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rfgeo.dataset import GenConfig, _place_receivers, _sample_emitter
from rfgeo import waveform as wf, geometry as geo, channel as ch, crlb
from rfgeo.gcc import gcc_phat_features
from rfgeo.rank1 import fsgcc_rank_deviation_stack, N_BANDS
from rfgeo.impairments import apply_cfo, apply_sco, draw_impairments, HW_SHIFTS


def gen_sample_shift(cfg, rng, rx_pos, rx_vel, shift, rank_bands=N_BANDS):
    """One sample with channel/SNR shift applied per `shift` dict."""
    p_tx = _sample_emitter(cfg, rng)
    kind = rng.choice(cfg.waveforms)
    # SNR: shifted range if given, else training range
    snr_lo, snr_hi = shift.get("snr_range", cfg.snr_db_range)
    snr_db = rng.uniform(snr_lo, snr_hi)
    fading = shift.get("fading", str(rng.choice(cfg.fading_choices)))
    k_db = shift.get("k_db", rng.uniform(*cfg.k_db_range))
    n_taps = shift.get("n_taps", 3)
    max_delay = shift.get("max_delay_samples", 4)
    cfo_ppm = shift.get("cfo_ppm", 0.0)      # 하드웨어 손상 OOD 축
    sco_ppm = shift.get("sco_ppm", 0.0)
    # 이 shift 가 '채널 통계 자체를 바꾸는 축'인가(delay_spread / kfactor).
    # 그런 축은 apply_multipath 로 채널을 흔드는 게 목적. 그 외 축(snr/cfo/sco)은
    # 기준선(dataset.py)과 동일하게 apply_tdl(TDL-D)을 써야 _0 이 in-dist 와 일치.
    channel_is_the_axis = ("n_taps" in shift or "max_delay_samples" in shift
                           or "fading" in shift or "k_db" in shift)
    use_tdl = (cfg.channel_mode == "tdl") and not channel_is_the_axis

    src = wf.generate_waveform(str(kind), cfg.n_samples, cfg.fs, rng)
    tdoa, fdoa, ranges, vrad = geo.compute_tdoa_fdoa(
        p_tx, rx_pos, rx_vel, cfg.carrier_hz, ref=0)
    M = cfg.n_receivers
    cfo_hz, sco = draw_impairments(M, rng, cfo_ppm, sco_ppm, cfg.carrier_hz)
    X = np.empty((M, 2, cfg.n_samples), dtype=np.float32)
    for i in range(M):
        rx = geo.render_receiver_iq(src, tdoa[i], fdoa[i], cfg.fs)
        # 수신기 프론트엔드 손상(OOD): SCO(시간축) -> CFO(주파수축) 순
        if sco_ppm > 0:
            rx = apply_sco(rx, sco[i], cfg.fs)
        if cfo_ppm > 0:
            rx = apply_cfo(rx, cfo_hz[i], cfg.fs)
        if use_tdl:
            rx = ch.apply_tdl(rx, rng, profile=cfg.tdl_profile,
                              delay_spread_ns=cfg.tdl_delay_spread_ns, fs=cfg.fs)
        else:
            rx = ch.apply_multipath(rx, rng, fading=str(fading), k_db=k_db,
                                    n_taps=n_taps, max_delay_samples=max_delay)
        rx = ch.add_awgn(rx, snr_db, rng)
        X[i, 0] = rx.real.astype(np.float32); X[i, 1] = rx.imag.astype(np.float32)
    feats = gcc_phat_features(X, max_lag=128)
    # Model-independent corruption statistic, computed from the SAME IQ. This is a
    # diagnostic read-out only — it never enters the model input (CLAUDE.md §13 E14).
    rank_dev = fsgcc_rank_deviation_stack(X, n_bands=rank_bands)
    return feats, p_tx[:2].astype(np.float32), snr_db, rank_dev


def build_split(cfg, n, rng, rx_pos, rx_vel, shift, rank_bands=N_BANDS):
    F, Y, S, R = [], [], [], []
    for _ in range(n):
        f, y, s, rd = gen_sample_shift(cfg, rng, rx_pos, rx_vel, shift, rank_bands)
        F.append(f); Y.append(y); S.append(s); R.append(rd)
    return np.stack(F), np.stack(Y), np.array(S), np.array(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output_phase6")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--channel-mode", default="tdl", choices=["random","tdl"],
                    help="기준선 데이터와 반드시 일치시킬 것(기본 tdl)")
    ap.add_argument("--tdl-profile", default="TDL-D")
    ap.add_argument("--tdl-delay-spread-ns", type=float, default=100.0)
    ap.add_argument("--std-pn", choices=["gold","mseq"], default="gold",
                    help="기준선과 일치. 기본 gold")
    ap.add_argument("--rank-bands", type=int, default=N_BANDS,
                    help="FS-GCC rank-1 이탈도의 대역수 L (진단 지표 전용, 모델 입력 불변). "
                         "기본 32 = 사전등록값(RANK1_OOD_PREREG.md)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = GenConfig(channel_mode=args.channel_mode,
                    tdl_profile=args.tdl_profile,
                    tdl_delay_spread_ns=args.tdl_delay_spread_ns)
    if args.std_pn is not None:
        from rfgeo.waveform_std import register_std_dsss
        register_std_dsss(args.std_pn)
        print(f"[std-pn] DSSS -> standard {args.std_pn}", flush=True)
    print(f"[gen_shift] channel_mode={cfg.channel_mode} profile={args.tdl_profile}", flush=True)
    rng = np.random.default_rng(args.seed)
    rx_pos, rx_vel = _place_receivers(cfg, rng)

    # graded shift definitions
    shifts = {
        # A. delay spread severity (in-dist -> severe multipath)
        "delay_0": {"n_taps":3, "max_delay_samples":4},      # in-distribution
        "delay_1": {"n_taps":5, "max_delay_samples":8},
        "delay_2": {"n_taps":8, "max_delay_samples":16},
        "delay_3": {"n_taps":12,"max_delay_samples":32},
        # B. SNR below training floor
        "snr_0":   {"snr_range":(-10,20)},                    # in-distribution
        "snr_1":   {"snr_range":(-15,-10)},
        "snr_2":   {"snr_range":(-20,-15)},
        # C. extreme NLOS (K far below training 3 dB)
        "kf_0":    {"fading":"rician","k_db":6.0},            # in-distribution
        "kf_1":    {"fading":"rician","k_db":0.0},
        "kf_2":    {"fading":"rician","k_db":-6.0},
        "kf_3":    {"fading":"rayleigh"},
        # D/E. 하드웨어 손상 (CFO/SCO) — 독립 OOD 축
        **HW_SHIFTS,
    }
    manifest = {}
    for name, sh in shifts.items():
        F, Y, S, R = build_split(cfg, args.n, rng, rx_pos, rx_vel, sh, args.rank_bands)
        np.savez_compressed(os.path.join(args.out, f"shift_{name}.npz"),
                            feats=F, y=Y, snr=S, rank_dev=R)
        manifest[name] = sh
        print(f"  {name:9s} feats {F.shape} snr[{S.min():.0f},{S.max():.0f}] "
              f"rank_dev median {np.median(R):.3f}", flush=True)
    json.dump(manifest, open(os.path.join(args.out,"shift_manifest.json"),"w"), indent=2)
    print("[phase6] shift sets generated", flush=True)

if __name__ == "__main__":
    main()
