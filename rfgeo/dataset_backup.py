"""
dataset.py — Fixed-transmitter RF geolocation dataset generator.

Orchestrates waveform -> geometry -> channel -> noise -> CRLB metadata into a
multi-receiver IQ dataset for the PHASE 1 (fixed transmitter) scenario.

Per sample
----------
  X : (M, 2, N) real array  — M receivers, [I,Q] channels, N samples each
  y : (2,) or (3,) float    — emitter ground position (regression target)
  meta : dict               — SNR(dB) per/global, channel params, geometry,
                              TDOA/FDOA truth, CRLB stds, waveform kind

Splits
------
train / calibration / test. The calibration split is held out *physically*
(distinct emitter positions and noise realizations) because Conformal
Prediction's coverage guarantee assumes an exchangeable, untouched calib set.

Coordinate convention (PHASE 1)
-------------------------------
Local ENU meters on a flat patch. Emitter z=0 (ground). Receivers placed in a
fixed geometry around an area of interest. This is deliberately simple and
auditable; ECEF + SGP4 orbital geometry enters at PHASE C (LEO).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict

from . import waveform as wf
from . import channel as ch
from . import geometry as geo
from . import crlb


@dataclass
class GenConfig:
    # signal
    fs: float = 2.0e6                 # sample rate, Hz
    n_samples: int = 4096             # samples per receiver snapshot
    carrier_hz: float = 1.5e9         # L-band primary
    waveforms: tuple = ("random", "dsss", "fhss", "burst")
    # geometry (local ENU, meters)
    n_receivers: int = 4
    area_radius_m: float = 8_000.0    # emitter placement radius; = rx_layout_radius
                                      # so emitters span inside AND just outside the
                                      # array, giving the GDOP diversity that makes
                                      # elliptical CP efficient (radius sweep 5..15 km
                                      # + 8/9 km x10-seed study: 8 km is the point
                                      # where ellipse area-reduction is realised while
                                      # coverage stays valid and median error stays low)
    rx_layout_radius_m: float = 8_000.0
    rx_height_m: float = 0.0          # ground-plane sensors. Height is immaterial to
                                      # the UQ result: a 0/50/100/500 m x5-seed sweep
                                      # left area-reduction, coverage, pos-level
                                      # Spearman and median error unchanged within
                                      # seed std (height is a systematic, learnable
                                      # part of the geometry, not a random nuisance).
    rx_speed_mps: float = 0.0         # static-static Scenario A: FDOA = 0 by
                                      # construction (GCC-PHAT extracts TDOA only, so
                                      # receiver motion has no representable role here;
                                      # FDOA is deferred to the satellite extension).
    # channel
    channel_mode: str = "random"      # "random" (legacy) | "tdl" (3GPP TR 38.901)
    fading_choices: tuple = ("rician", "rayleigh")
    k_db_range: tuple = (3.0, 10.0)   # Rician K-factor range (ITU-R P.681-ish)
    # TDL mode (channel_mode == "tdl")
    tdl_profile: str = "TDL-D"        # LOS nominal (satellite-link) baseline
    tdl_delay_spread_ns: float = 100.0  # 'Nominal' scaling, TR 38.901 Table 7.7.3-1
    # SNR (mixed-SNR dataset)
    snr_db_range: tuple = (-10.0, 20.0)
    # dataset size / splits
    n_train: int = 4000
    n_calib: int = 1000              # held out for Conformal Prediction
    n_test: int = 1000
    seed: int = 20260628


def _place_receivers(cfg: GenConfig, rng: np.random.Generator):
    """Fixed circular receiver layout at a configurable height.

    Static-static Scenario A: with cfg.rx_speed_mps == 0 the velocity block below
    yields zero velocity, so FDOA = 0. The velocity code is retained (not deleted)
    so the satellite extension can re-enable receiver motion simply by raising
    cfg.rx_speed_mps, without restructuring this function.
    """
    ang = np.linspace(0, 2 * np.pi, cfg.n_receivers, endpoint=False)
    rx_pos = np.stack([
        cfg.rx_layout_radius_m * np.cos(ang),
        cfg.rx_layout_radius_m * np.sin(ang),
        np.full_like(ang, cfg.rx_height_m),   # ground-plane sensors (rx_height_m=0)
    ], axis=1)
    v_dir = rng.standard_normal((cfg.n_receivers, 3))
    v_dir[:, 2] *= 0.1                 # mostly horizontal motion (inert when speed=0)
    v_dir /= np.linalg.norm(v_dir, axis=1, keepdims=True)
    rx_vel = v_dir * cfg.rx_speed_mps  # == 0 in static-static Scenario A
    return rx_pos, rx_vel


def _sample_emitter(cfg: GenConfig, rng: np.random.Generator) -> np.ndarray:
    """Uniformly place emitter within a disk of radius area_radius_m, z=0."""
    r = cfg.area_radius_m * np.sqrt(rng.uniform())
    th = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(th), r * np.sin(th), 0.0])


def generate_sample(cfg: GenConfig, rng: np.random.Generator,
                    rx_pos: np.ndarray, rx_vel: np.ndarray) -> dict:
    """Generate one multi-receiver geolocation sample."""
    p_tx = _sample_emitter(cfg, rng)
    kind = rng.choice(cfg.waveforms)
    snr_db = rng.uniform(*cfg.snr_db_range)
    fading = rng.choice(cfg.fading_choices)
    k_db = rng.uniform(*cfg.k_db_range)

    # 1) source waveform (unit power, phase preserved)
    src = wf.generate_waveform(str(kind), cfg.n_samples, cfg.fs, rng)

    # 2) geometry truth
    tdoa, fdoa, ranges, vrad = geo.compute_tdoa_fdoa(
        p_tx, rx_pos, rx_vel, cfg.carrier_hz, ref=0)

    # 3) render per-receiver IQ: delay+doppler, multipath, path-loss-aware SNR
    M = cfg.n_receivers
    X = np.empty((M, 2, cfg.n_samples), dtype=np.float32)
    pl_db = ch.freespace_path_loss_db(cfg.carrier_hz, ranges)  # (M,) metadata
    for i in range(M):
        rx_iq = geo.render_receiver_iq(src, tdoa[i], fdoa[i], cfg.fs)
        if cfg.channel_mode == "tdl":
            rx_iq = ch.apply_tdl(rx_iq, rng, profile=cfg.tdl_profile,
                                 delay_spread_ns=cfg.tdl_delay_spread_ns,
                                 fs=cfg.fs)
        else:
            rx_iq = ch.apply_multipath(rx_iq, rng, fading=str(fading), k_db=k_db)
        rx_iq = ch.add_awgn(rx_iq, snr_db, rng)
        X[i, 0] = rx_iq.real.astype(np.float32)
        X[i, 1] = rx_iq.imag.astype(np.float32)

    # 4) CRLB metadata from the clean source (citation-grounded error reference)
    cr = crlb.summarize_crlb(src, cfg.fs, snr_db, cfg.carrier_hz)

    meta = {
        "waveform": str(kind),
        "snr_db": float(snr_db),
        "fading": str(fading),
        "k_db": float(k_db),
        "tdoa_s": tdoa.astype(np.float32),
        "fdoa_hz": fdoa.astype(np.float32),
        "ranges_m": ranges.astype(np.float32),
        "pathloss_db": pl_db.astype(np.float32),
        "brms_hz": cr["brms_hz"],
        "trms_s": cr["trms_s"],
        "sigma_tdoa_s": cr["sigma_tdoa_s"],
        "sigma_tdoa_m": cr["sigma_tdoa_m"],
        "sigma_fdoa_hz": cr["sigma_fdoa_hz"],
    }
    return {"X": X, "y": p_tx[:2].astype(np.float32), "meta": meta}


def generate_split(cfg: GenConfig, n: int, rng: np.random.Generator,
                   rx_pos, rx_vel):
    Xs, ys, metas = [], [], []
    for _ in range(n):
        s = generate_sample(cfg, rng, rx_pos, rx_vel)
        Xs.append(s["X"]); ys.append(s["y"]); metas.append(s["meta"])
    return np.stack(Xs), np.stack(ys), metas


def generate_dataset(cfg: GenConfig):
    """
    Generate train / calibration / test splits with *independent* RNG streams
    so the calibration set is statistically untouched by training data
    (required for valid Conformal Prediction coverage).
    """
    ss = np.random.SeedSequence(cfg.seed)
    rng_layout, rng_tr, rng_cal, rng_te = [
        np.random.default_rng(s) for s in ss.spawn(4)]

    rx_pos, rx_vel = _place_receivers(cfg, rng_layout)

    train = generate_split(cfg, cfg.n_train, rng_tr, rx_pos, rx_vel)
    calib = generate_split(cfg, cfg.n_calib, rng_cal, rx_pos, rx_vel)
    test = generate_split(cfg, cfg.n_test, rng_te, rx_pos, rx_vel)

    return {
        "config": asdict(cfg),
        "rx_pos": rx_pos, "rx_vel": rx_vel,
        "train": train, "calib": calib, "test": test,
    }
