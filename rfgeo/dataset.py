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
    # Receiver azimuth layout. "uniform" is the verified Scenario A baseline
    # (evenly spaced, 360/M deg apart) and stays bit-exact reproducible: the
    # uniform branch consumes the layout RNG exactly as before. "minsep_random"
    # draws ONE random azimuth set per dataset -- still fixed for every sample, so
    # the backbone's fixed-geometry assumption is preserved (per-sample re-drawing
    # was tested twice, with and without receiver coords, and collapses
    # prediction) -- while breaking the 90-deg symmetry that pins GDOP near-constant.
    rx_layout_mode: str = "uniform"   # "uniform" | "minsep_random"
    rx_min_sep_deg: float = 40.0      # minimum angular gap between adjacent receivers
    rx_layout_seed: int = -1          # >=0 draws the layout from this seed alone, so the
                                      # layout can be varied independently of cfg.seed
                                      # (-1 = use the dataset's own layout RNG stream)
    # channel
    channel_mode: str = "random"      # "random" (legacy) | "tdl" (3GPP TR 38.901)
    fading_choices: tuple = ("rician", "rayleigh")
    k_db_range: tuple = (3.0, 10.0)   # Rician K-factor range (ITU-R P.681-ish)
    # TDL mode (channel_mode == "tdl")
    tdl_profile: str = "TDL-D"        # LOS nominal (satellite-link) baseline
    tdl_delay_spread_ns: float = 100.0  # 'Nominal' scaling, TR 38.901 Table 7.7.3-1
    # SNR (mixed-SNR dataset)
    snr_db_range: tuple = (-10.0, 20.0)
    # dataset size / splits  (ADOPTED 2026-07: session experiments E6/E7/E8)
    n_train: int = 8000              # 4000->8000 halves the shipped region: median
                                     # ellipse area 1.118->0.428 km^2 (paired t=-7.9),
                                     # two-force reduction 67->84%, while coverage
                                     # stays exact (z@90 ~ 0) and OOD detection holds
                                     # (AUROC 0.965). 12000 was REJECTED: efficiency
                                     # gain flattens but epistemic-error Spearman
                                     # collapses 0.550->0.408. See CLAUDE.md 13 (E6).
    n_calib: int = 4000              # held out for Conformal Prediction. 1000->4000
                                     # is adopted ONLY for Mondrian: per-SNR-bin
                                     # calibration goes from ~167 to ~667 points/bin,
                                     # cutting conditional-coverage deviation 55%
                                     # (0.0318->0.0144, 5-seed). It does NOT change Q,
                                     # region area or OOD (all confounded on n_train,
                                     # not n_calib -- E8 iso arm, effect ratio 43x).
    n_test: int = 2000               # measurement precision only (binomial coverage
                                     # SE 0.0095->0.0067); never changes a true value.
    seed: int = 20260628


def _minsep_azimuths(n: int, min_sep_deg: float,
                     rng: np.random.Generator) -> np.ndarray:
    """`n` random azimuths (rad) with every adjacent gap >= min_sep_deg.

    Exact construction, no rejection sampling: the n gaps around the circle must
    sum to 360 and each be at least min_sep, so reserve n*min_sep up front, spread
    the remaining slack uniformly over the simplex (Dirichlet(1,...,1)), and add it
    back. A random global rotation removes the arbitrary starting azimuth. The
    wrap-around gap is constrained too, which plain `sort(uniform(0, 2pi))` does
    not do -- that is what lets two receivers land nearly on top of each other and
    degenerate the array to M-1 effective sensors.
    """
    slack = 360.0 - n * min_sep_deg
    if slack < 0:
        raise ValueError(
            f"rx_min_sep_deg={min_sep_deg} is infeasible for {n} receivers: "
            f"needs {n * min_sep_deg:.0f} deg of 360. Max is {360.0 / n:.1f}.")
    gaps = min_sep_deg + rng.dirichlet(np.ones(n)) * slack
    start = rng.uniform(0.0, 360.0)
    az = (start + np.concatenate([[0.0], np.cumsum(gaps)[:-1]])) % 360.0
    return np.deg2rad(np.sort(az))


def _place_receivers(cfg: GenConfig, rng: np.random.Generator):
    """Fixed circular receiver layout at a configurable height.

    Static-static Scenario A: with cfg.rx_speed_mps == 0 the velocity block below
    yields zero velocity, so FDOA = 0. The velocity code is retained (not deleted)
    so the satellite extension can re-enable receiver motion simply by raising
    cfg.rx_speed_mps, without restructuring this function.

    The layout is drawn ONCE per dataset and shared by train/calib/test, in both
    modes. Only the azimuths differ; radius, height and the velocity block are
    untouched, so "uniform" reproduces the verified baseline bit-for-bit (it does
    not touch `rng` before the velocity draw).
    """
    if cfg.rx_layout_mode == "minsep_random":
        rng_layout = (np.random.default_rng(cfg.rx_layout_seed)
                      if cfg.rx_layout_seed >= 0 else rng)
        ang = _minsep_azimuths(cfg.n_receivers, cfg.rx_min_sep_deg, rng_layout)
    elif cfg.rx_layout_mode == "uniform":
        ang = np.linspace(0, 2 * np.pi, cfg.n_receivers, endpoint=False)
    else:
        raise ValueError(f"unknown rx_layout_mode: {cfg.rx_layout_mode!r} "
                         f"(expected 'uniform' or 'minsep_random')")
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
