"""
channel.py — Propagation channel: multipath fading + path loss + noise.

Two layers:
  1. Small-scale multipath fading: Rician (LOS present) or Rayleigh (NLOS).
  2. Large-scale path loss: ITU-R free-space basic transmission loss, with an
     optional excess-loss term for NLOS scenarios.

Then AWGN is added to realize the per-sample target SNR (mixed-SNR dataset).

References
----------
[ITU-R P.525]  "Calculation of free-space attenuation."
               Lfs(dB) = 32.45 + 20·log10(f_MHz) + 20·log10(d_km)
[ITU-R P.1057] Statistical fading distributions (Rician/Rayleigh).
[ITU-R P.681 / P.682] Land/aeronautical mobile-satellite propagation
               (Rician K-factor ranges used for LEO/GEO links).

We keep path loss as an *informational* metadata field by default: the SNR
label is what actually scales the noise. Path loss is recorded so that the
GEO-extension story ("near-zero FDOA + severe path loss -> low SNR") is grounded
in P.525 numbers rather than asserted.
"""

from __future__ import annotations
import numpy as np

C_LIGHT = 299_792_458.0


def freespace_path_loss_db(freq_hz: float, dist_m: float) -> float:
    """ITU-R P.525 free-space basic transmission loss in dB."""
    f_mhz = freq_hz / 1e6
    d_km = np.maximum(np.asarray(dist_m, dtype=float), 1.0) / 1e3
    return 32.45 + 20.0 * np.log10(f_mhz) + 20.0 * np.log10(d_km)


def apply_multipath(iq: np.ndarray, rng: np.random.Generator,
                    fading: str = "rician", k_db: float = 6.0,
                    n_taps: int = 3, max_delay_samples: int = 4,
                    fs: float | None = None) -> np.ndarray:
    """
    Apply a small-scale multipath fading channel.

    fading = 'rician' : dominant LOS component + diffuse taps (K-factor in dB).
    fading = 'rayleigh': no LOS, all taps complex Gaussian (K -> -inf).

    The multi-tap delay spread is what creates frequency-selective fading and
    is the physical reason the Gaussian-noise assumption of classical
    Kalman-family estimators is violated — a core motivation of the research.
    """
    if fading == "rayleigh":
        k_lin = 0.0
    else:
        k_lin = 10.0 ** (k_db / 10.0)

    # LOS (specular) power vs diffuse power split from K-factor.
    p_los = k_lin / (k_lin + 1.0)
    p_dif = 1.0 / (k_lin + 1.0)

    taps = np.zeros(max_delay_samples + 1, dtype=complex)
    # tap 0 = LOS + a diffuse contribution
    taps[0] = np.sqrt(p_los) + np.sqrt(p_dif / max(n_taps, 1)) * (
        rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    # remaining diffuse taps at random delays
    delays = rng.choice(np.arange(1, max_delay_samples + 1),
                        size=min(n_taps - 1, max_delay_samples),
                        replace=False) if n_taps > 1 else []
    for d in np.atleast_1d(delays):
        taps[d] = np.sqrt(p_dif / max(n_taps, 1)) * (
            rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)

    # normalize channel to unit energy (preserve SNR bookkeeping)
    e = np.sum(np.abs(taps) ** 2)
    if e > 0:
        taps /= np.sqrt(e)
    out = np.convolve(iq, taps, mode="same")
    return out


# ---------------------------------------------------------------------------
# 3GPP TR 38.901 v19.2.0 (2026-02) Tapped Delay Line profiles.
# Values transcribed verbatim from Table 7.7.2-4 (TDL-D) / 7.7.2-1..5.
# TDL-D/E are LOS profiles: tap 0 is a specular (Ricean) LOS path; the
# remaining taps are Rayleigh diffuse. Delays are *normalized* and get scaled
# by a desired RMS delay spread DS (ns) per Clause 7.7.3, eq. (7.7-1).
# ---------------------------------------------------------------------------

# (normalized_delay, power_dB, is_LOS)
_TDL_D = {
    "los_k_db": 13.3,   # NOTE in Table 7.7.2-4: first tap Ricean, K1 = 13.3 dB
    "taps": [
        (0.000, 0.0, True),    # LOS specular path (0 dB mean power)
        (0.000, -13.5, False),
        (0.035, -18.8, False),
        (0.612, -21.0, False),
        (1.363, -22.8, False),
        (1.405, -17.9, False),
        (1.804, -20.1, False),
        (2.596, -21.9, False),
        (1.775, -22.9, False),
        (4.042, -27.8, False),
        (7.937, -23.6, False),
        (9.424, -24.8, False),
        (9.708, -30.0, False),
        (12.525, -27.7, False),
    ],
}

_TDL_PROFILES = {"TDL-D": _TDL_D}


def _fractional_delay_channel(iq: np.ndarray, taps_norm, powers_db, is_los,
                              los_k_db: float, ds_ns: float, fs: float,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Apply a 3GPP TDL channel using frequency-domain fractional delays.

    Each tap has a normalized delay scaled to seconds by DS (Clause 7.7.3),
    then to a (generally non-integer) sample delay applied as a linear phase
    ramp in the frequency domain — so sub-sample delays are preserved rather
    than snapped to an integer grid (which would corrupt TDOA truth).

    The LOS tap is specular (fixed unit phasor, scaled by sqrt(K/(K+1)));
    diffuse taps are complex-Gaussian, jointly scaled by sqrt(1/(K+1)) so that
    the LOS/diffuse power split matches the profile K-factor. The whole channel
    is normalized to unit energy to preserve SNR bookkeeping.
    """
    n = iq.shape[0]
    freqs = np.fft.fftfreq(n)                       # cycles/sample
    IQ = np.fft.fft(iq)
    out_f = np.zeros(n, dtype=complex)

    k_lin = 10.0 ** (los_k_db / 10.0)
    los_scale = np.sqrt(k_lin / (k_lin + 1.0))
    dif_scale = np.sqrt(1.0 / (k_lin + 1.0))

    # linear tap gains from dB powers
    lin_pow = np.array([10.0 ** (p / 10.0) for p in powers_db])
    dif_mask = ~np.asarray(is_los, dtype=bool)
    dif_norm = np.sqrt(lin_pow[dif_mask].sum()) if dif_mask.any() else 1.0

    for tau_n, plin, los in zip(taps_norm, lin_pow, is_los):
        tau_samp = (tau_n * ds_ns * 1e-9) * fs      # fractional sample delay
        phase = np.exp(-2j * np.pi * freqs * tau_samp)
        if los:
            gain = los_scale * np.sqrt(plin)         # specular, fixed phase
        else:
            g = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
            gain = dif_scale * (np.sqrt(plin) / dif_norm) * g
        out_f += gain * phase * IQ

    out = np.fft.ifft(out_f)
    # unit-energy normalization on the effective channel (SNR bookkeeping)
    e_in = np.sum(np.abs(iq) ** 2)
    e_out = np.sum(np.abs(out) ** 2)
    if e_out > 0 and e_in > 0:
        out *= np.sqrt(e_in / e_out)
    return out


def apply_tdl(iq: np.ndarray, rng: np.random.Generator,
              profile: str = "TDL-D", delay_spread_ns: float = 100.0,
              fs: float = 2.0e6) -> np.ndarray:
    """
    Apply a 3GPP TR 38.901 TDL channel (default TDL-D, LOS satellite-link
    nominal). Delay spread default 100 ns = the 'Nominal delay spread' scaling
    value from Table 7.7.3-1. Taps placed via fractional (sub-sample) delay.
    """
    prof = _TDL_PROFILES[profile]
    taps_norm = [t[0] for t in prof["taps"]]
    powers_db = [t[1] for t in prof["taps"]]
    is_los = [t[2] for t in prof["taps"]]
    return _fractional_delay_channel(
        iq, taps_norm, powers_db, is_los,
        prof["los_k_db"], delay_spread_ns, fs, rng)


def add_awgn(iq: np.ndarray, snr_db: float,
             rng: np.random.Generator) -> np.ndarray:
    """
    Add complex AWGN to reach a target SNR (dB), measured over active samples.

    Signal power is estimated from samples above 1% of peak power so that BURST
    (low duty cycle) waveforms get noise scaled to the *on-time* SNR, matching
    the SNR label semantics used everywhere else.
    """
    p_inst = np.abs(iq) ** 2
    thresh = 0.01 * np.max(p_inst) if np.max(p_inst) > 0 else 0.0
    active = p_inst[p_inst > thresh]
    p_sig = np.mean(active) if active.size > 0 else np.mean(p_inst)
    snr_lin = 10.0 ** (snr_db / 10.0)
    p_noise = p_sig / snr_lin if snr_lin > 0 else p_sig
    noise = np.sqrt(p_noise / 2.0) * (
        rng.standard_normal(iq.shape) + 1j * rng.standard_normal(iq.shape))
    return iq + noise
