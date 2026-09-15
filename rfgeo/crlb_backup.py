"""
crlb.py — Cramér-Rao Lower Bound utilities for TDOA/FDOA geolocation.

This module is the *citation root* for measurement-error injection. We never
inject arbitrary noise; every measurement error variance is derived from the
CRLB of the corresponding estimator. This gives reviewers a principled
ground-truth reference against which EDL aleatoric uncertainty can later be
validated.

Key references
--------------
[Stein 1981]  S. Stein, "Algorithms for ambiguity function processing,"
              IEEE Trans. ASSP, 1981. (TDOA/FDOA CRLB via BT product.)
[Ho & Chan 1997] K.C. Ho, Y.T. Chan, "Solution and performance analysis of
              geolocation by TDOA," IEEE Trans. AES, 1997.
[Knapp & Carter 1976] "The generalized correlation method for estimation of
              time delay," IEEE Trans. ASSP, 1976. (GCC / TDOA variance.)

Definitions
-----------
RMS bandwidth  Brms = sqrt( ∫ f^2 |S(f)|^2 df / ∫ |S(f)|^2 df )
RMS duration   Trms = sqrt( ∫ t^2 |s(t)|^2 dt / ∫ |s(t)|^2 dt )

For a single coherent integration with effective input SNR `gamma_eff`
(post-integration, dimensionless, *not* dB) the standard results are:

  var(tau_hat)   >= 1 / ( (2π)^2 * Brms^2 * gamma_eff )        [time delay]
  var(nu_hat)    >= 1 / ( (2π)^2 * Trms^2 * gamma_eff )        [Doppler]

These are the classic Stein/Knapp-Carter forms. We expose them directly so the
data generator can set per-pair measurement σ from physics, not by hand.
"""

from __future__ import annotations
import numpy as np

C_LIGHT = 299_792_458.0  # m/s, exact (SI definition)


def db_to_linear(snr_db: float | np.ndarray) -> float | np.ndarray:
    """Convert SNR in dB to linear ratio."""
    return 10.0 ** (np.asarray(snr_db) / 10.0)


def rms_bandwidth(iq: np.ndarray, fs: float) -> float:
    """
    RMS (root-mean-square) bandwidth of a complex baseband signal, in Hz.

    Brms = sqrt( ∫ f^2 |S(f)|^2 df / ∫ |S(f)|^2 df )

    Computed about the spectral centroid so a frequency offset does not inflate
    the value. This is the bandwidth that actually drives TDOA accuracy
    (wider effective bandwidth -> sharper correlation peak -> better delay est).
    """
    iq = np.asarray(iq)
    n = iq.size
    S = np.fft.fftshift(np.fft.fft(iq))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    psd = np.abs(S) ** 2
    total = np.sum(psd)
    if total <= 0:
        return 0.0
    f_centroid = np.sum(f * psd) / total
    brms = np.sqrt(np.sum((f - f_centroid) ** 2 * psd) / total)
    return float(brms)


def rms_duration(iq: np.ndarray, fs: float) -> float:
    """
    RMS duration of a complex baseband signal, in seconds.

    Trms = sqrt( ∫ t^2 |s(t)|^2 dt / ∫ |s(t)|^2 dt )

    Drives FDOA (Doppler) accuracy: longer coherent observation -> finer
    Doppler resolution.
    """
    iq = np.asarray(iq)
    n = iq.size
    t = np.arange(n) / fs
    p = np.abs(iq) ** 2
    total = np.sum(p)
    if total <= 0:
        return 0.0
    t_centroid = np.sum(t * p) / total
    trms = np.sqrt(np.sum((t - t_centroid) ** 2 * p) / total)
    return float(trms)


def effective_snr(snr_db: float, n_samples: int) -> float:
    """
    Post-integration (effective) SNR, linear.

    Coherent integration of N samples raises the effective SNR by a factor N
    relative to per-sample input SNR. gamma_eff = N * gamma_in.
    This is the `gamma_eff` that enters the CRLB expressions.
    """
    return float(db_to_linear(snr_db) * n_samples)


def tdoa_crlb_seconds(brms_hz: float, gamma_eff: float) -> float:
    """
    CRLB standard deviation of a single TDOA estimate, in seconds.

    sigma_tau = 1 / ( 2π * Brms * sqrt(gamma_eff) )

    Stein (1981) / Knapp-Carter (1976) single-pair form.
    """
    if brms_hz <= 0 or gamma_eff <= 0:
        return np.inf
    return 1.0 / (2.0 * np.pi * brms_hz * np.sqrt(gamma_eff))


def fdoa_crlb_hz(trms_s: float, gamma_eff: float) -> float:
    """
    CRLB standard deviation of a single FDOA estimate, in Hz.

    sigma_nu = 1 / ( 2π * Trms * sqrt(gamma_eff) )
    """
    if trms_s <= 0 or gamma_eff <= 0:
        return np.inf
    return 1.0 / (2.0 * np.pi * trms_s * np.sqrt(gamma_eff))


def tdoa_crlb_meters(brms_hz: float, gamma_eff: float) -> float:
    """TDOA CRLB expressed as a range-difference std (meters): c * sigma_tau."""
    return C_LIGHT * tdoa_crlb_seconds(brms_hz, gamma_eff)


def summarize_crlb(iq: np.ndarray, fs: float, snr_db: float,
                   carrier_hz: float | None = None) -> dict:
    """
    Convenience: compute the full CRLB picture for one waveform realization.

    Returns Brms, Trms, gamma_eff, and the TDOA/FDOA CRLB stds. The FDOA-to-
    velocity conversion needs the carrier; if not given, only Hz is returned.
    """
    n = iq.size
    brms = rms_bandwidth(iq, fs)
    trms = rms_duration(iq, fs)
    g = effective_snr(snr_db, n)
    out = {
        "brms_hz": brms,
        "trms_s": trms,
        "gamma_eff": g,
        "sigma_tdoa_s": tdoa_crlb_seconds(brms, g),
        "sigma_tdoa_m": tdoa_crlb_meters(brms, g),
        "sigma_fdoa_hz": fdoa_crlb_hz(trms, g),
    }
    if carrier_hz is not None and carrier_hz > 0:
        # FDOA (Hz) -> radial-velocity-difference std (m/s): nu = f0 * v / c
        out["sigma_vrad_mps"] = out["sigma_fdoa_hz"] * C_LIGHT / carrier_hz
    return out
