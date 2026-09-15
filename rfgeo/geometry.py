"""
geometry.py — Geolocation geometry: TDOA / FDOA / AOA embedded into IQ.

This is the heart of "geolocation, not just modulation classification": we take
a single emitter waveform and produce the IQ that *each receiver* would observe,
with the inter-receiver delay (TDOA), Doppler difference (FDOA), and array phase
(AOA) physically imprinted on the samples — not merely attached as labels.

Fixed-transmitter scenario (PHASE 1)
------------------------------------
  - Emitter at unknown ground position p_tx (the regression target).
  - A set of receivers at known positions {p_rx}. For PHASE 1 these are static
    sensors with optional small velocities (so FDOA is exercised but the
    scenario stays "fixed transmitter"). Satellite motion comes in PHASE C/D.

Quantities
----------
  range r_i        = || p_rx_i - p_tx ||
  TOA  tau_i       = r_i / c
  TDOA_{i,ref}     = tau_i - tau_ref                         (seconds)
  radial vel v_i   = -(v_rx_i · (p_rx_i - p_tx)) / r_i       (closing +)
  Doppler  f_d,i   = f0 * v_i / c                            (Hz)
  FDOA_{i,ref}     = f_d,i - f_d,ref
  AOA              = azimuth/elevation of p_tx seen from receiver (for arrays)

We embed TDOA by fractional-sample delay (frequency-domain phase ramp) and FDOA
by a complex carrier multiply. This keeps phase intact (raw-IQ first design).

References
----------
[Ho & Chan 1997] TDOA geolocation geometry & CRLB.
[Stein 1981]     Joint TDOA/FDOA (cross-ambiguity) processing.
"""

from __future__ import annotations
import numpy as np

C_LIGHT = 299_792_458.0


def fractional_delay(iq: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Apply a (possibly fractional) sample delay via frequency-domain phase ramp.
    Positive delay shifts the signal later in time. Phase-preserving.
    """
    n = iq.size
    f = np.fft.fftfreq(n)
    ramp = np.exp(-2j * np.pi * f * delay_samples)
    return np.fft.ifft(np.fft.fft(iq) * ramp)


def apply_doppler(iq: np.ndarray, fdoa_hz: float, fs: float) -> np.ndarray:
    """Apply a Doppler/FDOA shift (Hz) as a complex carrier multiply."""
    n = iq.size
    t = np.arange(n) / fs
    return iq * np.exp(2j * np.pi * fdoa_hz * t)


def ranges_and_velocities(p_tx: np.ndarray, rx_pos: np.ndarray,
                          rx_vel: np.ndarray):
    """
    Compute per-receiver range (m) and radial velocity (m/s, closing positive).

    p_tx   : (3,) emitter position
    rx_pos : (M,3) receiver positions
    rx_vel : (M,3) receiver velocities
    """
    d = rx_pos - p_tx[None, :]                 # (M,3) vectors rx<-tx
    r = np.linalg.norm(d, axis=1)              # (M,)
    los = d / np.clip(r[:, None], 1e-9, None)  # unit LOS rx<-tx
    # radial velocity of receiver toward emitter (closing positive):
    v_rad = -np.sum(rx_vel * los, axis=1)
    return r, v_rad


def compute_tdoa_fdoa(p_tx: np.ndarray, rx_pos: np.ndarray, rx_vel: np.ndarray,
                      f0_hz: float, ref: int = 0):
    """
    Return TDOA (s) and FDOA (Hz) of every receiver relative to `ref`.
    """
    r, v_rad = ranges_and_velocities(p_tx, rx_pos, rx_vel)
    toa = r / C_LIGHT
    fd = f0_hz * v_rad / C_LIGHT
    tdoa = toa - toa[ref]
    fdoa = fd - fd[ref]
    return tdoa, fdoa, r, v_rad


def aoa_az_el(p_tx: np.ndarray, rx_pos: np.ndarray):
    """
    Azimuth/elevation (rad) of the emitter as seen from each receiver, in a
    local ENU-like frame aligned with the global axes (x=E, y=N, z=Up).
    Returned for completeness/AOA ablation; PHASE 1 default pipeline is TDOA/FDOA.
    """
    d = p_tx[None, :] - rx_pos
    az = np.arctan2(d[:, 0], d[:, 1])                       # from North toward East
    el = np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1))
    return az, el


def render_receiver_iq(src_iq: np.ndarray, tdoa_s: float, fdoa_hz: float,
                       fs: float) -> np.ndarray:
    """
    Produce the IQ seen at one receiver from the source waveform by imprinting
    its TDOA (as a fractional-sample delay) and FDOA (as a Doppler multiply).
    """
    delay_samples = tdoa_s * fs
    delayed = fractional_delay(src_iq, delay_samples)
    return apply_doppler(delayed, fdoa_hz, fs)
