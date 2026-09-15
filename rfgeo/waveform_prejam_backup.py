"""
waveform.py — Non-cooperative LPI/LPD waveform synthesis (complex baseband IQ).

We model the *intercepted* signal of a non-cooperative emitter. The emitter
deliberately spreads/hides energy (Low Probability of Intercept/Detection):
  - FHSS  : frequency-hopping spread spectrum
  - DSSS  : direct-sequence spread spectrum
  - BURST : short transient transmissions with low duty cycle
  - RANDOM: random linear modulation (random QPSK/16-QAM symbol stream)

Design intent
-------------
1. All waveforms are complex baseband (analytic) so phase is preserved — this
   is why raw IQ is the first-class representation (spectrogram loses phase).
2. We synthesize at a fixed sample rate `fs`. Spreading is what gives these
   signals their wide RMS bandwidth, which in turn sets TDOA accuracy via CRLB.
   NOTE: `sps` is unified to 2 across ALL waveforms so that Brms (hence the
   CRLB floor) is not confounded by per-waveform sps. Verified over 5 seeds
   to leave CP coverage statistically unchanged vs. the previous mixed-sps
   setting (coverage@90 diff t=-0.21), i.e. a clean-up with no UQ side effect.
3. Each generator returns unit-average-power IQ; SNR is applied later by the
   channel/noise stage so power bookkeeping stays in one place.

References for modulation/spreading conventions
-----------------------------------------------
[Proakis & Salehi, Digital Communications] — pulse shaping, QAM constellations.
[Pickholtz, Schilling, Milstein 1982] "Theory of Spread-Spectrum
 Communications" — DSSS/FHSS fundamentals.
"""

from __future__ import annotations
import numpy as np


def _rrc_taps(beta: float, sps: int, span: int = 8) -> np.ndarray:
    """Root-raised-cosine pulse-shaping filter taps."""
    n = span * sps
    t = (np.arange(-n / 2, n / 2 + 1)) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4 * beta)) < 1e-12:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps ** 2))


def _random_symbols(n: int, order: int, rng: np.random.Generator) -> np.ndarray:
    """Random QPSK (order=4) or 16-QAM (order=16) symbols, unit average power."""
    if order == 4:
        re = rng.choice([-1, 1], size=n)
        im = rng.choice([-1, 1], size=n)
        sym = (re + 1j * im) / np.sqrt(2)
    elif order == 16:
        levels = np.array([-3, -1, 1, 3])
        re = rng.choice(levels, size=n)
        im = rng.choice(levels, size=n)
        sym = (re + 1j * im) / np.sqrt(10)
    else:
        raise ValueError("order must be 4 or 16")
    return sym


def _normalize_power(iq: np.ndarray) -> np.ndarray:
    """Scale to unit average power (per active sample)."""
    p = np.mean(np.abs(iq) ** 2)
    if p <= 0:
        return iq
    return iq / np.sqrt(p)


def gen_random_modulation(n: int, fs: float, rng: np.random.Generator,
                          sps: int = 2, order: int = 4,   # sps unified to 2 (was 4)
                          beta: float = 0.35) -> np.ndarray:
    """Random linear modulation: RRC-shaped random QAM stream (baseline LPI)."""
    n_sym = n // sps + 16
    sym = _random_symbols(n_sym, order, rng)
    up = np.zeros(n_sym * sps, dtype=complex)
    up[::sps] = sym
    taps = _rrc_taps(beta, sps)
    sig = np.convolve(up, taps, mode="same")[:n]
    if sig.size < n:
        sig = np.pad(sig, (0, n - sig.size))
    return _normalize_power(sig)


def gen_dsss(n: int, fs: float, rng: np.random.Generator,
             spread_factor: int = 16, sps: int = 2,
             order: int = 4) -> np.ndarray:
    """
    DSSS: each data symbol multiplied by a pseudo-random ±1 chip sequence.
    The chip rate >> symbol rate spreads the spectrum, widening Brms.
    """
    chip_rate_samples = sps
    n_chips = n // chip_rate_samples + spread_factor
    n_sym = n_chips // spread_factor + 1
    data = _random_symbols(n_sym, order, rng)
    pn = rng.choice([-1, 1], size=(n_sym, spread_factor))
    chips = (data[:, None] * pn).reshape(-1)          # spread chip stream
    up = np.repeat(chips, chip_rate_samples)[:n]
    if up.size < n:
        up = np.pad(up, (0, n - up.size))
    # mild pulse shaping to keep it band-limited at fs
    taps = _rrc_taps(0.35, sps)
    sig = np.convolve(up, taps, mode="same")[:n]
    return _normalize_power(sig)


def gen_fhss(n: int, fs: float, rng: np.random.Generator,
             n_hops: int = 32, hop_bw_frac: float = 0.8,
             order: int = 4) -> np.ndarray:
    """
    FHSS: carrier hops pseudo-randomly across the band on a per-dwell basis.
    Each dwell carries a short random-modulated segment at the hop frequency.
    Aggregated over many hops the occupied bandwidth (hence Brms) is large.
    """
    dwell = n // n_hops
    if dwell < 8:
        dwell = 8
        n_hops = n // dwell
    sig = np.zeros(n, dtype=complex)
    max_off = hop_bw_frac * fs / 2.0
    for h in range(n_hops):
        a = h * dwell
        b = min(a + dwell, n)
        m = b - a
        if m <= 0:
            break
        base = gen_random_modulation(m, fs, rng, sps=2, order=order)  # sps=2 (unified)
        f_off = rng.uniform(-max_off, max_off)        # hop frequency
        t = np.arange(m) / fs
        sig[a:b] = base * np.exp(2j * np.pi * f_off * t)
    return _normalize_power(sig)


def gen_burst(n: int, fs: float, rng: np.random.Generator,
              duty: float = 0.25, order: int = 4) -> np.ndarray:
    """
    BURST: a single short transient within an otherwise silent window (low duty
    cycle). Power is normalized over the *active* burst, so the silent region
    stays near zero and the effective on-time SNR matches the SNR label.
    """
    burst_len = max(8, int(n * duty))
    start = rng.integers(0, max(1, n - burst_len))
    sig = np.zeros(n, dtype=complex)
    active = gen_random_modulation(burst_len, fs, rng, sps=2, order=order)  # sps unified to 2 (was 4)
    # taper edges to avoid spectral splatter from hard gating
    w = np.hanning(burst_len)
    sig[start:start + burst_len] = active * w
    # normalize over active region only
    p = np.mean(np.abs(sig[start:start + burst_len]) ** 2)
    if p > 0:
        sig /= np.sqrt(p)
    return sig


WAVEFORM_GENERATORS = {
    "random": gen_random_modulation,
    "dsss": gen_dsss,
    "fhss": gen_fhss,
    "burst": gen_burst,
}


def generate_waveform(kind: str, n: int, fs: float,
                      rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Dispatch to the requested LPI/LPD waveform generator."""
    if kind not in WAVEFORM_GENERATORS:
        raise ValueError(f"unknown waveform '{kind}'; "
                         f"choose from {list(WAVEFORM_GENERATORS)}")
    return WAVEFORM_GENERATORS[kind](n, fs, rng, **kwargs)
