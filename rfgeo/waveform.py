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


# =====================================================================
# Jammer-class emitters (ARM ONLY — never mixed into the 4-waveform baseline)
# =====================================================================
# Signal models: Morales Ferre, de la Fuente, Lohan, "Jammer classification in
# GNSS bands via machine learning algorithms", Sensors 19(22):4841, 2019,
# Eqs. (2)-(6).
#
# ⚠️ Parameters are NOT taken verbatim from that paper's Table 1. Three entries
#    there cannot be used here:
#      - chirp  U(5,20) MHz / U(5,20) us : sweeps our 2 MHz passband in
#        0.5-8 us = 1-16 samples at fs=2 MHz -> not representable.
#      - NB     U(20, 2000) MHz          : contradicts "narrowband" and lies
#        outside our +-1 MHz observation band.
#      - pulse  f_r U(1e11, 19e11) THz   : 1e23 Hz, physically impossible.
#    We substitute the values MEASURED from orbit by Clements, Humphreys, Ellis,
#    "Dual-Satellite Geolocation of Terrestrial GNSS Jammers from LEO",
#    IEEE/ION PLANS 2023, pp. 458-469 (chirp 200 kHz / 8 ms and 20 kHz / 100 ms;
#    narrowband content at the tens-of-kHz scale), and a physically sane DME
#    pulse repetition rate.
#
# 🔴 These MUST stay out of the default `GenConfig.waveforms`. Section 5.1 of
#    CLAUDE.md unified sps=2 across the four baseline waveforms precisely to
#    REMOVE per-waveform Brms confounding; in this arm the Brms spread is the
#    independent variable (8 kHz .. 519 kHz, a 65x range). Mixing the two
#    families in one dataset would break that settled decision.
#
# Design constants (deterministic from the definitions, computed before the
# experiment by scripts/data/jammer_waveform_probe.py, SNR 10 dB, 100 draws):
#     waveform    Brms(kHz)   occ@-20dB   sigma_tdoa(m)
#     jam_cw          8.0       0.0013        26.09      <- degenerate case
#     jam_am        416.4       0.0044         0.568     <- wide Brms, empty band
#     jam_chirp      32.1       0.034         12.65      <- Clements-measured
#     jam_pulse     137.0       0.177         16.21      <- low duty -> gamma_eff
#     jam_nb         32.5       0.053          7.55

def gen_jam_cw(n: int, fs: float, rng: np.random.Generator,
               f_max_frac: float = 0.4) -> np.ndarray:
    """Eq. (2) with K=1: a single tone.

    Deliberately degenerate: Brms collapses to the record-length limit, so the
    TDOA CRLB blows up. Used to check that conformal validity survives when the
    estimator itself is powerless (the cost should appear as region size only).
    """
    t = np.arange(n) / fs
    f = rng.uniform(-f_max_frac, f_max_frac) * fs / 2.0
    return _normalize_power(np.exp(2j * np.pi * f * t
                                   + 1j * rng.uniform(0.0, 2.0 * np.pi)))


def gen_jam_am(n: int, fs: float, rng: np.random.Generator,
               k_min: int = 3, k_max: int = 5,
               f_max_frac: float = 0.9) -> np.ndarray:
    """Eq. (2) with K>1: multi-tone AM jammer.

    Tones are spread across the band, so Brms stays LARGE (~416 kHz, above the
    baseline waveforms) while spectral occupancy collapses to ~0.4%. This is the
    only waveform in the project with that combination, and it is the sharpest
    available test of the PHAT-whitening question (Donohue 2007 reports PHAT
    degrading for narrowband/comb-like spectra; PHAT lifts the ~99.6% of bins
    that carry noise only).
    """
    t = np.arange(n) / fs
    K = int(rng.integers(k_min, k_max + 1))
    sig = np.zeros(n, dtype=complex)
    for _ in range(K):
        f = rng.uniform(-f_max_frac, f_max_frac) * fs / 2.0
        sig += np.exp(2j * np.pi * f * t + 1j * rng.uniform(0.0, 2.0 * np.pi))
    return _normalize_power(sig)


def gen_jam_chirp(n: int, fs: float, rng: np.random.Generator,
                  f_swp: float = 200e3, t_swp: float = 8e-3) -> np.ndarray:
    """Eq. (3): saw-tooth linear chirp, Clements-measured parameters.

    At the default 200 kHz / 8 ms the 2.048 ms snapshot covers 25.6% of one
    sweep, i.e. a slowly drifting FM tone spanning ~51 kHz.
    """
    t = np.arange(n) / fs
    b = 1.0 if rng.random() < 0.5 else -1.0
    tm = np.mod(t + rng.uniform(0.0, t_swp), t_swp)        # saw-tooth phase
    f0 = -f_swp / 2.0
    phase = 2.0 * np.pi * (f0 * tm + b * 0.5 * (f_swp / t_swp) * tm ** 2)
    return _normalize_power(np.exp(1j * phase
                                   + 1j * rng.uniform(0.0, 2.0 * np.pi)))


def gen_jam_chirp_wb(n: int, fs: float, rng: np.random.Generator,
                     f_swp_hz: tuple = (5e6, 20e6),
                     t_swp_s: tuple = (5e-6, 20e-6),
                     guard: float = 0.05) -> np.ndarray:
    """Eq. (3) with Morales Table 1 parameters: WIDEBAND saw-tooth chirp.

    F_swp ~ U(5, 20) MHz and T_swp ~ U(5, 20) us, i.e. the dominant civilian
    "personal privacy device" class. Distinct from `jam_chirp`, which uses the
    NARROWBAND values Clements et al. measured from LEO (200 kHz / 8 ms). The
    two are different physics and must be kept as separate classes.

    ⚠️ Partial-band observation. A receiver of bandwidth fs sees a sweep wider
    than fs only while the instantaneous frequency is inside the passband. We
    model that literally: the sample is zeroed whenever |f_inst(t)| exceeds
    (1-guard)*fs/2. Gating in TIME is the correct model for a pure linear FM --
    generating the out-of-band part and relying on the FFT would alias it back
    into the passband, which a real front end never does.

    In-band duty is therefore ~ min(1, fs / F_swp); with fs = 8 MHz this spans
    0.4 (F_swp = 20 MHz) to 1.0 (F_swp <= 8 MHz).
    Power is normalized over the ACTIVE (in-band) samples, matching the
    burst/pulse rule so the SNR label refers to on-time power.
    """
    f_swp = rng.uniform(*f_swp_hz)
    t_swp = rng.uniform(*t_swp_s)
    b = 1.0 if rng.random() < 0.5 else -1.0
    t = np.arange(n) / fs
    tm = np.mod(t + rng.uniform(0.0, t_swp), t_swp)        # saw-tooth phase
    rate = f_swp / t_swp                                    # Hz per second
    f_inst = b * (-f_swp / 2.0 + rate * tm)                 # instantaneous freq
    phase = 2.0 * np.pi * b * (-f_swp / 2.0 * tm + 0.5 * rate * tm ** 2)
    sig = np.exp(1j * phase + 1j * rng.uniform(0.0, 2.0 * np.pi))
    inband = np.abs(f_inst) <= (1.0 - guard) * fs / 2.0
    sig = sig * inband
    p = np.mean(np.abs(sig[inband]) ** 2) if np.any(inband) else 0.0
    return sig / np.sqrt(p) if p > 0 else sig


def gen_jam_pulse(n: int, fs: float, rng: np.random.Generator,
                  tau_us: tuple = (1.0, 19.0),
                  fr_khz: tuple = (0.1, 2.0),
                  fc_frac: float = 0.3) -> np.ndarray:
    """Eq. (5): pulse / DME-like jammer.

    Short pulses widen Brms, but the low duty cycle shrinks the number of active
    samples and hence gamma_eff. This exercises the active-SNR CRLB patch (§5.4)
    on a waveform it was never tuned on -- the patch was built from `burst`.
    Power is normalized over the ACTIVE region only, matching the burst rule.
    """
    tau = rng.uniform(*tau_us) * 1e-6
    fr = rng.uniform(*fr_khz) * 1e3
    w = max(2, int(round(tau * fs)))
    period = max(w + 1, int(round(fs / fr)))
    t = np.arange(n) / fs
    fc = rng.uniform(-fc_frac, fc_frac) * fs / 2.0
    env = np.zeros(n)
    for a in range(0, n, period):
        env[a:min(a + w, n)] = 1.0
    sig = np.exp(2j * np.pi * fc * t) * env
    act = env > 0
    p = np.mean(np.abs(sig[act]) ** 2) if np.any(act) else 0.0
    return sig / np.sqrt(p) if p > 0 else sig


def gen_jam_nb(n: int, fs: float, rng: np.random.Generator,
               bw_khz: tuple = (20.0, 200.0),
               fc_frac: float = 0.3) -> np.ndarray:
    """Eq. (6): band-limited noise jammer (noise-FM).

    Continuous narrow spectrum, in contrast to the discrete lines of `jam_am`.
    """
    bw = rng.uniform(*bw_khz) * 1e3
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    f = np.fft.fftfreq(n, 1.0 / fs)
    fc = rng.uniform(-fc_frac, fc_frac) * fs / 2.0
    mask = np.abs(f - fc) <= bw / 2.0
    if mask.sum() < 2:                       # guarantee a non-degenerate band
        mask[int(np.argmin(np.abs(f - fc)))] = True
    return _normalize_power(np.fft.ifft(np.fft.fft(x) * mask))


#: Baseline LPI/LPD emitters. `GenConfig.waveforms` defaults to exactly these.
# =====================================================================
# Held-out LPI classes — WAVEFORM-FAMILY OOD AXIS (never trained on)
# =====================================================================
# Why these exist
# ---------------
# Every OOD axis in this project so far perturbs the *channel* or the
# *hardware* (SNR, CFO, SCO, delay spread, K-factor, TDL profile). None of
# them perturbs the *waveform*. But "non-cooperative" centrally means the
# transmit waveform is unknown, and the baseline model is trained on the very
# four classes it is tested on. These generators supply an unseen-waveform
# axis so that claim can be measured instead of asserted.
#
# They are NOT in BASELINE_WAVEFORMS, so the baseline dataset is bit-identical.
#
# Design constraints (all three are in-band and unit-power, like the baseline):
#   * occupancy must be high at fs=2 MHz — the `jam_*` narrowband classes
#     failed precisely because they occupied 0.1-19% of the band, so PHAT
#     whitened mostly noise (see JAMMER_WB_ARM_RESULT_2026-08-14.md).
#   * no parameter may exceed the observation band, or the receiver would be
#     modelling a signal it physically cannot see.
#
# References for the waveform families
# ------------------------------------
# LFM / FMCW as the canonical LPI *radar* waveform (large time-bandwidth
#   product, low peak power): Pace, "Detecting and Classifying Low Probability
#   of Intercept Radar", Artech House. Cited by Vankayalapati & Kay,
#   Proc. SPIE 7706:77060U (2010), which is our nearest LPI-localization work.
# MSK as a constant-envelope, spectrally-compact modulation:
#   Proakis & Salehi, "Digital Communications".
# OFDM as the dominant modern wideband intercept target.
#
# NOTE the family split this exposes: our four baseline waveforms are
# communications LPI/LPD (DSSS/FHSS/burst, Pickholtz et al. 1982), whereas the
# LPI *radar* literature means FMCW/LFM/polyphase. `lfm_inband` is the bridge.

def gen_lfm_inband(n: int, fs: float, rng: np.random.Generator,
                   bw_frac: tuple = (0.55, 0.85),
                   t_swp_s: tuple = (2.0e-4, 8.0e-4)) -> np.ndarray:
    """
    LPI *radar* LFM/FMCW: linear sweep kept entirely inside the passband.

    Unlike `jam_chirp_wb` (Morales jammer parameters, sweep 5-20 MHz, i.e. far
    wider than our band, so most of it is gated away) this sweeps only a
    fraction of fs and is therefore always fully observable. Time-bandwidth
    product is order 1e2-1e3, which is what makes an LFM low-probability-of-
    intercept in the first place.
    """
    bw = rng.uniform(*bw_frac) * fs
    t_swp = rng.uniform(*t_swp_s)
    up = rng.random() < 0.5
    t = np.arange(n) / fs
    tm = np.mod(t + rng.uniform(0.0, t_swp), t_swp)
    rate = bw / t_swp
    sign = 1.0 if up else -1.0
    phase = 2.0 * np.pi * sign * (-bw / 2.0 * tm + 0.5 * rate * tm ** 2)
    sig = np.exp(1j * (phase + rng.uniform(0.0, 2.0 * np.pi)))
    return _normalize_power(sig)


def gen_msk(n: int, fs: float, rng: np.random.Generator,
            sps: int = 2) -> np.ndarray:
    """
    MSK: constant-envelope CPM. Same symbol rate as the baseline (sps unified
    to 2, cf. §5.1) so the comparison isolates *pulse shape*, not rate.
    """
    n_sym = n // sps + 4
    bits = rng.choice([-1.0, 1.0], size=n_sym)
    # phase advances by +-pi/2 per symbol, linearly interpolated within it
    step = (np.pi / 2.0) * np.repeat(bits, sps)[:n] / sps
    phase = np.cumsum(step) + rng.uniform(0.0, 2.0 * np.pi)
    return _normalize_power(np.exp(1j * phase))


def gen_ofdm(n: int, fs: float, rng: np.random.Generator,
             n_fft: int = 64, cp_frac: float = 0.25,
             used_frac: float = 0.8) -> np.ndarray:
    """
    OFDM with cyclic prefix: dense, near-Gaussian spectrum and high PAPR.
    The most common modern wideband intercept target, and spectrally the
    opposite of the narrowband `jam_*` classes.
    """
    n_used = max(2, int(n_fft * used_frac) // 2 * 2)
    cp = int(n_fft * cp_frac)
    out = []
    while sum(len(b) for b in out) < n:
        sym = np.zeros(n_fft, dtype=complex)
        idx = np.arange(1, n_used // 2 + 1)
        d = _random_symbols(2 * len(idx), 4, rng)
        sym[idx] = d[:len(idx)]
        sym[-idx] = d[len(idx):]
        x = np.fft.ifft(sym) * np.sqrt(n_fft)
        out.append(np.concatenate([x[-cp:], x]))
    return _normalize_power(np.concatenate(out)[:n])


#: Held-out LPI classes for the waveform-family OOD axis. Never trained on.
LPI_OOD_WAVEFORMS = ("lfm_inband", "msk", "ofdm")

BASELINE_WAVEFORMS = ("random", "dsss", "fhss", "burst")
#: Jammer-class emitters, selected explicitly via `--waveforms` (arm only).
JAMMER_WAVEFORMS = ("jam_cw", "jam_am", "jam_chirp", "jam_chirp_wb",
                    "jam_pulse", "jam_nb")

WAVEFORM_GENERATORS = {
    "random": gen_random_modulation,
    "dsss": gen_dsss,
    "fhss": gen_fhss,
    "burst": gen_burst,
    # --- arm only; adding these leaves the four above bit-identical ---
    "jam_cw": gen_jam_cw,
    "jam_am": gen_jam_am,
    "jam_chirp": gen_jam_chirp,
    "jam_chirp_wb": gen_jam_chirp_wb,
    "jam_pulse": gen_jam_pulse,
    "jam_nb": gen_jam_nb,
    # --- held-out LPI classes (waveform-family OOD axis, never trained on) ---
    "lfm_inband": gen_lfm_inband,
    "msk": gen_msk,
    "ofdm": gen_ofdm,
}


def generate_waveform(kind: str, n: int, fs: float,
                      rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Dispatch to the requested LPI/LPD waveform generator."""
    if kind not in WAVEFORM_GENERATORS:
        raise ValueError(f"unknown waveform '{kind}'; "
                         f"choose from {list(WAVEFORM_GENERATORS)}")
    return WAVEFORM_GENERATORS[kind](n, fs, rng, **kwargs)
