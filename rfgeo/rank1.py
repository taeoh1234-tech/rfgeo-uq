"""
rank1.py — FS-GCC rank-1 deviation: a MODEL-INDEPENDENT corruption statistic.

Why this module exists
----------------------
E9 established that the NIW-DER head's epistemic uncertainty can INVERT out of
distribution: on the delay-spread axis the true error rises ~4x while epistemic
falls to ~0.45x (confidently wrong). E14 showed that the rank structure of the
frequency-sliding GCC matrix (Cobos 2020) moves the CORRECT way on that same axis,
because it measures inter-band disagreement rather than effective SNR.

Cobos 2020 (TASLP), eq. 24: for an ideal noiseless single-path observation the
FS-GCC matrix is rank-1 (R = phi_0 e^H); noise and multipath scatter the per-band
peaks and raise the rank (eq. 27). The deviation from rank-1 is therefore a direct
physical measure of "how corrupted is this observation".

IMPORTANT — this is a DIAGNOSTIC READ-OUT, not a preprocessing change:
  * The CNN is still fed GCC-PHAT curves only (gcc.gcc_phat_features).
  * No low-rank approximation is ever applied to the data, so the CRLB anchor that
    justifies the aleatoric validation stays intact (CLAUDE.md §16.3 excludes
    FS-GCC as a *preprocessing* precisely because learned/low-rank reconstruction
    would make the error floor data-dependent; a read-out does not).

Known limits (CLAUDE.md §13 E14 step 5):
  * Deviation is a function of d/L and vanishes when the multipath delay d is an
    integer multiple of the band count L (all bands see the same phase pattern).
  * It is fundamentally noise-dominated (Spearman vs SNR ~= -0.91); multipath moves
    it an order of magnitude less than noise does. Use SNR-conditional thresholds.

Reference
---------
[Cobos et al. 2020] "Frequency-sliding generalized cross-correlation: A sub-band
time delay estimation approach," IEEE/ACM TASLP.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations

N_BANDS = 32          # Cobos 2020 default; pre-registered value (RANK1_OOD_PREREG.md)
EPS = 1e-8


def fsgcc_rank_deviation(xi: np.ndarray, xj: np.ndarray,
                         n_bands: int = N_BANDS, eps: float = EPS) -> float:
    """
    Rank-1 deviation of the FS-GCC matrix for one receiver pair.

    Whitened cross-spectrum -> fftshift -> split into n_bands contiguous,
    non-overlapping frequency slices -> IFFT each slice (slice extraction IS the
    baseband demodulation that preserves the rank-1 structure) -> SVD.

    Ideal (noiseless, single path): R(f) = exp(-j2*pi*f*tau0), so slice l equals
    exp(-j2*pi*f_l*tau0) * (common vector) -> outer product -> rank 1 -> 0.0.

    Returns
    -------
    1 - sigma_1^2 / sum_i sigma_i^2   (0 = perfectly rank-1, larger = more corrupt)
    """
    n = xi.shape[-1]
    Xi = np.fft.fft(xi, n=n)
    Xj = np.fft.fft(xj, n=n)
    R = Xi * np.conj(Xj)
    R = R / (np.abs(R) + eps)              # PHAT whitening (identical to gcc.py)
    R = np.fft.fftshift(R)                 # monotonic frequency axis before slicing
    nb = n // n_bands
    M = np.fft.ifft(R[:nb * n_bands].reshape(n_bands, nb), axis=1)
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2
    tot = p.sum()
    if tot <= 0:
        return float("nan")
    return float(1.0 - p[0] / tot)


def fsgcc_rank_deviation_stack(iq_rx: np.ndarray, n_bands: int = N_BANDS) -> float:
    """
    Median rank-1 deviation over all receiver pairs of one sample.

    Parameters
    ----------
    iq_rx : (M, 2, L) real array — the same per-sample IQ format gcc_phat_features
            consumes, so both can be computed from one generated sample.
    """
    cplx = iq_rx[:, 0, :] + 1j * iq_rx[:, 1, :]
    vals = [fsgcc_rank_deviation(cplx[i], cplx[j], n_bands=n_bands)
            for i, j in combinations(range(iq_rx.shape[0]), 2)]
    return float(np.median(vals))


def indist_percentile_map(reference: np.ndarray):
    """
    Build x -> empirical in-distribution percentile (0..1).

    Deployment use: calibrate on in-distribution data, then a test sample's
    percentile is a calibration-free score comparable across detectors.
    """
    s = np.sort(np.asarray(reference, float))
    n = max(len(s), 1)
    return lambda x: np.searchsorted(s, np.asarray(x, float), side="right") / n
