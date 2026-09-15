"""
gcc.py — Generalized Cross-Correlation with Phase Transform (GCC-PHAT).

Why this module exists
----------------------
PHASE 2 showed that a CNN fed raw concatenated IQ struggles to recover emitter
position. The position information lives in the *relative inter-receiver delay*
(TDOA), and a translation-invariant CNN has no natural mechanism to compare
absolute alignment between receiver channels. The acoustic source-localization
literature solves exactly this by feeding inter-channel correlation features
instead of (or in addition to) raw signals.

GCC-PHAT (Knapp & Carter 1976) is the standard such feature:

    R_ij(tau) = IFFT( (X_i(f) * conj(X_j(f))) / |X_i(f) * conj(X_j(f))| )

The PHAT weighting (dividing by magnitude) whitens the spectrum so the
correlation peak is sharp and robust to multipath/coloration — the peak lag is
the TDOA between receivers i and j. We hand the network a stack of these
cross-correlation curves (one per receiver pair), so the "compare alignments"
operation is provided in the input rather than something the CNN must invent.

This is a *representation* change at the backbone input. It does not touch the
DER/CP methodology contribution that sits on the output side.

Reference
---------
[Knapp & Carter 1976] "The generalized correlation method for estimation of
time delay," IEEE Trans. ASSP. (Also the same citation already used in crlb.py
for TDOA variance — internally consistent.)
"""

from __future__ import annotations
import numpy as np
from itertools import combinations


def gcc_phat_pair(xi: np.ndarray, xj: np.ndarray, max_lag: int | None = None,
                  eps: float = 1e-8) -> np.ndarray:
    """
    GCC-PHAT cross-correlation between two complex IQ snapshots.

    Parameters
    ----------
    xi, xj : (L,) complex arrays (receiver i and j)
    max_lag : if set, return only lags in [-max_lag, +max_lag] (centered)

    Returns
    -------
    cc : real-valued correlation curve, length (2*max_lag+1) if max_lag set,
         else L (fftshifted so zero-lag is centered).
    """
    n = xi.shape[-1]
    Xi = np.fft.fft(xi, n=n)
    Xj = np.fft.fft(xj, n=n)
    R = Xi * np.conj(Xj)
    R /= np.abs(R) + eps              # PHAT weighting (whitening)
    cc = np.fft.ifft(R, n=n)
    cc = np.fft.fftshift(cc).real     # zero-lag at center
    center = n // 2
    if max_lag is not None:
        cc = cc[center - max_lag: center + max_lag + 1]
    return cc.astype(np.float32)


def gcc_phat_features(iq_rx: np.ndarray, max_lag: int = 128) -> np.ndarray:
    """
    Build the full stack of pairwise GCC-PHAT curves for one sample.

    Parameters
    ----------
    iq_rx : (M, 2, L) real array — M receivers, [I,Q], L samples
            (the PHASE 1 per-sample input format)
    max_lag : half-window of lags to keep around zero. With fs=2 MHz and
              receivers ~8 km apart, |TDOA| <= ~baseline/c. Max range-difference
              across the array is bounded by the inter-receiver baseline; a
              few-hundred-sample window comfortably covers it. 128 -> +-64 us.

    Returns
    -------
    feats : (P, 2*max_lag+1) array, P = M*(M-1)/2 receiver pairs.
            Each row is one pair's GCC-PHAT curve. Treated as P input channels
            by the 1D CNN.
    """
    M = iq_rx.shape[0]
    # reconstruct complex IQ per receiver
    cplx = iq_rx[:, 0, :] + 1j * iq_rx[:, 1, :]       # (M, L)
    rows = []
    for i, j in combinations(range(M), 2):
        rows.append(gcc_phat_pair(cplx[i], cplx[j], max_lag=max_lag))
    return np.stack(rows, axis=0)                      # (P, 2*max_lag+1)


def n_pairs(M: int) -> int:
    return M * (M - 1) // 2
