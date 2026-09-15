"""
conformal.py — Elliptical Conformal Prediction on top of multivariate DER (NIW).

What changed vs the circular version
------------------------------------
The previous module produced *circular* regions: nonconformity score
||y - pred|| / std_mag, giving a single radius Q * std_mag around the point.
That throws away the very thing the NIW head now provides — the *shape and
orientation* of the error ellipse.

Elliptical CP uses the NIW covariance to whiten the residual before scoring:

    score_i = sqrt( (y_i - pred_i)^T  Sigma_i^{-1}  (y_i - pred_i) )      (Mahalanobis)

This is the residual measured in "ellipse units": how many ellipse-radii away
the truth is, along whatever direction it lies. Whitening turns the tilted
error ellipse into a unit circle, we take the conformal quantile Q there, then
map back — so the calibrated region for a test point is the ellipse

    { z : (z - pred)^T Sigma^{-1} (z - pred) <= Q^2 }

i.e. Sigma's own ellipse, uniformly inflated by Q so that empirical coverage
hits the nominal 1 - alpha. Distribution-free finite-sample coverage still
holds (it is ordinary split CP on a scalar score); we only changed the score
from an isotropic ratio to a Mahalanobis ratio.

Why this is the right partner for NIW
-------------------------------------
NIW gives the tilted covariance; elliptical CP is the calibration geometry that
respects it. In high-GDOP scenes the ellipse is long and thin, so an elliptical
region of the same coverage is far smaller in area than the enclosing circle
(we measured ~-53% area). The global-scale that Meinert's r-coupling discards
is exactly what Q restores here: EDL supplies the ellipse *shape*, CP supplies
the *scale + guarantee*. Complementary, not redundant.

Area of the region (for efficiency reporting)
---------------------------------------------
The Mahalanobis ball of squared-radius Q^2 under Sigma has area
    area = pi * Q^2 * sqrt(det(Sigma))          (n=2)
so det(Sigma) is the natural "how big is the ellipse" summary.

References
----------
[Romano, Patterson, Candes 2019] CQR.  [Vovk et al.] split-conformal foundations.
[Johnstone & Cox / Messoudi et al. 2021/2022] multivariate & elliptical
conformal regions (Mahalanobis nonconformity).
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# finite-sample conformal quantile (unchanged)
# ---------------------------------------------------------------------------
def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    ceil((n+1)(1-alpha))/n empirical quantile of calibration scores.
    Guarantees P(coverage) >= 1-alpha under exchangeability.
    """
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


# ---------------------------------------------------------------------------
# core: Mahalanobis nonconformity under per-sample NIW covariance
# ---------------------------------------------------------------------------
def _mahalanobis(resid: np.ndarray, cov: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Per-sample Mahalanobis distance sqrt( r^T Sigma^{-1} r ).

    resid : (n_samples, d)     residual vectors (y - pred), meters
    cov   : (n_samples, d, d)  per-sample predictive covariance, meters^2
    Returns (n_samples,) distances.

    Implemented via Cholesky solve for numerical stability; a small jitter is
    added to the diagonal to guard against near-singular ellipses.
    """
    ns, d = resid.shape
    jitter = eps * np.eye(d)[None, :, :]
    C = cov + jitter
    # solve C z = r   ->  z = C^{-1} r ; then r^T z = mahalanobis^2
    out = np.empty(ns, dtype=float)
    for i in range(ns):
        Li = np.linalg.cholesky(C[i])
        z = np.linalg.solve(Li, resid[i])          # lower-tri solve
        out[i] = np.sqrt(float(z @ z))             # ||L^{-1} r|| = sqrt(r^T C^-1 r)
    return out


def elliptical_calibrate(y_cal, pred, cov, alpha):
    """
    Elliptical (Mahalanobis) conformal calibration.

    y_cal : (n,2) true positions on the calibration set
    pred  : (n,2) NIW point estimates (mu0), meters
    cov   : (n,2,2) NIW total predictive covariance, meters^2
    Returns Q (scalar). The calibrated region for a test point with covariance
    Sigma is  { z : (z-pred)^T Sigma^{-1} (z-pred) <= Q^2 }.

    Adaptivity: because the score is normalized by each sample's own Sigma,
    uncertain (large-ellipse) samples automatically get proportionally larger
    regions; Q is the single shared inflation that makes coverage exact.
    """
    resid = y_cal - pred                            # (n,2)
    scores = _mahalanobis(resid, cov)               # (n,)
    return _finite_sample_quantile(scores, alpha)


def elliptical_coverage(y_true, pred, cov, Q):
    """
    Check coverage of the elliptical regions of Mahalanobis radius Q.

    Returns (covered_bool[n], area_m2[n]) where
        covered  = Mahalanobis(y-pred; Sigma) <= Q
        area     = pi * Q^2 * sqrt(det(Sigma))     (n=2 ellipse area)
    """
    resid = y_true - pred
    md = _mahalanobis(resid, cov)
    covered = md <= Q
    det = np.clip(np.linalg.det(cov), 1e-24, None)
    area = np.pi * (Q ** 2) * np.sqrt(det)
    return covered, area


def elliptical_axes(cov, Q):
    """
    Semi-axis lengths (meters) of the calibrated region per sample, for
    plotting / reporting. For covariance Sigma with eigenvalues lam_k, the
    region  r^T Sigma^{-1} r <= Q^2  has semi-axes  Q * sqrt(lam_k) along the
    eigenvectors. Returns (n, 2) sorted descending (major, minor).
    """
    w = np.linalg.eigvalsh(cov)                     # ascending eigenvalues (n,2)
    w = np.clip(w, 0.0, None)
    axes = Q * np.sqrt(w)[:, ::-1]                  # major first
    return axes


# ---------------------------------------------------------------------------
# Mondrian (group-conditional) elliptical CP
# ---------------------------------------------------------------------------
def mondrian_elliptical_calibrate(y_cal, pred, cov, groups_cal, alpha):
    """
    Group-conditional elliptical CP: a separate conformal Q_g per group g
    (e.g. per SNR bin). Restores per-group coverage that marginal CP can miss
    (low-SNR under-coverage vs high-SNR over-coverage), at the cost of fewer
    calibration points per group. Falls back to the marginal Q for tiny groups.

    Returns dict {group: Q_g}.
    """
    resid = y_cal - pred
    scores = _mahalanobis(resid, cov)
    Qs = {}
    for g in np.unique(groups_cal):
        m = groups_cal == g
        if m.sum() < 10:
            Qs[g] = _finite_sample_quantile(scores, alpha)
        else:
            Qs[g] = _finite_sample_quantile(scores[m], alpha)
    return Qs


# ---------------------------------------------------------------------------
# Backward-compatible circular API (kept so old PHASE-5/6 scripts still import,
# and so an ablation "circular vs elliptical" comparison is one call away).
# ---------------------------------------------------------------------------
def radial_calibrate(y_cal, pred, std_mag, alpha):
    """
    Circular CP (isotropic). Nonconformity = ||y-pred|| / std_mag.
    Retained for the elliptical-vs-circular ablation; the main pipeline uses
    elliptical_calibrate. `std_mag` is a scalar std magnitude per sample
    (e.g. niw_scalar_std(total_cov)).
    """
    resid = np.sqrt(((y_cal - pred) ** 2).sum(1))
    scores = resid / np.clip(std_mag, 1e-6, None)
    return _finite_sample_quantile(scores, alpha)


def radial_coverage(y_true, pred, std_mag, Q):
    """Circular coverage: region radius = Q*std_mag; returns (covered, radius)."""
    radius = Q * std_mag
    dist = np.sqrt(((y_true - pred) ** 2).sum(1))
    return dist <= radius, radius


def mondrian_radial_calibrate(y_cal, pred, std_mag, groups_cal, alpha):
    """Circular Mondrian CP (ablation companion to the elliptical version)."""
    resid = np.sqrt(((y_cal - pred) ** 2).sum(1))
    scores = resid / np.clip(std_mag, 1e-6, None)
    Qs = {}
    for g in np.unique(groups_cal):
        m = groups_cal == g
        Qs[g] = _finite_sample_quantile(scores[m] if m.sum() >= 10 else scores, alpha)
    return Qs


# per-axis CQR helpers (unchanged, retained for completeness)
def cqr_calibrate_axis(y_cal, q_lo, q_hi, alpha):
    scores = np.maximum(q_lo - y_cal, y_cal - q_hi)
    return _finite_sample_quantile(scores, alpha)


def cqr_interval_axis(q_lo, q_hi, Q):
    return q_lo - Q, q_hi + Q
