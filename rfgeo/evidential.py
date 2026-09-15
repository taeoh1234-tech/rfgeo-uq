"""
evidential.py — Multivariate Deep Evidential Regression (Meinert & Lavin 2021,
Normal-Inverse-Wishart prior), replacing the univariate Amini NIG head.

Why NIW instead of NIG
----------------------
Emitter position is a *2D coordinate* (x, y). The previous NIG head treated the
two axes as independent — it output a separate scalar variance per axis and had
no way to express that the x-error and y-error are correlated. In geolocation
they almost always are: receiver geometry (GDOP) makes the error ellipse tilted
and elongated along a preferred direction. Modelling x and y independently is
like insisting the dart-cluster on a target is always an axis-aligned rectangle
when it is really a slanted ellipse.

Meinert's multivariate DER places a Normal-Inverse-Wishart (NIW) prior over the
full 2x2 covariance Sigma of a bivariate Gaussian likelihood. The network now
predicts the whole covariance structure (shape + orientation of the ellipse),
not two independent variances. This is exactly the information the downstream
elliptical Conformal Prediction consumes.

Parameterization (Meinert 2021, Eqs. 12-15)
-------------------------------------------
For n=2 targets the network outputs, per sample:
    mu0  in R^2                      -> point prediction (gamma)
    ell  in R^{n(n+1)/2}=R^3         -> lower-triangular Cholesky factor L of
                                        the scale matrix, with positive diagonal
    p_nu in R                        -> raw scalar mapped to nu (DoF)

The scale matrix is  Psi = nu * Sigma0 = nu * L L^T  (Eq. 13 uses L L^T
directly). L is built as (Meinert Eq. 15):
    L[j,j] = exp(ell_j)      (positive diagonal, guarantees PD)
    L[j,k] = ell_jk  (j>k)   (free lower off-diagonal)
    L[j,k] = 0       (j<k)

nu is constrained > n+1 so the covariance mean is finite. Following Wu/Ye's
multivariate ERN activation (needed to keep nu safely above the n+1 pole):
    nu = n(n+5)/2 + softplus-like(p_nu) ... we use the tanh form from the MERN
    write-up: nu = n(n+5)/2 + tanh(p_nu) * n(n+3)/2 + 1  (> n+1 by construction).

nu = r * kappa COUPLING (Eq. 12) — the key stabilizer
-----------------------------------------------------
Meinert shows the marginal t-distribution cannot disentangle (kappa, nu, Sigma0)
from data alone (shape-parameter non-identifiability). Coupling nu = r*kappa
with a *constant* hyperparameter r removes one redundant DoF, which is what
tamed the lambda-sensitivity we saw with NIG (x34.7 -> x1.3 in our pipeline).
Because we parameterize by nu directly here, kappa is defined implicitly as
kappa = nu / r. With the coupling, the evidence regularizer of Amini is no
longer needed: minimizing the NLL alone is sufficient (Meinert Sec. 3).

Cost of the coupling: we lose the *global* scale of aleatoric/epistemic
uncertainty (Meinert notes this explicitly). That is fine here — the downstream
Conformal Prediction calibration constant Q re-absorbs the global scale, so the
NIW output only needs to get the *shape and ordinal ranking* of uncertainty
right, which is precisely what EDL is good at.

Uncertainty decomposition (Eq. 14)
-----------------------------------
    E[mu]     = mu0                                  (prediction)
    E[Sigma]  = nu/(nu - n - 1) * L L^T              (aleatoric covariance)
    Var[mu]   = E[Sigma] / nu                        (epistemic covariance)

Both are full 2x2 matrices now — tilted ellipses, not per-axis scalars.

References
----------
[Meinert & Lavin 2021] "Multivariate Deep Evidential Regression."
[Amini et al. 2020]     "Deep Evidential Regression" (the univariate NIG this
                         file replaces).
[Ye et al. 2024 / Wu AAAI-24] multivariate ERN activation for nu.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Head: feature vector -> NIW parameters (mu0, L, nu) for n=2 coordinates.
# ---------------------------------------------------------------------------
class NIWHead(nn.Module):
    """
    Maps a feature vector to Normal-Inverse-Wishart parameters for `n_out`
    coordinates (n_out=2 for 2D geolocation).

    Output neuron budget (Meinert): n(n+3)/2 + 1.
      n=2 -> 2*5/2 + 1 = 6 outputs:
        2 for mu0 (gamma, unconstrained),
        3 for ell (Cholesky factor of the n x n scale matrix),
        1 for p_nu (raw scalar mapped to the DoF nu).

    Coupling: nu = r * kappa. We predict nu directly and treat kappa = nu / r
    implicitly; r is a fixed hyperparameter (default 1.0, as in Meinert's
    benchmark experiment).
    """
    def __init__(self, in_dim: int, n_out: int = 2, r: float = 1.0):
        super().__init__()
        self.n = n_out
        self.r = float(r)
        self.n_tri = n_out * (n_out + 1) // 2          # 3 for n=2
        self.n_params = n_out * (n_out + 3) // 2 + 1    # 6 for n=2
        self.fc = nn.Linear(in_dim, self.n_params)
        # tril indices for scattering ell into L
        self._tril_idx = torch.tril_indices(n_out, n_out)

    def forward(self, x):
        out = self.fc(x)                                # (B, n_params)
        n, ntri = self.n, self.n_tri
        mu0 = out[:, :n]                                # (B, n) prediction
        ell = out[:, n:n + ntri]                        # (B, ntri) Cholesky raw
        p_nu = out[:, n + ntri]                         # (B,) raw DoF scalar

        L = self._build_L(ell)                          # (B, n, n) lower-tri, PD
        nu = self._activate_nu(p_nu)                    # (B,) > n+1
        return mu0, L, nu

    def _build_L(self, ell):
        """
        Build lower-triangular L with positive diagonal (Meinert Eq. 15):
          diagonal   -> exp(ell_j)   (strictly positive => Sigma0 = L L^T is PD)
          off-diag   -> ell_jk       (free)
        `ell` is ordered by torch.tril_indices row-major: for n=2 the order is
        (0,0), (1,0), (1,1).
        """
        B = ell.shape[0]
        n = self.n
        L = ell.new_zeros(B, n, n)
        idx = self._tril_idx.to(ell.device)
        rows, cols = idx[0], idx[1]
        for t in range(idx.shape[1]):
            j, k = int(rows[t]), int(cols[t])
            if j == k:
                L[:, j, k] = torch.exp(ell[:, t].clamp(max=8.0))  # bounded exp
            else:
                L[:, j, k] = ell[:, t]
        return L

    def _activate_nu(self, p_nu):
        """
        nu = n(n+5)/2 + tanh(p_nu) * n(n+3)/2 + 1  (MERN activation; > n+1).
        For n=2: base = 7, span = 5, so nu in (7-5, 7+5)+... => nu in (3, 13),
        always > n+1 = 3. We add a tiny epsilon to stay strictly above the pole.
        Also cap nu (Meinert warns nu->inf is numerically unstable).
        """
        n = self.n
        base = n * (n + 5) / 2.0            # 7 for n=2
        span = n * (n + 3) / 2.0            # 5 for n=2
        nu = base + torch.tanh(p_nu) * span + 1.0
        return nu.clamp(min=n + 1.0 + 1e-3, max=1.0e3)


# ---------------------------------------------------------------------------
# Loss: multivariate NIW negative log-likelihood (Meinert Eq. 13).
# ---------------------------------------------------------------------------
def niw_nll(y, mu0, L, nu, r: float = 1.0):
    """
    Multivariate NIW NLL (Meinert 2021, Eq. 13), per sample.

      L_i = logGamma((nu - n + 1)/2) - logGamma((nu + 1)/2)
            + (n/2) log(r + nu)
            - nu * sum_j log L[j,j]                       (= (nu/2) log|LL^T| term,
                                                            since log|LL^T| = 2 sum log L_jj)
            + (nu + 1)/2 * log | L L^T + (1/(r+nu)) (y-mu0)(y-mu0)^T |

    Notes
    -----
    * We use the kappa = nu/r coupling, so the factor that in the general form
      is (1+kappa)/kappa * ... collapses: Meinert's Eq. (13) already absorbs the
      coupling and writes the data term with 1/(r+nu). We follow Eq. (13)
      verbatim.
    * The `- nu * sum_j log L_jj` term is the Eq. (13) `- nu * sum_j ell_j`
      written in terms of L's diagonal (ell_j = log L_jj by construction).
    * const (n/2 * log pi etc.) dropped: constant in the parameters.

    Shapes: y,(B,n)  mu0,(B,n)  L,(B,n,n)  nu,(B,)
    """
    n = y.shape[-1]
    diff = (y - mu0).unsqueeze(-1)                       # (B, n, 1)
    Sig0 = torch.matmul(L, L.transpose(-1, -2))          # (B, n, n) = L L^T

    # data-scaled inner matrix:  M = L L^T + 1/(r+nu) (y-mu0)(y-mu0)^T
    r_nu = (r + nu).clamp(min=1e-6)                      # (B,)
    outer = torch.matmul(diff, diff.transpose(-1, -2))   # (B, n, n)
    M = Sig0 + outer / r_nu.view(-1, 1, 1)

    logdet_M = torch.logdet(M)                           # (B,)
    # log|LL^T| = 2 * sum_j log L_jj ; the Eq.(13) term is -nu * sum_j log L_jj
    log_diag_L = torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12))
    sum_log_diag = log_diag_L.sum(-1)                    # (B,)

    nll = (
        torch.lgamma((nu - n + 1) / 2.0)
        - torch.lgamma((nu + 1) / 2.0)
        + (n / 2.0) * torch.log(r_nu)
        - nu * sum_log_diag
        + (nu + 1) / 2.0 * logdet_M
    )
    return nll


def evidential_loss(y, mu0, L, nu, r: float = 1.0, lam: float = 0.0):
    """
    Total loss. With the nu=r*kappa coupling the evidence regularizer is
    unnecessary (Meinert Sec. 3): default lam=0.0 -> loss = mean NLL.

    `lam` is kept in the signature for API compatibility with the old NIG
    trainer; if a nonzero lam is passed we add the multivariate analogue of the
    Amini regularizer |y-mu0| * total_evidence, total_evidence = kappa + nu.
    """
    nll = niw_nll(y, mu0, L, nu, r=r)
    if lam and lam > 0.0:
        kappa = nu / r
        total_evidence = kappa + nu                      # Meinert Phi' = kappa+nu
        reg = torch.linalg.norm(y - mu0, dim=-1) * total_evidence
        loss = (nll + lam * reg).mean()
        return loss, nll.mean(), reg.mean()
    return nll.mean(), nll.mean(), torch.zeros((), device=y.device)


# ---------------------------------------------------------------------------
# Uncertainty decomposition (Meinert Eq. 14) — full covariance matrices.
# ---------------------------------------------------------------------------
def niw_uncertainties(L, nu):
    """
    Return (aleatoric_cov, epistemic_cov), each (B, n, n).

      E[Sigma]  = nu / (nu - n - 1) * L L^T        (aleatoric covariance)
      Var[mu]   = E[Sigma] / nu                    (epistemic covariance)

    These are tilted 2x2 ellipses. For scalar summaries (e.g. an SNR table or an
    error-vs-epistemic Spearman check) reduce each matrix with a scale like
    sqrt(trace) or sqrt(det)** (see niw_scalar_std below).
    """
    n = L.shape[-1]
    Sig0 = torch.matmul(L, L.transpose(-1, -2))          # (B, n, n)
    denom = (nu - n - 1).clamp(min=1e-3).view(-1, 1, 1)
    aleatoric = nu.view(-1, 1, 1) / denom * Sig0
    epistemic = aleatoric / nu.view(-1, 1, 1)
    return aleatoric, epistemic


def niw_total_cov(L, nu):
    """Total predictive covariance = aleatoric + epistemic (B, n, n)."""
    al, ep = niw_uncertainties(L, nu)
    return al + ep


def niw_scalar_std(cov, mode: str = "trace"):
    """
    Collapse a (B,n,n) covariance to a (B,) scalar 'std magnitude' in the same
    units as position (meters, once scaled), for ranking / SNR-table use.

      mode='trace' : sqrt(trace(cov))   ~ total spread (isotropic-equivalent std)
      mode='det'   : det(cov)**(1/(2n)) ~ geometric-mean std (ellipse 'radius')

    'trace' matches the old sqrt(al+ep) magnitude convention most closely.
    """
    if mode == "det":
        n = cov.shape[-1]
        return torch.det(cov).clamp(min=1e-24) ** (1.0 / (2 * n))
    return torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1).clamp(min=1e-24))


# ---------------------------------------------------------------------------
# Backward-compatibility shim so old scripts importing NIG names still run.
# ---------------------------------------------------------------------------
def nig_uncertainties(*args, **kwargs):  # pragma: no cover - guard
    raise RuntimeError(
        "nig_uncertainties() is from the retired Amini NIG head. Use "
        "niw_uncertainties(L, nu) which returns full 2x2 covariances, or "
        "niw_total_cov(L, nu) + niw_scalar_std(cov) for a scalar magnitude."
    )
