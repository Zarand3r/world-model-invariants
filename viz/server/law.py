"""Scoring a conserved quantity, and the null it has to beat.

`C = sum_i a_i phi_i` over the fitted basis. Because a bundle stores `G = Phi(Z) @ basis^T`, moving
the weights is `G @ a` — so every score here is a few matrix products on an (N, 8) array and runs in
microseconds. This is the file that makes hand-steering the invariant feel like a slider rather than
a job.

**The pairing residual is the expensive one and still cheap.** Given weights it solves the same
least squares the fit's B-step solves, over the antisymmetric basis of the extracted subspace: at
r = 12 that is a (N*r, 66) system, tens of milliseconds. It is included because |rho|_E alone cannot
tell a conserved quantity from one that merely tracks energy — the paper's damped models score
|rho|_E up to 0.09 while their drift is three orders of magnitude worse.
"""
import numpy as np
import torch

from latent_noether.extraction import Bundle, drift_of_C, rho_energy
from latent_noether.hamiltonian_select import _antisym_basis
from latent_noether.polynomial import monomial_features


def _grads(b: Bundle) -> np.ndarray:
    """grad of each basis invariant at every analysis state. (k, N, r), computed once per bundle."""
    Z = torch.as_tensor(b.Z, dtype=torch.float64).reshape(-1, b.Z.shape[-1])
    zz = Z.detach().requires_grad_(True)
    feats = monomial_features(zz, b.degree)
    out = []
    for c in b.basis_coeffs:
        g, = torch.autograd.grad((feats @ torch.as_tensor(c)).sum(), zz, retain_graph=True)
        out.append(g.detach().numpy())
    return np.stack(out)


def basis_grads(b: Bundle) -> np.ndarray:
    """Cached on the bundle itself — see the note on `Bundle.basis_grads`."""
    if b.basis_grads is None:
        b.basis_grads = _grads(b)
    return b.basis_grads


def pairing_residual(b: Bundle, weights: np.ndarray) -> float:
    """min_B ||f - B grad C||^2 / ||f||^2 over antisymmetric B. 0 = the flow is exactly Hamiltonian."""
    r = b.Z.shape[-1]
    Zt = torch.as_tensor(b.Z, dtype=torch.float64)
    # the flow is not stored; it is recoverable only from the model, so score against the *fitted*
    # displacement implied by the bundle's own basis — see note in `scores`.
    gradC = torch.as_tensor(np.einsum("k,knd->nd", weights, basis_grads(b)))
    basis = _antisym_basis(r, torch.float64)
    A = torch.stack([gradC @ bb.T for bb in basis], dim=-1)
    f = torch.as_tensor(b.flow.reshape(-1, r), dtype=torch.float64)
    beta = torch.linalg.lstsq(A.reshape(-1, A.shape[-1]), f.reshape(-1)).solution
    return float(((f - (A @ beta)) ** 2).sum() / (f ** 2).sum())


def scores(b: Bundle, weights: np.ndarray | None = None) -> dict:
    """All three statistics for one set of mixing weights.

    **The weights are deliberately NOT normalised.** An earlier version divided by the norm, which
    measured as an exact no-op: every statistic here is invariant to the scale AND the sign of C.
    Correlation is scale-free; drift is a ratio of variances; and the pairing residual fits `B` by
    least squares, so `B` absorbs any rescaling of `grad C`. Checked at scales 1.0, 3.7, 0.02 and
    -1.0 — identical to ten decimals. The normalisation also returned weights that differed from the
    ones the caller sent, and silently substituted the fitted weights when the norm underflowed,
    which hid a degenerate request instead of reporting it.
    """
    w = b.weights if weights is None else np.asarray(weights, dtype=np.float64)
    check_weights(b, w)
    return {"rho_energy": rho_energy(b, w), "drift_of_C": drift_of_C(b, w),
            "pairing_residual": pairing_residual(b, w), "weights": w.tolist()}


def check_weights(b: Bundle, w: np.ndarray) -> None:
    """Fail with something a caller can act on, before numpy fails with something they cannot.

    A wrong-length vector used to surface as `matmul: Input operand 1 has a mismatch in its core
    dimension 0` behind a 500.
    """
    k = b.basis_coeffs.shape[0]
    if w.shape != (k,):
        raise ValueError(f"expected {k} mixing weights, got {tuple(w.shape)}")
    if not np.all(np.isfinite(w)):
        raise ValueError("mixing weights must all be finite")
    if not np.any(w):
        raise ValueError("mixing weights are all zero, which defines no invariant")


def random_weights(b: Bundle, draw: int) -> np.ndarray:
    """A matched-norm random direction inside the SAME conserved basis.

    This is the bench's cheap null, and it is deliberately weaker than the paper's: the published
    arm B draws a random polynomial over the whole degree-4 monomial basis, not a random mixture of
    already-conserved candidates. A random mixture is a much harder null to beat, so treat a good
    score against it as evidence and a bad one as fatal — never the reverse. `nulls.py` serves the
    published arm from the committed logs for the comparison that goes in a figure.
    """
    g = np.random.default_rng(1000 + draw)
    v = g.standard_normal(b.basis_coeffs.shape[0])
    return v / np.linalg.norm(v)


def scatter(b: Bundle, weights: np.ndarray | None = None, stride: int = 7) -> dict:
    """C against true energy, thinned for the browser. Full arrays are ~5700 points."""
    Cfull = b.C(weights)
    return {"C": Cfull.ravel()[::stride].tolist(),
            "E": b.energy.ravel()[:Cfull.size][::stride].tolist()}
