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


_grad_cache: dict[int, np.ndarray] = {}


def basis_grads(b: Bundle) -> np.ndarray:
    gid = id(b)
    if gid not in _grad_cache:
        _grad_cache[gid] = _grads(b)
    return _grad_cache[gid]


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
    w = b.weights if weights is None else np.asarray(weights, dtype=np.float64)
    n = float(np.linalg.norm(w))
    w = w / n if n > 1e-30 else b.weights
    return {"rho_energy": rho_energy(b, w), "drift_of_C": drift_of_C(b, w),
            "pairing_residual": pairing_residual(b, w), "weights": w.tolist()}


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
    C = b.C(weights).ravel()[::stride]
    E = b.energy.ravel()[:b.C(weights).size][::stride]
    return {"C": C.tolist(), "E": E.tolist()}
