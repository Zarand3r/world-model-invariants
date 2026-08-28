"""Balance-law extraction: from `dC = 0` to `dC = P(z, a)`.

Pre-registered in `docs/F1_PREREG.md`. The free-evolution search finds a scalar whose change under
the transition is small. Under actuation the target is instead a PAIR `(C, P)` with

    C(z_{t+1}) - C(z_t)  ~=  P(z_t, a_t),        P(z, a) = a * (monomials(z) . q)

`P` is linear in the action with a state-dependent coefficient, which is the correct functional form:
true power is `tau * thetadot`, linear in the torque with a coefficient that is a function of state.

**No alternating least squares is needed**, though the prereg allowed for it. Writing
`D = M(z_{t+1}) - M(z_t)` and `R = diag(a) M(z_t)`, the objective is `||D c - R q||^2`. For fixed `c`
the inner `q` is an ordinary least-squares solve, so substituting it back gives

    minimise  || (I - P_R) D c ||^2   subject to ||c|| = 1,

with `P_R` the orthogonal projector onto the column space of `R`. That is a symmetric eigenproblem in
`c` alone: project out everything the action can explain, then find the direction whose remaining
change is smallest. Forcing `q = 0` recovers the free-evolution search exactly, which is what makes
the P3 comparison ("does the balance law beat a mere constant?") a like-for-like one.

The unit-norm constraint on `c` is what rules out the trivial `C = 0`.
"""
from __future__ import annotations

import numpy as np


def _projector_complement(R: np.ndarray, rcond: float = 1e-10):
    """I - R R^+, built from an SVD so a rank-deficient power basis is handled without pseudo-inverse
    blowup. Returns a function applying the projector, never the dense n x n matrix."""
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    keep = s > (s.max() * rcond if s.size and s.max() > 0 else 0.0)
    Uk = U[:, keep]
    return lambda X: X - Uk @ (Uk.T @ X)


def fit_balance_pair(MZ: np.ndarray, MZn: np.ndarray, a: np.ndarray, MP=None, ridge: float = 1e-10):
    """Jointly fit (C, P). MZ/MZn are monomial features at z_t and z_{t+1}; `a` is the action.

    Returns dict with `c`, `q`, the balance residual, and the residual of the best CONSERVED scalar
    on the same data (q forced to zero) for the registered P3 comparison.
    """
    MZ = np.asarray(MZ, float); MZn = np.asarray(MZn, float)
    a = np.asarray(a, float).reshape(-1, 1)
    # POWER BASIS, separate from the C basis. Ground-truth validation (2026-08-28) showed that
    # letting P use the full degree-4 family makes the joint fit DEGENERATE: it drives the residual
    # low while recovering nothing, |rho(C, E)| = 0.51 and |rho(power coef, thetadot)| = 0.07. With
    # a degree-1 power basis the same solver returns |rho(C, shadow H~)| = 1.0000 and
    # |rho(power coef, thetadot)| = 0.9973. `MP` defaults to MZ only to preserve the original call
    # signature; callers should pass a low-degree basis.
    MP = MZ if MP is None else np.asarray(MP, float)
    # Centre the features. The constant monomial is exactly conserved and would otherwise be the
    # trivial minimiser; centring removes it from contention along with any constant offset.
    mu = MZ.mean(0, keepdims=True)
    MZc, MZnc = MZ - mu, MZn - mu
    D = MZnc - MZc
    R = a * (MP - MP.mean(0, keepdims=True))    # power basis: action times state monomials

    # GENERALISED eigenproblem, not an ordinary one. Minimising ||D c|| alone selects whichever
    # direction varies least in absolute terms -- a near-constant combination -- which on ground
    # truth returned |rho(C, E)| = 0.02 and no gain over the conserved-only fit. Constraining
    # c' T c = 1 with T the total covariance makes the objective the *invariance ratio*: change under
    # the transition relative to the spread the scalar actually has. That is the same normalisation
    # the free-evolution search uses, so the P3 comparison stays like-for-like.
    T = MZc.T @ MZc / max(len(MZc), 1) + ridge * np.eye(MZ.shape[1])
    L = np.linalg.cholesky(T)
    Li = np.linalg.inv(L)

    def _smallest(A):
        A = 0.5 * (A + A.T)
        w, V = np.linalg.eigh(Li @ A @ Li.T)
        return Li.T @ V[:, 0], float(w[0])

    Pc = _projector_complement(R)
    c, eig_bal = _smallest(D.T @ Pc(D) / max(len(D), 1))
    q, *_ = np.linalg.lstsq(R, D @ c, rcond=None)
    c_con, eig_con = _smallest(D.T @ D / max(len(D), 1))
    w_bal, w_con = [eig_bal], [eig_con]

    def _resid(cc, qq):
        r = D @ cc - (R @ qq if qq is not None else 0.0)
        # normalised by the spread of C itself, so the number is scale-free in C
        return float(np.median(np.abs(r)) / max(np.std(MZc @ cc), 1e-30))

    return {"c": c, "q": q,
            "residual_balance": _resid(c, q),
            "residual_conserved_only": _resid(c_con, None),
            "c_conserved_only": c_con,
            "eig_balance": float(w_bal[0]), "eig_conserved": float(w_con[0])}
