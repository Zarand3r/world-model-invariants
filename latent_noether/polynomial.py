"""Closed-form invariant discovery.

An invariant is a scalar constant along trajectories but varying across them. For a
candidate restricted to a polynomial family, `C(z) = cᵀφ(z)`, our invariance ratio

    within-trajectory variance / total variance = cᵀWc / cᵀTc

is a Rayleigh quotient — so minimizing it is a GENERALIZED EIGENVALUE PROBLEM, not an
optimization. One eigendecomposition returns the entire ranked family at once: each
near-zero eigenvalue is an independent invariant, each eigenvector is a printable
polynomial, and the gap to the remaining eigenvalues is a principled detection
threshold. For a 4-D state at degree 2 this is a 15×15 problem — microseconds.

This is the low-capacity rung of the capacity ladder: if a quadratic form already
captures the invariant, an MLP that "finds" it was not reading out structure the model
holds so much as constructing it.
"""
import itertools

import torch


def _exponent_tuples(d: int, degree: int) -> list[tuple[int, ...]]:
    """All exponent vectors with 1 <= total degree <= `degree`. The constant term is
    excluded on purpose: it is trivially invariant and makes the problem degenerate."""
    out = []
    for total in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), total):
            exps = [0] * d
            for i in combo:
                exps[i] += 1
            out.append(tuple(exps))
    return out


_EXP_CACHE: dict = {}


def monomial_features(z: torch.Tensor, degree: int) -> torch.Tensor:
    """z: (..., d) -> (..., n_features) monomials of total degree 1..degree.

    Vectorised over the exponent table rather than looping in Python. At degree 4 in 12 dimensions
    the table has 1819 rows, and the loop version cost 48 ms per call on GPU against 0.014 ms here.
    That matters because the intervention rollouts call this twice per step: the A2c confirmation
    went from an estimated 20 GPU-hours to minutes.

    Powers are built by repeated MULTIPLICATION and gathered, not by `z ** e` with a float exponent.
    The first vectorised attempt put the exponent table in `z.dtype`, which sends `z ** 4.0` through
    exp/log; on negative coordinates that loses enough precision to raise the joint-fit residual on
    a clean Hamiltonian flow from below 1e-3 to 8.7e-02, and three tests caught it. Integer powers
    by multiplication are exact and just as fast.
    """
    d = z.shape[-1]
    key = (d, degree, z.device)
    E = _EXP_CACHE.get(key)
    if E is None:
        E = torch.tensor(_exponent_tuples(d, degree), dtype=torch.long, device=z.device)
        _EXP_CACHE[key] = E
    powers = [torch.ones_like(z)]
    for _ in range(degree):
        powers.append(powers[-1] * z)
    Pw = torch.stack(powers, dim=-2)                       # (..., degree+1, d)
    idx = torch.arange(d, device=z.device)
    return Pw[..., E, idx].prod(dim=-1)                    # (..., n_features)


def monomial_names(d: int, degree: int, var_names: list[str] | None = None) -> list[str]:
    names = var_names or [f"z{i + 1}" for i in range(d)]
    out = []
    for exps in _exponent_tuples(d, degree):
        parts = [names[i] if e == 1 else f"{names[i]}^{e}" for i, e in enumerate(exps) if e]
        out.append("*".join(parts))
    return out


def _expression(coeffs: torch.Tensor, names: list[str], tol: float = 1e-3) -> str:
    """Human-readable polynomial, normalized to its largest coefficient."""
    c = coeffs / coeffs.abs().max()
    terms = []
    for v, n in zip(c.tolist(), names):
        if abs(v) < tol:
            continue
        sign = "-" if v < 0 else "+"
        mag = abs(v)
        body = n if abs(mag - 1.0) < tol else f"{mag:.3g}*{n}"
        terms.append(f"{sign} {body}")
    if not terms:
        return "0"
    text = " ".join(terms)
    return text[2:] if text.startswith("+ ") else text


def design_features(z: torch.Tensor, degree: int,
                    include_trig: bool = False) -> torch.Tensor:
    """The design matrix the invariant search actually fits on.

    Shared by the fit and by held-out scoring, so a coefficient vector is always
    evaluated against the same basis that produced it — building the two separately is
    how `validated_invariants` silently disagreed with `polynomial_invariants` under
    `include_trig`.
    """
    if not include_trig:
        return monomial_features(z, degree)
    # needed for energies like the pendulum's H = w^2/2 + (g/L)(1 - cos(theta))
    return torch.cat([monomial_features(z, degree), torch.sin(z), torch.cos(z)], dim=-1)


def polynomial_invariants(traj: torch.Tensor, degree: int = 2, ridge: float = 1e-8,
                          var_names: list[str] | None = None,
                          max_results: int = 8, include_trig: bool = False) -> list[dict]:
    """Find invariants of the trajectories in closed form.

    traj: (n_traj, T, d) states. Returns results sorted best-first, each with
    `ratio` (the invariance ratio in [0, 1]; ~0 means conserved), `coeffs`, `names`,
    and `expression`.
    """
    if traj.ndim != 3:
        raise ValueError(f"traj must be (n_traj, T, d), got {tuple(traj.shape)}")
    n_traj, T, d = traj.shape
    phi = design_features(traj, degree, include_trig)           # (n_traj, T, F)
    flat = phi.reshape(-1, phi.shape[-1])

    # standardize features: raw monomials differ by orders of magnitude in scale
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-12)
    phi = (phi - mu) / sd
    flat = phi.reshape(-1, phi.shape[-1])

    # W: mean within-trajectory covariance. T_: total covariance.
    centered = phi - phi.mean(dim=1, keepdim=True)
    W = torch.einsum("ntf,ntg->fg", centered, centered) / (n_traj * T)
    fc = flat - flat.mean(0)
    T_ = fc.T @ fc / flat.shape[0]
    eye = torch.eye(W.shape[0], dtype=W.dtype)
    T_ = T_ + ridge * torch.diagonal(T_).mean() * eye

    # generalized symmetric eigenproblem W c = lam T c, via Cholesky of T
    # Degenerate latents (a collapsed or untrained representation) make the monomial
    # features linearly dependent and T_ singular. Escalate the ridge rather than crashing.
    used_ridge = ridge
    for _ in range(12):
        try:
            L = torch.linalg.cholesky(T_)
            break
        except torch.linalg.LinAlgError:
            T_ = T_ + used_ridge * torch.diagonal(T_).mean() * eye
            used_ridge *= 10.0
    else:
        raise torch.linalg.LinAlgError(
            "feature covariance stayed singular; the data is rank-deficient in this basis")
    Linv = torch.linalg.solve_triangular(L, eye, upper=False)
    M = Linv @ W @ Linv.T
    vals, vecs = torch.linalg.eigh((M + M.T) / 2)

    names = monomial_names(d, degree, var_names)
    if include_trig:
        vn = var_names or [f"z{i + 1}" for i in range(d)]
        names = names + [f"sin({v})" for v in vn] + [f"cos({v})" for v in vn]
    out = []
    for k in range(min(max_results, vals.shape[0])):
        c_std = Linv.T @ vecs[:, k]
        c_raw = c_std / sd                       # undo standardization -> raw monomials
        out.append({
            "ratio": float(vals[k].clamp_min(0.0)),
            "coeffs": c_raw.tolist(),
            "names": names,
            "expression": _expression(c_raw, names),
        })
    return out


def predicted_invariant_counts(A: torch.Tensor, tol: float = 1e-3) -> dict:
    """How many polynomial invariants the linearized dynamics z -> A z admits.

    Linear invariant cᵀz  <=>  Aᵀc = c        (left eigenvector with eigenvalue 1).
    Quadratic invariant zᵀPz <=> AᵀPA = P, whose solutions correspond to eigenvalue
    pairs with λᵢλⱼ = 1 — so every conjugate pair on the unit circle contributes one.

    Run this BEFORE searching: it says how many invariants to expect and at what degree.
    """
    lam = torch.linalg.eigvals(A)
    linear = int(((lam - 1).abs() < tol).sum())
    n = lam.shape[0]
    quadratic = sum(1 for i in range(n) for j in range(i, n)
                    if abs(complex(lam[i] * lam[j]) - 1.0) < tol)
    return {"linear": linear, "quadratic": quadratic,
            "eigenvalue_moduli": lam.abs().tolist()}


def validated_invariants(traj: torch.Tensor, degree: int = 2, n_fit: int | None = None,
                         **kwargs) -> list[dict]:
    """`polynomial_invariants` with an out-of-sample check on disjoint trajectories.

    The in-sample invariance ratio is **not trustworthy at high degree**: a rich enough
    monomial basis can always find a combination that is constant along the trajectories
    it was fitted to. Measured on a 4-D central force, degree 8 scores an in-sample ratio
    of exactly 0 while its held-out ratio is 7.7e-03 — nine orders of magnitude of
    overfitting, and indistinguishable from a genuine invariant if you only look at the
    fit. (The degree-2 result used throughout this project holds up: 1.8e-12 held out.)

    Adds `heldout_ratio` to each result. Prefer it over `ratio` for any reported claim,
    and treat a large `ratio → heldout_ratio` inflation as evidence of overfitting rather
    than of conservation.
    """
    if traj.ndim != 3:
        raise ValueError(f"traj must be (n_traj, T, d), got {tuple(traj.shape)}")
    n = traj.shape[0]
    n_fit = n_fit if n_fit is not None else max(2, n // 2)
    if n_fit >= n:
        raise ValueError(f"need at least one held-out trajectory: n_fit={n_fit}, n={n}")
    fit, test = traj[:n_fit], traj[n_fit:]
    results = polynomial_invariants(fit, degree=degree, **kwargs)
    d = traj.shape[-1]
    feats = design_features(test.reshape(-1, d), degree,
                            kwargs.get("include_trig", False))
    for r in results:
        c = torch.as_tensor(r["coeffs"], dtype=traj.dtype)
        vals = (feats @ c).reshape(test.shape[0], test.shape[1])
        total = vals.reshape(-1).var().clamp_min(1e-30)
        r["heldout_ratio"] = float(vals.var(dim=1).mean() / total)
    return results
