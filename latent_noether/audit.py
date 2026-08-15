"""The invariance audit: which readouts survive a change of latent coordinates?

The central claim of this project is that a frozen model's latent coordinate system is
unfixed, so a structural claim about the model is admissible only if its readout is
invariant under reparameterization. This module turns that from an argument into a
measurement: apply random invertible changes of coordinates to the latents, re-run every
instrument, and tabulate what moves.

Transformation laws (see docs/THEORY.md §1). Under `z → Az`:

    vector field  g → A g          gradient  ∇C → A⁻ᵀ ∇C
    bivector      B → A B Aᵀ       Jacobian  J → A J A⁻¹

so any readout built only from conjugation-invariants (eigenvalues, ranks), from scalars
(`∇Cᵀ B ∇C`), or from quantities decoded through the model's own readout should be flat —
while anything measured with a Euclidean norm in latent coordinates, or expressed as
coefficients in a fixed basis, should move.

`audit` takes a dict of named callables `readout(Z, A) -> float` and reports the relative
spread of each across sampled `A`. The prediction is that the spreads partition cleanly
by invariance class; a readout that lands on the wrong side of that partition is either
mis-derived or measuring something other than what it claims.
"""
import torch


def random_gl(d: int, generator: torch.Generator, strength: float = 1.0,
              dtype=None) -> torch.Tensor:
    """A well-conditioned random element of GL(d).

    Biased toward the identity so the transformed latents stay in a numerically sane
    range: a near-singular `A` would make every readout look unstable for reasons that
    have nothing to do with invariance.
    """
    dtype = dtype or torch.get_default_dtype()
    A = torch.eye(d, dtype=dtype) + strength * torch.randn(d, d, generator=generator,
                                                           dtype=dtype)
    # reject ill-conditioned draws rather than silently reporting their noise
    for _ in range(50):
        if float(torch.linalg.cond(A)) < 20.0:
            return A
        A = torch.eye(d, dtype=dtype) + 0.5 * strength * torch.randn(
            d, d, generator=generator, dtype=dtype)
    raise RuntimeError("could not draw a well-conditioned A")


def relative_spread(values: list[float]) -> float:
    """Scale-free dispersion: (max − min) / |median|, or (max − min) when the median is
    ~0. Used because readouts live on different scales and only their *stability* is
    being compared."""
    v = torch.tensor(values, dtype=torch.get_default_dtype())
    med = v.median().abs()
    return float((v.max() - v.min()) / med) if med > 1e-9 else float(v.max() - v.min())


def audit(readouts: dict, Z: torch.Tensor, n_draws: int = 6, seed: int = 0,
          strength: float = 0.6) -> dict:
    """Run every readout under `n_draws` random changes of latent coordinates.

    readouts: name -> callable(Z_transformed, A) -> float. Each callable receives the
    transformed latents *and* the transformation, so a correctly-derived readout can
    transport its own fitted objects rather than silently refitting in a new gauge —
    refitting with a coordinate-dependent regularizer reintroduces a preferred gauge and
    is a common way to fake invariance.

    Returns per-readout baseline value, values across draws, and relative spread.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be (n, d), got {tuple(Z.shape)}")
    d = Z.shape[1]
    g = torch.Generator().manual_seed(seed)
    mats = [torch.eye(d, dtype=Z.dtype)] + [random_gl(d, g, strength, Z.dtype)
                                            for _ in range(n_draws)]
    out = {}
    for name, fn in readouts.items():
        vals, errors = [], []
        for A in mats:
            try:
                vals.append(float(fn(Z @ A.T, A)))
            except Exception as e:                      # surface, never swallow
                errors.append(f"{type(e).__name__}: {e}")
        if not vals:
            out[name] = {"error": errors[0] if errors else "no values"}
            continue
        out[name] = {"baseline": vals[0], "values": vals,
                     "relative_spread": relative_spread(vals),
                     "n_failed": len(errors)}
    return out


def random_diffeomorphism(d: int, generator: torch.Generator, strength: float = 0.3,
                          dtype=None):
    """A smooth, exactly invertible NONLINEAR change of latent coordinates.

    Uses an additive coupling layer (RealNVP-style): the first half of the coordinates is
    passed through untouched, and the second half is shifted by a smooth function of the
    first. Inversion is exact and closed-form — no optimization, no approximation — which
    matters because an approximate inverse would contaminate the measurement it is meant to
    make.

    This closes the audit's scope limitation. Linear draws test `GL(d)` invariance; a
    diffeomorphism is what actually raises polynomial degree, so it is the only way to test
    whether a degree-bounded family is gauge-invariant in the full sense.
    """
    dtype = dtype or torch.get_default_dtype()
    h = d // 2
    if h == 0:
        raise ValueError("need at least 2 dimensions for a coupling map")
    W = strength * torch.randn(h, h, generator=generator, dtype=dtype)
    b = strength * torch.randn(h, generator=generator, dtype=dtype)

    def fwd(z):
        a, c = z[..., :h], z[..., h:]
        return torch.cat([a, c + torch.tanh(a @ W.T + b)], dim=-1)

    def inv(z):
        a, c = z[..., :h], z[..., h:]
        return torch.cat([a, c - torch.tanh(a @ W.T + b)], dim=-1)

    fwd.inverse = inv
    return fwd


def audit_nonlinear(readouts: dict, Z: torch.Tensor, n_draws: int = 4, seed: int = 0,
                    strength: float = 0.3) -> dict:
    """As `audit`, but with nonlinear coordinate changes.

    Readouts receive `(Z_transformed, phi)` where `phi` is the map and `phi.inverse` its
    exact inverse — so a readout can transport its fitted objects by composing with
    `phi.inverse` rather than refitting in the new chart.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be (n, d), got {tuple(Z.shape)}")
    d = Z.shape[1]
    g = torch.Generator().manual_seed(seed)
    identity = lambda z: z
    identity.inverse = lambda z: z
    maps = [identity] + [random_diffeomorphism(d, g, strength, Z.dtype)
                         for _ in range(n_draws)]
    out = {}
    for name, fn in readouts.items():
        vals, errors = [], []
        for phi in maps:
            try:
                vals.append(float(fn(phi(Z), phi)))
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        if not vals:
            out[name] = {"error": errors[0] if errors else "no values"}
            continue
        out[name] = {"baseline": vals[0], "values": vals,
                     "relative_spread": relative_spread(vals), "n_failed": len(errors)}
    return out


def polynomial_coupling(d: int, generator: torch.Generator, degree: int = 2,
                        strength: float = 0.3, dtype=None):
    """An exactly invertible NONLINEAR change of coordinates that is *polynomial*.

    `random_diffeomorphism` uses tanh, so a polynomial invariant pushed through it is no
    longer polynomial at any finite degree — fine for testing invariance, useless for
    calibration. This map is an additive coupling with a homogeneous degree-`degree`
    polynomial shift:

        forward   (a, c) -> (a, c + (a**degree) W^T)
        inverse   (a, c) -> (a, c - (a**degree) W^T)

    Both directions are polynomial of degree `degree`, so the effect on an invariant is
    predictable in closed form: a degree-2 invariant becomes degree `2*degree`. That known
    answer is what lets `k*` be calibrated rather than merely sanity-checked.
    """
    dtype = dtype or torch.get_default_dtype()
    h = d // 2
    if h == 0:
        raise ValueError("need at least 2 dimensions for a coupling map")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    W = strength * torch.randn(d - h, h, generator=generator, dtype=dtype)

    def fwd(z):
        a, c = z[..., :h], z[..., h:]
        return torch.cat([a, c + (a ** degree) @ W.T], dim=-1)

    def inv(z):
        a, c = z[..., :h], z[..., h:]
        return torch.cat([a, c - (a ** degree) @ W.T], dim=-1)

    fwd.inverse = inv
    return fwd
