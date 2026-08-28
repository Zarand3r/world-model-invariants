"""Closed-form polynomial invariant discovery must, on systems with known answers:
recover the exact invariant as a printable expression, count independent invariants via
the eigenvalue spectrum, refuse when none exists, and reproduce the degree at which each
invariant becomes expressible."""
import numpy as np
import pytest
import torch

from latent_noether.envs import (CentralForceParams, OscillatorParams,
                                 sample_central_initial_conditions,
                                 sample_initial_conditions, simulate, simulate_central)
from latent_noether.polynomial import (monomial_names, monomial_features,
                                       polynomial_invariants, predicted_invariant_counts)


def _as_t(x):
    return torch.as_tensor(x, dtype=torch.get_default_dtype())


def test_monomial_features_and_names_agree():
    z = torch.randn(5, 3)
    feats = monomial_features(z, degree=2)
    names = monomial_names(3, degree=2, var_names=["a", "b", "c"])
    assert feats.shape == (5, len(names))
    assert "a" in names and "a*b" in names and "a^2" in names
    # no constant term: a constant is trivially invariant and would be degenerate
    assert "1" not in names
    i = names.index("a*b")
    torch.testing.assert_close(feats[:, i], z[:, 0] * z[:, 1])


def test_recovers_angular_momentum_on_central_force():
    """L = x*vy - y*vx is exactly degree 2, so it must appear at the quadratic tier."""
    rng = np.random.default_rng(0)
    traj = simulate_central(CentralForceParams(eps=0.6),
                            sample_central_initial_conditions(64, rng), 200)
    res = polynomial_invariants(_as_t(traj), degree=2,
                                var_names=["x", "y", "vx", "vy"])
    best = res[0]
    assert best["ratio"] < 1e-4
    # the recovered coefficient vector must be parallel to L's
    names = best["names"]
    target = torch.zeros(len(names))
    target[names.index("x*vy")] = 1.0
    target[names.index("y*vx")] = -1.0
    c = torch.as_tensor(best["coeffs"], dtype=target.dtype)
    cos = abs(float((c @ target) / (c.norm() * target.norm())))
    assert cos > 0.99, f"recovered {best['expression']}"


def test_recovers_energy_on_harmonic_oscillator():
    """H = p^2/2 + w^2 q^2/2 is degree 2 in (q, p)."""
    rng = np.random.default_rng(0)
    w = 1.3
    traj = simulate(OscillatorParams(omega=w), sample_initial_conditions(64, rng), 300)
    res = polynomial_invariants(_as_t(traj), degree=2, var_names=["q", "p"])
    best = res[0]
    assert best["ratio"] < 1e-6
    names, c = best["names"], torch.as_tensor(best["coeffs"])
    ratio = float(c[names.index("q^2")] / c[names.index("p^2")])
    assert abs(ratio - w ** 2) < 1e-2      # H ∝ p^2 + w^2 q^2


def test_refuses_on_damped_oscillator():
    """No exact invariant exists; every eigenvalue must stay well away from zero."""
    rng = np.random.default_rng(0)
    init = sample_initial_conditions(64, rng)
    good = polynomial_invariants(_as_t(simulate(OscillatorParams(), init, 300)), degree=2)
    damped = polynomial_invariants(
        _as_t(simulate(OscillatorParams(zeta=0.1), init, 300)), degree=2)
    assert damped[0]["ratio"] > 100 * good[0]["ratio"]
    assert damped[0]["ratio"] > 1e-2


def test_counts_independent_invariants_via_spectrum_gap():
    """Central force has two independent invariants (energy, L); at degree 2 only L is
    expressible, so exactly one eigenvalue should be near zero."""
    rng = np.random.default_rng(0)
    traj = simulate_central(CentralForceParams(eps=0.6),
                            sample_central_initial_conditions(64, rng), 200)
    res = polynomial_invariants(_as_t(traj), degree=2)
    ratios = [r["ratio"] for r in res]
    assert ratios[0] < 1e-4 < ratios[1]      # exactly one, with a clear gap


def test_quartic_tier_finds_the_anharmonic_energy():
    """E = v^2/2 + r^2/2 + eps*r^4/4 is degree 4: invisible at degree 2, found at 4."""
    rng = np.random.default_rng(0)
    traj = simulate_central(CentralForceParams(eps=0.6),
                            sample_central_initial_conditions(48, rng), 150)
    res = polynomial_invariants(_as_t(traj), degree=4)
    small = [r for r in res if r["ratio"] < 1e-4]
    assert len(small) >= 2, [r["ratio"] for r in res[:4]]


def test_predicted_counts_from_jacobian_spectrum():
    """Linear invariants <-> left eigenvectors with lambda = 1; quadratic <-> eigenvalue
    pairs with lambda_i * lambda_j = 1 (a conjugate pair on the unit circle gives one)."""
    th = torch.tensor(0.4)
    c, s = torch.cos(th), torch.sin(th)
    rot = torch.tensor([[c, -s], [s, c]])                  # |lambda| = 1 pair
    assert predicted_invariant_counts(rot)["quadratic"] >= 1
    assert predicted_invariant_counts(rot)["linear"] == 0

    contract = 0.9 * rot                                    # damped: no invariant
    counts = predicted_invariant_counts(contract)
    assert counts["quadratic"] == 0 and counts["linear"] == 0

    shift = torch.tensor([[1.0, 0.0], [0.0, 0.5]])          # one linear invariant
    assert predicted_invariant_counts(shift)["linear"] == 1


def test_heldout_ratio_exposes_high_degree_overfitting():
    """The in-sample ratio is untrustworthy at high degree: a rich basis can always fit a
    'conserved' combination on the trajectories it saw. Held-out scoring catches it."""
    from latent_noether.polynomial import validated_invariants
    rng = np.random.default_rng(0)
    traj = _as_t(simulate_central(CentralForceParams(eps=3.0),
                                  sample_central_initial_conditions(64, rng), 120))
    low = validated_invariants(traj, degree=2)[0]
    high = validated_invariants(traj, degree=8)[0]
    # the genuine degree-2 invariant survives out of sample
    assert low["heldout_ratio"] < 1e-8
    # degree 8 looks perfect in-sample and is orders of magnitude worse held out
    assert high["ratio"] < 1e-9
    assert high["heldout_ratio"] > 100 * low["heldout_ratio"]


def test_validated_invariants_requires_a_holdout():
    from latent_noether.polynomial import validated_invariants
    rng = np.random.default_rng(0)
    traj = _as_t(simulate_central(CentralForceParams(), sample_central_initial_conditions(4, rng), 60))
    with pytest.raises(ValueError):
        validated_invariants(traj, degree=2, n_fit=4)


def test_validated_invariants_supports_trig_basis():
    """The held-out design must be built with the SAME basis as the fit. Scoring a
    trig-augmented coefficient vector against a monomial-only design is a size mismatch
    that crashes -- and the pendulum's energy needs cos(theta), so this path is real."""
    from latent_noether.polynomial import validated_invariants

    rng = np.random.default_rng(0)
    traj = _as_t(simulate(OscillatorParams(), sample_initial_conditions(32, rng), 200))
    results = validated_invariants(traj, degree=2, include_trig=True)
    assert results and all("heldout_ratio" in r for r in results)
    assert results[0]["heldout_ratio"] < 1e-2       # energy is genuinely conserved here


def test_validated_invariants_exposes_overfitting_the_fit_ratio_hides():
    """A basis rich enough to fit anything must be caught by held-out scoring: the
    in-sample ratio understates the true one. This is the check that voided the
    separatrix result, so it is the check that must not regress."""
    from latent_noether.polynomial import validated_invariants

    rng = np.random.default_rng(1)
    traj = _as_t(simulate(OscillatorParams(), sample_initial_conditions(16, rng), 120))
    hi = validated_invariants(traj, degree=8)[0]
    assert hi["heldout_ratio"] >= hi["ratio"]        # held-out is never the optimistic one


def test_returned_coefficients_live_in_RAW_monomial_space():
    """`polynomial_invariants` fits in standardised feature space and must convert back.

    The eigenproblem standardises columns because raw degree-4 monomials span many orders of
    magnitude, then returns `c_raw = c_std / sd`. **Every consumer in `scripts/` applies these
    coefficients to raw `monomial_features`.** If the conversion were dropped, no error would be
    raised anywhere -- the numbers would silently change, in every experiment at once.

    This is not hypothetical. On 2026-08-28 a throwaway probe applied returned coefficients to
    *standardised* features and reported |rho(C, E)| = 0.58 for a model whose true value is 0.97,
    which read as evidence of a problem with the paper until the probe itself was found to be wrong.
    Nothing in the suite pinned the convention.

    The check: reconstructing C from the returned coefficients on raw features must reproduce the
    reported invariance ratio.
    """
    torch.manual_seed(0)
    n_traj, T = 24, 40
    t = torch.linspace(0.0, 6.0, T)
    amp = torch.rand(n_traj, 1) * 1.5 + 0.5
    phase = torch.rand(n_traj, 1) * 6.283
    traj = torch.stack([amp * torch.cos(t + phase), -amp * torch.sin(t + phase)], dim=-1).double()

    res = polynomial_invariants(traj, degree=2, max_results=1)[0]
    c = torch.as_tensor(res["coeffs"], dtype=traj.dtype)

    feats = monomial_features(traj.reshape(-1, traj.shape[-1]), 2)      # RAW, as consumers use
    C = (feats @ c).reshape(n_traj, T)

    within = C.var(dim=1).mean()
    total = C.reshape(-1).var()
    ratio = float(within / total)

    assert ratio == pytest.approx(res["ratio"], abs=1e-6, rel=1e-3), (
        f"reported ratio {res['ratio']:.3e} but raw-space reconstruction gives {ratio:.3e}; "
        "the returned coefficients are no longer in raw monomial space"
    )
    assert ratio < 1e-6, f"harmonic energy should be near-perfectly conserved, got {ratio:.3e}"
