"""The selection instrument must pick the Hamiltonian out of a family of conserved
quantities. If it cannot do that where the answer is known analytically, its verdict on a
network's latent means nothing.

**Nonlinearity is required, and settling this took three attempts.** The first measurement
said so, from a float32 printout whose own numbers contradicted the ratio it printed. Re-doing
it in float32 suggested the opposite. Only float64 resolves it, because the residuals live near
1e-30 where single precision is pure cancellation noise.

Measured in float64 (float32 cannot settle this — the residuals sit near 1e-30, far
below single precision, and ratios of them are pure noise):

    eps=0.0 (LINEAR)   energy 3.9e-32   L 2.5e-31   discrimination 6.3x     -- NONE
    eps=0.5            energy 9.0e-29   L 1.4e-02   discrimination 1.5e26
    eps=3.0            energy 1.7e-29   L 8.8e-02   discrimination 5.1e27

At eps=0 angular momentum generates the flow exactly as well as energy does, because for a
quadratic C of a LINEAR system `B = A M_C^-1` solves the pairing and is antisymmetric for
both. **Nonlinearity is a requirement of the method**, not a preference.

The separate failure on the harmonic case was candidate SELECTION — the conserved family's top
member is a mixture — which is what `fit_hamiltonian_pair` exists to solve. The two problems
were conflated at first.
"""
import numpy as np
import pytest
import torch

from latent_noether.envs import (CentralForceParams, sample_central_initial_conditions,
                                 simulate_central)
from latent_noether.hamiltonian_select import (fit_hamiltonian_pair,
                                              select_generating_invariant)

# Energy of the anharmonic central force is 0.5|v|^2 + 0.5 r^2 + 0.25 eps r^4 --
# QUARTIC. A degree-2 search cannot represent it, and returns a best pairing
# residual of 0.088 instead of ~1e-9. The search basis must contain the target.
DEGREE = 4


def _central(eps=3.0, n_traj=96, T=150, seed=0):
    """(traj, flow, energy, angular_momentum) for the anharmonic central force.

    The flow is the system's exact vector field, not a finite difference, so the test
    isolates the selection criterion from integration error.
    """
    rng = np.random.default_rng(seed)
    tr = simulate_central(CentralForceParams(eps=eps),
                          sample_central_initial_conditions(n_traj, rng), T)
    x, y, vx, vy = tr[..., 0], tr[..., 1], tr[..., 2], tr[..., 3]
    r2 = x ** 2 + y ** 2
    flow = np.stack([vx, vy, -(1 + eps * r2) * x, -(1 + eps * r2) * y], axis=-1)
    E = 0.5 * (vx ** 2 + vy ** 2) + 0.5 * r2 + 0.25 * eps * r2 ** 2
    L = x * vy - y * vx
    to_t = lambda z: torch.as_tensor(z, dtype=torch.get_default_dtype())
    return to_t(tr), to_t(flow), to_t(E), to_t(L)


def _corr(a, b):
    return abs(float(torch.corrcoef(torch.stack([a.reshape(-1), b.reshape(-1)]))[0, 1]))


def test_selects_energy_not_angular_momentum():
    """THE test. Both E and L are exactly conserved, so conservation cannot distinguish them.
    Only E generates the flow through a constant antisymmetric B, so the pairing must."""
    traj, flow, E, L = _central()
    res = select_generating_invariant(traj, flow, degree=DEGREE, n_candidates=8)
    best = res["best"]
    d = traj.shape[-1]
    from latent_noether.polynomial import monomial_features
    vals = (monomial_features(traj.reshape(-1, d), DEGREE)
            @ torch.as_tensor(best["coeffs"], dtype=traj.dtype)).reshape(traj.shape[:2])
    assert _corr(vals, E) > 0.9, f"selected quantity is not energy: |rho|_E={_corr(vals, E):.3f}"
    assert _corr(vals, E) > _corr(vals, L), "selected L over E"


def test_nonlinearity_is_required_for_discrimination():
    """Pins the requirement, in float64. In float32 these residuals are below precision and
    the ratio is noise — which is how this claim got asserted, retracted, and reinstated."""
    from latent_noether.poisson import _antisymmetric_basis

    def ratio(eps):
        traj, flow, _, _ = _central(eps=eps)
        s = traj.reshape(-1, 4).double()
        f = flow.reshape(-1, 4).double()
        x, y, vx, vy = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        r2 = x ** 2 + y ** 2
        gE = torch.stack([x * (1 + eps * r2), y * (1 + eps * r2), vx, vy], dim=1)
        gL = torch.stack([vy, -vx, -y, x], dim=1)
        basis = _antisymmetric_basis(4).double()

        def res(G):
            A = torch.stack([G @ b.T for b in basis], dim=-1)
            sol = torch.linalg.lstsq(A.reshape(-1, A.shape[-1]), f.reshape(-1)).solution
            return float(((f - (A @ sol)) ** 2).sum() / (f ** 2).sum())
        return res(gL) / max(res(gE), 1e-300)

    assert ratio(0.0) < 1e3, f"linear case should NOT discriminate, got {ratio(0.0):.2e}"
    assert ratio(3.0) > 1e10, f"nonlinear case should discriminate, got {ratio(3.0):.2e}"


def test_joint_fit_beats_rank_and_test():
    """The conserved family's top member is a mixture; the answer is a direction inside the
    subspace. Solving for C and B together must recover energy far better than selecting."""
    traj, flow, E, L = _central()
    from latent_noether.polynomial import monomial_features
    r = fit_hamiltonian_pair(traj, flow, degree=DEGREE, n_basis=8)
    vals = (monomial_features(traj.reshape(-1, 4), DEGREE)
            @ torch.as_tensor(r["coeffs"], dtype=traj.dtype)).reshape(traj.shape[:2])
    assert r["residual"] < r["rank_and_test_residual"] / 100, (
        f'joint {r["residual"]:.2e} vs rank-and-test {r["rank_and_test_residual"]:.2e}')
    assert _corr(vals, E) > 0.99 and _corr(vals, E) > 5 * _corr(vals, L)


def test_the_generating_quantity_is_not_the_most_conserved_one():
    """The instrument only earns its keep if the two rankings differ. If the flow-generating
    quantity were always the top-ranked conserved one, plain invariant search would do."""
    traj, flow, _, _ = _central()
    res = select_generating_invariant(traj, flow, degree=DEGREE, n_candidates=8)
    ratios = [r["ratio"] for r in res["ranked"]]
    pairings = [r["pairing_residual"] for r in res["ranked"]]
    assert min(ratios) < 1e-6, "candidates should include genuinely conserved quantities"
    # the whole family is conserved, but they must NOT all generate the flow
    assert max(pairings) > 10 * min(pairings), (
        f"pairing failed to discriminate: residuals {pairings}")


def test_joint_fit_rejects_a_dissipative_flow():
    """With damping the system is not Hamiltonian, so no conserved quantity generates the
    flow. A method that still reports a low residual is fitting B's free parameters."""
    traj, flow, _, _ = _central()
    damped = flow.clone()
    damped[..., 2:] -= 0.3 * traj[..., 2:]
    clean = fit_hamiltonian_pair(traj, flow, degree=DEGREE)["residual"]
    diss = fit_hamiltonian_pair(traj, damped, degree=DEGREE)["residual"]
    assert diss > 100 * clean, f"dissipative flow not rejected: {diss:.3e} vs {clean:.3e}"


def test_joint_fit_rejects_a_random_flow():
    traj, flow, _, _ = _central()
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(traj.shape, generator=g) * float(flow.std())
    clean = fit_hamiltonian_pair(traj, flow, degree=DEGREE)["residual"]
    rand = fit_hamiltonian_pair(traj, noise, degree=DEGREE)["residual"]
    assert rand > 100 * clean, f"random flow not rejected: {rand:.3e} vs {clean:.3e}"


def test_rejects_mismatched_shapes():
    traj, flow, _, _ = _central()
    with pytest.raises(ValueError):
        select_generating_invariant(traj, flow[:, :-1], degree=DEGREE)
    with pytest.raises(ValueError):
        select_generating_invariant(traj.reshape(-1, 4), flow.reshape(-1, 4), degree=DEGREE)
