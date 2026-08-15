"""`k*` must be calibrated before it is allowed to carry a claim.

A degree-bounded search that fails to find an invariant is ambiguous between "the
coordinates are warped" (the Phase-2 hypothesis) and "the instrument is too weak" (risk R7,
and the reason the separatrix experiment went VOID). These tests pin the instrument against
known answers so that ambiguity is bounded rather than assumed away.
"""
import math

import numpy as np
import pytest
import torch

from latent_noether.audit import polynomial_coupling
from latent_noether.envs import OscillatorParams, sample_initial_conditions, simulate
from latent_noether.gauge import decodability, k_star


def _osc(n_traj=48, T=120, seed=0):
    rng = np.random.default_rng(seed)
    traj = simulate(OscillatorParams(), sample_initial_conditions(n_traj, rng), T)
    return torch.as_tensor(traj, dtype=torch.get_default_dtype())


# ---- calibration: does k* measure what it claims? ---------------------------------------

def test_k_star_is_2_in_native_coordinates():
    """The oscillator's energy is quadratic, so an unwarped latent must give k* = 2."""
    k, trace = k_star(_osc(), max_degree=4)
    assert k == 2, f"expected 2, got {k}; trace={trace}"


def test_k_star_tracks_an_injected_warp_of_known_degree():
    """THE calibration test. A degree-m polynomial coupling turns a degree-2 invariant into
    a degree-2m one, in closed form. k* must follow it there -- otherwise a rising k* in
    Phase 2 could not be read as rising gauge complexity."""
    z = _osc()
    phi = polynomial_coupling(2, torch.Generator().manual_seed(0), degree=2, strength=0.25)
    warped = phi(z)
    k, trace = k_star(warped, max_degree=6)
    assert k == 4, f"degree-2 coupling should give k*=4, got {k}; trace={trace}"


def test_k_star_is_flat_under_a_linear_embedding():
    """Risk R3: if k* moved with latent dimension it would confound gauge complexity with
    model size, and the Phase-2 ladder would be uninterpretable. A linear embedding changes
    the dimension while preserving content, and degree-bounded families are GL-invariant
    (Table 1), so k* must not move."""
    z = _osc()
    g = torch.Generator().manual_seed(1)
    base, _ = k_star(z, max_degree=4)
    for d in (4, 6):
        E = torch.linalg.qr(torch.randn(d, 2, generator=g))[0]      # isometric embedding
        k, trace = k_star(z @ E.T, max_degree=4)
        assert k == base, f"dim {d}: k* moved {base} -> {k}; trace={trace}"


def test_k_star_reports_infinity_rather_than_clipping():
    """An undiscoverable invariant must be reported as inf -- PHASE2_PLAN.md reads that as
    *undetermined*, never as evidence. Silently returning max_degree would turn a failure
    into a finding."""
    g = torch.Generator().manual_seed(2)
    noise = torch.randn(20, 60, 3, generator=g)      # no invariant exists
    k, trace = k_star(noise, max_degree=3)
    assert k is math.inf and len(trace) == 3


def test_k_star_trace_reports_the_whole_curve():
    """D_disc is a curve over degree, not a single number: reporting only the crossing would
    hide whether the fit was marginal or decisive."""
    _, trace = k_star(_osc(), max_degree=4)
    assert [r["degree"] for r in trace] == [1, 2]
    assert all("heldout_ratio" in r and "ratio" in r for r in trace)


def test_k_star_rejects_bad_input():
    with pytest.raises(ValueError):
        k_star(torch.randn(10, 5), max_degree=3)          # not (n_traj, T, d)
    with pytest.raises(ValueError):
        k_star(_osc(), max_degree=0)


# ---- decodability ------------------------------------------------------------------------

def test_decodability_recovers_a_linear_target_and_is_held_out():
    z = _osc()
    energy = (z ** 2).sum(dim=-1)                      # not linear in z
    linear = 3.0 * z[..., 0] - 2.0 * z[..., 1]         # exactly linear in z
    assert decodability(z, linear) > 0.99
    assert decodability(z, energy) < 0.5               # a linear probe should fail here


def test_decodability_is_scored_on_unseen_trajectories():
    """An in-sample probe R2 on a wide latent is close to meaningless. Fitting pure noise
    must not produce a high score."""
    g = torch.Generator().manual_seed(3)
    latents = torch.randn(24, 40, 30, generator=g)
    target = torch.randn(24, 40, generator=g)
    assert decodability(latents, target) < 0.3


def test_decodability_rejects_mismatched_shapes():
    z = _osc()
    with pytest.raises(ValueError):
        decodability(z, torch.randn(5, 5))
    with pytest.raises(ValueError):
        decodability(z.reshape(-1, 2), torch.randn(10))


def test_whitening_makes_recovery_invariant_to_reparameterisation():
    """D13: the pairing residual's magnitude is chart-dependent, and on exact dynamics one
    random reparameterisation in three destroyed recovery outright. Whitening canonicalises
    the chart and must restore stability -- this is the fix, so it needs a regression test."""
    from latent_noether.audit import random_gl
    from latent_noether.gauge import whiten
    from latent_noether.hamiltonian_select import fit_hamiltonian_pair
    from latent_noether.polynomial import monomial_features
    from tests.test_hamiltonian_select import _central

    traj, flow, E, _ = _central()
    g = torch.Generator().manual_seed(0)
    rhos = []
    for k in range(3):
        A = (torch.eye(4, dtype=traj.dtype) if k == 0
             else random_gl(4, g, strength=0.6, dtype=traj.dtype))
        Zt, Ft = traj @ A.T, flow @ A.T
        Zw, W = whiten(Zt)
        Fw = Ft @ W
        fit = fit_hamiltonian_pair(Zw, Fw, degree=4, n_basis=8)
        c = torch.as_tensor(fit["coeffs"], dtype=Zw.dtype)
        C = (monomial_features(Zw.reshape(-1, 4), 4) @ c).reshape(Zw.shape[:2])
        n = min(C.shape[1], E.shape[1])
        rhos.append(abs(float(torch.corrcoef(torch.stack(
            [C[:, :n].reshape(-1), E[:, :n].reshape(-1)]))[0, 1])))
    assert min(rhos) > 0.95, f"whitening failed to stabilise recovery: {rhos}"
