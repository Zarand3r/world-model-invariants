"""The Noether pairing must work in ANY dimension and must not depend on the learned
coordinate system.

A symmetry generator g pairs with a conserved quantity C when g = B∇C for some
antisymmetric (Poisson) structure B. Assuming the canonical B = J presupposes canonical
(q,p) coordinates, which learned latents are not — that assumption is what made the old
2-D-only test meaningless in 4-D. Solving for B instead is covariant: under z -> Az the
objects transform as g -> Ag, ∇C -> A⁻ᵀ∇C, B -> ABAᵀ, and B∇C = g is preserved.
"""
import torch

from latent_noether.poisson import noether_pairing_residual


def _canonical_2d(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 2, generator=g)
    energy = lambda q: 0.5 * (q ** 2).sum(-1)
    rotation = lambda q: torch.stack([-q[:, 1], q[:, 0]], dim=1)   # = J grad(E)
    radial = lambda q: q.clone()                                   # not a Hamiltonian field
    return z, energy, rotation, radial


def test_matched_pair_has_zero_residual():
    z, energy, rotation, _ = _canonical_2d()
    out = noether_pairing_residual(rotation(z), energy, z)
    assert out["residual"] < 1e-8


def test_mismatched_pair_is_rejected():
    """The radial field cannot be written B*grad(E) for ANY antisymmetric B: B*grad(E)
    is always orthogonal to grad(E), while the radial field is parallel to it."""
    z, energy, _, radial = _canonical_2d()
    out = noether_pairing_residual(radial(z), energy, z)
    assert out["residual"] > 0.9


def test_pairing_is_coordinate_invariant():
    """The failure the review found: the same objects in different coordinates must give
    the SAME answer. The old canonical-J test swung across 0.15-0.52 here."""
    z, energy, rotation, _ = _canonical_2d()
    base = noether_pairing_residual(rotation(z), energy, z)["residual"]
    gen = torch.Generator().manual_seed(3)
    for _ in range(4):
        A = torch.randn(2, 2, generator=gen) + 3.0 * torch.eye(2)
        Ainv = torch.linalg.inv(A)
        z_t = z @ A.T
        C_t = lambda q: energy(q @ Ainv.T)
        g_t = rotation(z) @ A.T
        assert abs(noether_pairing_residual(g_t, C_t, z_t)["residual"] - base) < 1e-6


def test_works_in_four_dimensions():
    """4-D: angular momentum L = x*vy - y*vx pairs with the phase-space rotation
    g = (-y, x, -vy, vx) under the canonical structure — and the solver must find it
    without being told the structure."""
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(768, 4, generator=gen)
    L = lambda s: s[:, 0] * s[:, 3] - s[:, 1] * s[:, 2]
    rot = torch.stack([-z[:, 1], z[:, 0], -z[:, 3], z[:, 2]], dim=1)
    assert noether_pairing_residual(rot, L, z)["residual"] < 1e-8
    # a field that is not a Hamiltonian field of L is rejected
    assert noether_pairing_residual(z.clone(), L, z)["residual"] > 0.9


def test_recovered_structure_is_antisymmetric():
    z, energy, rotation, _ = _canonical_2d()
    B = noether_pairing_residual(rotation(z), energy, z)["B"]
    torch.testing.assert_close(B, -B.T, atol=1e-6, rtol=0)
