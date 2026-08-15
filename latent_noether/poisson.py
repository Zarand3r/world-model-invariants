"""The Noether pairing, without presupposing canonical coordinates.

Noether's correspondence in Hamiltonian mechanics says a symmetry generator g and its
conserved quantity C are related through the Poisson structure: g = B∇C, with B
antisymmetric. The canonical choice B = J = [[0,I],[-I,0]] is only valid in canonical
(q,p) coordinates — and a network's learned latents are an arbitrary invertible image of
whatever the true state is. Assuming J there produced a score that moved between 0.15 and
0.52 for the SAME generator and invariant merely re-expressed in different coordinates
(see scripts/review_checks.py); that made the 4-D test meaningless.

Solving for B instead is covariant in the right sense, but the claim needs care. Under z -> Az:
    g -> A g,     ∇C -> A⁻ᵀ ∇C,     B -> A B Aᵀ
so B∇C = g is preserved and the EXISTENCE question is chart-free.

**The residual's MAGNITUDE is not.** It is ‖g − B∇C‖²/‖g‖², a ratio of Euclidean norms in
latent coordinates, which A distorts: the expression becomes ‖A(g − B∇C)‖²/‖Ag‖². Only the
zero is invariant. Measured 2026-08-04: a well-conditioned random reparameterisation
(cond < 20) moved a residual of 0.164 to 0.50–0.66 — across the whole gap separating
conservative from dissipative models.

This is exactly what Table 1 predicts for anything measured with a Euclidean norm in latent
coordinates. Comparisons of nonzero residuals are therefore only meaningful **under a fixed
extraction procedure**, which is what every experiment here uses (PCA on the model's own
latents). State it that way; do not claim chart-independence for the magnitude. This is the right primitive for asking whether a
symmetry found inside a network carries a Noether charge.

The fit is a linear least-squares problem in B's d(d-1)/2 free parameters: closed form,
no optimizer, in the same spirit as `polynomial.py`.
"""
import torch


def _antisymmetric_basis(d: int, dtype=None) -> torch.Tensor:
    """(m, d, d) basis of antisymmetric matrices, m = d(d-1)/2.

    `dtype` is honoured because the basis is contracted against caller-supplied tensors. Building
    it at the global default silently forced float32 and raised
    "expected scalar type Float but found Double" the first time a caller used float64
    (the DreamerV3 extraction, 2026-08-11). `hamiltonian_select._antisym_basis` always took a
    dtype; this one did not, so the two disagreed.
    """
    mats = []
    for j in range(d):
        for k in range(j + 1, d):
            E = torch.zeros(d, d, dtype=dtype)
            E[j, k], E[k, j] = 1.0, -1.0
            mats.append(E)
    return torch.stack(mats)


def _grad(scalar_fn, z: torch.Tensor) -> torch.Tensor:
    zg = z.detach().requires_grad_(True)
    return torch.autograd.grad(scalar_fn(zg).sum(), zg)[0]


def noether_pairing_residual(g_vals: torch.Tensor, C, z: torch.Tensor) -> dict:
    """Does the generator g arise from the invariant C through SOME Poisson structure?

    g_vals: (n, d) generator evaluated at z. C: callable (n, d) -> (n,). z: (n, d).

    Returns the scale-free residual min_B E‖g − B∇C‖² / E‖g‖² in [0, 1] — 0 means g is
    exactly the Hamiltonian vector field of C for the recovered B (a genuine Noether
    pair), ~1 means no antisymmetric B relates them at all — together with that B.

    Note the test has real teeth: B∇C is always orthogonal to ∇C, so any generator with a
    component along ∇C (a field that CHANGES C) is rejected no matter what B is chosen.
    """
    if g_vals.shape != z.shape:
        raise ValueError(f"g_vals {tuple(g_vals.shape)} must match z {tuple(z.shape)}")
    n, d = z.shape
    if d < 2:
        raise ValueError("a Poisson structure needs at least 2 dimensions")

    v = _grad(C, z)                                   # (n, d) gradients of the invariant
    basis = _antisymmetric_basis(d, dtype=z.dtype)    # (m, d, d)
    # column p of the design maps parameter p to its contribution (E_p v_i) for every i.
    # Index order must be (sample, component, parameter) so the reshape lines up with the
    # flattened target: an (i, p, j) ordering silently coincides with (i, j, p) only when
    # d = 2 (one basis matrix), which is how an earlier version passed 2-D and failed 4-D.
    design = torch.einsum("pjk,ik->ijp", basis, v).reshape(n * d, basis.shape[0])
    target = g_vals.reshape(n * d)
    coeffs = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
    B = torch.einsum("p,pjk->jk", coeffs, basis)

    pred = v @ B.T
    denom = (g_vals ** 2).sum(-1).mean().clamp_min(1e-30)
    return {"residual": float(((g_vals - pred) ** 2).sum(-1).mean() / denom),
            "B": B,
            "generator_norm": float(g_vals.norm(dim=-1).mean())}
