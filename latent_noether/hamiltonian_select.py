"""Select the physical invariant by asking which one GENERATES the flow.

The instrument this project needed and did not have.

**The problem it solves.** Unsupervised invariant search does not recover physical laws. Any
rich latent admits many conserved combinations — at PCA dimension 20 with degree 2 there are
230 candidate features — and the eigenproblem returns whichever scores the lowest invariance
ratio. Measured: the selected quantity's correlation with true energy swung 0.935 → 0.063
under a change of extraction alone, and reached 0.998 on an *untrained* latent. **"Most
conserved" and "physical" are unrelated criteria**, so ranking by conservation alone cannot
work, however well it is implemented.

**The fix, from Hamiltonian mechanics.** For a Hamiltonian system, every conserved quantity
`C` satisfies `{C, H} = 0`, but only the Hamiltonian itself generates the flow:

    f = B∇H          the Hamiltonian vector field
    f ≠ B∇C          for a generic other conserved C

So among a family of conserved quantities, the pairing residual `min_B E‖f − B∇C‖² / E‖f‖²`
is small for `H` and large for the others. **The Noether pairing is not merely a further
claim to test — it is the selection criterion that picks the physical invariant out of an
arbitrary family.**

Two properties follow, and both matter for the controls:

1. It tests the geometry of the model's **own transition map**, not the value of a quantity.
   Whether a quantity is decodable from the latent — which for any function of the trajectory
   is nearly always true, trained or not — says nothing about whether the flow has Hamiltonian
   structure with respect to it.
2. **Nonlinearity is required.** On a LINEAR system `B = A M_C⁻¹` solves the pairing and is
   antisymmetric for *any* quadratic conserved `C`, so the criterion is empty. Measured in
   float64 — single precision cannot settle it, the residuals sit near 1e-30:

       eps=0.0 (linear)   energy 3.9e-32   L 2.5e-31   discrimination 6.3x  -- NONE
       eps=3.0            energy 1.7e-29   L 8.8e-02   discrimination 5.1e27

3. `B` is *solved*, never assumed. Assuming canonical `J` scored the same object anywhere from
   0.15 to 0.52 under a change of coordinates. See `poisson.py`.
"""
import torch

from latent_noether.poisson import _antisymmetric_basis, noether_pairing_residual
from latent_noether.polynomial import monomial_features, polynomial_invariants


def _poly(coeffs: torch.Tensor, degree: int):
    def C(z: torch.Tensor) -> torch.Tensor:
        return monomial_features(z, degree) @ coeffs
    return C


def select_generating_invariant(traj: torch.Tensor, flow: torch.Tensor, degree: int = 2,
                                n_candidates: int = 8, candidates=None, **kwargs) -> dict:
    """Rank conserved quantities by how well each generates the flow.

    traj: (n_traj, T, d) latent trajectories. flow: (n_traj, T, d) the model's own vector
    field at those points — its transition displacement, not a fitted surrogate.

    Returns the ranked candidates, each with its invariance ratio (how conserved) and its
    pairing residual (how well it generates the flow), sorted by the latter. `best` is the
    lowest-residual candidate: the estimate of the Hamiltonian.

    The two rankings differing is the point. If they agreed, conservation alone would
    suffice and this instrument would be unnecessary.
    """
    if traj.ndim != 3 or flow.shape != traj.shape:
        raise ValueError(f"traj and flow must both be (n_traj, T, d); "
                         f"got {tuple(traj.shape)} and {tuple(flow.shape)}")
    d = traj.shape[-1]
    z = traj.reshape(-1, d)
    f = flow.reshape(-1, d)

    # `candidates` lets a caller that has already solved the eigenproblem pass it in.
    # `fit_hamiltonian_pair` computes exactly this and used to trigger a second, identical
    # eigendecomposition purely to report the rank-and-test comparison.
    if candidates is None:
        candidates = polynomial_invariants(traj, degree=degree, max_results=n_candidates, **kwargs)
    out = []
    for rank, cand in enumerate(candidates):
        coeffs = torch.as_tensor(cand["coeffs"], dtype=traj.dtype)
        pairing = noether_pairing_residual(f, _poly(coeffs, degree), z)
        out.append({"conservation_rank": rank, "ratio": cand["ratio"],
                    "pairing_residual": pairing["residual"], "coeffs": cand["coeffs"],
                    "expression": cand.get("expression", "")})
    out.sort(key=lambda r: r["pairing_residual"])
    return {"best": out[0], "ranked": out,
            # If the flow-generating quantity is also the most-conserved one, the extra
            # machinery bought nothing on this system; worth reporting either way.
            "selection_changed_the_answer": out[0]["conservation_rank"] != 0}


def fit_hamiltonian_pair(traj: torch.Tensor, flow: torch.Tensor, degree: int = 2,
                         n_basis: int = 8, n_iter: int = 30, candidates=None, **kwargs) -> dict:
    """Solve for the conserved quantity AND the Poisson structure *together*.

    Ranking conserved quantities and then testing each one's pairing does not work, and the
    reason is structural rather than numerical. The conserved family at degree 4 contains
    `E`, `L`, `E²`, `EL`, `L²`, … all with excellent invariance ratios, so the eigenproblem's
    top candidate is generally a **mixture**. Measured on the anharmonic central force: the
    best-conserved candidate had ratio 1.86e-09 and correlated 0.9882 with true energy, yet
    paired at 5.97e-02 where exact energy pairs at 1.49e-09.

    **A 1.2% error in the invariant costs seven orders of magnitude in the pairing.** The
    answer is a *direction inside* the conserved subspace that no single eigenvector isolates,
    so it must be searched for, not selected.

    `f = B∇C` with `C = Σ aᵢφᵢ` is bilinear in `(a, B)`: fixing either makes the other a
    closed-form least-squares problem. Alternating between them converges quickly, and the
    scale degeneracy (scaling `C` inversely scales `B`) is absorbed by renormalising `a`.

    Returns the fitted coefficients over the conserved basis, the final residual, and the
    residual of the plain rank-and-test approach for comparison.
    """
    if traj.ndim != 3 or flow.shape != traj.shape:
        raise ValueError(f"traj and flow must both be (n_traj, T, d); "
                         f"got {tuple(traj.shape)} and {tuple(flow.shape)}")
    d = traj.shape[-1]
    z, f = traj.reshape(-1, d), flow.reshape(-1, d)

    # `candidates` lets a caller reuse a solved eigenproblem. It depends only on `traj`, so any
    # null that perturbs the FLOW alone (e.g. shuffling F against Z) yields an identical conserved
    # basis -- and at degree 4 in 12 dimensions that basis is a 1819x1819 eigenproblem costing ~69 s.
    # Passing it in turns a 173-minute null study into a tractable one.
    cands = candidates if candidates is not None else polynomial_invariants(
        traj, degree=degree, max_results=n_basis, **kwargs)
    coeffs = torch.stack([torch.as_tensor(c["coeffs"], dtype=traj.dtype) for c in cands])

    # gradients of each basis invariant, by autograd on the monomial map
    zz = z.detach().requires_grad_(True)
    feats = monomial_features(zz, degree)
    grads = []
    for c in coeffs:
        g, = torch.autograd.grad((feats @ c).sum(), zz, retain_graph=True)
        grads.append(g.detach())
    G = torch.stack(grads)                                     # (k, n, d)

    basis = _antisym_basis(d, traj.dtype)                      # (m, d, d)
    a = torch.zeros(len(coeffs), dtype=traj.dtype)
    a[0] = 1.0
    def b_step(a):
        """Given C = sum a_i phi_i, solve min_B ||f - B grad C|| and return (B, A, beta, residual)."""
        gradC = torch.einsum("k,knd->nd", a, G)
        A = torch.stack([gradC @ b.T for b in basis], dim=-1)   # (n, d, m)
        beta = torch.linalg.lstsq(A.reshape(-1, A.shape[-1]), f.reshape(-1)).solution
        return (torch.einsum("m,mij->ij", beta, basis), A, beta,
                float(((f - (A @ beta)) ** 2).sum() / (f ** 2).sum()))

    # The B-step is hoisted out of the loop because each iteration ended by recomputing exactly
    # what the next one's B-step would redo -- a third of the work, for identical numbers.
    B, _, _, resid = b_step(a)
    for _ in range(n_iter):
        # a-step: given B, solve min_a ||f - sum_i a_i B grad phi_i||
        M = torch.stack([g @ B.T for g in G], dim=-1)           # (n, d, k)
        a = torch.linalg.lstsq(M.reshape(-1, M.shape[-1]), f.reshape(-1)).solution
        a = a / a.norm().clamp_min(1e-30)
        B, _, _, new = b_step(a)
        if abs(resid - new) < 1e-12:
            resid = new
            break
        resid = new

    return {"residual": resid, "basis_coeffs": a.tolist(),
            "coeffs": (a @ coeffs).tolist(),
            "rank_and_test_residual": select_generating_invariant(
                traj, flow, degree=degree, n_candidates=n_basis, candidates=cands, **kwargs
            )["best"]["pairing_residual"]}


# One implementation, re-exported under the name callers already use. Two copies of this existed
# and disagreed on dtype handling, which crashed every float64 caller the first time one appeared
# (D39). Keeping the alias avoids churning the scripts that import it.
_antisym_basis = _antisymmetric_basis
