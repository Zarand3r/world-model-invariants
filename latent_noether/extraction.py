"""The extraction preamble, in one place: latent -> subspace -> flow -> the fitted invariant.

Five scripts each carried their own copy of this sequence — `run_dreamer_extraction`,
`run_dreamer_edit`, `run_dreamer_leverage`, `run_dreamer_residual_decomp`, `run_edit_compactness` —
and the copies had already drifted on which split they took and whether the warmup was dropped before
or after the PCA. A probe UI that recomputes it a sixth way would make its numbers unreviewable
against the paper, so the sequence lives here and every caller gets the same one.

**What a `Bundle` is.** Everything downstream of one (checkpoint, extraction dimension, degree)
choice that does not depend on which law you are enforcing or where a rollout starts. The fit is the
expensive part — a 1819x1819 eigenproblem at LD=12, degree 4, measured at 22.5 s — so it is computed
once and cached, and anything interactive is expressed as arithmetic on the result.

**The one design decision worth stating.** The conserved candidates come back as `k` basis
invariants phi_i and mixing weights a_i with `C = sum_i a_i phi_i`. Rather than store the collapsed
coefficient vector alone, a bundle keeps the basis and the projected features

    G = Phi(Z) @ basis_coeffs.T          (N, k)

so C for ANY mixing weights is `G @ a` — a matrix-vector product on an (N, 8) array instead of a
refit. That is what makes hand-steering the invariant interactive: re-mixing is microseconds, and
the 22.5 s is paid only when the checkpoint, the dimension or the degree changes.
"""
import dataclasses

import numpy as np
import torch

from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features, polynomial_invariants

G_ACC, M_ACC, L_ACC = 10.0, 1.0, 1.0      # gymnasium Pendulum-v1 constants


def true_energy(states: np.ndarray) -> np.ndarray:
    """E = 1/2 (m l^2 / 3) thetadot^2 + m g (l/2) cos theta, theta from upright.

    Identical to the copy in `run_dreamer_extraction.py`, which is the definition the paper scores
    against. Gymnasium's semi-implicit integrator conserves a shadow Hamiltonian rather than this,
    which shows up as ~12% oscillation with no secular drift and is treated as a noise floor.
    """
    th, thd = states[..., 0], states[..., 1]
    return 0.5 * (M_ACC * L_ACC ** 2 / 3) * thd ** 2 + M_ACC * G_ACC * (L_ACC / 2) * np.cos(th)


@dataclasses.dataclass
class Bundle:
    """One extraction. Arrays are numpy so this serialises to a plain .npz."""
    model_key: str              # the registry key; `ckpt` is an absolute path and moves with the repo
    ckpt: str
    ld: int
    degree: int
    warmup: int
    split_start: int
    h_mean: np.ndarray          # (D,)      latent mean, subtracted before projecting
    P: np.ndarray               # (D, r)    U @ R, the projection into the extracted subspace
    P_pinv: np.ndarray          # (r, D)    maps a correction in the subspace back into h
    Z: np.ndarray               # (n, T, r) projected trajectories
    flow: np.ndarray            # (n, T, r) one-step displacement under the transition
    G: np.ndarray               # (n*T, k)  basis invariants evaluated on Z — see module docstring
    basis_coeffs: np.ndarray    # (k, m)    the k candidate invariants over the monomial basis
    weights: np.ndarray         # (k,)      fitted mixing weights; C = sum_i weights_i phi_i
    energy: np.ndarray          # (n, T)    true pendulum energy, for scoring only
    eigenvalues: np.ndarray     # (r,)      latent covariance spectrum, descending
    participation_ratio: float
    residual: float             # pairing residual of the fitted law
    rank_and_test_residual: float

    # Derived, cached on the instance rather than in a module-level dict. A dict keyed on
    # `id(bundle)` looked equivalent and is not: 3000 short-lived Bundles reuse 7 distinct ids under
    # CPython, so as soon as one is evicted and collected the next bundle inherits its gradients.
    # Hanging the cache off the object ties its lifetime to the thing it describes and cannot alias.
    basis_grads: "np.ndarray | None" = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def coeffs(self) -> np.ndarray:
        """The collapsed degree-`degree` coefficient vector of the fitted C."""
        return self.weights @ self.basis_coeffs

    def C(self, weights: np.ndarray | None = None) -> np.ndarray:
        """C on the analysis trajectories, for any mixing weights. Shape (n, T)."""
        w = self.weights if weights is None else np.asarray(weights, dtype=self.G.dtype)
        return (self.G @ w).reshape(self.Z.shape[:2])


def encode_split(model, frames: torch.Tensor, warmup: int) -> torch.Tensor:
    """Teacher-forced latents with the warmup dropped. frames: (n, T, 64, 64, 3) in [-0.5, 0.5]."""
    with torch.no_grad():
        return model.encode(frames)[:, warmup:].detach()


def build_bundle(model, frames: torch.Tensor, states: np.ndarray, *, ckpt: str = "",
                 model_key: str = "", ld: int = 12, degree: int = 4, warmup: int = 10,
                 split_start: int = 204, n_basis: int = 8) -> Bundle:
    """Run the full extraction and return everything downstream of it.

    `frames` and `states` are ALREADY restricted to the analysis trajectories; `split_start` is
    recorded so the bundle says which split it came from. The split is applied in exactly one place
    (`viz.server.assets.analysis_split`) rather than re-derived here, so a caller cannot fit on one
    slice and score on another — and passing only the split keeps 1.5 GB of unused training frames
    off the GPU.
    """
    if frames.shape[0] != states.shape[0]:
        raise ValueError(f"frames and states disagree on trajectory count: "
                         f"{frames.shape[0]} vs {states.shape[0]}")
    H = encode_split(model, frames, warmup)                           # (n, T, D)
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)

    U = pca_subspace(H, ld)
    Z = (H - h_mean) @ U
    R = effective_rank_basis(Z)
    Z = Z @ R
    P = U @ R

    with torch.no_grad():
        nxt = model.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = ((nxt - h_mean) @ P) - Z                                      # one-step displacement

    cov = torch.cov(Z.reshape(-1, Z.shape[-1]).T)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    p = ev / ev.sum().clamp_min(1e-30)
    pr = float(1.0 / (p ** 2).sum().clamp_min(1e-30))

    Zc, Fc = Z.double().cpu(), F.double().cpu()
    # Solve the eigenproblem ONCE and hand it to the fit. It is the whole cost of this function
    # (1819x1819 at LD=12, degree 4), it depends only on `Z`, and we need the basis afterwards to
    # build `G`. Letting `fit_hamiltonian_pair` solve its own would run it twice for one answer.
    cands = polynomial_invariants(Zc, degree=degree, max_results=n_basis)
    fit = fit_hamiltonian_pair(Zc, Fc, degree=degree, n_basis=n_basis, candidates=cands)
    basis = np.stack([np.asarray(c["coeffs"], dtype=np.float64) for c in cands])
    weights = np.asarray(fit["basis_coeffs"], dtype=np.float64)

    Phi = monomial_features(Zc.reshape(-1, Zc.shape[-1]), degree)     # (N, m)
    Gmat = (Phi @ torch.as_tensor(basis.T, dtype=Phi.dtype)).numpy()  # (N, k)

    return Bundle(
        model_key=model_key, ckpt=ckpt, ld=ld, degree=degree, warmup=warmup,
        split_start=split_start,
        h_mean=h_mean.detach().cpu().numpy(), P=P.detach().cpu().numpy(),
        P_pinv=torch.linalg.pinv(P).detach().cpu().numpy(),
        Z=Zc.numpy(), flow=Fc.numpy(), G=Gmat, basis_coeffs=basis, weights=weights,
        energy=true_energy(states)[:, warmup:],
        eigenvalues=ev.detach().cpu().numpy(), participation_ratio=pr,
        residual=float(fit["residual"]),
        rank_and_test_residual=float(fit["rank_and_test_residual"]),
    )


def rho_energy(bundle: Bundle, weights: np.ndarray | None = None) -> float:
    """|rho| between C and true energy — the paper's headline statistic."""
    C = bundle.C(weights)
    n = min(C.shape[1], bundle.energy.shape[1])
    return abs(float(np.corrcoef(C[:, :n].ravel(), bundle.energy[:, :n].ravel())[0, 1]))


def drift_of_C(bundle: Bundle, weights: np.ndarray | None = None) -> float:
    """Within-trajectory variance of C as a fraction of its total variance. 0 = conserved.

    The paper's "drift of C" row: 0.7-1.3e-4 on conservative models, 0.203-0.266 on damped ones.

    Two details are copied from `run_dreamer_refusal._drift` rather than chosen, because a UI that
    reports a differently-defined "drift" beside the paper's table would be quietly comparing two
    quantities: it is scored on the HELD-OUT half of the trajectories, and the variances are
    unbiased (torch's default, ddof=1). Using all trajectories and numpy's biased default moved this
    by 6% of its own value on the conservative models.
    """
    C = bundle.C(weights)
    te = C[C.shape[0] // 2:]
    within = float(np.mean(np.var(te, axis=1, ddof=1)))
    return within / max(float(np.var(te.ravel(), ddof=1)), 1e-30)
