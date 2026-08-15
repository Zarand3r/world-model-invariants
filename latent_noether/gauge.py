"""Gauge complexity `k*`, and decodability — the two Phase-2 measurements.

`k*` is the minimum polynomial degree at which an invariant becomes discoverable from the
latent. It exists because the Phase-2 hypothesis needs a *gauge-invariant* notion of "the
law got simpler", and raw simplicity is not one: the nonlinear audit measured a refit
invariant moving 2.55–4.80 under a diffeomorphism while a transported one stayed at exactly
0.0000. The required degree is a property of the chart, so it is the thing to measure.

**`k*` is only admissible as evidence once calibrated.** A degree-bounded search failing
tells you either that the coordinates are warped (the claim) or that the instrument is too
weak (a confound we have already been bitten by — the separatrix went VOID for exactly that
reason). `tests/test_gauge.py` therefore pins two properties before any experiment uses it:

1. **It tracks an injected warp of known degree.** Compose the latent with a polynomial
   coupling map of degree `m`; a degree-2 invariant becomes degree `2m`, and `k*` must find
   it there. Calibration, not a sanity check.
2. **It is flat under padding.** Embedding the same content in more dimensions must not move
   `k*`, or the Phase-2 ladder would confound gauge complexity with latent size (risk R3).

`k* = inf` is reported, never silently clipped, and per PHASE2_PLAN.md means *undetermined*
rather than evidence for the hypothesis.
"""
import math

import numpy as np
import torch

from latent_noether.polynomial import monomial_features, validated_invariants
from latent_noether.probe import fit_probe


def effective_rank_basis(traj: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis (d, r) of the directions the data actually occupies.

    A latent that is rank-deficient — which a closed subspace extracted from a much larger
    hidden state routinely is — contains directions that are identically constant on the
    data. Those are *trivially* conserved: they score an in-sample invariance ratio of 0
    while carrying no content, and the generalized eigenproblem prefers them to the real
    invariant. Measured on a 2-D oscillator isometrically embedded in 4-D, this alone moved
    `k*` from 2 to infinity.

    Singular values are compared to the largest, so the threshold is scale-free.
    """
    flat = traj.reshape(-1, traj.shape[-1])
    flat = flat - flat.mean(dim=0)
    _, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    keep = S > tol * S[0].clamp_min(1e-30)
    return Vh[keep].T


def k_star(traj: torch.Tensor, max_degree: int = 6, gate: float = 1e-2,
           rank_tol: float = 1e-6, **kwargs) -> tuple[float, list[dict]]:
    """Minimum degree at which an invariant is discoverable, and the per-degree trace.

    traj: (n_traj, T, d). Returns `(k, trace)` where `k` is the smallest degree whose best
    **held-out** invariance ratio is below `gate`, or `math.inf` if none is. The trace holds
    one row per degree so the whole `D_disc` curve is reported rather than only its
    threshold crossing — a single fixed degree would smuggle in a gauge assumption.

    Scored held out throughout. In-sample ratios are not admissible: degree 8 has been
    measured at an in-sample ratio of exactly 0 with a held-out ratio of 7.7e-03.

    The latent is first restricted to its occupied subspace (`rank_tol`); without that,
    rank deficiency alone drives `k*` to infinity — see `effective_rank_basis`.
    """
    if traj.ndim != 3:
        raise ValueError(f"traj must be (n_traj, T, d), got {tuple(traj.shape)}")
    if max_degree < 1:
        raise ValueError(f"max_degree must be >= 1, got {max_degree}")
    # Restrict to the occupied subspace before searching, or degenerate directions win.
    traj = traj @ effective_rank_basis(traj, rank_tol)
    trace, found = [], math.inf
    for degree in range(1, max_degree + 1):
        best = validated_invariants(traj, degree=degree, **kwargs)[0]
        trace.append({"degree": degree, "ratio": best["ratio"],
                      "heldout_ratio": best["heldout_ratio"]})
        if found is math.inf and best["heldout_ratio"] < gate:
            found = degree
            break                      # smallest passing degree is the definition
    return found, trace


def decodability(latents: torch.Tensor, target: torch.Tensor, degree: int = 1,
                 n_fit: int | None = None, ridge: float = 1e-6) -> float:
    """Held-out R² of a probe reading `target` from degree-`degree` features of `latents`.

    **`degree` is not optional in practice, and matching it to the discovery degree is the
    whole ballgame.** Decodability is degree-dependent exactly as discoverability is: angular
    momentum is quadratic, so a linear probe scores R² ≈ −0.27 on a latent from which the
    invariant eigenproblem recovers it cleanly. Phase 1 already paid for this once — a linear
    probe scored R² = 0.005 on the same quantity and its causal τ was meaningless.

    Comparing `D_dec` at one capacity against `D_disc` at another does not measure
    "decodable but not discoverable"; it measures the capacity gap. Any Phase-2 claim must
    hold them equal.

    latents: (n_traj, T, D); target: (n_traj, T) or (n_traj, T, k). Split by trajectory, so
    the probe is scored on trajectories it never saw — an in-sample probe R² on a
    high-dimensional latent is close to meaningless.

    **`ridge` defaults to 1e-6, not 0.05, and that change retracted a result.** With ridge at
    0.05 an untrained latent decoded energy at 0.011; at 1e-8 it decoded the same quantity at
    0.988. The regularizer was suppressing a genuine signal by two orders of magnitude, and a
    reported "training effect" of +0.202 turned out to be the suppression. Ridge scales to mean
    feature variance, so it penalises exactly the low-variance directions a conserved quantity
    tends to occupy. Raise it only with a measured reason, and sweep it before believing any
    separation.

    This is `D_dec`. **Never use it alone**: an untrained network is a random feature map and
    position history implicitly contains velocity, which is how the retracted Newton result
    reached 0.7%. The Phase-2 gate uses `D_dec(trained) − D_dec(untrained)`.
    """
    if latents.ndim != 3:
        raise ValueError(f"latents must be (n_traj, T, D), got {tuple(latents.shape)}")
    tgt = target.unsqueeze(-1) if target.ndim == 2 else target
    if tgt.shape[:2] != latents.shape[:2]:
        raise ValueError(f"target {tuple(tgt.shape[:2])} does not match "
                         f"latents {tuple(latents.shape[:2])}")
    n = latents.shape[0]
    n_fit = n_fit if n_fit is not None else max(2, n // 2)
    if n_fit >= n:
        raise ValueError(f"need at least one held-out trajectory: n_fit={n_fit}, n={n}")

    feats = monomial_features(latents, degree) if degree > 1 else latents
    flat = lambda x, s: x[s].reshape(-1, x.shape[-1]).detach().cpu().numpy()
    H_fit, t_fit = flat(feats, slice(None, n_fit)), flat(tgt, slice(None, n_fit))
    H_te, t_te = flat(feats, slice(n_fit, None)), flat(tgt, slice(n_fit, None))

    W, _, _ = fit_probe(H_fit, t_fit, ridge=ridge)
    pred = (H_te - H_fit.mean(axis=0)) @ W.T
    tc = t_te - t_fit.mean(axis=0)
    ss_res = ((tc - pred) ** 2).sum(axis=0)
    ss_tot = ((tc - tc.mean(axis=0)) ** 2).sum(axis=0)
    return float((1.0 - ss_res / np.maximum(ss_tot, 1e-30)).mean())


def pca_subspace(latents: torch.Tensor, d: int) -> torch.Tensor:
    """Top-`d` principal directions of the latent, as a (D, d) orthonormal basis.

    The default extraction for Phase 2, replacing `fit_sufficient_closed_subspace`.

    **Why closure was replaced, measured rather than assumed.** Closure selects the subspace
    the transition preserves *and that carries the readout*. At low dimension the readout term
    dominates, and instantaneous prediction is best served by fast oscillatory directions — so
    a slowly-inferred per-episode constant gets crowded out even though the dynamics depend on
    it. Measured on a trained pendulum whose raw latent decodes the hidden parameter at 0.987:

        closure d=3   0.051      PCA d=5    0.390
        closure d=5   0.282      PCA d=12   0.835
        closure d=12  0.398
        closure d=20  0.773

    PCA beats closure at every matched dimension. This also gives a mechanism for the Phase-1
    Wedge-1 result, where closure lost to a linear probe: closure was discarding content the
    probe retained.

    Closure remains correct for what it claims — the subspace the transition preserves — and
    stays available as an ablation. It is simply the wrong extraction when the quantity of
    interest varies slowly.
    """
    if latents.ndim != 3:
        raise ValueError(f"latents must be (n_traj, T, D), got {tuple(latents.shape)}")
    flat = latents.reshape(-1, latents.shape[-1])
    flat = flat - flat.mean(dim=0)
    return torch.linalg.svd(flat, full_matrices=False)[2][:d].T


def whiten(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalise the latent chart: zero mean, identity covariance.

    **SCOPE — read before using (D16).** Valid only for *well-conditioned* latents. It must NOT
    be applied to the PCA-extracted GRU/RSSM latents used throughout this project: those measure
    a covariance condition number of ~180 and a participation ratio of **2.13 of 6 dimensions**,
    so whitening amplifies four near-null transient directions by ~12x and gives them equal
    weight in a norm-based residual. Measured on the 10 recorded models:

        conservative  unwhitened 0.088-0.407   whitened 0.600-0.797
        damped        unwhitened 0.444-0.689   whitened 0.683-0.846
        separation    p = 0.0079 (perfect)     p = 0.095 (destroyed)

    Recovery degrades too: |rho|_E fell to 0.302 and 0.097 on seeds where the unwhitened
    procedure gives >= 0.9988. A floored variant `W = V diag(1/sqrt(max(lam, tau*lam_max))) V^T`
    was swept over tau in [0, 1] and **no value satisfies both goals**: the best chart spread
    (0.116) lands where recovery has already collapsed to 0.307, and recovery is non-monotone in
    tau. Chart-invariance and noise-robustness are in genuine conflict for these latents.

    Use the D13 scoping instead — compare residuals only under a fixed extraction procedure.

    The rest of this docstring describes the regime where whitening IS correct, and the
    regression test below covers exactly that regime (exact dynamics, well-conditioned).

    The pairing residual is a ratio of Euclidean
    norms in latent coordinates, so its magnitude is chart-dependent (D13): a well-conditioned
    random reparameterisation moved a residual of 0.164 to 0.50–0.66, across the whole gap
    separating conservative from dissipative models. Worse, on *exact* dynamics with an exactly
    recoverable energy, one reparameterisation in three destroyed the recovery outright
    (|ρ| 1.000 → 0.036).

    Whitening removes the free `GL(d)` action up to rotation, so the search sees the same chart
    regardless of how the extraction happened to orient it. Measured on true dynamics across
    four random reparameterisations:

        raw       |ρ| 1.000, 0.036, 0.992, 0.945    <- one total failure
        whitened  |ρ| 0.999, 0.995, 0.995, 1.000    <- stable

    Returns `(whitened_latents, W)`. **The flow must be transformed by the same `W`** — it is a
    tangent vector, so `F -> F @ W`, and applying it to the state but not the flow silently
    breaks the pairing.
    """
    if latents.ndim != 3:
        raise ValueError(f"latents must be (n_traj, T, d), got {tuple(latents.shape)}")
    flat = latents.reshape(-1, latents.shape[-1])
    mu = flat.mean(dim=0)
    centred = flat - mu
    cov = (centred.T @ centred) / centred.shape[0]
    ev, V = torch.linalg.eigh(cov)
    W = V @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ V.T
    return (latents - mu) @ W, W
