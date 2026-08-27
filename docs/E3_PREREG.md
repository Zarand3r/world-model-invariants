# Pre-registration — E3, normal vs tangent decomposition of one-step error

**Written 2026-08-26, before any E3 quantity was computed.** Promoted to Stage 1 the same day after
E2 returned Outcome B. Claim addressed: **C3 as amended** — that the transition carries a
near-constant systematic violation of `C`, rather than losing conservation off-manifold.

## Why this is now the confirmatory test

E2 established that the local defect `r(z) = C(T(z)) - C(z)` does not grow with rollout depth. That
is consistent with the transition having a persistent error component **transverse to the level set
of `C`**. E3 measures that component directly, and it is the geometric statement of which E1's
projection is the remedy: Hairer §IV.4 projection removes exactly the normal displacement.

If the amended C3 is right, one-step error decomposes into a tangent part (which moves the state
along the level set — a phase/timing error, no energy consequence) and a normal part (which changes
`C`, and integrates into physical error).

## The measured object

At each observation-conditioned state, compare the autonomous step with the true next encoded state:

    dz_t = T(z_t^obs) - z_{t+1}^obs

With `g_t = grad C(z_t)`, decompose

    dz_perp = ((g^T dz) / ||g||^2) g          (normal to the level set)
    dz_par  = dz - dz_perp                    (tangent)

## Primary metric — fixed now

**The normal fraction of one-step error:**

    f_perp(t) = ||dz_perp(t)||^2 / ||dz(t)||^2

reported as the median over states, with a bootstrap CI over trajectories.

The registered comparison is against the **null expectation for an isotropic error in the extracted
subspace**, which is `1/LD = 1/12 = 0.0833`: a one-step error with no preferential orientation
relative to `grad C` puts that fraction of its energy in the normal direction by chance.

- **Registered prediction (amended C3):** `f_perp` is **greater** than 1/12, with a CI excluding it.
  The transition's error is preferentially transverse to the level set.
- **Falsifier:** `f_perp` is at or below 1/12. Then one-step error is not preferentially normal, the
  geometric story is wrong, and the projection's benefit needs a different explanation.

## Secondary, registered but not decisive

1. **Depth independence.** `f_perp` measured at rollout depth k, k = 0..49. Amended C3 predicts flat.
2. **Does normal error predict rollout failure better than total error?** Spearman correlation of
   per-trajectory accumulated `||dz_perp||` vs accumulated `||dz||` against final decoded-energy
   error. Registered as a comparison of two correlations, not a threshold.
3. **Does the projection remove the normal component selectively?** `f_perp` recomputed with the
   E1 edit active at alpha = 0.4. Prediction: `f_perp` falls, `||dz_par||` roughly unchanged.
4. **Tangent/phase association.** Whether `||dz_par||` tracks decoded *phase* error while
   `||dz_perp||` tracks decoded *energy* error. Registered as descriptive; a clean double
   dissociation would be strong, but its absence falsifies nothing on its own.

## Controls

Same three as E2, since the normal fraction is only meaningful relative to them:

- **Random `C`** (20 norm-matched draws). A random constraint's gradient has no relation to the
  transition, so its `f_perp` should sit at 1/12. **This is the control that matters**: if random
  constraints also show `f_perp` >> 1/12, the effect is a property of the latent geometry, not of
  the recovered invariant.
- **Untrained models.** Registered expectation: `f_perp` near 1/12.
- **Damped models**, once trained.

## Splits, checkpoints, exclusions

Analysis split `204:`, `WARMUP = 10`, LD = 12, `C` frozen from
`fit_hamiltonian_pair(..., degree=4, n_basis=8)` — identical to E1 and E2 so all three compose.
Checkpoints are the Stage 1 milestone grid; hashes recorded per row. Exclusion only by the
pre-existing training acceptance checks.

## Known limitation, stated in advance

`dz` compares an autonomous step against the *encoder's* next state. Encoder and transition are
different maps, so `dz` contains encoder-representation mismatch as well as transition error. E3
therefore measures the orientation of the combined discrepancy. That is the right object for
explaining why the projection helps — the projection acts on exactly this quantity — but it is not a
clean measurement of transition error alone, and no claim about the transition in isolation will be
made from it.
