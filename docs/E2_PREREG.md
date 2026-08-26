# Pre-registration — E2, where conservation fails

**Written before any E2 result was produced. 2026-08-26**, while Stage 1 training was still running,
so no E2 quantity had been computed. Governed by `docs/ROADMAP.md` Phase I. Claim addressed:
**C3 — failure mechanism.**

## The question

The paper shows `C` is nearly constant on observation-conditioned trajectories and drifts under
autonomous imagination. It does not say **why**. Two very different mechanisms produce that same
summary, and the roadmap treats separating them as the potential centrepiece:

- the transition is locally faithful to `C` everywhere, but a small systematic bias accumulates; or
- the transition is locally faithful to `C` only on states the observation distribution supports,
  and recursive rollout carries the state off that support into a region where the model's own
  learned physics no longer holds.

The second is the interesting one and is the mechanism the roadmap's target claim asserts.

## The measured object

The **local conservation defect**, one autonomous step applied at a state:

    r(z) = C(T(z)) - C(z)

This is deliberately *local*: it asks what the transition does in one step from a given state, and
so separates "the map is locally wrong here" from "errors have accumulated over many steps". The
existing drift statistic cannot make that separation.

Measured on two state distributions:

- `r_obs(t) = C(T(z_t^obs)) - C(z_t^obs)` at every real encoded state
- `r_auto(k) = C(T(z_k^auto)) - C(z_k^auto)` at rollout depth `k`, started from the same states

## Normalisation, fixed now

`r` is reported in units of the across-trajectory standard deviation of `C` on the analysis split,
so it is comparable across models and seeds:

    rho(k) = median_traj |r_auto(k)| / std_traj(C)

with `rho_obs` the same statistic on observation-conditioned states. `rho_obs` is the floor: it is
what the transition does on states the model was trained to represent.

## Primary metric

**The growth of `rho(k)` with rollout depth**, summarised by the OLS slope of `rho(k)` against `k`
over `k = 0..49`, and by the ratio `rho(49) / rho_obs`.

Registered thresholds, chosen before seeing any value:

- **Outcome A (support loss)** — `rho(49) / rho_obs >= 3` and the slope is positive with a bootstrap
  CI excluding 0.
- **Outcome B (systematic bias)** — `rho(49) / rho_obs < 3` and the slope CI includes 0, i.e. the
  defect is nonzero but roughly flat in depth.
- **Outcome C (phase coupling)** — neither, and `r_auto(k)` has a dominant oscillatory component:
  registered as the case where a single sinusoid at the trajectory's own libration frequency
  explains `>= 0.5` of the variance of `r_auto(k)`. If C holds, E5/E11 are prioritised over E4.

These are not mutually exhaustive by construction; if the data land outside all three the result is
reported as unclassified rather than forced into a box.

## Off-support measurement — one metric, fixed now

Distance of `z_k^auto` from the observation-conditioned latent distribution, as the **whitened
nearest-neighbour distance**: whiten `z` by the covariance of the observation-conditioned `Z` on the
analysis split, then take the Euclidean distance from `z_k^auto` to its nearest neighbour among
those states. Reported as the median over trajectories at each depth `k`.

One metric only, chosen in advance, because the temptation with an unregistered family of distance
measures is to report whichever rises fastest.

The claim E2 can support with this is **association**: whether the depth at which `rho(k)` rises is
the depth at which the state leaves support. E2 does **not** establish that leaving support *causes*
the defect — a state could be off-support for reasons unrelated to `C`. Establishing direction needs
an intervention that moves a state off support while holding `C` fixed, which is E3/E12 territory and
is not claimed here.

## Controls

- **Damped models.** `r` is measured with each damped model's own recovered `C`. On a dissipative
  system there is no conserved quantity to lose, so `rho_obs` should already be large and the
  depth-growth should be weak or absent. This distinguishes "recursion breaks conservation" from
  "our defect statistic grows with depth for any model".
- **Untrained models.** Same pipeline. Registered expectation: `rho_obs` already large.
- **Random `C`.** The same 20 norm-matched draws used elsewhere, on the conservative models. A random
  polynomial is not conserved anywhere, so `rho_obs` should be large and the ratio near 1.

## Splits, checkpoints, exclusions

Analysis split `204:`, `WARMUP = 10`, depth 0..49 — identical to E1 so the two are directly
comparable. `C` frozen from `fit_hamiltonian_pair(..., degree=4, n_basis=8)`, LD=12, exactly as in
`run_dreamer_edit.py`. Checkpoints are the Stage 1 models; hashes recorded per row.

Exclusion only by the pre-existing training acceptance checks. No exclusion on E2 outcome.

## What would falsify the roadmap's target claim

If `rho_obs` is already comparable to `rho(49)` — the transition violates `C` just as badly on real
encoded states as deep in a rollout — then the "imagination leaves the region where its physics
holds" story is wrong, and the honest description is that the model never respected `C` step-to-step
at all and the observation-conditioned constancy is an artefact of being re-anchored by real frames
at every step. That is a publishable negative result and will be reported as one.

## Ordering note

E2 depends on the same frozen `C` as E1 but not on E1's outcome, and is worth running even if E1
fails its gate: if the intervention turns out to be a latent regulariser, E2 still says whether the
latent scalar is locally preserved by the transition. It is therefore not gated behind E1.
