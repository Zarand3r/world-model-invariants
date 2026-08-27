# Pre-registration — E4, counterfactual invariant dialing

**Written 2026-08-27, before any E4 quantity was computed.** Governed by `docs/ROADMAP.md` Stage 2.
Claim addressed: **C4 — causal physical control.**

## Why this is the headline experiment

Everything measured so far is *restorative*: set `C` back to `C_0` and the imagined rollout gets more
physical. That establishes that violating `C` costs accuracy. It does **not** establish that `C` is a
control variable — that the model's imagined physics is *governed* by it.

E4 sets `C` to values it was never at and asks whether the imagined world changes in the
quantitatively predicted direction. Correction shows drift is harmful; dialing would show the scalar
is causal.

## Two protocols

### E4a — donor-level intervention (primary)

Take a recipient state and an independent donor trajectory. Move the recipient minimally along the
local `C`-normal until `C(z_edited) = C(z_donor)`, then roll out autonomously and decode.

**The intervention never sees true energy.** `C(z_donor)` is read off the model's own latent. Ground
truth enters only afterwards, when the decoded rollout's physical energy is compared with the
donor trajectory's true energy.

This is the interchange form: it stays inside the distribution of `C` values the model actually
produces, and so cannot be dismissed as pushing the latent somewhere meaningless.

### E4b — synthetic sweep (secondary)

Set `C -> C_0 + dC` for a **preregistered** grid of offsets, expressed in units of the
across-trajectory standard deviation of `C` on the analysis split:

    dC / std_traj(C) in {-1.0, -0.5, -0.25, 0, +0.25, +0.5, +1.0}

Fixed now so that "which offsets to report" cannot be chosen after seeing the outcome.

## Primary metric — fixed now

**Transfer correlation**: Spearman rho between the intended change in `C` and the realised change in
decoded physical energy, across donor-recipient pairs (E4a) and across offsets (E4b).

- **Registered prediction:** rho > 0 with a bootstrap CI excluding 0, on every conservative seed.
- **Falsifier:** rho is 0 or negative, or its CI includes 0. Then `C` is not a control variable, the
  intervention is restorative only, and C4 is unsupported.

## `C` is not assumed to be energy

`C` may be a nonlinear monotone function of physical energy, so the registered evidence is
**monotonicity and transfer correlation**, not slope-1 agreement. A calibration from `C` to `E`, if
reported at all, is fitted **only on the Dreamer training trajectories (0:204)** — the same split
that calibrated the pixel readout — and frozen before any E4 evaluation. Predicted-vs-realised slope
and R^2 are reported as secondary, conditional on that frozen calibration.

## Secondary metrics, registered but not decisive

1. **Amplitude and period.** For librating trajectories, decoded `theta_max` and period against the
   analytic pendulum values at the realised energy.
2. **Separatrix crossing.** Fraction of librating trajectories pushed into rotation at the largest
   positive offset, and whether the new regime **persists** autonomously for the remaining rollout
   without further forcing. Registered as descriptive: a nonzero persistent crossing rate would be
   strong, its absence falsifies nothing on its own.
3. **Directional asymmetry.** Whether raising and lowering `C` are equally effective. Spies et al.
   (arXiv:2412.11867) report activation is easier than suppression in world models, so both
   directions are reported separately and never pooled.

## Controls

- **Random `C`** (20 draws, magnitude-matched as in the direction-matched null). A random constraint
  dialed by the same amount should produce **no coherent** energy change. This is the control that
  matters: if random directions also move decoded energy monotonically, the effect is latent
  geometry, not the invariant.
- **Equal-norm tangent** edits. By construction these cannot change `C` to first order, so they
  should not change decoded energy either.
- **Damped models**, once three exist.
- **Untrained models.**

## Splits, checkpoints, exclusions

Analysis split `204:` for the primary run and `runs/pendulum_pixels_eval.npz` for a disjoint
replication, following E9. `C`, `h_mean`, `U`, `R` frozen exactly as in E1/E9. Step magnitude is
whatever is required to reach the target `C`, so E4 is **not** magnitude-matched by construction —
which is why the random-`C` control is dialed to the same `C` offset rather than the same step size.

Exclusions: a donor-recipient pair is dropped if the required edit exceeds 5x the median edit norm
(an unreachable target), a rule fixed now to prevent a long tail of extreme edits from dominating.
Trajectories are excluded only if the geometric readout fails on their rendered frames.

## Known limitation

Moving along the `C`-normal changes the latent microstate, not only `C`. A positive transfer
correlation is evidence that the recovered **subspace** is causally deployed for physical energy, not
proof that the model maintains an internal energy register. E5's interpretation rule applies here
too, and no stronger claim will be made from E4 alone.
