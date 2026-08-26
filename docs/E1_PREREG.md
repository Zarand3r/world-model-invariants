# Pre-registration — E1, decoded physical energy from imagined pixels

**Written before any E1 result was produced. 2026-08-26.**
Governed by `docs/ROADMAP.md` (Phase I, gate experiment). Claim addressed: **C2 — physical validity.**

## The question

`scripts/run_dreamer_edit.py` scores the intervention with pixel MSE and nothing else. The paper
therefore cannot distinguish two possibilities:

1. projecting the latent back onto the level set of `C` makes the model's **imagined physics more
   physical**; or
2. it is a latent regulariser that happens to lower pixel error while the decoded world drifts in
   true energy just as badly.

Wang (arXiv:2606.24945) states the objection directly: a model "can conserve [a learned latent
Hamiltonian or a learned scalar witness] while drifting in true energy." E1 decides it.

## The readout — non-learned, and validated before use

A **geometric** image-to-angle estimator, not a learned probe:

- rod orientation from image moments about the known pivot in the 64x64 render
- unwrap `theta` across the rollout
- `thetadot` by central finite differences at the simulator `dt = 0.05`
- energy from the existing `energy()` in `scripts/make_pendulum_pixels.py`
  (`E = 0.5 (m l^2 / 3) thetadot^2 + m g (l/2) cos theta`, g=10, m=1, l=1)

A learned probe is forbidden as the primary result: it would refit the very quantity under test.

**Validation, reported before any intervention number.** The estimator is scored on
(a) true rendered frames and (b) Dreamer reconstructions of observation-conditioned frames, against
the `states` array stored in the dataset `.npz`. We report median and 90th-percentile absolute error
in `theta`, `thetadot`, and `E`. These define the **noise floor**. An intervention effect smaller
than the floor is reported as "below readout resolution", never as an effect.

## Primary metric — fixed now

**Secular drift of decoded energy**, per rollout:

    D_sec = OLS slope of Ehat_k against k, k = 0..H-1

normalised by the across-trajectory standard deviation of true `E` in the analysis set, so it is
dimensionless and comparable across models. Reported as the **median over trajectories**, with a
bootstrap CI over trajectories nested within model seed.

Rationale for choosing drift rather than `|Ehat_k - E_ref_k|`: gymnasium's semi-implicit integrator
conserves a shadow Hamiltonian, so textbook `E` oscillates ~8-12% with **no secular component**
(`scripts/make_pendulum_pixels.py` docstring). Oscillation is therefore signal-free noise here;
secular drift is the thing the intervention should remove. The reference `E_ref(t)` from the matched
simulator trajectory is reported alongside, but is not the primary statistic.

## Scoring rule — inherited from D9, non-negotiable

`alpha` is swept over the **fixed grid (0, 0.05, 0.1, 0.2, 0.4)**, identical for every arm, the full
curve is reported, and the statistic is the **slope of `D_sec` versus alpha**. Best-over-alpha is
forbidden: D9 exists because it once reported -5.9% for a damped arm whose curve rose monotonically.

## Secondary metrics (reported, not decisive)

pixel MSE (for comparability with the published result) · mean absolute `theta` error · mean
absolute `thetadot` error · `|Ehat_k - E_ref_k|` · phase error.

## Arms

| arm | model | constraint enforced | registered expectation |
|---|---|---|---|
| **A** | conservative | its own recovered `C` | `D_sec` **decreases** with alpha |
| **B** | conservative | 20 norm-matched random degree-4 draws | no improvement, or harm |
| **C** | damped | its own recovered `C` | no improvement, or harm |
| **D** | conservative | no edit (alpha = 0) | baseline, shared with all arms |

Arm B is a **distribution, not a draw** (M24): 20 independent draws, same 20 across models so the
comparison is paired. Reported statistic is the null median and the recovered arm's percentile in it.

## Splits, checkpoints, seeds

- Datasets regenerated per `docs/REPRODUCE.md` (conservative seed 0, damped seed 11, eval seed 777).
- `C` is fitted exactly as in `run_dreamer_edit.py` — `fit_hamiltonian_pair(..., degree=4,
  n_basis=8)` on the analysis split, LD=12, WARMUP=10 — and **frozen** before any E1 metric is
  computed.
- Horizon `H = 50`, matching the published intervention, so E1 is comparable to it. Extension to
  other horizons is E6, not E1.
- **These are newly trained checkpoints, not the ones behind arXiv:2608.23526.** Training is capped
  by wall clock, so numbers land near but not exactly on the committed run logs. Every E1 artefact
  records its checkpoint hash.

## Exclusion rules

A model is excluded only by the **pre-existing** training acceptance checks (KL > 1 nat, one-step
decoding at least 4x better than the dataset mean, finite rollouts). No exclusion on E1 outcome.
A trajectory is excluded only if the geometric estimator fails on the **real** frames of that
trajectory (occlusion / degenerate moment), a condition determined before any rollout is decoded.

## Registered outcomes

**Gate passes** if arm A's `D_sec` slope versus alpha is negative with a bootstrap CI excluding 0,
and arm A's percentile in the arm-B null is extreme in the registered direction.

**Gate fails** if arm A's `D_sec` slope is >= 0, or its CI includes 0, or arm B improves `D_sec` as
much as arm A.

**On failure the roadmap is explicit:** stop describing the intervention as repairing physical
dynamics, reinterpret it as a latent regulariser, and investigate why pixel error improves while
physics does not. That is a publishable negative result and will be reported as one, not buried.

**Ambiguous outcome** — pixel MSE improves, `D_sec` improves by less than the readout noise floor —
is reported as ambiguous, and E1 is re-run at longer horizon (H = 100, 200) before any claim is made,
since secular drift is easier to resolve over more steps.
