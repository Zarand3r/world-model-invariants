# F13 — Does the recovered coefficient follow the MODEL or the evaluation data?

**Registered 2026-09-07, before generating the dataset or training anything.**

## The question this settles

F6 found that models trained at timestep `dt` recover a correction coefficient tracking the
parameter-free prediction `c*(dt) = 2.5 dt` — slope `2.484 +/- 0.058` against a predicted `2.500`,
argmin exactly on prediction for 10 of 12 models, nothing fitted. **Whether that is a fact about the
model or about the data it was measured on is untested**, because every F6 model was only ever
evaluated on its own timestep's data.

Two attempts to separate them failed. Cross-evaluating on another timestep's data puts the model far
out of distribution (one-step error `0.007 -> 0.6` rad, nine times the separation the test needs).
Measuring along the model's own imagined rollout has no discriminative power (flat sweep; the
recovered optimum wanders the full grid as horizon changes).

## The design, and why it separates them where those could not

Train a world model **conditioned on `dt`**, then evaluate on **one fixed dataset** while sweeping
only the conditioning input. The data is *literally identical* across arms — same frames, same
states, same encoder inputs — so there is nothing left for a data confound to act on. If the argmin
moves with the conditioning, it is the model.

**Precondition, already run and passed (recorded below, not assumed).** This only works if the
conserved quantity is a property of the *map* rather than of the trajectory. Ground-truth check:
apply a one-step map of size `dt'` to states drawn from a `dt = 0.05` trajectory and sweep the
shadow family. The argmin tracks the **map** exactly at every timestep tested:

| map `dt'` | predicted `c*(dt')` | argmin `c` |
|---|---|---|
| 0.02 | 0.0500 | **0.0500** |
| 0.035 | 0.0875 | **0.0875** |
| 0.05 | 0.1250 | **0.1250** |
| 0.08 | 0.2000 | **0.2000** |

This is the check that four withdrawn experiments skipped: the difference being tested demonstrably
exists in what the measurement can see, *before* any GPU time.

## Method

- **Data.** Concatenate F6's four existing pixel datasets (`dt` in {0.02, 0.035, 0.05, 0.08}, 256
  trajectories x 120 steps each) into one mixed set of 1024 trajectories. Each trajectory carries its
  own `dt` as a constant per-timestep value in the adapter's existing 1-D conditioning channel — the
  same channel F1 used for torque, with the same indexing convention.
- **Models.** 3 seeds, otherwise identical to F6's training configuration.
- **Evaluation.** `runs/pend_dt0.05.npz` **only**, held fixed, sweeping the conditioning value over
  the four training timesteps. Repeated on `runs/pend_dt0.02.npz` as a control that the result does
  not depend on which fixed dataset is chosen.

## Gates, all registered before running

- **G0 — the model must actually use `dt` (HARD GATE).** Following F1's validated action-use check:
  compare one-step prediction error with the true `dt` against the same values shuffled across
  trajectories. Required: `mse_true / mse_shuffled <= 0.9` on at least **2 of 3** seeds. **If G0
  fails, P1 is not readable and must not be reported as a scheme effect** — a model that ignores the
  conditioning cannot answer the question, and that is an uninformative outcome rather than a
  negative one.
- **G1 — the sweep must discriminate.** At each conditioning value, contrast between the best and
  worst `rho_obs` across the grid must be `>= 2x`. F8 produced a perfectly alive measurement whose
  sweep was flat, and its argmin was noise; this gate exists because that was not registered then.
- **G2 — observable difference.** Passed above, before training.

## Registered predictions

- **P1 (primary).** With the evaluation data held fixed, the recovered argmin `c` increases with the
  conditioning `dt`: Spearman(conditioning `dt`, argmin `c`) `>= 0.8` on at least **2 of 3** seeds.
- **P2 (quantitative).** The slope of argmin `c` against conditioning `dt` is within a factor of 2 of
  the parameter-free `2.5`.
- **P3 (robustness).** P1 holds on both fixed evaluation datasets.

## Falsifier

If the argmin does not move with the conditioning while the data is held fixed, then the recovered
coefficient is a property of the **evaluation data** and not of the model. F6's and E19's model-side
claims would then fail, the paper's title claim goes, and the work narrows to the probing-methodology
result. That is the outcome to report if it happens.

## Direction

None stated. Three stated expectations this session were wrong, one of them in a way I had not
imagined.
