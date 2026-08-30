# F11 --- Forced choice on the TIMESTEP, the axis that is actually observable

**Registered 2026-08-30, before writing the measurement. No training.**

## Why this axis and not the last one

F7, F7b, F9 and F10 compared semi-implicit Euler against velocity Verlet. Those eliminate to the
**same** three-term position recurrence and differ only in which finite difference is labelled the
velocity --- a variable the pixels never render. The models could not have distinguished them, and
all four experiments are withdrawn (`tests/test_observable_difference.py` now pins the reason).

The **timestep** is different. It enters the recurrence directly:

    th_{t+1} = 2 th_t - th_{t-1} + a(th_t) dt^2

so `dt` changes what the model observes, and F6's question --- does a pixel-trained model recover a
coefficient tracking its own simulator's timestep? --- is well posed. That is why F6 reverts to
unresolved rather than withdrawn. This is the forced-choice test of it.

## Measurement

Cross F6's `dt = 0.02` and `dt = 0.08` checkpoints (4x apart) against both datasets. For each
consecutive true position pair `(th_{t-1}, th_t)` in the evaluation data, form two counterfactual
next positions from the **same** pair:

| candidate | value |
|---|---|
| the **model's** timestep | `2 th_t - th_{t-1} + a(th_t) dt_model^2` |
| the **data's** timestep | `2 th_t - th_{t-1} + a(th_t) dt_data^2` |

Decode the model's own predicted next position from `m.transition(h_t)` and score a **hit** when it
lands nearer the *model's* timestep candidate. Null is 50%.

**Only off-diagonal cells are meaningful**: when `dt_model = dt_data` the two candidates are
identical by construction. Diagonal cells are reported but not scored, and this is stated now rather
than discovered later.

## Registered predictions

- **P1.** Off-diagonal, models prefer their **own** timestep's candidate at above 50%
  (binomial, two-sided, `p < 0.01`) on at least **2 of 3** seeds in each of the two families.
- **Falsifier.** If models prefer the **data's** timestep, their one-step prediction is driven by the
  observed sequence rather than by learned dynamics, and F6's model-side claim fails --- this time on
  an axis where the difference genuinely exists in the observations.

## Readability gates, pre-committed

Both are stated before running, because in F8 and F10 I met a gate I had not registered.

- **G1 (separation).** The two candidates differ by `|a(th)| |dt_model^2 - dt_data^2|`, about
  `0.09 |sin th|` rad here, against a one-step prediction error of roughly `0.0066` rad measured in
  F9 --- an expected margin near `8x`, far better than the scheme axis ever offered. Required:
  median separation `>` median model error, **per cell**. A cell failing this is not read.
- **G2 (out of distribution).** A `dt = 0.02` model shown `dt = 0.08` data sees four times the
  per-frame motion. The earlier cross-timestep control (F7 amendment 3) went degenerate for exactly
  this reason. If a model's prediction error on crossed data exceeds the separation, the cell is
  **not readable**, and that is an honest "cannot test here", **not** evidence against F6.

## Direction

None stated. My last two stated expectations were both wrong, one of them in a way I had not
imagined.
