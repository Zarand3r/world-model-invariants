# Experiment roadmap and results ledger

*Ordered by dependency, not by interest. Updated as each lands — see CLAUDE.md.*
*Status vocabulary: **queued** / **running** / **done** / **negative** / **void** / **closed**.*

## Order, with what each gates

| # | experiment | status | result | gates |
|---|---|---|---|---|
| **0** | **F14 — cross-fit audit.** Fit `C` and the supervised probe on one set, freeze, score on a disjoint set. | **done** | **survives: 6.22× median cross-fit vs 6.7× in-sample** (7.27/5.93/6.22), P1 3/3, frozen `C` still `ρ_E` 0.92–0.98. Falsifier did not fire. | — |
| **0b** | **F6 presentation fix.** Slope and argmin are algebraically one result. State the two-parameter fit and that P3 failed. | **done** | Applied to the derived summaries. `paper1.2/` prose left alone — changing what it claims is the author's call. | write-up only |
| **1** | **F13 — timestep attribution.** Conditioned on `dt`; evaluate on fixed data sweeping only the conditioning. | **running** | seed 4 trained, **G0 passed** (dt-use 0.854 ≤ 0.90). Seeds 3, 5 training. No verdict yet. | 2, 3 |
| **2** | **Paper 1 v2.** Corrected null, 2-DoF, E18, the five negatives. | **queued** | — | submission |
| **3** | **Newton Stage A.** Recover `Q,P` with `ΔQ ∝ P`, `ΔP ∝ f(Q)`; inspect the recovered `f`. | **queued** | — | 4 |
| **4** | **Newton Stage B.** Variable-mass environment, held-out `(F₂,m₂)`. | **queued** | — | 5 |
| **5** | **Newton Stage C.** Interventions; `F→λF, m→λm` leaving `a` unchanged is the *primary* prediction. | **queued** | — | — |
| **6** | **Benchmark replication** (DMC). Answers "this is a pendulum". | **queued** | — | — |

## Why this order

**#0 gates everything and costs a day.** Every downstream claim rests on the 6.7× and 767×, and both
are currently measured with `C` fitted and scored on the same 52 trajectories. Running this after the
paper is written would be the worst possible ordering.

**#3 must answer F1 before it runs.** F1 already asked whether the model implements a dynamical
*relation* it demonstrably obeys — the power balance law — and the action's power explained **0.31%**
of the variance in `dC`. That is the closest prior attempt and it was negative. Stage A asks about
free dynamics and the whole vector field rather than one balance law, so it is not settled, but the
preregistration must state why it expects a different answer.

**#3 also needs #1's result**, because Stage A inherits the same attribution problem (fitting `f` on
real trajectories) and the hold-data-fixed-vary-the-conditioning pattern is F13's.

## Closed

| experiment | outcome |
|---|---|
| **F12** model selection by `rho_obs` | **closed, negative.** Validation loss ranks checkpoints better (+0.82 vs +0.47); after controlling for training step every ranker collapses to ~0. Third failed downstream use. |
| **F5** control return | **negative.** 0.24% of an episode's return spread. |
| **F2** online trust signal | **negative.** Beaten 3/3 by plain latent displacement. |
| **F3** constraint extraction | **negative.** A quantity that never varies carries no information. |
| **F1** balance law under actuation | **negative.** Power explains 0.31% of the variance in `dC`. |
| **F7, F7b, F9, F10** integration *scheme* | **void — invalid axis.** The two schemes share one position recurrence and differ only in a velocity label the pixels never show. |
| **F8** imagined-rollout measurement | **void — no power.** Sweep flat; argmin spans the full grid across horizons. |
| **F11** timestep forced choice | **not readable.** Crossed models 90× out of distribution. A pre-committed gate stopped a p<1e-100 false positive. |
