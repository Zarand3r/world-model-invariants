# Executive summary

*Updated 2026-08-30. Kept current after every significant development — see CLAUDE.md.*

We are testing whether a world model trained only on video of a swinging pendulum learns physics it
actually obeys. The distinction that matters: a probe can read a quantity out of the model's internal
state almost perfectly, and the model's own predictions can still ignore that quantity completely.
Paper 1 (published on arXiv) showed this happens. Since then we have been trying to make a stronger
claim — that the model learns the specific approximation its simulator uses — and that attempt has
not succeeded. Everything runs on small physics toys, with predictions and failure conditions written
down before each experiment.

## Roadmap

- **Now: correct and strengthen paper 1, post it as v2.** One published result is wrong (see
  Results); the fix makes it stronger. Add the two-degrees-of-freedom system, the supervised-probe
  control, and the four negatives.
- **Next, cheapest: use our measurement to *choose* between models.** No new training. Would give the
  paper the practical use it currently lacks. Two earlier attempts at a practical use both failed.
- **Then: settle the big open question by training a model that is told its own timestep.** Needs new
  training. This is the only route left to the stronger claim.
- **Biggest lever, highest cost: repeat the result on a standard benchmark**, not a pendulum. The
  objection every reviewer will raise.
- **Not doing:** more seeds, more toy systems, or anything comparing the two integration schemes —
  that comparison is provably impossible (see Results).

## Results

**Established.**

- A probe fitted to the pendulum's true energy reads it almost perfectly (0.9999) yet the model's own
  transition preserves it **6.7× worse** than a quantity found without labels — and forcing the
  model to obey the probe makes its physics *worse*, never better.
- Nudging the model back toward the label-free quantity during imagination **cuts physical error
  55–76%** on trajectories it never saw, where 0 of 60 matched random controls help. Survives being
  applied 50 steps into imagination.
- The effect is a property of the **architecture**, not the data: a different network on *identical*
  trajectories is **767× worse**. This is our cleanest result, because the data is held fixed.
- Outside the training range, energy stays readable while conservation collapses (155–267×).
- Everything replicates on a second, two-degrees-of-freedom system.

**Negative, and useful.** Four things we predicted would work and measured not working, each with a
diagnosed cause: it does not improve control performance, it is not a usable live warning signal, it
does not learn the balance law under actuation, and fixed constraints cannot be extracted at all.

**Open.** Whether the model tracks its simulator's *timestep* is unresolved. The measurement showing
it might be reading the training data rather than the model, and the tests that would tell them apart
do not work with the models we have. Not disproven — untested.

**Dead.** The claim that the model learns its simulator's integration *scheme* is impossible to test
from video, and always was: the two schemes we compared produce identical motion and differ only in
bookkeeping the pixels never show. Four experiments were built on that gap before it was noticed.

**A defect in the published paper.** Its check that "random directions don't help" was not a fair
comparison — the random ones were pushed 29× harder. Corrected, the result gets *stronger* (2 of 3
seeds becomes 3 of 3). Anyone using the old code should apply the fix in
`docs/UPGRADE_FROM_PUBLISHED.md`.

**Odds of a main-track conference acceptance.** ~30% for the paper as it stands. ~45–60% if the
timestep question is settled. These estimates have moved a lot; the ranking of what to do next is
more reliable than the numbers.
