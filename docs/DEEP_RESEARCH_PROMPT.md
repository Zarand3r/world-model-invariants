# Deep research prompt — finding extensions from adjacent literature

*Paste the block below into a deep-research tool. It is self-contained: a researcher with no prior
context should be able to execute it. Written 2026-08-31, against the state in `docs/SUMMARY.md`.*

---

## Role and objective

You are helping a machine-learning researcher decide what to do next on a world-model
interpretability project that has hit three specific walls. Your job is **not** to summarise the
field. It is to find **transferable methods, framings, and results from adjacent literature that
would unblock one of the three named problems below**, and to say concretely how each would apply.

Prefer five findings that change a decision over fifty that are merely relevant.

## What the project has established

A DreamerV3 world model trained only on pendulum video is examined for quantities its own learned
transition approximately preserves. The central, replicated finding is a **dissociation between
decodability and dynamical preservation**:

- A linear probe fitted to the pendulum's *true* energy reads it at correlation 0.9999, yet the
  model's own one-step transition preserves it **6.7x worse** than a quantity found without labels.
  Forcing the model toward the probe's quantity during imagination makes its physics *worse*.
- Nudging the latent state back toward the label-free quantity **cuts decoded physical error 55-76%**
  over 100-step imagined rollouts on unseen trajectories, with 0 of 60 magnitude-matched random
  directions and 0 of 15 equal-norm tangent controls helping. The effect survives being applied 50
  steps into autonomous imagination.
- The same procedure on a conv-GRU trained on **identical** trajectories gives a conservation
  statistic **767x** worse — so the effect is a property of the architecture, not the data.
- Out of distribution, energy stays decodable (probe 0.999) while conservation degrades 155-267x.
- Everything replicates on a second, two-degrees-of-freedom system.

The measurement is `rho_obs` = median |C(T(z)) - C(z)| / std(C), where `T` is the model's own latent
transition and `C` is a degree-4 polynomial in a PCA frame of the latent, fitted without labels.

## The three walls — this is what you are solving

**Wall 1 — attribution: is the measurement reading the model, or the data?**
`rho_obs` is computed from a one-step prediction applied to encoded *real* frames. A model that
predicts well reproduces the next state of whatever trajectory it is shown, so "which quantity is
best preserved" may reflect the evaluation trajectories rather than anything the model learned. Two
attempts to separate them failed: cross-evaluating a model on another simulator's data puts it badly
out of distribution (one-step error rises 0.007 -> 0.6 rad), and measuring along the model's own
imagined rollout has no discriminative power (the sweep is flat, and the recovered optimum wanders
across the entire search range as the horizon changes).

> **Find:** how other fields separate "property of the learned operator" from "property of the
> evaluation distribution". Candidate framings to check: Koopman operator approximation and its
> known spectral pollution / spurious-eigenvalue diagnostics; backward error analysis and modified
> equation analysis for learned integrators; system identification with persistently exciting inputs;
> causal-abstraction and activation-patching methodology in interpretability, which faces the same
> "is this the model or the input distribution?" problem; off-policy evaluation, where the whole field
> is about attributing an estimate to a policy rather than to the data that produced it.

**Wall 2 — no downstream use. Three preregistered attempts have failed.**
(a) Using the conserved quantity to correct plans gives 0.24% of an episode's return spread.
(b) Its drift is not a usable online trust signal — beaten by plain latent displacement.
(c) It does not rank checkpoints better than validation loss (Spearman +0.47 against +0.82), and
after controlling for training step no measure carries signal at all.
The measured reason for (a) and (b) is the same: the effect is tiny next to the spread of whatever
decision is being made.

> **Find:** settings where a conservation or invariance measure over a *learned* dynamics model has
> demonstrably paid off — long-horizon rollout stability, model-predictive control, sim-to-real
> transfer, detecting distribution shift, or curriculum/data selection. Be specific about the regime
> that made it work, since the failures above all share "effect small relative to decision spread".
> Equally valuable: credible evidence that such measures *systematically* fail to transfer, which
> would let the project state a bounded negative rather than keep testing.

**Wall 3 — toy systems.** Two smooth Hamiltonian systems, 13.5M parameters, no released checkpoint.

> **Find:** work that took an interpretability result from a toy dynamical system to a standard
> benchmark (DMC, Atari, robotics) and what had to change. Specifically: what *decodable and
> dynamically relevant* quantity plays the role of energy in a benchmark environment with contacts
> and non-smooth dynamics? A prior attempt here failed because the chosen quantity (a rod-length
> constraint) never varies, so it carries no information and lives in decoder weights. Quantities
> that vary are required.

## What is dead — do not propose it

The claim that the model learns its simulator's integration *scheme* is provably untestable from
pixels: semi-implicit Euler and velocity Verlet reduce to the identical position recurrence
`th_{t+1} = 2 th_t - th_{t-1} + a(th_t) dt^2` and differ only in which finite difference is labelled
the velocity — a variable the rendered frames never show. Four experiments were built on that gap.
Anything relying on distinguishing integrators from position observations alone is out.

The *timestep* remains observable (it enters through `a dt^2`) and that question is open.

## Fields worth mining, and why

Rank your own search, but these are the ones the project has *not* systematically read:

- **Backward error analysis / modified equations** — the shadow-Hamiltonian framing came from here,
  but only the textbook result. Recent work on what neural integrators actually converge to may
  speak directly to Wall 1.
- **Koopman / DMD operator learning** — the same object (a learned linear-ish operator and its
  invariants) with a mature diagnostic literature about spurious modes and data dependence.
- **Probing critiques and causal interpretability** — amnesic probing, causal scrubbing, activation
  patching. The project's central claim is a probing critique; this literature will say how much of
  it is already known, which matters for novelty.
- **Conserved-quantity discovery** (AI Feynman, Noether-networks, symmetry discovery) — adjacent
  method, different goal; may supply attribution controls the project lacks.
- **Model-based RL evaluation** — how the field measures whether a world model is *good*, and whether
  anything beyond validation loss has ever won. Directly targets Wall 2.
- **Hamiltonian / Lagrangian / symplectic neural networks** — these *impose* structure where this
  project *measures* it. Their failure modes may predict this project's.

## What counts as a useful finding

**Good:** a specific paper or method, what it would let the project do that it currently cannot, the
experiment it suggests, and its most likely failure mode here. A result that would let the project
*close* a question counts as much as one that opens a direction.

**Not useful:** general surveys; "physics-informed ML is a growing field"; anything requiring a
capability the project lacks (large-scale training, proprietary environments); or restating the
project's own findings back to it.

**Especially valuable:** evidence that the central claim is *less* novel than believed — prior work
already showing decodability-versus-dynamical-preservation in learned models. The project needs to
know this before submission, not from a reviewer.

## Output format

1. **Top 5 findings**, ranked by how much each changes a decision. For each: the source, the wall it
   addresses, the concrete experiment it suggests, and what would make it fail here.
2. **Novelty assessment** — closest prior work to the dissociation claim, and how close.
3. **Dead ends confirmed** — anything above that the literature says will not work, with the reason.
4. **One paragraph** on any field not listed above that turned out to be more relevant than expected.
