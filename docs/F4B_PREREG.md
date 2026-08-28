# Pre-registration — F4b, is the conservation gap the architecture or the training?

**Written 2026-08-28, before any F4b model was trained.**

## The confound F4 left

The ConvGRU conserves the recovered scalar **730x worse** than DreamerV3's RSSM (`rho_obs` 1.75-9.02
against ~7e-03), and recovery itself fails on 1 of 3 seeds. But the two are **not matched on how much
their transitions are trained**:

| | prior-training signal |
|---|---|
| DreamerV3 RSSM | `kl_dyn` trains the prior at **every** step of a 64-step sequence |
| ConvGRU (F4) | open-loop term over **8** of 64 steps |

So F4 cannot say whether the gap is the **latent architecture** (categorical stochastic + KL versus
a deterministic GRU state) or simply **how much the transition was trained**. That distinction
decides whether the paper's scope claim is about latent design or about training objectives, so it
is worth one clean experiment.

## Design

Identical model, identical data, identical step-capped contract. **Only the open-loop horizon
changes**: 8 -> 56 of a 64-step sequence, so the transition is trained over nearly the whole
sequence, comparable to `kl_dyn`'s every-step signal.

Everything downstream is frozen: LD = 12, degree 4, `n_basis` = 8, WARMUP = 10, same acceptance
checks including the rollout-motion criterion added after F4's first run.

## Primary metric

`rho_obs` of the recovered scalar at step 60,000, median over 3 seeds.

- **Training explains the gap** if `rho_obs` comes within **10x** of the RSSM's ~7e-03, i.e.
  **< 7e-02**. The ConvGRU currently sits at 1.75-9.02, so this requires a 25-130x improvement —
  large, but the untrained-to-8-step change was already 10-40x.
- **Architecture matters** if `rho_obs` stays above 7e-02 despite near-full-sequence prior training.

Reported alongside: `|rho_E|` and how many of 3 seeds clear the 0.8 recovery bar, since F4 failed
that on 1 of 3 and the two-factor account predicts recovery should improve if conservation does.

## Registered prediction

**Partial closure.** `rho_obs` improves substantially but does **not** reach 7e-02, and recovery
improves to 3 of 3. Reasoning: the untrained-to-8-step change bought 10-40x and the remaining gap is
~250x, so a further 7x from 8 -> 56 steps seems plausible but not sufficient. Predicting a partial
result rather than a clean one is deliberate — the honest expectation, not the tidy one.

**Either clean outcome is more informative than the prediction being right.** Full closure would say
the RSSM's latent design is incidental and only prior training matters; no closure would isolate the
architecture.

## Known limit

Matching "amount of prior training" between an open-loop reconstruction term and a KL between prior
and posterior is approximate — they are different losses, not the same loss at different strengths.
A residual gap after this experiment cannot be attributed to architecture with certainty, only
narrowed toward it.
