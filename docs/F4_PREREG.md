# Pre-registration — F4, a second world-model family

**Written 2026-08-28, before any F4 model was trained.** `docs/ROADMAP.md` Phase V, unlocked by its
own condition: *"After the phenomenon is established in a second physical system, replicate it in
another world-model family."* The 2-DoF system is established at n = 3, so F4 is now in order.

## The question

Every result in this project comes from **one architecture** — DreamerV3's RSSM, with categorical
stochastic latents, KL balancing and unimix. A reviewer's obvious question is whether the recovered
invariant is a property of *learned world models* or a property of *this particular latent design*.

## The contrast, chosen to be maximal within the same interface

A **deterministic convolutional GRU autoencoder**: conv encoder -> GRU -> conv decoder, trained on
pure reconstruction with **no stochastic latent, no KL term, no unimix**. Every mechanism that makes
an RSSM an RSSM is removed; what remains is the minimum needed to be a pixel world model.

The extraction requires exactly three methods — `encode`, `transition`, `readout_from_h` — so the
entire analysis pipeline runs **unchanged**. That is the point: any difference in result is
attributable to the architecture and not to the measurement.

## Frozen, exactly as for DreamerV3

LD = 12, degree 4, `n_basis` = 8, WARMUP = 10, step-capped at 60,000 with the E8 milestone grid.
`--max-hours` non-binding. Same datasets, same analysis split, same direction-matched intervention
protocol and `eps` grid.

Hidden size chosen so the GRU's recurrent state is **512**, matching DreamerV3's `deter`, so the
latent the extraction operates on has the same dimensionality. Parameter count will differ and is
reported, not matched — matching capacity would require changing the architecture, which defeats the
comparison.

## Acceptance, before any analysis

The same registered training checks the RSSM had to pass, minus the KL check, which is undefined for
a deterministic model: one-step decoding at least 4x better than predicting the dataset mean, and
finite non-collapsed open-loop rollouts. **A model failing these is not evidence about the method**
and its extraction results are discarded, not reported.

## Registered predictions

1. **Recovery**: `|rho_E| > 0.8` on the pendulum at some checkpoint, with an invariance ratio within
   two orders of magnitude of the RSSM's.
2. **Repair**: direction-matched projection reduces decoded physical energy drift, with 0 of 20
   magnitude-matched random directions beating it.
3. **Refusal**: the damped arm shows the same collapse of specificity the RSSM showed.

**Falsifier, and it is the informative outcome:** if recovery fails on a model that passes the
acceptance checks, the invariant is a property of the RSSM's latent structure rather than of learned
world models generally, and **every claim in the project must be qualified to that architecture**.
That would be a significant negative result and will be reported as one.

## Scope

Pendulum first, 3 seeds, since that is where the RSSM baseline is most thoroughly characterised
(n = 3 across recovery, repair, disjoint evaluation, E8 and E4). The 2-DoF system is a later
extension and is not part of this registration.

## Known limit

One alternative architecture is not "learned world models in general". A negative result would be
decisive for the general claim; a positive result establishes two families rather than a class, and
will be worded that way.
