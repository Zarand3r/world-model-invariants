# F12 — Does the operator statistic pick better checkpoints than validation loss?

**Registered 2026-08-30, before writing the measurement. No training.**

## Why

The paper has no downstream use, and two attempts to give it one both failed: the correction does not
improve control return (F5, `0.24%` of an episode's spread) and drift is not a usable online trust
signal (F2, beaten 3/3 by plain latent displacement). Both tried to **fix** a rollout, where the
effect is tiny beside the spread the decision is made over.

**Choosing between models is a different regime**, and it is where the signal is already known to be
large: F4b separates two architectures by `767x` on identical data. This asks whether that
discrimination is useful at the granularity a practitioner actually works at.

## The question

Given several checkpoints of the same model trained on the same data, does ranking them by
`rho_obs` predict **long-horizon rollout fidelity** better than ranking them by **validation
reconstruction loss** — the thing everyone already has — or by **probe accuracy** `rho_E`?

## Pool

`dreamer_ref` seeds 3/4/5 at steps 1000, 3000, 6500, 15000, 30000, 60000 — **18 checkpoints**, one
architecture, one dataset (`runs/pendulum_pixels.npz`). Validation loss comes from the recorded
training histories at the matching step.

## The target, and why it is not the obvious one

**Target: open-loop pixel fidelity** — roll the model forward 100 steps from a warmup state, decode,
and compare to the true frames. Lower is better.

**Deliberately not energy drift.** `rho_obs` is a conservation statistic; scoring it against a
conservation target would be close to circular and the result would mean nothing. Pixel fidelity is
what a practitioner actually selects on, and it is independent of conservation by construction.

## Registered predictions

- **P1 (primary).** `|Spearman(rho_obs, fidelity)| > |Spearman(val_recon, fidelity)|` across the 18
  checkpoints.
- **P2.** `rho_obs` also beats probe accuracy `rho_E`.
- **P3 (the harder test).** P1 still holds **within** each seed, where all six checkpoints share
  initialisation and data and differ only in training duration — at least **2 of 3** seeds.

## Gates, pre-committed

- **G1 (the target must vary).** If open-loop fidelity is flat across the pool there is nothing to
  rank and no prediction is readable. Required: the best and worst checkpoint differ by at least
  `2x` in pixel fidelity.
- **G2 (not a training-duration proxy).** Both `rho_obs` and `val_recon` will improve with training,
  so any ranker correlates with the target simply by tracking step count. P1 is only interesting if
  `rho_obs` beats `val_recon` **after partialling out training step**. Report the partial
  correlations alongside the raw ones; if `rho_obs` wins only on the raw ranking, say so and do not
  claim a selection result.

G2 is the gate I would most likely have skipped. F11 is the reason it is written down first.

## Falsifier

If `val_recon` ranks checkpoints as well or better, the statistic has no selection value, the
paper stays diagnostic-only, and direction 2 is closed. That is a publishable negative in the same
sense F2 and F5 are: it bounds the claim with a number.
