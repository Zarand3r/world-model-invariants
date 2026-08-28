# Pre-registration — E12c, mid-rollout interchange (the dormant-pathway test)

**Written 2026-08-28, before any E12c quantity was computed.**

## What is still open

Makelov, Lange & Nanda (ICLR 2024) show a subspace edit can produce the expected output change
through a **dormant pathway the model does not normally use**. Level-set projection is a subspace
intervention, so this is the sharpest live objection to E1 and E4.

Two partial answers exist. **E12 failed as registered** — its within-trajectory correlation had no
power, since `C` varies by only 3-4% of its across-trajectory spread within a rollout. **E12b**
showed `C`'s drift predicts decoded-energy drift on **unedited** rollouts (+0.74/+0.45/+0.74 against
random +0.02), which is on-pathway evidence but correlational.

What neither does is an **interchange at depth**: edit the `C` subspace *in the middle of an
autonomous rollout* and ask whether the model's subsequent imagined physics follows.

## Design

Roll autonomously for `k = 50` steps. At step `k`, set `C(z_k)` to the value of an **independent
donor trajectory** by minimal displacement along the local `C`-normal — the E4a protocol, applied
mid-rollout rather than at `t = 0`. Then continue rolling free for another 50 steps and decode.

The distinction from E4 matters: E4 edits an **encoder-produced** state, which the model has been
trained to represent. E12c edits a state the model reached **on its own**, 50 steps into imagination.
A dormant pathway would be expected to behave differently there.

## Primary metric

Spearman between the intended change in `C` and the realised change in decoded physical energy over
the post-edit segment, across donor-recipient pairs.

- **Registered prediction:** `rho > 0.5` with a bootstrap CI excluding 0, on all three seeds. If the
  subspace is genuinely used by the forward pass, editing it mid-imagination should steer the
  imagined physics much as editing it at `t = 0` does (E4 gave +0.808 to +0.916).
- **Falsifier:** `rho` near zero or inconsistent in sign while E4's `t = 0` edit still works. That is
  the dormant-pathway signature — the subspace is reachable from encoder states but not used by the
  model's own dynamics — and E1/E4 would need reinterpretation.

## Controls

- **Equal-norm tangent edits** at the same depth: cannot change `C` to first order, so should not
  steer energy.
- **Magnitude-matched random directions**, 10 draws.
- Reported alongside: the same statistic at `k = 0`, which is E4's protocol, so the two depths are
  compared within one run rather than across log entries.

## Scope

Pendulum seeds 3/4/5 at step 6,500 — where E4 is characterised at n = 3. Analysis split, `C` and the
coordinate frame frozen, no refitting.

## Known limit

A positive result shows the recovered **subspace** is causally deployed at depth, not that the model
maintains an internal energy register. The interpretation rule from `E4_PREREG` carries over
unchanged.
