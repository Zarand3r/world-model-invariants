# Pre-registration — F3, from orbit-label invariants to constraint residuals

**Written 2026-08-29, before any F3 quantity exists. NOT YET APPROVED TO RUN.**

Roadmap `docs/ROADMAP.md` F3, which states the requirement plainly: *"do not simply run the current
invariant extractor on limb-length or contact constraints… First generalize the method."*

## Why the current method cannot find a constraint, stated precisely

The extraction minimises the **invariance ratio**, within-trajectory variance divided by **total**
variance, as a generalised eigenproblem `W a = lambda T a`. That objective is built for
*orbit-label* invariants: quantities constant **within** a trajectory that **vary between**
trajectories. Energy is one.

A constraint is the opposite kind of object. `G(z) = 0` holds for **every** state of **every**
trajectory, so its total variance is ~0 and it lies in `T`'s null space -- precisely the degenerate
direction the ridge exists to suppress. **The current method does not merely miss constraints; it
actively excludes them.** Running it on limb lengths would return whatever the regulariser happened
to leave, which is the failure mode this project has already been bitten by twice.

## The generalisation

Seek `G` minimising the **second moment** `E[G(z)^2]` subject to `||a|| = 1`, rather than a variance
ratio. The smallest eigenvector of the feature second-moment matrix, not of `T^-1 W`.

That objective is trivially satisfiable by numerical degeneracy, which is the trap F1's balance
extractor fell into (`cond(T) = 9.9e38`, a residual readable as anything from 0.97x to 5445x). Three
defences, all carried over from lessons already paid for here:

1. **Standardise the monomial features** and use a **trace-relative ridge**, as
   `polynomial_invariants` does and as `balance.py` originally failed to.
2. **Held-out evaluation.** `G` is fitted on one half of the trajectories and its residual reported
   on the other, as E9 established and F1's amendment 2 had to retrofit.
3. **A positive control on a known constraint**, below. Without it, "we found a direction with small
   residual" is unfalsifiable.

## Stage A -- validate on a constraint we already have

**The pendulum has an exact algebraic constraint in its own pixels: the rod length is fixed.** The
ink centroid's distance from the calibrated pivot is constant across every frame of every
trajectory, and the frozen readout in `pixel_readout.py` already measures both. So

    G_true(state) = ||centroid - pivot|| - L

is known exactly, is genuinely a constraint rather than an orbit label, and requires no new data, no
new model, and no external checkpoint. Stage A runs on the **existing three reference checkpoints**.

### Registered predictions (Stage A)

**A1 -- recovery.** The recovered `G` correlates with `G_true` at `|rho| >= 0.5` on at least 2 of 3
seeds, **on held-out trajectories**.

**A2 -- it is a constraint, not an invariant.** The recovered `G` has held-out second moment at least
`5x` smaller than the best *orbit-label* invariant the existing search returns on the same latent.
If the existing method does as well, the generalisation is empty.

**A3 -- specificity.** `G`'s held-out second moment is at least `5x` below that of 20 random
directions at matched coefficient norm.

> **Gate: if A1 fails, STOP.** A method that cannot recover a constraint we can write down exactly,
> in a model we already trust, has no business being pointed at a walker. Report as a negative about
> the method and do not proceed to Stage B.

## Stage B -- a released checkpoint, only if Stage A passes

Apply the validated extraction to a public DreamerV3 checkpoint and ask whether it represents its
own kinematic constraints. **Registered infrastructure risks, so a failure there is not misread as a
scientific result:**

- `dreamer_adapter.py` hardcodes `num_actions=1` and a 64x64 single-camera observation. A walker
  checkpoint has a 6-dimensional action space and different encoder shapes. Loading one is an
  adapter change, not a config change.
- If no released checkpoint can be loaded and verified to reconstruct its own training frames, **that
  is an infrastructure blocker and F3 Stage B is reported as not run** -- not as a negative result
  about constraints.

Stage B gets its own registered predictions once Stage A's outcome is known. Registering them now
would be registering against a method that may not exist.

## Scope and honesty constraints

- 3 seeds; the model seed is the independent unit, as everywhere in this project.
- `G_true` is used **only** for evaluation, after the fit is frozen, per the roadmap's rule at line 79.
- Records carry the `latent_noether.provenance` stamp.
- **A negative is a real outcome.** If a world model trained on pendulum video does not represent its
  own rod-length constraint as a low-residual latent direction, that is worth reporting: it would say
  the representation encodes the *state* without encoding the *kinematics* that generate it, which is
  a sharper version of the paper's existing thesis.

## Status

**AWAITING APPROVAL.** Stage A needs no new training, no new data and no external download -- it runs
on checkpoints already in hand. Estimated under an hour.
