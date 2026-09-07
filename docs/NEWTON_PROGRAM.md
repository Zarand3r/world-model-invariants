# Transition interpretability — recovering the latent vector field, and eventually Newton

**Status: NOT explored. Documented 2026-09-07 for later.** Proposal from Richard's collaborator;
this file records it, checks it against what the repository already contains, and flags the two
pieces of our own evidence that bear on it most.

## The idea

Move from **feature** interpretability to **transition** interpretability.

Our existing result recovers a conserved scalar — a *first integral* of the learned dynamics,
`C(s_{t+1}) ~ C(s_t)`. Newton's second law is the complementary object: not a conserved scalar but
the **vector field generating the transition**. So the sequel is not "probe for `F`, `m`, `a`" but:

> Can we recover the coordinate system and vector field the frozen world model itself uses, and do
> they correspond to physical law?

A hierarchy of claims, which is the framing worth keeping:

| level | claim | test | strength |
|---|---|---|---|
| 1 | **carried** | a probe decodes `q, v, F, m` | weak |
| 2 | **predicts correctly** | rollouts approximately obey `F = ma` in pixel/state space | behavioural |
| 3 | **dynamics-native** | latent coordinates exist whose *transition* satisfies the equation | strong |
| 4 | **causally used** | manipulating those coordinates changes the transition per the law | strongest |

**Our E18 result is precisely a demonstration that levels 1 and 3 can disagree dramatically** —
energy is decodable at `rho = 0.9999` yet an unlabelled scalar is `6.7x` better preserved by the
model's own transition. That is the methodological foundation the rest of this program stands on, and
it is worth noting it is **not** one of the results withdrawn this year.

## Stage A — recover the latent equation of motion (existing pendulum model)

Search for learned scalars `Q(h)`, `P(h)` such that under the model's transition `T`:

    (Q(T(h)) - Q(h)) / dt  ~  P(h)
    (P(T(h)) - P(h)) / dt  ~  f(Q(h))

then **inspect the recovered `f`**. If `f(Q) ~ -k sin(Q)` without `Q` ever having been supervised
with the true angle, that is a latent coordinate system *plus* a latent equation of motion, recovered
from the model rather than imposed on it. Afterwards align `Q` with physical `theta` and ask whether
`k` corresponds to `g/l`.

This is the vector-field analogue of the unlabelled-invariant experiment.

## Stage B — a deliberately identifiable Newton environment

The existing pendulum **cannot** establish `F = ma`, and the reason is structural rather than
practical: for an ideal pendulum `m l^2 theta'' = -m g l sin(theta)`, so `theta'' = -(g/l)
sin(theta)` and **mass cancels**. A model trained on one pendulum has no reason to represent `m` at
all, and many latent explanations produce identical observations.

So Stage B needs a new environment: a visually observed point mass, `m x'' = F`, with independent
variation over mass, applied force, initial position and initial velocity — and a **held-out
combination**. Train on `(F1,m1)`, `(F1,m2)`, `(F2,m1)`; withhold `(F2,m2)`.

Then search, without telling the method what mass or force are, for

    Q_{t+1} - Q_t ~ P_t ,      P_{t+1} - P_t ~ G(U_t, M_t)

and ask whether the recovered `G` has the separable form `G(U,M) ∝ U/M`.

## Stage C — the interventions, which are the actual evidence

A supervised probe reporting `R^2(m) = 0.9999` means almost nothing on its own — that is the whole
point of E18. The claim rests on intervening on the *dynamics-native* mass coordinate:

- `M(h) -> 2 M(h)` with force held fixed should give `a -> a/2`
- `F -> 2F` should give `a -> 2a`
- **the symmetry test**, and the strongest of the three: `F -> lambda F` together with
  `m -> lambda m` should leave acceleration **unchanged**

The third is much harder to explain through a generic correlated feature than either of the first
two, and should be the primary registered prediction rather than a supporting check.

## What the repository already has, and what it does not

**Partly exists.** `latent_noether/poisson.py::fit_hamiltonian_pair` already fits a *vector field*,
not just a scalar: it finds an antisymmetric `B` with `F ~ B grad C`, where `F` is the latent
one-step displacement. Stage A can build on this rather than start from nothing. **What has never
been done is inspecting the recovered field's functional form** — `B` and `grad C` have only ever
been used as a means of finding `C`.

**Does not exist.** No `Q, P` coordinate recovery, no inspection of `f`, and — checked — **no
environment with variable mass**. Every system in `latent_noether/envs.py` is unit-mass. Stage B
requires a new environment, which is real work but not large.

## The two results of ours that bear on this most

**1. F1 is the closest prior attempt and it was NEGATIVE.** F1 asked whether the model implements a
dynamical *relation* it demonstrably obeys — the power balance law under actuation, `dC/dt = P`. On
models that verifiably use their actions (0.740--0.761 true vs shuffled), the action's power explains
**0.31%** of the variance in `dC`, with the observed change 4.5--5.4x larger than predicted. The
conclusion recorded at the time: *the model represents the quantity and does not implement the
relation it obeys.*

That is directly relevant. Stage A asks a similar question in a different regime — free dynamics
rather than actuated, and the whole vector field rather than one balance law — so F1 does not settle
it. But **any Stage A preregistration must state why it expects a different answer**, and if it
cannot, that is a reason to reconsider before spending the compute.

**2. The attribution trap applies to Stage A but not to Stage C.** Recovering `f` by fitting to the
model's transition **on real trajectories** inherits exactly the problem that left F6 unresolved: a
model that predicts well reproduces the next state of whatever it is shown, so the recovered `f`
could be a property of the evaluation data. Stage A must be designed with that separation built in
from the start — the F13 pattern (hold the data fixed, vary a conditioning input) is the template.

**Stage C is immune by construction**, in the same way E1's interventions are: it edits a latent
coordinate and measures the consequence under the model's own rollout against a matched control.
That is a genuine strength of this program — it has a level-4 test built in, and level 4 is the level
our confounds cannot reach.

## Cost, roughly

| stage | needs | estimate |
|---|---|---|
| A | no new training; new fitting code on existing checkpoints | days |
| B | new environment, new dataset, ~3 seeds trained | ~1 week |
| C | no new training beyond B; intervention harness exists (E1/E4 pattern) | days |

## Sequencing note

The proposal suggests attempting this in Dreamer before a larger video-diffusion model, because a
recurrent latent gives a compact, explicit transition state to interrogate while a 14B diffusion
architecture makes localisation much harder. That ordering is right, and for an additional reason
specific to us: every measurement we own is defined against an explicit one-step transition `T`, and
none of it ports to a model without one.
