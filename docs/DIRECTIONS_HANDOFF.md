# Directions handoff — where the odds actually move

**2026-08-30.** Companion to `docs/REVIEWER_HANDOFF.md` (current state) and
`docs/REFRAME_PROPOSAL.md` (what the paper can claim). This one is forward-looking: what to run
next, ranked by how much it shifts a main-track decision.

Rendered version: <https://claude.ai/code/artifact/998f0b86-4325-4847-90fc-3fa2ee30cf69>

## 1. The odds

Main-track acceptance, single submission, no resubmission cycle.

| state of the paper | ICLR | NeurIPS | ICML | what decides it |
|---|---|---|---|---|
| **as drafted** | -- | -- | -- | **Unsubmittable.** The title asserts the model learns its simulator's integrator; the attribution to the model rather than to the data it was measured on is untested. |
| **reframed, as-is** | 25--30% | 20--25% | 15--20% | A probing-methodology result with an unusually well-controlled negative section. Two toy systems and no downstream benefit are the binding constraints. |
| **reframed + directions 1 & 2** | 45--55% | 40--50% | 35--45% | Removes both binding constraints: the result stops being about a pendulum, and the paper gains a downstream use it currently lacks. |

**Weight these lightly.** Over two days this week I called a result the cleanest in the project,
recommended withdrawing two others on the strength of it, and retracted that recommendation --- all
on one evidence base. The *ranking* below is more reliable than the percentages, because it turns on
which reviewer objections are binding rather than on how severe I guess they are.

## 2. Why those numbers

**Carrying it:** a properly separated model-vs-data result (conv-GRU vs RSSM on *identical*
trajectories, `767x`); causal interventions that were **measured** immune, not argued immune (repair
3/3 with `C` fitted on a different dataset, `-38.8%` vs `-38.3%`); four preregistered negatives each
with a measured cause; 33 preregistrations with falsifiers written before results.

**Holding it back:** two smooth Hamiltonian toy systems at 13.5M parameters with no released
checkpoint --- the most predictable reviewer sentence there is; **no downstream benefit**, tested
twice and negative both times; a core claim that overlaps known critique (that probes can be
epiphenomenal is not new --- the operator-side alternative and the causal confirmation are, and that
has to be argued rather than assumed); `n = 3` throughout; E10b not established.

## 3. Directions, ranked by leverage

The first two are worth more than the rest combined, because each removes one binding constraint.

### 1. Show the dissociation on a standard benchmark  *(cost high, +15--20 pts)*

Run the probe-versus-operator comparison on a DMC or publicly released DreamerV3 checkpoint. The
claim needs no conserved scalar --- only a quantity that is **decodable and dynamically relevant**
(contact state in walker, centre-of-mass energy in cheetah). Largest single lever: it converts a
pendulum paper into a world-model paper, and it is the objection all three venues raise
independently.

*Kill condition:* if the dissociation vanishes at scale that is a real, publishable negative, but it
ends the generality claim. Heed F3's warning --- pick a quantity that **varies**; one that never
varies carries no information and lives in decoder weights.

### 2. Model selection by the operator statistic  *(cost low, +8--12 pts)* -- **start here**

Both previous downstream attempts tried to **fix** rollouts (a planning correction, a per-rollout
trust signal) and both failed for the same measured reason: the effect is tiny beside the spread the
decision is made over. Choosing **between** models is a different regime, and it is where the signal
is already known to be enormous --- F4b separates two architectures by `767x` on identical data.

Concrete test: does ranking checkpoints by `rho_obs` predict long-horizon rollout fidelity better
than ranking by validation loss or by probe accuracy? Every asset exists --- F6's twelve models
across four timesteps, F4b's conv-GRUs at two budgets, the step-60,000 arms. **No new training.**

*Kill condition:* if validation loss ranks them just as well, the statistic has no practical claim.
Register that comparison as the primary outcome **before** running.

### 3. Mixed-timestep or `dt`-conditioned training  *(cost medium, +10--15 conditional)*

The only route to settling the attribution. Cross-evaluation cannot do it: one-step error rises from
`~0.007` rad on a model's own timestep to `~0.6` rad on another's --- nine times the separation the
test needs. Highest ceiling here and the highest variance, on the axis that already cost a week.

*Precondition, non-negotiable:* run `tests/test_observable_difference.py` against the design
**before** any GPU time. Four experiments died because the difference tested lived in the simulator's
bookkeeping rather than in the pixels the model consumes.

### 4. A third architecture family  *(cost medium, +4--6)*

The `767x` gap is RSSM against conv-GRU. A transformer world model makes the axis a spectrum and
tests whether the preserved direction is a property of recurrent state at all. Do it only if 1 is
blocked.

### 5. A non-physics invariant  *(cost medium, +3--5)*

Everything so far is a conserved scalar in a Hamiltonian system. The same dissociation for a discrete
or logical property would show the finding is about probing rather than physics. Largely subsumed by
direction 1 if that lands somewhere with non-physical structure.

## 4. Not worth running

| candidate | why not |
|---|---|
| more seeds | `n = 3` is not what is doubted; effects are well outside their controls |
| a third toy system | adds a system, not a domain --- and confirms the objection |
| another intervention variant | the causal case is already the strongest part |
| a sixth dissociation axis | reviewers count claims, not axes |
| re-running the scheme comparison | the two schemes reduce to one position recurrence; withdrawn and pinned by a test |

## 5. If you only do one thing

**Direction 2.** No training, existing checkpoints, and it fills the hole two previous experiments
failed to fill --- the hole a reviewer names right after "this is a pendulum".

If you can afford two, add **direction 1** and write the reframe while it runs. The reframe rests
only on already-separated evidence, so it is robust to however 1 and 3 turn out --- and if 3 later
succeeds, the integrator result returns to the headline of a paper that was already sound, rather
than propping up one that was not.
