# World-Model Invariants: Experimental Roadmap

**Status:** adopted 2026-08-26. This document governs all further experimental work.
Every implementation decision should serve or falsify the target claim stated at the end.

## Mission

The existing paper shows that a conventionally trained DreamerV3 world model, trained only on
conservative pendulum pixels, contains a label-free recovered scalar `C(z)` that:

1. is highly correlated with true pendulum energy;
2. is approximately constant on observation-conditioned trajectories;
3. is absent as a comparable invariant in matched damped models;
4. drifts during autonomous imagination;
5. when restored toward its initial level set, improves 50-step pixel rollouts by roughly 3%;
6. outperforms random polynomial constraints on two of three trained seeds.

**The current paper is not yet the final scientific story.**

The goal of all further work is to determine whether the following stronger mechanism is true:

> A standard world model spontaneously learns physically meaningful dynamical structure. Its
> transition behaves consistently with that structure on states supported by real observations, but
> recursive imagination drives the model into states where that structure is no longer respected.
> Violations accumulate into long-horizon physical and prediction error. The learned physical
> structure can both diagnose this failure and causally control or repair imagined dynamics.

**Do not optimize for the number of experiments. Optimize for decisive evidence for or against this
chain.**

---

## Claim hierarchy

Treat the project as six progressively stronger claims. Every experiment must map explicitly to one.

| | Claim | Statement | Status |
|---|---|---|---|
| **C1** | Emergence | Ordinary pixel-prediction training makes a physical invariant reproducibly emerge. | Supported; needs more seeds and training-to-convergence. |
| **C2** | Physical validity | The recovered latent quantity corresponds to actual physics in decoded imagined worlds, not merely to a self-consistent latent scalar. | **Not yet established decisively.** |
| **C3** | Failure mechanism | ~~The model locally respects the learned law on observation-supported states but loses this property during recursive autonomous rollout.~~ **Amended 2026-08-26 (provisional, seed 3 only, approved by Richard):** the model's transition carries a near-constant systematic violation of the learned law at every step, on and off the observation-supported manifold, which integrates into large long-horizon physical error. | **Original form falsified by E2 on seed 3.** Amended form confirmed by E3, and E2 was subsequently run on seeds 4 and 5 (`e2_s4_depth100`, `e2_s5_depth100`) -- see the execution-status table above. |
| **C4** | Causal physical control | Changing the recovered invariant changes the physical regime imagined by the model in the quantitatively expected direction. | **Not yet established.** |
| **C5** | Predictive utility | Invariant violation provides an internal early-warning signal for later rollout failure. | Preliminary evidence, weak. |
| **C6** | Generality | The complete phenomenon survives beyond a 1-DoF pendulum. | **Not yet established.** |

---

## Phase 0 — Freeze methodology before new results

Before running each major experiment, create a preregistration file containing:

- scientific hypothesis
- primary metric
- exact dataset split
- exact checkpoints
- exact seeds
- intervention grid
- horizons
- exclusion rules
- control arms
- expected qualitative result
- result that would falsify the hypothesis

**Never replace a preregistered metric after seeing results.**

Store:

- immutable raw rows
- model/config hashes
- random seeds
- git commit
- derived summaries **separately** from raw results

All final figures and tables must regenerate from committed raw rows.

Treat **model seed — not trajectory — as the independent experimental unit** for model-level claims.
Use hierarchical/bootstrap uncertainty when trajectories are nested within models.

Physical labels may be used **only after invariant extraction is frozen**, for evaluation and
explicitly declared calibration experiments.

---

# PHASE I — Critical experiments on existing checkpoints

Highest-information experiments. Do these before broadening the project.

## E1 — Decode true physical energy from imagined pixels

**Priority: P0. Gate experiment.** Claim: C2.

Current rollout evaluation uses pixel MSE only. Establish whether invariant correction actually
improves *physical* dynamics.

Implement a **non-learned** image-to-angle estimator using renderer geometry:

- estimate rod orientation from pixels using image moments / known center
- unwrap `theta`
- estimate `thetadot` using central finite differences
- compute physical energy using the simulator's existing `energy()` implementation

**Do not use a learned probe as the primary result.**

Validate the estimator separately on:

1. true rendered frames
2. Dreamer reconstructions of observation-conditioned frames

Quantify the readout noise floor.

Because Gym's numerical integrator produces oscillation in textbook energy, compare predicted
decoded energy against the **matched simulator reference `E_ref(t)`** rather than assuming textbook
`E` should be perfectly constant.

For each `alpha`, report:

- physical energy error versus time
- secular physical energy drift
- pixel MSE
- angle error
- angular-velocity error

**Arms:** recovered `C` · matched random constraints · damped model · no edit.

### Gate

If latent-`C` correction improves pixel MSE but **does not** improve decoded physical energy, stop
describing the intervention as repairing physical dynamics. Reinterpret it as a latent regularizer
and investigate why.

If physical energy improves, proceed aggressively.

---

## E2 — Locate exactly where conservation fails

**Priority: P0. Potential mechanistic centerpiece.** Claim: C3.

Missing from both the current paper and prior reviewer roadmaps.

Define the local conservation defect:

    r(z) = C(T(z)) - C(z)

Measure `r` on two distributions.

**Teacher-forced / observation-supported states.** At every real encoded state `z_t^obs`, apply the
autonomous transition once:

    r_obs(t) = C(T(z_t^obs)) - C(z_t^obs)

**Self-generated states.** Start from the same initial states and roll autonomously. At depth `k`:

    r_auto(k) = C(T(z_k^auto)) - C(z_k^auto)

Measure for increasing rollout depth.

Also quantify distance of `z_k^auto` from the empirical observation-conditioned latent distribution
using at least one fixed, **preregistered** metric such as whitened nearest-neighbor distance.

### Distinguish three mechanisms

- **Outcome A:** `r_auto` grows strongly with rollout depth.
  *Interpretation:* recursive imagination moves into states on which the learned transition no
  longer respects its own physical structure. **This is the strongest result.**
- **Outcome B:** `r_auto` remains roughly constant but nonzero.
  *Interpretation:* a small systematic integration bias accumulates over time. Still interesting,
  but a different mechanism.
- **Outcome C:** `r_auto` oscillates coherently with phase.
  *Interpretation:* invariant drift may be coupled to phase/timing error. Then prioritize E5/E11.

**Report all outcomes honestly.**

---

## E3 — Decompose one-step error normal vs tangent to the invariant surface

Claim: C3.

For an observation-conditioned state, compare the autonomous next state with the next
observation-conditioned state:

    dz_t = T(z_t^obs) - z_{t+1}^obs

With `g_t = grad C(z_t)`, decompose `dz_t = dz_perp_t + dz_par_t`, where

    dz_perp_t = ((g_t^T dz_t) / ||g_t||^2) g_t

Ask:

- Does `||dz_perp||` predict future rollout error better than `||dz||`?
- Does tangent error mostly affect phase while normal error changes physical energy?
- Does the existing correction selectively remove the normal component?

This gives the projection a geometric interpretation: **invariant correction approximately removes
transition error transverse to the learned physical manifold.**

---

## E4 — Counterfactual invariant dialing

**Priority: P0/P1. Potential headline result.** Claim: C4.

Do not merely restore `C = C_0`. Set the level to new values. Prefer two protocols.

### E4a — In-distribution donor-level intervention

Take a recipient state and an independent donor trajectory. Change the recipient state minimally
along the local `C`-normal direction so that

    C(z_edited) = C(z_donor)

**The intervention itself must never use true energy.**

Then decode the resulting imagination and ask whether its measured physical energy shifts toward the
donor trajectory's true energy. This avoids arbitrary synthetic scales and connects naturally to
causal interchange.

### E4b — Synthetic sweep

Choose preregistered positive and negative offsets `C_0 + dC`. Measure:

- realized physical energy
- amplitude
- period
- direction/regime
- persistence of the new regime

Where possible, test the pendulum **separatrix**. The dataset contains both libration and rotation
trajectories. Ask whether increasing `C` can move an imagined trajectory from libration to rotation,
and whether the new regime persists without continued forcing.

**Do not assume `C = E`.** It may be a nonlinear monotone function of physical energy. Primary
evidence should therefore be:

- monotonicity
- transfer correlation
- calibrated predicted-vs-realized relationship
- regime transitions

If an `E <-> C` calibration is needed, fit it **only on a dedicated calibration split** and freeze it
before evaluation.

---

## E5 — Causal deployment / natural-value interchange

Claim: C4.

Liao & Cao motivate this experiment, but **do not mechanically copy their linear-direction
protocol**. Our `C(z)` is nonlinear and has a state-dependent normal direction.

Implement a local, minimal-norm **natural-value** edit that transfers a donor trajectory's `C` value
to a recipient. Then measure how a subsequent transition responds.

Possible scalar outputs:

- analytically decoded next-step physical energy
- next-step `C`
- physically interpretable next-frame quantities

Compute a transfer statistic analogous to transfer-correlation between the donor-induced change and
the post-transition physical change.

**Controls:** matched tangent edit · matched random direction · recovered `C` in damped models ·
high-energy-decodability quantities that are poorly conserved · untrained Dreamers.

### Important interpretation rule

A positive transfer statistic alone does **not** prove that Dreamer explicitly computes with
"energy" as an independent internal variable. The edit changes the latent microstate.

Call this evidence for **causal deployment of the recovered physical subspace**, not proof of a
literal internal energy register, unless stronger localization evidence supports that statement.

---

## E6 — Promote and extend the horizon mechanism

Claim: C5.

Preliminary evidence suggests accumulated invariant violation becomes more predictive at longer
horizons.

Generate fresh long evaluation trajectories if necessary; the current 120-frame dataset cannot
support arbitrary `H = 200` evaluations from all start states.

Evaluate, where meaningful, `H in {10, 25, 50, 100, 200}`. For every horizon measure:

- pixel error
- physical energy error
- phase error
- accumulated invariant defect
- correction benefit

Test whether **correction benefit(H) grows with horizon**.

Compare predictors of final rollout error:

1. time / generic power law
2. accumulated invariant violation
3. instantaneous invariant violation
4. latent distance from observation-supported states
5. one-step prediction error
6. phase drift

**Do not oversell the existing `R^2 ~ 0.19`.** The important question is whether the relationship
becomes reproducible and mechanistically specific.

---

# PHASE II — Make the result statistically defensible

Run partly in parallel with Phase I.

## E7 — Expand seeds

Claim: C1.

Train approximately **20 conservative** and **10-20 damped** models.

**Do not discard inconvenient successful-training seeds.**

Report distributions of:

- `|rho(C, E)|`
- invariance ratio
- local conservation defect
- autonomous drift
- physical-energy correction benefit
- pixel correction benefit
- recovered-vs-null percentile

Use confidence intervals. This resolves whether the current 2-of-3 specificity result is typical or
accidental.

---

## E8 — Training-to-saturation checkpoint sweep

Claim: C1, C3.

Current models were trained for only ~6.5k updates. Train representative models substantially longer
and save checkpoints at `{1k, 3k, 6.5k, 15k, 30k, 60k}` until validation performance clearly
saturates.

At every checkpoint measure: one-step prediction error · reconstruction quality · invariant recovery
· local conservation · autonomous drift · physical rollout error · correction benefit.

### Critical result

The strongest outcome is: **predictive training converges, the invariant remains strongly
represented and locally conserved, but autonomous violation persists and correction remains
beneficial.** That establishes that learning the law and reliably integrating according to it are
distinct capabilities.

If invariant drift disappears with adequate training, **revise the paper rather than hiding this**:
invariant drift is then a diagnostic of incompletely trained world models.

---

## E9 — Truly disjoint causal evaluation

Claim: C2, C4.

Do not continue using the same 52 trajectories for invariant fitting and intervention scoring.

Create `D_train`, `D_discover`, `D_calibration/val`, `D_test`:

- train Dreamer on `D_train`
- discover `C` on `D_discover`
- calibrate any physical readouts / intervention values on `D_calibration`
- report final intervention effects on untouched `D_test`

Existing preregistered `alpha` grids remain fixed unless a new preregistration explicitly changes
them.

This can be tested immediately on existing models by generating fresh intervention trajectories
while keeping their already-fitted `C` frozen.

---

# PHASE III — Strong nulls and rival mechanisms

Run after the central mechanism has survived E1-E9.

## E10 — Decodability-matched but conservation-mismatched null

**One of the most important controls.** Claim: C2, C4.

The present random polynomial null answers *"does an arbitrary latent constraint help?"* It does not
answer *"is conservation specifically what matters?"*

Use the generalized eigenspectrum already produced by the invariant search. Find candidate scalars
`Q_i` whose true-energy decodability is close to that of `C` but whose invariance ratios differ.
Compare intervention benefit at approximately matched energy decodability.

Desired analysis: **intervention benefit vs invariance quality, conditional on energy
decodability.**

A decisive result would show: equally energy-decodable quantities do not provide equal corrective
value; benefit tracks dynamical conservation. This directly supports the paper's probe-vs-dynamics
thesis.

---

## E11 — Phase-drift rival

Claim: C3.

Recent work shows autoregressive dynamics errors can be overwhelmingly phase errors. Test the
strongest simple rival explanation.

Construct a **preregistered** phase/timing baseline learned only on calibration trajectories, for
example a global transition-speed or phase-rate correction.

Compare: no correction · phase correction · invariant correction · combined correction. Measure both
physical-energy and phase error.

Possible outcomes, all scientifically useful:

- **Energy correction reduces energy error first, then phase error.** Strong evidence that phase
  drift is downstream of physical invariant drift.
- **Phase correction matches the invariant intervention.** The mechanism may primarily be timing.
- **Combination helps substantially.** Both are independent rollout-error modes.

---

## E12 — Faithfulness / subspace-illusion controls

Claim: C4.

Because successful activation edits can activate dormant pathways, **do not infer mechanism from
intervention success alone.**

Use: small local edits · natural donor values · equal-norm tangent edits · matched random
directions/subspaces · multiple edit magnitudes · verification that `C` predicts behavior on
unmodified trajectories · comparison of edit-induced states with the normal latent data
distribution.

Ask whether arbitrary equal-rank interventions can reproduce the same physical effects.

---

## E13 — Full causal matrix: trained x damped x untrained

Claim: C1-C4. Candidate headline figure.

Push the untrained controls through the **entire** pipeline, not merely invariant recovery.

| Model | Energy decodable? | Conserved? | Physical correction helps? | Counterfactual transfer? |
|---|---|---|---|---|
| trained conservative | | | | |
| trained damped | | | | |
| untrained | | | | |

The ideal pattern:

- **untrained:** sometimes decodable, not conserved, not causally useful
- **damped:** physical state decodable, no conserved energy-like scalar, correction useless
- **conservative trained:** decodable, conserved locally, causally useful

---

## E14 — Held-out energy / OOD tests

Claim: C6.

The model may be doing case-based generalization rather than learning broadly valid physical
structure.

First, with existing models, generate physically valid trajectories whose energies lie outside the
training distribution while staying safely below simulator clipping. Freeze `C` and test: invariant
recovery · local conservation · counterfactual dialing · correction benefit.

A later, stronger version: deliberately retrain Dreamer with an **excluded energy band** and test
whether `C` and the repair extrapolate through that missing band.

---

## E15 — Damping dose-response

Lower priority. Claim: C1.

Sweep damping strength `zeta` instead of using only one dissipative condition. Ask whether
increasing dissipation causes a systematic transition:

    strong approximate invariant -> weak approximate invariant -> no useful invariant

This may reveal an **invariance timescale** rather than a binary conservative/dissipative
distinction.

Do not assume monotonic energy correlation must occur; the finite observation horizon matters.

---

## E16 — Basis/subspace robustness

Lower priority unless reviewers attack PCA dependence.

Test whether recovery depends critically on choosing top-variance PCA coordinates. Possible
controls: multiple extraction dimensions · random low-dimensional projections from the full
recurrent state · alternative fixed linear subspaces.

**Note:** merely rotating coordinates within the same 12-D PCA span is *not* a meaningful test — the
degree-4 polynomial function class is essentially invariant to such rotations. The real question is
whether the selected high-variance span is **necessary**.

---

# PHASE IV — Generalize the central phenomenon

## E17 — Coupled anharmonic oscillator

**Highest-value generalization experiment.** Claim: C6.

Do this for a scientific reason, not benchmark breadth. The pendulum has one degree of freedom,
where energy is essentially the only nontrivial scalar that stays constant within a trajectory while
differing across trajectories.

Use a 2-DoF coupled anharmonic system where:

- energy transfers between subsystems
- total energy remains conserved
- energy no longer uniquely specifies the trajectory
- anharmonicity removes trivial independent normal-mode conservation
- observations remain simple enough for reliable pixel rendering

Prefer a Hamiltonian of the schematic form

    H = (1/2)(p1^2 + p2^2) + V(q1, q2)

with nonlinear onsite terms and coupling. Use a **symplectic** numerical integrator.

Run the same pipeline with minimal modification:

    recover -> matched dissipation control -> local conservation -> autonomous failure
      -> decoded physical error -> correction -> counterfactual dialing

**First attempt the already-fixed extraction hyperparameters. Do not silently modify the method
until it succeeds.** If adaptation is required, record the original failure and treat the adapted
method as a separate experiment.

If the full phenomenon reproduces here, the 1-DoF objection is substantially weakened.

---

# PHASE V — Follow-up research program

Should not delay the core paper unless core experiments finish early.

## F1 — From conservation laws to balance laws under action

Introduce nonzero torque. Energy now obeys a balance relation rather than `dE = 0`:

    dE/dt = input power - dissipation

Ask whether an ordinary action-conditioned world model spontaneously learns a latent scalar obeying
this accounting relation. Then test whether correcting toward the **predicted energy balance**,
rather than toward a constant level set, improves imagination.

This is the natural bridge to model-based RL.

## F2 — Invariant drift as an online trust signal  **[ATTEMPTED 2026-08-28 — NEGATIVE]**

> **Result: does not work.** Accumulated invariant drift at rollout step 25 predicts decoded physical
> energy error at step 100 with Spearman -0.02 / +0.09 / -0.24, beaten on all three models by simple
> latent displacement (+0.27 / +0.29 / +0.31) and beaten on two of three by a random-constraint
> control that was registered to sit near zero. The registered falsifier fired. The physical trust
> horizon below is **not supported** and must not be claimed. See `docs/F2_PREREG.md` and the
> execution log entry of 2026-08-28.

Test whether the model can know its imagination is becoming unreliable *before* large visible error
appears.

At rollout time estimate: local conservation defect · accumulated conservation defect · distance
from observation-supported latent states. Predict future rollout failure.

Compare against: reconstruction error · one-step error · latent norm · ensemble uncertainty if
available · any existing recurrent confusion signal.

Define a practical **physical trust horizon**: stop trusting autonomous imagination when learned
physical consistency crosses a calibrated threshold.

## F3 — Larger / public world models

A released DreamerV3 walker-walk checkpoint is attractive for scale, but **do not simply run the
current invariant extractor on limb-length or contact constraints.**

The current objective searches for scalars that (1) remain constant within a trajectory and (2) vary
between trajectories. Fixed limb lengths and contact constraints are a different mathematical
object: **algebraic/kinematic constraints rather than orbit-label invariants.**

First generalize the method from

    C(z_t) = constant along trajectory

to constraint residuals such as

    G(z_t) = 0

Then public control checkpoints become a meaningful testbed. This is a separate methodological
extension.

## F4 — Second architecture

After the phenomenon is established in a second physical system, replicate it in another world-model
family. Lower priority than eliminating the 1-DoF objection.

The question: is the phenomenon specific to Dreamer's RSSM, or characteristic of autoregressive
learned world models more broadly?

---

# Execution order

---

# Execution status  *(maintained; last updated 2026-08-28)*

This section is **generated by hand from the run records and `docs/EXECUTION_LOG.md`**, and exists
because the plan below was written before anything ran. The plan is left intact; outcomes are
recorded here rather than by editing the original intent, so what was predicted stays legible next to
what happened. Every number is checked mechanically by `scripts/verify_paper_numbers.py`.

| item | status | outcome |
|---|---|---|
| **E1** repair | **DONE** | Repair confirmed on decoded physical energy, 3/3 seeds. The *original* norm-matched null was **defective** -- the projection is scale-invariant in `C`, so random draws took 29x larger steps. Re-run with a magnitude-matched null; specificity improved from 2/3 to 3/3. |
| **E2** depth | **DONE -- Outcome B** | Local defect does **not** grow with rollout depth. Falsified C3's original form; C3 amended to a near-constant per-step violation. |
| **E3** tangent/normal | **DONE** | Confirmatory for amended C3. A stated "unexplained mechanism" weakness was **withdrawn**: the invariance ratio *is* normal error, so low `f_perp` is the objective being met, not a finding. |
| **E4** dialing | **DONE** | Setting `C` steers imagined true energy, rank correlation 0.80--0.91, 0/25 controls beating it. Re-derived from scratch 2026-08-28. |
| **E8** saturation | **DONE** | Repair persists at step 60,000; not an undertraining artifact. |
| **E9** disjoint eval | **DONE** | 55--76% drift reduction on unseen trajectories, 0/60 magnitude-matched controls. Re-derived from scratch 2026-08-28 (-75.92% reproduced exactly). |
| **E10** matched decodability | **NOT CONSTRUCTIBLE** | 1 of 250 candidates above `\|rho_E\| = 0.3` on the pendulum; no band can be populated at any pool size. |
| **E10b** (its 2-DoF replacement) | **DONE -- NEGATIVE** | Spearman +0.19 / +0.57 / +0.19 at n=3; CI excludes zero on **1 of 3**. The previously reported +0.71 came from an **uncommitted script and does not reproduce**. On one seed the band is unconstructible (19 of 400). |
| **E11 / E12** controls | **DONE** | Phase rival and faithfulness controls pass. |
| **E12c** interchange | **DONE** | Effect survives interchange 50 steps into imagination (+0.92 / +0.85, 0/13 controls at both depths). Re-derived 2026-08-28. |
| **E14b** OOD | **DONE** | Decodable-but-not-conserved outside the training band: free probe 0.999 while conservation degrades **155--267x**. |
| **E17** 2-DoF | **DONE** | Recovery and repair both transfer to two degrees of freedom, 3/3 seeds. |
| **E17b** bootstrap | **WITHDRAWN** | Bootstrap-with-replacement is **invalid** for a within-vs-across-trajectory statistic; duplicated trajectories contribute zero within-trajectory variance. The sample-size concern it raised was withdrawn. |
| **E18** supervised probe | **DONE -- headline** | A probe fitted to *true* energy reaches `\|rho_E\| = 0.9999` yet is **6.7x** less conserved by the transition, and **never repairs** (increases drift on 2 of 3, reduces on none) where the label-free scalar repairs 3/3 at median **-42.2%**. |
| **E19** shadow Hamiltonian | **DONE -- strongest result** | The data comes from a symplectic integrator conserving `H~ = H + O(dt)`, not `H`. Sweeping the target family, `rho_obs` is minimised at **exactly** the coefficient the integrator predicts (`c* = 0.125`, no free parameters), on **3/3** seeds, **591x** better conserved than textbook `E` on ground truth, and correcting that `O(dt)` term flips the intervention from harmful to repairing on every seed. Residual gap to the label-free scalar: **1.10x** -- it accounts for essentially the whole of E18's gap. |
| **F1** balance laws | **IN PROGRESS -- analysis blocked** | Gate 0 passed (relation closes at 0.017 with shadow + midpoint power; the **textbook** balance law explains almost nothing, residual 0.90). Three seeds trained and accepted. The balance **extractor** is at fault, not the model: the paper's validated search finds energy at `rho = 0.91` on the same latent where the balance fit returns 0.18. **Not a negative about the model.** |
| **F2** trust signal | **DONE -- NEGATIVE** | Accumulated drift predicts rollout failure at Spearman -0.02 / +0.09 / -0.24, **beaten on 3/3 by plain latent displacement** (+0.27 / +0.29 / +0.31). Registered falsifier fired. |
| **F4 / F4b** architecture | **DONE** | Conservation transfers to **no** conv-GRU seed: `rho_obs` 4.83--5.55 against the RSSM's ~7e-3, a **767x** gap. 7x more prior training moved the median from 5.33 to 5.26, so **architecture matters, not training amount**. Identification transfers on 2 of 3 under *both* budgets. |

**Not yet attempted:** F3 (released checkpoints / constraint residuals `G(z) = 0`), contacts,
multiple interacting objects, and any downstream demonstration that the privileged direction helps a
planner. The last of these is the paper's most reviewer-visible gap and is recorded in
`paper1.2/sections/limits.tex`.

## Stage 1 — Immediate gates, existing checkpoints

> **Amended 2026-08-26, approved by Richard.** The repo carries no checkpoints, so Stage 1 had to
> retrain. `train_dreamer_pendulum.py`'s own M28 note forbids wall-clock capping for models that will
> be compared, but `REPRODUCE.md` used it and it binds (741 steps/min on this GPU => ~22k steps vs the
> paper's ~6.5k, and a step count that varies with machine load). Training is now capped by
> **optimizer steps**, and each run saves the **E8 milestone grid**
> `{1k, 3k, 6.5k, 15k, 30k, 60k}` on a single optimisation path. **E8 therefore moves from Stage 3
> into Stage 1**, at almost no extra GPU cost, and every Stage 1 experiment can be run both at the
> paper-comparable 6,500-step checkpoint and at saturation.

0. **E8** training-saturation sweep — *now produced as a by-product of Stage 1 training*
1. **E1** physical-energy readout
2. **E2** local conservation failure vs rollout depth
3. **E3** tangent/normal decomposition — **PROMOTED 2026-08-26.** E2's Outcome B makes this the
   confirmatory test of the amended C3: a depth-independent defect should appear as a persistent
   normal component of one-step error that the projection removes.
4. **E9** fresh untouched intervention set
5. **E6** extended horizon analysis

Kick off **E7** seed training in parallel.

**Do not proceed blindly if E1 fails.**

## Stage 2 — Headline causal experiments

1. **E4** invariant dialing
2. **E5** natural-value causal deployment
3. **E13** full trained/damped/untrained causal matrix

These determine whether the paper can make a strong causal claim.

## Stage 3 — Robustness

1. **E8** training saturation
2. **E10** decodability-matched null
3. **E11** phase rival
4. **E12** faithfulness controls
5. **E14** OOD energy
6. **E15 / E16** only as needed

## Stage 4 — Generality

**E17** coupled anharmonic oscillator. If the complete mechanism replicates, integrate it into the
main paper rather than relegating it to an appendix.

## Stage 5 — New research program

Actions/balance laws, trust horizons, constraints, larger checkpoints, architecture generalization.

---

# Decision tree

| Condition | Action |
|---|---|
| **E1 fails** | Pixel improvement is not demonstrably physical. Stop claiming physical repair. Investigate why the latent edit regularizes predictions. |
| **E1 succeeds but E2 shows no off-manifold increase** | Mechanism is likely accumulated integrator bias rather than loss of physical structure under distribution shift. Refocus accordingly. |
| **E2 shows sharply increasing local defect on self-generated states** | This becomes the central mechanism: the transition is locally physical on the data manifold but recursively leaves the region where its learned physics is valid. |
| **E4 succeeds strongly** | Make counterfactual control a headline result. |
| **E4 fails but correction still improves true physical energy** | Keep the narrower repair claim. |
| **E8 shows drift vanishes with adequate training** | Do not claim a fundamental world-model failure. Reframe as an undertraining diagnostic. |
| **E10 shows decodability predicts correction as well as conservation** | The "dynamically privileged invariant" thesis is weakened. |
| **E11 shows phase correction explains everything** | Study invariant drift as a cause or correlate of phase error rather than asserting an independent mechanism. |
| **E17 reproduces the full chain** | The paper has evidence for a general phenomenon rather than a pendulum-specific curiosity. |

---

# What not to optimize for

Do not spend substantial time on:

- dozens of additional toy environments
- prettier latent visualizations
- extra polynomial degrees without a specific question
- reporting best-over-hyperparameter results
- a second architecture before the mechanism is understood
- claiming `C` is literally energy merely because correlation is high
- claiming causal "use" merely because an activation edit changes output
- hiding negative seeds
- adding experiments because they are easy rather than because they separate hypotheses

---

# Desired final scientific picture

> **Amended 2026-08-26, provisional, approved by Richard.** E2 on seed 3 returned Outcome B: the
> local conservation defect does not grow with rollout depth (ratio 1.18-1.65 against a registered
> threshold of 3; slope CI includes zero). The state does leave observation support, but the
> transition violates `C` by about the same amount on and off the manifold. The two struck-through
> lines below are therefore replaced. E1's repair and specificity results are unaffected.

The strongest paper should support this sequence:

    ordinary pixel prediction
      -> physical invariant emerges
      -> transition respects it far better than any random constraint (rho_obs 6.3e-03 vs 5.7e-01)
      -> BUT carries a near-constant per-step violation, on and off the manifold
      ~~-> recursive imagination leaves that support~~
      ~~-> local physical consistency deteriorates~~
      -> that constant violation INTEGRATES over the rollout
      -> violations accumulate into physical + pixel error
      -> violation predicts future rollout failure
      -> restoring the invariant repairs the trajectory
      -> changing the invariant changes the imagined physics

The target contribution is **not** "we discovered energy inside Dreamer," nor "projection onto a
conservation law improves prediction." Both ideas have substantial prior art.

The target contribution is:

> ~~A conventionally trained world model can spontaneously learn a correct physical constraint yet
> fail to remain within the region where its own transition respects that constraint during
> recursive imagination.~~
>
> **Amended 2026-08-26 (provisional):** A conventionally trained world model can spontaneously learn
> a correct physical constraint -- one its own transition preserves two orders of magnitude better
> than any matched random constraint -- and still violate that constraint by a small, systematic,
> depth-independent amount at every step of its own imagination. Integrated over a rollout, that
> violation is a specific and repairable mechanism of long-horizon physical error, and the constraint
> is both a causal control variable and a model-internal signal for deciding when to distrust
> imagination.
>
> This is a claim about the learned *operator*, not about distribution shift, which places it beside
> the energy-drift literature in machine-learned force fields and the projection methods of
> geometric integration (Hairer §IV.4) rather than beside exposure bias.

Every implementation decision should serve or falsify that claim.
