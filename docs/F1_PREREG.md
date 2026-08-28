# Pre-registration — F1, from conservation laws to balance laws under action

**Written 2026-08-28, before any F1 quantity was computed.** Roadmap Stage 5, specified at
`docs/ROADMAP.md:552`.

## The question

Every result in this project so far concerns **free evolution**, where the target relation is
`C(z_{t+1}) - C(z_t) = 0`. Under actuation, energy is no longer conserved; it obeys an accounting
relation. For the actuated pendulum with applied torque `tau`,

    I*thetaddot = m g (l/2) sin(theta) + tau        =>        dE/dt = tau * thetadot

with no dissipation term, since the simulator is undamped. So the question becomes:

> Does an ordinary action-conditioned world model, trained only on pixels and actions, spontaneously
> learn a latent scalar whose change per step is accounted for by the action, rather than one that
> stays constant?

This is a strictly harder question than conservation. A constant is a degenerate balance law, and a
search for constants would simply fail here.

## What makes this worth running

If the answer is yes, the paper's object generalises from "conserved quantity" to "quantity with a
learned source term", which is the form physical bookkeeping actually takes in any controlled system
and the natural bridge to model-based RL. If the answer is no, that is a sharp boundary on the whole
programme: the method would be shown to work only where the true relation is `dC = 0`.

**No direction is predicted.** Free-evolution success does not obviously transfer, because the model
must learn a *state-dependent coupling* to the action rather than an invariant direction.

## Design

### Data

Gymnasium Pendulum, undamped, with **nonzero torque**. Actions are piecewise-constant random torques,
held for `K` steps and redrawn, with magnitude bounded so that `|thetadot|` stays clear of the
simulator's `|thetadot| = 8` clip — the clip breaks the balance relation outright, exactly as it
would break conservation. The action sequence is stored with the frames and states, and the torque
bound and hold length are recorded in the run record.

Free-evolution data is **not** reused: a model trained on `tau = 0` has never seen an action.

### Method: balance-law extraction

Generalise the free-evolution fit. Instead of seeking `C` with `Delta C = 0`, seek a pair `(C, P)`
minimising the **balance residual**

    R  =  || C(z_{t+1}) - C(z_t)  -  P(z_t, a_t) ||^2

over the same degree-4 monomial family for `C`. The power term is parameterised as **linear in the
action with a state-dependent coefficient**,

    P(z, a)  =  a * ( monomials(z) . q )

which is the correct functional form: the true power `tau * thetadot` is linear in `tau` with a
coefficient that is a function of state. `C` and `P` are fitted jointly by alternating least squares,
mirroring `fit_hamiltonian_pair`, and normalised so `R` is scale-free in `C`.

## Gate 0 — does the balance law close on the GROUND TRUTH? (no model)

**Run before any training.** On ground-truth states and actions, compute the normalised residual of

    E_{t+1} - E_t  -  tau_t * thetadot_t * dt

E19 established that this simulator conserves a **shadow** Hamiltonian rather than the textbook one,
so the discrete balance relation will carry an `O(dt)` correction for the same reason. We therefore
evaluate the residual for textbook `E` **and** for the E19 shadow `H~ = E + 0.125 * thetadot *
sin(theta)`, and for the power evaluated at `thetadot_t` and at `thetadot_{t+1}` (the semi-implicit
update uses the updated velocity, so the second is the candidate that should close).

**G0 passes** if some combination drives the normalised residual **below 0.05**, i.e. the discrete
balance law closes to within 5% of the per-step energy change.

> **If G0 fails, STOP.** The relation being tested would not hold in the data, so no result about the
> model could be interpreted. Report as a negative and do not train anything.

This gate is the F1 analogue of E1's readout gate and E19's P1, and it exists because E19 showed that
assuming the textbook relation holds discretely is exactly the mistake that makes a perfect fit
meaningless.

## Registered predictions, conditional on G0

**P1 — identification.** The jointly fitted `C` correlates with the ground-truth energy at
`|rho_E| >= 0.8` on at least 2 of 3 seeds.

**P2 — the source term is physical.** The fitted power coefficient `monomials(z) . q` correlates with
true `thetadot` at `|rho| >= 0.8` on at least 2 of 3 seeds. This is what distinguishes a genuine
balance law from a curve fit: the coupling to the action must be the physical one.

**P3 — the balance law beats conservation.** The balance residual `R` of the fitted `(C, P)` is at
least `5x` lower than the residual of the **best conserved** scalar found by the free-evolution
search on the same latent, i.e. forcing `P = 0`. If a constant scalar does just as well, the model
has not learned a source term and the "balance law" framing adds nothing.

**P4 — control.** `R` is at least `5x` lower than for 20 random `(C, P)` pairs drawn at matched
coefficient norm.

## Falsifiers, stated plainly

- **G0 fails** -> the relation does not hold in the data; stop, report negative, train nothing.
- **P1 fails** -> the search does not find energy under actuation; the free-evolution result does not
  transfer, and that is the finding.
- **P2 fails while P1 passes** -> `C` is energy-like but its action coupling is not physical; the fit
  is absorbing the action as a nuisance term rather than learning power. Report as a **negative**:
  this is the most likely way to get a misleadingly good `R`.
- **P3 fails** -> a conserved scalar explains the data as well as a balance law; the generalisation
  is empty on this system.

## Scope and honesty constraints, fixed now

- 3 independently trained seeds, step-capped exactly as in Stage 1, with the E8 checkpoint grid.
- The evidence-base guard will flag this claim until it reaches n = 3.
- Physical labels (`E`, `thetadot`, `tau`) are used **only after the fit is frozen**, for evaluation,
  per the roadmap's rule at line 79. The fit sees `z` and `a` only.
- The second half of the roadmap's F1 item — "correcting toward the predicted energy balance improves
  imagination" — is **deliberately not registered here**. It is only meaningful if P1-P3 pass, and
  registering an intervention before knowing whether the object exists would repeat the error E10b
  was caught by. It will get its own preregistration.

## Provenance

All records carry the `latent_noether.provenance` stamp: argv, git HEAD and dirty state, and the
sha256 of every input. This is the first experiment in the project preregistered *after* the
provenance audit, and it is expected to be reproducible from the record alone.

---

## Amendment, 2026-08-28 -- the power basis, fixed on GROUND TRUTH before any model was touched

**Written after validating the extractor on ground-truth states, and before it has been applied to
any trained checkpoint.** No F1 model quantity exists yet.

### What the validation found

The registered method parameterised the power term over "the same degree-4 monomial family" as `C`.
Validated on ground-truth states -- where the answer is known exactly and is representable -- that
method **fails**:

| power basis degree | terms | residual | ratio vs conserved-only | `rho(C, E)` | `rho(C, H~)` | `rho(q, thetadot)` |
|---|---|---|---|---|---|---|
| 1 | 3 | 0.00111 | **55.6x** | 0.9849 | **1.0000** | **0.9973** |
| 2 | 9 | 0.00085 | 72.8x | 0.9848 | 1.0000 | 0.9972 |
| 3 | 19 | 0.00236 | 26.2x | 0.7813 | 0.7959 | 0.7460 |
| **4 (as registered)** | 34 | 0.00273 | 22.6x | **0.5068** | **0.5107** | **0.0745** |

At degree 4 the fit achieves a *low residual while recovering nothing*. This is exactly the failure
the original registration named as most dangerous -- "the fit is absorbing the action as a nuisance
term rather than learning power" -- and P2 is what detects it. A 34-term power basis multiplied by
the action has enough freedom to explain `Delta C` for the wrong reasons.

Two further notes from the validation, both worth recording:

- An earlier version of the solver used an ordinary rather than **generalised** eigenproblem and
  returned `rho(C, E) = 0.02`: minimising `||D c||` alone selects a near-constant direction. The
  objective must be the invariance ratio, normalised by the spread `C` actually has.
- The validation must be done in **periodic coordinates** `(cos th, sin th, thetadot)`, not raw
  `theta`. The pendulum rotates so `theta` is unbounded, and `cos theta` is not a polynomial in it.
  The model never sees `theta` either -- a frame determines orientation. Using raw `theta` made the
  extractor look broken when the harness was.

### The change

**The reported object becomes the power-degree sweep itself, degrees 1 to 4, not a single choice.**

Picking one degree and reporting it would be indistinguishable from tuning, and the ground-truth
sweep above is the natural reference: a correct extraction should look like the top rows and degrade
toward the bottom. The model's sweep is reported against it.

`P1`, `P2` and `P3` are evaluated at **power degree 1**, the minimal basis containing the true form
(`tau * thetadot`, linear in the coordinates), with degrees 2-4 reported as sensitivity.

**Registered now, so it cannot be decided later:** if degree 1 fails on the model, we do **not** climb
the degree ladder looking for a pass. A higher degree that "works" after degree 1 fails is the
degeneracy above, not a discovery. The response to failure at degree 1 is to report it, with the
sweep, as a negative.

### Why this is not tuning

The change was made on ground-truth states with no model involved, the failure it fixes is one the
original registration had already named, and the fix is frozen here before any checkpoint is
analysed. The ground-truth numbers are committed in this document so the reference curve cannot be
retro-fitted.

---

## Amendment 2, 2026-08-28 -- residuals must be evaluated on HELD-OUT trajectories

**Written before the registered step-60,000 read. Forced by a validity failure, not by a result.**

### What went wrong

The registered analysis evaluates the balance residual **in sample**, on the same trajectories the
`(C, P)` pair was fitted to. At the latent dimension this project uses that is not sound. Degree-4
monomials in `LD = 12` give **1,819 coefficients for `C`**, against 5,720 available samples -- and
2,860 if the data is split. Measured on `f1_act_s3_step30000`:

| power degree | balance in-sample | balance held-out | conserved in | conserved held-out | ratio in | ratio **held-out** |
|---|---|---|---|---|---|---|
| 1 | 0.00635 | 0.02594 | 0.00949 | 0.00939 | 1.49x | **0.36x** |
| 2 | 0.00365 | 0.02127 | 0.00949 | 0.00939 | 2.60x | **0.44x** |
| 4 | **0.00000** | 0.04128 | 0.00949 | 0.00939 | **711764x** | **0.23x** |

At degree 4 the fit is **exact in sample** -- residual 0.00000, ratio 7e5 -- and useless out of
sample. The apparent advantage of the balance term is **entirely memorisation**. The conserved-only
fit, by contrast, generalises essentially perfectly (0.00949 -> 0.00939), so the defect is specific
to the extra freedom the power term adds.

The ground-truth validation could not have caught this: there `LD = 3` gives 35 coefficients against
47,600 samples, oversampled by three orders of magnitude more than the model setting.

### The change

**All F1 residuals -- balance, conserved-only and random -- are fitted on one half of the analysis
trajectories and evaluated on the other half.** `P3` and `P4` are read from held-out residuals.
`P1` and `P2` are correlations of a frozen `C` and `q` against labels and are reported on the
held-out half for consistency.

This is not a new discipline for this project. **E9 already established disjoint evaluation** -- "fit
`C` and the whole coordinate frame on `data`, score on `eval_data`" -- and the F1 registration simply
failed to apply a control the project uses elsewhere. Applying it is consistency, not a moved
goalpost.

### Direction of the change, stated plainly

This amendment makes F1 **harder to pass, not easier**. In sample the balance term looked like a
1.5x-4.6x improvement; held out it is a 0.2x-0.4x *degradation*. The change is being made because
the measurement was invalid, and it moves the expected result against the hypothesis F1 was written
to test. Recording that here so the direction cannot be misread later.

### What is now expected, and registered

On the evidence so far -- one seed, half training -- `P3` will **fail**, and F1's likely finding is
that the model represents an approximately conserved, energy-like scalar even under actuation, with
the action term adding nothing that generalises. That is a legitimate negative and it will be
reported as one. The registered read remains step 60,000 across 3 seeds; nothing is concluded from
the preliminary checkpoint.

---

## Amendment 3, 2026-08-28 -- a positive control for the extraction at the model's operating point

**Written before the control was run, with both outcomes interpreted in advance.**

Seed 3 at step 60,000 fails P1, P2 and P3 on held-out data. Before that is reported as "the model
does not learn a balance law", one alternative must be excluded: **the extraction may simply be
underpowered at the model's operating point.** The ground-truth validation that justified the method
ran at `LD = 3` (35 coefficients, 47,600 samples). The model runs at `LD = 12` (1,819 coefficients,
~2,860 training samples) -- three orders of magnitude less favourable.

### The control

Take ground-truth states in the periodic coordinates `(cos th, sin th, thetadot)`, where the balance
law is known and exactly representable, and **embed them in 12 dimensions with a random linear map**,
then run the *identical* pipeline -- same PCA to `LD = 12`, same effective-rank basis, same degree-4
`C` basis, same power-degree sweep, same held-out split, same trajectory and step counts as the
analysis set.

The embedding is information-preserving, so a method adequate at the model's operating point must
still recover the law.

### Registered interpretation, both directions

- **Control RECOVERS the law** (`rho(C, H~) >= 0.8`, `rho(q, thetadot) >= 0.8`, held-out ratio
  `>= 5x`): the extraction is adequate at `LD = 12` with this sample size, and F1's failure is a
  property of **the model**, not the method. The negative stands as a finding.
- **Control FAILS**: the extraction is underpowered at this operating point and **F1 is inconclusive,
  not negative.** No claim about the model may be made from it; the honest report is that the method
  does not scale to this latent dimension at this sample size, and F1 needs a lower-dimensional
  extraction or more data before its question can be asked.

Recording this before running so the outcome cannot be reinterpreted afterwards. The second outcome
is the more damaging one for the F1 programme and it is registered as freely as the first.

---

## Amendment 4, 2026-08-28 -- repairing the positive control so it reaches rank 12

**The registered interpretation of amendment 3 is unchanged.** Only the control's *construction* is
repaired, because as built it could not reach the operating point it was written to test.

### Why the first control could not work

Embedding `(cos th, sin th, thetadot)` through a random linear map preserves information but not
rank: `effective_rank_basis` correctly collapsed it back to **3**, since a linear map cannot
manufacture degrees of freedom the system does not have. The model's latent retains **12**.

### The repair

Build the control's latent from **time-lagged** ground-truth coordinates -- the state at
`t, t-1, t-2, t-3`, giving 12 components. Lagged coordinates of a nonlinear system are *not*
linearly dependent (`cos th_{t-1}` is a nonlinear function of the state at `t`), so the linear rank
is genuinely higher while the information content is unchanged and the balance law remains exactly
representable through the `t` block.

This also matches the model more honestly than the first attempt: an RSSM's deterministic state
encodes **history**, not the instantaneous physical state, so a lagged embedding is the closer
analogue of what the model actually stores.

### Interpretation, unchanged from amendment 3

- **Control recovers** (`rho(C, H~) >= 0.8`, `rho(q, thetadot) >= 0.8`, held-out ratio `>= 5x`, at
  the same power degree the model is read at): the extraction is adequate at rank 12 with ~2,860
  training samples, and **F1's failure is a property of the model**. The negative stands.
- **Control fails**: **F1 is inconclusive, not negative**, and no claim about the model is made.

Registered before running, as before. Reported at power degrees 1 and 2, since the degree-1 read was
calibrated at a 17x larger sample size and the rank-3 control recovered only at degree 2.
