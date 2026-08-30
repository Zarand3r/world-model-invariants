# Reframe proposal — what the paper can claim on the surviving evidence

**A proposal for Richard, written 2026-08-30. Nothing in `paper1.2/` has been changed.**

The open decision is: (a) run mixed-timestep training to settle the attribution behind the current
headline, or (b) reframe the paper around what the evidence already supports. Option (b) is hard to
weigh in the abstract, so this is what it would concretely look like. Option (a) remains open and
this does not argue against it.

---

## 1. Why the current framing cannot ship as written

The title claims the model learns its simulator's **integrator**. What is established is that models
trained at timestep `dt` yield a recovered coefficient matching `c*(dt)`. Whether that reflects the
**model** or the **data it was measured on** is untested, and F11 showed it is not testable by
cross-evaluation with the present assets (a model's one-step error rises from `~0.007` to `~0.6` rad
across a 4x timestep change --- nine times the separation the test needs). Details in
`docs/REVIEWER_HANDOFF.md` §0.

A reviewer who asks "how do you know this is the model and not the data?" currently has no answer.

---

## 2. What survives, graded by how well it is separated

### Strong --- properly separated, model-versus-data

- **F4b, the architecture gap.** Conv-GRU against RSSM on **identical trajectories**: `rho_obs`
  4.83--5.55 against `~7e-3`, a **767x** gap, at two training budgets. This is the cleanest
  model-dependence evidence in the project, because the data is held fixed by construction. Seven
  times more training moves the median from 5.33 to 5.26, so it is architecture, not training amount.
- **The causal interventions (E1, E4, E9, E12c).** Measured on the model's own imagined rollout
  against magnitude-matched nulls, so no evaluation trajectory is tracked and the confound cannot
  arise. This week they were *measured* immune rather than argued immune: repair survives **3/3**
  with `C` fitted on an entirely different dataset (median `-38.8%` matched against `-38.3%`
  mismatched). Drift cut **55--76%** on unseen trajectories, **0 of 60** matched random controls;
  steering at rank correlation **0.802--0.914**; survives interchange **50 steps** into imagination.

### Strong --- same model, same data, so not exposed to the confound

- **E18, the probing dissociation.** A probe fitted to *true* energy reaches `|rho| = 0.9999` yet is
  **6.7x** less preserved by the model's own transition than a label-free scalar, and enforcing it
  during imagination makes the model's physics **worse**, never better.

  **State the limit honestly**: the label-free scalar is fitted to *be* conserved, so its winning on
  conservation is partly by construction. The non-trivial half is the other direction --- a quantity
  that is almost perfectly **decodable** is not **dynamically preserved**, and no probe hygiene
  detects this, because nothing is wrong with the probe.
- **E14b.** Outside the training band the free probe reaches 0.999 while conservation degrades
  **155--267x**: decodable and not conserved, in the same model, at the same time.

### Load-bearing negatives, each with a measured cause

F5 (no control benefit: largest arm `0.0014` against return SD `0.575`); F2 (drift is not a trust
signal, beaten 3/3 by plain latent displacement); F1 (the action's power explains **0.31%** of the
variance in `dC`); F3 (a quantity that never varies carries no information). These bound the claim
rather than decorate it.

---

## 3. The proposed claim

> **Decodability does not imply dynamical preservation.** A world model can carry a quantity that a
> linear probe recovers almost perfectly while its own learned transition does not preserve it ---
> and the converse: a label-free search against the transition finds a direction that *is* preserved
> and that causally repairs imagination. Probing measures what is **representable**; it does not
> measure what the dynamics **respects**.

**Why it is worth saying.** Probing is the field's default interpretability instrument, and it is
routinely read as evidence a model "has" a quantity. This gives a concrete, measured case where that
inference fails in a model that is otherwise working, with the failure invisible to every standard
probe diagnostic --- and it supplies an alternative statistic that does not fail the same way.

**What changes if it is true.** Probe accuracy stops being sufficient evidence for representational
claims about dynamics; an operator-side check becomes necessary.

**Strongest alternative, and the answer.** *The label-free direction wins because it is optimised for
the statistic being reported.* Partly true, and it must be conceded in the text. What survives it:
the supervised probe **never repairs** and makes physics worse (E18), the conv-GRU shows the
preserved direction is a property of the architecture and not of the data (F4b), and the repair is
causal under matched nulls at depth 50 (E12c).

**Deliberately not claimed:** that the model learns its simulator's integrator, its numerical scheme,
or its timestep.

---

## 4. What this costs

- The **title and abstract** are rewritten; the integrator framing goes.
- **F6 and E19 move to a bounded observation**: the recovered coefficient matches `c*(dt)`, with the
  attribution stated as untested and the reason given. This is a *smaller* result but a defensible
  one, and it stops being the load-bearing claim.
- The paper becomes a **probing-methodology contribution with an unusually well-controlled negative
  section**, rather than a positive general result about world models and physics.
- Realistically this is a weaker paper than the one drafted, and I am not going to put a number on
  how much weaker --- my forecasts this week were wrong in both directions within two days.

## 5. What it does not cost

No experiment is discarded. Every number in `paper1.2/` still reproduces; what changes is which of
them the argument leans on. F6/E19 keep their sections, demoted and correctly caveated.

---

## 6. Recommendation

**Do (b) now and keep (a) open.** The reframe rests only on evidence that is already separated, so it
can be written and checked immediately, and it is robust to whatever mixed-timestep training later
shows. If (a) then succeeds, the integrator result can be *restored to the headline* on top of a
paper that was already sound --- which is a much better position than submitting on it now and having
a reviewer ask the attribution question I could not answer.

If you prefer (a) first, the training is roughly F6's cost and I would want the observable-difference
test (`tests/test_observable_difference.py`) run against the design **before** any GPU time, since
skipping that check is what cost the last week.
