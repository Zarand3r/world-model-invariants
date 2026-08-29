# Claim architecture — paper 1.2

**Not part of the manuscript.** Written first, per the build-by-compression order, so the prose has
something to serve. Every number below traces to a run record named in `docs/RESULTS.md`.

## One-sentence central claim

> Decodability is not dynamical structure: a physical quantity can be perfectly readable from a
> world model's latent while the model's own transition does not preserve it — and only a statistic
> computed on the transition operator tells the two apart.

## What changed from paper 1.0

Paper 1.0 argued *"a world model learns a physical constraint yet violates it when imagining
forward."* That claim survives, but it is now the **second** claim rather than the first, because a
stronger and more transferable one sits above it and because the evidence for the dissociation is
now five-fold and includes a decisive supervised control.

Paper 1.0 also contains a **defect**: it describes its random-constraint null as "matched in norm",
but the projection `z <- z - alpha (C - C_0) grad C / ||grad C||^2` is invariant under `C -> lambda C`,
so coefficient norm has no effect on the edit. Measured: random draws took steps **29x larger** than
the recovered constraint. The published specificity comparison was therefore unmatched. Re-run with a
magnitude-matched (fixed step size) null, specificity **improves** from 2/3 seeds to 3/3.

---

## Claim 1 — Probing over-reports; the operator statistic discriminates

**Claim.** Whether a latent *encodes* a physical quantity and whether the model's transition
*preserves* it are different properties, they dissociate, and the dissociation is directional:
decodability survives where conservation fails.

**Why surprising.** Probing is the field's default instrument for "does the model represent X?", and
the standard remedy for its known weaknesses is better probe hygiene — control tasks, selectivity,
random-init baselines. This says the remedy addresses the wrong failure. A probe fitted to the true
physical quantity, reaching essentially perfect correlation, still identifies a direction the
dynamics do not use.

**Decisive evidence.** E18: a supervised polynomial probe fitted to true energy reaches
`|rho_E| = 0.9999` and is **6.7x less conserved** by the transition than the label-free scalar
(median 4.58e-02 vs 6.85e-03); it **never repairs** rollouts (+26.8/-0.3/+33.4 per seed, so it
increases drift on 2 of 3 and reduces it on none) where the label-free scalar repairs on 3 of 3
(median -42.2%, per-seed -50.9/-42.2/-32.2). All aggregates are medians across the three seeds,
matching the figures. <!-- superseded: the mean-based 6.3x / +20.0% quoted here before 2026-08-27 -->

**Supporting, each an independent axis:** untrained models (`rho_obs` 74x worse at `rho_E` up to
0.908); out of distribution (free probe 0.999, conservation 55-267x worse); a second architecture
(`rho_E` 0.91 on 2 of 3 seeds, conservation 767x worse, n=3 at both training budgets); actuation (`rho_E` 0.72-0.88, the action's power explains 0.31% of the motion of `C`, n=3); an untrained transition (`rho_E` 0.97, conservation
~10,000x worse).

**Strongest alternative.** The supervised probe is worse simply because it is *different* from the
recovered scalar, not because it is less conserved.

**This alternative is NOT closed by E10b, and the paper says so.** E10b matched candidates on
decodability and varied conservation over five orders of magnitude; the rank correlation with repair
is +0.19, +0.57, +0.19 at n = 3, with the 95% CI excluding zero on only 1 of 3 seeds.
<!-- superseded: the +0.71 reported before 2026-08-28 came from an uncommitted script and does not reproduce; see the execution log -->
On a third seed the matched band cannot be built at all — 19 of 400 candidates — and
conservation and decodability stay rank-correlated even inside the band.

**What closes it instead is E19.** The shadow-Hamiltonian sweep shows the supervised probe is worse
for an identified reason rather than an unexplained one: it is fitted to a quantity the generating
integrator does not conserve. `rho_obs` is minimised at exactly the coefficient the integrator
predicts, on 3 of 3 seeds, 12x better than the wrong-sign control, and correcting that O(dt) term
flips the intervention from harmful to repairing on every seed. That is a mechanism, not a
correlation, and it does not depend on E10b.

---

## Claim 2 — The operator-privileged direction is causally deployed

**Claim.** The scalar the transition best preserves is not a descriptive correlate: enforcing it
repairs the imagined physics, and setting it steers the imagined physics quantitatively.

**Why surprising.** A conserved-looking latent scalar could be an artifact of the search. Causal
control with an external physical readout is what separates the two, and it is rare in
interpretability.

**Decisive evidence.**
- **Repair**, on *decoded true physical energy* rather than pixels: -32 to -51% at H = 100 and
  **-55 to -76% at H = 190 on 512 trajectories the method never saw**, with **0/60**
  magnitude-matched random and **0/15** equal-norm tangent directions beating it, n = 3.
- **Control**: setting `C` to an independent donor's value moves the imagined world's *true* energy
  toward that donor's true energy, Spearman **+0.81 to +0.92**, **0/75** controls, n = 3.
- **Deployed at depth**: the same interchange applied **50 steps into autonomous imagination** gives
  +0.76 to +0.85, 0/13 controls, n = 3 — the dormant-pathway objection (Makelov et al., ICLR 2024).

**Strongest alternative.** The projection is a generic regulariser. Ruled out by the
magnitude-matched null (0/60), by equal-norm tangent controls sitting at ~0, and by the damped arm,
where the same pipeline yields a direction **indistinguishable from random** (24/60 random draws beat
it; it harms on 2/3 seeds).

---

## Claim 3 — The phenomenon has boundaries, and they are informative

**Claim.** The conserved quantity emerges under specific conditions: it is bounded to the training
distribution and it is architecture-dependent.

**Why this is a claim and not a limitation.** Both boundaries are measurements with matched controls,
and each says something about *what produces* conservation.

**Evidence.**
- **Distribution-bounded**: out of distribution a fresh probe recovers energy at **0.999** while the
  transition conserves it **55-267x worse**, and the frozen `C` collapses to 0.03-0.31. The model's
  representation generalises; its conservation does not. n = 3.
- **Architecture-dependent**: on a deterministic conv-GRU trained to the same contract, identification
  transfers on **2 of 3** seeds and conservation on **none** (`rho_obs` 1.75-9.02 against the RSSM's
  ~7e-03). Training the transition 7x longer did not close it.

**What this forbids.** The paper may not claim the phenomenon holds for learned world models in
general. The supportable scope is RSSM-like models, in-distribution.

---

## Deliberately not claimed

- That the model "understands physics". It learned something locally valid that does not extrapolate.
- That the mechanism of the *fit* is surprising. Low normal error is what the invariance ratio
  optimises; the empirical content is that the resulting direction coincides with physical energy.
- Any downstream utility. No planning, control or task result exists.

## Figure 1 structure

`probe finds it -> operator says the model does not use it -> intervening confirms the operator`.
Panel A: supervised probe `rho_E` 0.9999 vs label-free 0.966. Panel B: `rho_obs` 4.58e-02 vs
6.85e-03. Panel C: repair +26.8% vs -42.2% (medians; supervised is +26.8/+33.4/-0.3 per seed -- it never repairs, but it only *increases* drift on 2 of 3). One figure, one message: the right answer, fitted
perfectly, does not work.
