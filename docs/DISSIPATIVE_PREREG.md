# Pre-registration — the dissipative DreamerV3 control

**Written before any damped data was generated. 2026-08-11.**

## The question

Our extraction recovers an energy-correlated invariant from a pixel-trained DreamerV3 at
|ρ|_E = 0.973 / 0.967 / 0.975. **Does it also do that when there is no invariant to find?**

Two distinct worries, and the control addresses both:

1. **Recovery may be automatic rather than learned.** The eigenproblem returns functions constant
   within a trajectory and varying across it. On a 1-DOF conservative pendulum energy is
   essentially the *only* such function, so any faithful encoding of (θ, θ̇) yields it. N1 already
   measured a **random feature map** recovering energy at ρ = 0.998. On a damped system no such
   function exists; a confident "invariant" there is confabulation.
2. **The Noether pairing — our distinctive instrument — may not work on this substrate.** Its
   headline GRU evidence *is* the dissipative separation (p = 5.4e-06, d = 3.10). But on Dreamer
   the residual's magnitude is already known to misbehave: 0.83–0.90 on a **conservative** system,
   worse than our toy **damped** models at 0.44–0.69, and moving opposite to recovery across LD.

## Registered predictions

Latent dimension **LD = 12**, carried over from the confirmed conservative run. Same renderer,
same training script, same adapter, same extraction — only gymnasium's dynamics change.

**P1 — REFUSAL (the primary criterion).** |ρ|_E < 0.7 on ≥ 2 of 3 damped seeds, against
0.973/0.967/0.975 conservative. 0.7 is the same threshold the conservative run had to *pass*, so
the test is symmetric and was not chosen after seeing damped numbers.
→ **FAIL means the pipeline reports a confident invariant where none exists**, which would
substantially undercut the conservative result rather than merely add a caveat.

**P2 — does the pairing residual separate at all?** Reported, **not** used as the primary
criterion, because we have already measured its magnitude to be unreliable here. Registering it as
the criterion and watching it fail would confound "the method cannot refuse" with "this statistic
is mis-scaled on this geometry" — different findings.
→ If it separates, the LD anti-correlation is a scaling problem and the instrument is sound.
→ If it does not, the pairing works as a selection criterion on a small deterministic GRU and not
on a modern stochastic pixel model. That is a serious negative and gets reported as one.

**P3 — invariance ratio.** The best candidate's held-out ratio should be materially worse on
damped than conservative.

## Gates (the test is VOID, not negative, if these fail)

**G1 — the model must have trained.** Same acceptance as the conservative run: raw KL > 1.0 nats,
1-step decode ≥ 4× better than predict-the-mean, rollouts finite and non-collapsed. A model that
did not train is not evidence about the method (M11).

**G2 — the latent must not be degenerate.** Damped trajectories spiral to a common fixed point, so
late-time states coincide across trajectories and the latent can go rank-deficient — which
manufactures *trivially* constant directions scoring a perfect invariance ratio while carrying no
content. That exact failure once moved `k*` from 2 to infinity. Report the participation ratio and
the retained rank; if the latent has collapsed, the refusal is an artefact of degeneracy and proves
nothing.

Damping is **ζ = 0.15**, matching the GRU dissipative experiments, over the same 120-step horizon
so trajectories decay without collapsing.

## Why this is worth doing before S4 or a new substrate

A method that cannot refuse is not a method. Everything downstream — the invariant edit, the
mechanism — presupposes that what we recover is real.

---

## Addendum — ζ selected by the G2 gate, before any model was trained

The initially proposed ζ = 0.15 (the GRU control value) **fails G2 on this substrate**, measured on
states alone with no rendering and no model, so no outcome was visible when this was decided.

ζ is a damping *ratio*, so energy falls by `1 − e^{−2πζ}` per period independently of ω₀. The
pendulum's ω₀ = 3.87 gives ~3.7 periods in the 120-step window, so ζ = 0.15 removes 61% of the
energy *per period* and the trajectories are dead well before the window ends: at step 119,
`std(θ̇)` = 0.161 and `std(E)` = 0.009 against the conservative 4.042 and 2.036 — **0% of the
reference**. Extraction on that would find trivially constant directions and "refuse" for a reason
that has nothing to do with the method.

**Registered selection rule:** take the largest ζ whose late-window across-trajectory spread stays
≥ 25% of the conservative reference on *both* `std(θ̇)` and `std(E)`.

| ζ | E loss/period | std(θ̇)@119 | std(E)@119 | % of ref | G2 |
|---|---|---|---|---|---|
| 0.15 | 61% | 0.161 | 0.009 | 0% | FAIL |
| 0.10 | 47% | 0.480 | 0.045 | 2% | FAIL |
| 0.07 | 36% | 1.031 | 0.142 | 7% | FAIL |
| 0.05 | 27% | 1.694 | 0.308 | 15% | FAIL |
| **0.03** | **17%** | **2.702** | **0.695** | **34%** | **OK** |
| 0.02 | 12% | 3.205 | 1.011 | 50% | OK |

**Selected: ζ = 0.03.** Unambiguously dissipative — energy decays monotonically, no conserved
quantity exists — while the latent stays non-degenerate. This is a gate-driven choice, not a
tuned one: the rule and the threshold are fixed above, and the selection used no model output.

**Caveat carried forward:** ζ = 0.03 is a *weaker* dissipation than the GRU control's effective
value. If the method refuses here, that is a stronger result than refusing at heavy damping. If it
fails to refuse, "the dissipation was too weak to detect" is a live alternative and must be tested
by re-running at ζ = 0.05 and 0.07 before concluding the method cannot refuse.

---

## Addendum 2 — the criterion, locked before any damped checkpoint was read

External review, correctly, rejected `|ρ|_E` alone as the refusal criterion. A quantity can track
*instantaneous* energy without being conserved, and the damped system has legitimate structure
(`dE/dt < 0`, a predictable decay envelope) that the model may well represent. Finding structure is
not a false positive. The question is narrower:

> **Does the conservative-law extractor identify a non-trivial, approximately conserved scalar with
> convincing flow-generation evidence, on a system that has none?**

### Test A — is the recovered `C` actually conserved?

Report `|ρ(C, E)|` **and the drift of `C` itself** — its held-out within-trajectory variance ratio.
Correlation with instantaneous energy is not conservation, and only the second quantity can tell
them apart. Registered: on damped, `C` must show materially larger drift than on conservative.

### Test B — is the candidate EXCEPTIONAL against matched nulls? *(the primary criterion)*

For every model, conservative and damped alike, score the recovered candidate against nulls drawn
**within that same model's own latent**:

1. **flow shuffle** — permute `F` against `Z`, destroying the dynamical pairing while preserving
   both marginals; refit end to end
2. **temporal permutation** — shuffle time within trajectories, destroying within-trajectory
   constancy; refit end to end
3. **random matched-complexity `C`** — random coefficient vectors over the same degree-4 basis

**Why this rescues the pairing residual.** Its *magnitude* is not comparable across substrates
(D13, and the LD anti-correlation). But a null comparison is computed inside a single fixed
extraction and chart — precisely the regime where D13 says residual comparisons *are* meaningful.
The null converts an uncalibrated number into a calibrated one.

**Registered:** the conservative candidate is exceptional against its own nulls (percentile ≤ 5%)
and the damped candidate is not. That, not any absolute residual, is the pass condition.

### Test C — causal sign *(deferred to S4)*

Enforcing `C` should improve conservative rollouts and give no benefit or harm on damped. This is
the strongest refusal check and recreates the GRU's strongest result; it is out of scope here.

### Outcomes, agreed in advance

- **Pass**: conservative exceptional, damped not. The method discriminates; proceed to S4.
- **Also a pass**: damped yields an energy-*correlated* `C` that is neither conserved nor
  exceptional. This reinforces N1 — decodability is not law.
- **Fail**: damped yields an equally convincing conserved candidate under the same criterion. Then
  the method cannot refuse on Dreamer, and that must be fixed before recovered invariants are used
  for anything downstream.
