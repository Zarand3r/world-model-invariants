# Pre-registration — S4, the causal edit on DreamerV3

**Written before any Dreamer edit was run. 2026-08-11.**

## The question

We now know a trained conservative DreamerV3 contains a genuinely conserved, energy-correlated
invariant (drift ~1e-4, |ρ|_E 0.973) that a matched dissipative DreamerV3 does not (drift 0.258,
|ρ|_E 0.048; D44). S4 asks whether that quantity **matters to the model's own future**:

> If we repair the recovered physical state during imagination, does Dreamer's rollout get better?

This is the identify → **intervene** → verify rung, and the strongest available evidence that a
recovered structure is real rather than fitted. There is no way for a spurious `C` to improve a
rollout.

## The edit — identical to the GRU protocol, no re-derivation

At each imagination step, project the latent back onto the level set of `C`:

    z <- z - α (C(z) - C₀) ∇C(z) / ‖∇C(z)‖²

then map the correction back to the hidden state through the pseudo-inverse of `U R`, so the
component of `h` outside the extracted subspace is untouched. Decoding-time only: no weight
surgery, no fine-tuning, frozen-model constraint preserved.

## Scoring — the SLOPE, not the best α (D9, non-negotiable)

`α` is swept over the **fixed grid (0, 0.05, 0.1, 0.2, 0.4)**, identical for every arm, and the
**full curve is reported**. The statistic is the direction of error-versus-α.

D9 exists because `best-over-α` once reported −5.9% for a damped arm whose curve rose monotonically
to twice its baseline: taking the most favourable point on a rising curve inverts the sign of the
finding. *"Enforcing a true law helps more the harder you enforce it; enforcing a false one hurts
more the harder you enforce it"* is a stronger claim than any single-α comparison and is nearly
impossible to produce by accident.

If `α` must be tuned per arm to see an effect, the result is a tuning artefact and is reported as one.

## Arms

| arm | model | law enforced | registered expectation |
|---|---|---|---|
| **A** | conservative | its own recovered `C` | rollout error **decreases** with α |
| **B** | conservative | **random** matched-complexity `C` | no improvement, or harm |
| **C** | damped | its own recovered `C` | no improvement, or harm |

**Arm B is the arm that can kill this.** If projection improves rollouts regardless of which `C` is
enforced, the edit is merely regularising the rollout — shrinking it toward a manifold, damping
divergence — and says nothing about physics. **A uniform improvement across arms is the main way
this dies**, which is why every arm gets identical treatment on an identical grid.

## Metrics

1. **Rollout pixel MSE** against the true frames — the model's own prediction quality.
2. **True-energy drift** of the rollout, read through the angle probe used in the mechanism work
   (measurement only, never extraction; its gates G1/G2 from `run_dreamer_mechanism` apply).

Both are reported per α.

## Registered outcome

**PASS** = arm A's error decreases with α **and** arms B and C do not (flat or rising). Anything
else is reported as it comes, including a uniform improvement, which would retire the edit as
evidence about physics.

## What a pass would establish

physics exists internally → the model lets that state drift → externally correcting it improves
prediction. That is the bridge to the architectural question: why repair physical state post hoc
if a world model can be built to preserve it during imagination?
