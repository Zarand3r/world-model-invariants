# F10 --- Forced choice: does the model's step land nearer its own scheme's next state?

**Registered 2026-08-30, before writing the measurement. No training.**

## Why this design

F9 measured the signal-to-noise: the models resolve the scheme difference by roughly `2x`, and about
three quarters of their apparent error is the **pixel readout**, not the dynamics. Both closed
instruments spend that margin --- the one-step statistic loses it to data-tracking, the rollout to
accumulation, and F9 shows both also pay a large decoding cost.

A two-alternative forced choice spends none of it, and has a null that requires no calibration.

## Measurement

For each analysis state `t` on semi-implicit data:

1. From the true `(theta_t, thetadot_t)`, compute **both** counterfactual next states --- the one
   semi-implicit Euler gives and the one velocity Verlet gives.
2. Render both to frames with the existing renderer and encode both with the model: `h_SI`, `h_VV`.
3. Compare the model's own predicted next latent `h_pred = m.transition(h_t)` against the two.
4. Score a **hit** when `h_pred` is nearer `h_SI` (its own training scheme) than `h_VV`.

Encoding is shared by all three latents, so readout error is common-mode and largely cancels ---
this is the factor F9 identified as the dominant loss.

The null is exactly **50%**, needing no model of the noise.

## Registered predictions

- **P1 (primary).** Semi-implicit-trained models (F6 `dt = 0.05`, seeds 3/4/5) score hit rates
  significantly above 50% (binomial, two-sided, `p < 0.01`) on at least **2 of 3**.
- **P2 (the control that makes P1 mean anything).** The **Verlet-trained** models (F7, seeds 3/4/5),
  run identically on the same semi-implicit data, must **not** show the same preference: their hit
  rate for `h_SI` must be below that of the semi-implicit models on at least 2 of 3. Without this,
  a hit rate above 50% could reflect the rendering or the encoder rather than the learned dynamics.
- **P3 (symmetry check).** Repeat the whole thing on Verlet data with the roles swapped. A real
  effect should reverse; an artefact of the counterfactual construction should not.
- **Falsifier.** If P1 fails, or if P2 shows Verlet-trained models preferring `h_SI` just as strongly,
  the model does not carry its integrator and F6/E19's model-side claims should be withdrawn.

## Pre-committed guard against the failure mode of the last three experiments

Each of the last three measurements failed in a way I had not registered: a verdict that did not
follow (F6 cross), an alive-but-flat sweep (F8), a readout that might have been blind (F9, caught
only because I added P3 after F8). The analogue here is a **degenerate comparison**: if `h_SI` and
`h_VV` are nearly identical in latent space, the choice is a coin flip regardless of the model.

Registered now, before seeing anything: report the median latent distance `||h_SI - h_VV||` against
the median `||h_pred - h_SI||`. **If the two counterfactual latents are closer to each other than the
model's prediction is to either, the comparison is degenerate and P1 must not be read.**

## Direction

None stated.

---

## Amendment 1 --- the comparison moves to `theta`, and why (registered before running)

Implementing the latent version exposed a defect in the design above. `encode` is a teacher-forced
sequence pass whose indexing is `state k has consumed obs[:k]`, so a latent is defined by its
*history*, not by a frame in isolation. "Encode the counterfactual frame and compare" therefore has
no well-defined meaning: the counterfactual latent would carry a posterior update at the wrong
index, and a mistake there would corrupt the test **silently**, which is precisely the class of
error that has cost me four experiments this week.

The forced choice is well-defined in `theta`, and needs no rendering and no counterfactual encoding:

    hit  <=>  |theta_pred - theta_SI|  <  |theta_pred - theta_VV|

where `theta_pred` is decoded from `m.transition(h_t)` and the two counterfactual angles are computed
analytically from the same true `(theta_t, thetadot_t)`. Every quantity already exists in F9.

**What this costs.** The latent version was chosen so that encoder error would be common-mode. In
`theta` it is not: `theta_pred` carries readout noise while the two targets are exact. So the test is
noisier than the design above intended.

**Why it still has power, computed before running.** F9 measured total one-step error `0.0066` rad
against a scheme separation of `0.0144` rad. A model sitting on its own scheme's next state is
misclassified only when noise exceeds half the separation, so the expected hit rate is roughly
`Phi(0.0072 / 0.0066) ~ 0.86` --- far from the 50% null. If the model instead learned the continuous
flow, it sits *between* the two schemes and the rate goes to ~50%.

**The control gains a sharper, signed prediction.** A Verlet-trained model run on semi-implicit data
should prefer `theta_VV`, so its hit rate for `theta_SI` should fall **below 50%**, not merely below
the semi-implicit models' rate. P2 is tightened accordingly: Verlet models score below 50% on at
least 2 of 3, and strictly below the semi-implicit models.

**Degeneracy guard, restated in these terms.** The comparison is degenerate if the separation is not
resolvable against the readout noise. F9 already establishes it is (`D_model/D_scheme = 0.46`), and
the guard is re-checked here per model: **if `D_model >= D_scheme` for a model, its hit rate is not
to be read.**
