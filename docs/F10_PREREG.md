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
