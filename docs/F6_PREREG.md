# Pre-registration — F6, does a pixel-trained world model encode the simulator's TIMESTEP?

**Written 2026-08-29, before any F6 quantity exists.** Approved to run.

## The claim being tested

E19 established that the label-free search recovers not textbook energy but the **shadow**
Hamiltonian of the integrator that generated the data,

    H~ = E + c* * thetadot * sin(theta),      c* = (dt/2) * m g (l/2) = 0.125 at dt = 0.05,

and that `rho_obs` is minimised at exactly that `c*` on 3 of 3 models with no fitted parameters.

`c*` is a property of the **discretisation**, not of pendulums. `gymnasium`'s Pendulum uses
semi-implicit (symplectic) Euler -- velocity updated from the old angle, position from the **new**
velocity (`pendulum.py:138-140`) -- and `dt` is a settable attribute. So the theory makes a
**parameter-free scaling prediction**:

> `c*(dt) = 2.5 * dt`, a line through the origin.

If a world model trained **only on pixels** reproduces that line, it has learned the simulator's
numerical scheme, not the physics that scheme approximates. That is a claim about learned simulators
generally, with direct bearing on sim-to-real transfer and model-based RL, and it is strictly
stronger than the diagnostic claim the paper currently makes.

## Design

Four datasets at **dt in {0.02, 0.035, 0.05, 0.08}** -- a 4x range, predicted `c*` spanning
`0.05`--`0.20`. Everything else is held fixed: 256 trajectories x 120 frames, identical initial-
condition distribution, identical clip-rejection, identical training contract (3 seeds, step-capped,
analysed at the paper's primary step-6,500 checkpoint).

**The confound this design accepts, stated now.** With frames fixed at 120, physical duration varies
with `dt`: 2.4 s at 0.02 to 9.6 s at 0.08 (1.5 to 5.9 oscillation periods). Fixing duration instead
would vary dataset size and the number of training frames, which is a worse confound for a
representation claim. `dt = 0.01` is **excluded** because 120 frames covers only 0.74 of a period,
so a failure there could not be attributed to the hypothesis.

### The sweep is relative, so the same test is applied at every dt

Rather than an absolute grid, sweep `r = c / (2.5 dt)` over
`{-1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3}`. The prediction is then the **same
value of `r` at every timestep**, which makes the claim a single statement rather than four.
`r = 0` is textbook energy; `r = -1` is the wrong-sign control that E19 registered.

## Registered predictions

**P1 -- physics gate, no model.** On ground-truth states at each `dt`, the invariance ratio of
`T_r` is minimised at `r = 1` (exactly, on this grid) and is at least **2x** lower there than at
`r = 0`.

> **If P1 fails at any dt, STOP for that dt.** The relation would not hold in the data, so nothing
> measured on a model could be interpreted.

**P2 -- the model finds it.** `argmin_r rho_obs = 1` (or an adjacent grid point) on at least 2 of 3
seeds, for at least 3 of the 4 timesteps.

**P3 -- the scaling law.** Regressing the recovered `c*` on `dt` across all 12 models gives a slope
within **20%** of 2.5 and an intercept whose confidence interval contains zero. This is the headline:
a parameter-free line, predicted from the integrator, recovered from pixels.

**P4 -- specificity.** `rho_obs` at `r = -1` exceeds that at `r = +1`, at every dt, on at least 2 of
3 seeds.

## Falsifiers, stated plainly

- **P2 fails** -> the model does not track the timestep. The E19 result would then be specific to
  `dt = 0.05` rather than general, which materially narrows it and must be reported.
- **P3 fails while P2 holds** -> the model finds *a* correction that is not the integrator's. That
  would be the most interesting negative available: it would say the model learns *some* shadow
  quantity whose coefficient is not `(dt/2) mg(l/2)`, and identifying what it is instead becomes the
  next question.
- **P1 fails** -> derivation or data wrong at that timestep; stop there.

**No direction is predicted for the model results.** The physics prediction is exact; whether a
pixel-trained network recovers it is exactly what is unknown.

## Scope and honesty constraints

- 3 seeds per timestep, 12 models; the model seed is the independent unit.
- Physical labels are used only after the fit is frozen, per the roadmap's rule at line 79.
- All records carry the `latent_noether.provenance` stamp.
- Analysed at step 6,500, the paper's primary checkpoint, so this is comparable to E18/E19 rather
  than a new regime.

## Cost

Data generation ~30 min; 12 models x ~11 min at 6,500 steps; analysis minutes. Under 4 GPU-hours.
