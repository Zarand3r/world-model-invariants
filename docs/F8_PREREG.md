# F8 --- Does the model itself carry the integrator? An imagined-rollout measurement

**Registered 2026-08-30, before writing the measurement or running anything. No training.**

## Why

F7's cross-evaluation control (`docs/F7_PREREG.md`, amendment 2) showed the measurement F6, E19 and
F7 all share is determined by the **evaluation dataset**, not the checkpoint: the argmin was `+1.0` in
all six semi-implicit-eval cells and `0.0` in all six Verlet-eval cells, regardless of which model
took the step. The cause is structural --- `rho_obs` is computed from a **one-step** prediction on
encoded **real** frames, and a model that predicts well reproduces the next state of whatever
trajectory it is shown.

That falsified F7. It left F6 and E19 **unresolved**: they use the same measurement, and the only
axis F6 varies (the timestep) puts a crossed model out of distribution, so the control that settled
F7 cannot be run on them (amendment 3: `0/6` follow the data, `5/6` pinned at grid edges).

This experiment removes the evaluation trajectory instead of crossing it.

## The measurement

Conservation along the model's **own imagined rollout**. The data supplies only the initial latent;
every subsequent state is produced by `m.transition`. There is no trajectory to track, so the
mechanism that defeated the one-step statistic cannot operate.

    h_0 = encode(real frames)[:, WARMUP]
    h_{t+1} = m.transition(h_t)                       for t = 0 .. T-1
    rho_img(r) = median_t |C_r(h_{t+1}) - C_r(h_t)| / std_t C_r(h_t)

`C_r` is fitted exactly as in F6/E19 (degree-4 monomials on the model's own PCA frame, regressed onto
`E + r c*(dt) theta_dot sin(theta)`). The fit is necessarily per-model, since each model has its own
latent space --- that is unavoidable and is how E19 and F6 already work. E1's amendment-4 control
showed the repair effect does not depend on the data used to *identify* `C`, so this is not the
lever under test.

## The decisive comparison

Semi-implicit-trained models (F6 `dt = 0.05`, seeds 3/4/5) and Verlet-trained models (F7, seeds
3/4/5), **rolling from the same real frames at the same timestep**. Only the model differs.

- **P1 (primary).** The two families separate: semi-implicit models give argmin `r >= 0.5`, Verlet
  models give argmin `r <= 0.5`, each on at least **2 of 3** seeds.
- **Falsifier.** If both families land at the same argmin, the model does not carry its integrator,
  and **F6's and E19's model-side claims fail** the same way F7's did. The paper's title claim would
  then be unsupported and must be withdrawn or rewritten.
- **P2 (validity gate --- read this before P1).** An imagined rollout that decays to a fixed point
  conserves everything trivially and `rho_img` becomes `0/0`. The measurement is only readable if the
  rollout stays alive. Registered: median `std_t C_r` along the imagined trajectory must be at least
  **20%** of `std C_r` on the real trajectories, and the rollout must stay finite, on every model.
  **If P2 fails, P1 is not to be read at all** --- the same rule as E1's hard gate.

## Direction

Stating none. I registered an expectation twice this week; once it was wrong, and once wrong in a way
I had not imagined. The measurement decides.

## Scope

This adjudicates whether the *model* carries the scheme. It does **not** by itself restore F7 ---
that experiment's registered analysis remains falsified. And a P1 pass would support F6/E19's
model-side claim without re-establishing anything about the *coefficient scaling law*, which would
need its own imagined-rollout replication across timesteps.

---

## Amendment 1 --- a validity gate I failed to register (POST-HOC, declared)

**Written 2026-08-30 after seeing the horizon-100 results.**

P2 was registered to catch a rollout that **collapses**. It did not anticipate an *alive* rollout
whose sweep is nearly **flat**, and that is what happened: rollouts are healthy (imagined spread
1.02--1.06x the real spread) but the contrast across the whole `r` grid is **1.12x--1.45x**, against
roughly **5.7x** for the one-step measure between `r = 0` and `r = 1` alone.

With a flat sweep the argmin is noise. My script's printed conclusion --- "the families do NOT
separate: F6/E19 model-side claims fail as F7's did" --- was therefore **wrong**, in the same way
`run_f6_cross.py`'s verdict was wrong yesterday: treating "no separation detected" as "no separation
exists" when the instrument cannot see either way.

I have added a contrast gate (`P3`, requiring `>= 2.0x`) and it fails. **This gate is post-hoc.** It
is declared as such, it can only ever *withhold* a conclusion rather than create one, and it makes
the result weaker (uninformative) rather than stronger. The registered outcome stands as:

> **F8 at horizon 100 is UNINFORMATIVE. It neither supports nor refutes F6 and E19.**

## Amendment 2 --- horizon sweep, and the confound check that must accompany it

**Registered before running.**

The two ends of the horizon axis are both useless for opposite reasons: at horizon 1 the measurement
*is* the confounded one-step statistic, and at horizon 100 accumulated rollout error swamps the
`O(dt)` distinction between shadow candidates. If a usable window exists it is in between.

Sweep horizons `{2, 3, 5, 10, 25, 50, 100}` on all six models, reusing one rollout per model and
evaluating over prefixes so nothing is recomputed.

- **P4 (power).** A horizon is *readable* only if sweep contrast is `>= 2.0x` on at least **4 of 6**
  models. Report the contrast at every horizon regardless.
- **P5 (separation).** At the shortest readable horizon, the families separate: semi-implicit argmin
  `>= 0.5` and Verlet argmin `<= 0.5`, each on at least **2 of 3**.
- **P6 (the confound check, mandatory before reading P5).** Any readable horizon is close enough to
  one step that the F7 confound must be ruled out *at that horizon*, not assumed away. Roll every
  model from initial latents taken from **both** datasets. P5 may only be read if the argmin follows
  the **model** rather than the dataset that supplied `h_0`, on at least 2 of 3 per family --- the
  direct analogue of F7 amendment 2.
- **Falsifier.** If no horizon is readable, the imagined-rollout approach cannot adjudicate F6 and
  E19, and they stay unresolved pending a different measurement.

No expected direction stated.
