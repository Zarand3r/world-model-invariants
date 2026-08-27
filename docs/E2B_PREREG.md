# Pre-registration — E2b, how invariant violation accumulates

**Written 2026-08-27, before any E2b quantity was computed.** Authorised by `docs/ROADMAP.md`'s own
decision tree, which for E2 Outcome B directs: "The mechanism is likely accumulated integrator bias
rather than loss of physical structure under distribution shift. **Refocus accordingly.**"
Claim addressed: **C3 as amended.**

## The question this settles

E2 established that the per-step conservation defect `r(z) = C(T(z)) - C(z)` is roughly constant
with rollout depth. E3 established that the component of one-step error normal to the level set is
small — about 2% of the error's energy, three to four times *less* than isotropic.

Those two facts are compatible with two different accumulation laws, and they make opposite
predictions:

- **systematic bias** — the per-step normal displacement has a consistent sign, so violations add
  coherently and `|C(z_k) - C(z_0)|` grows like `k^1`
- **random walk** — the per-step displacement is near-zero-mean, so violations add incoherently and
  `|C(z_k) - C(z_0)|` grows like `k^0.5`

A post-hoc check already reported in the log found the normal component **less** systematic than the
tangent one (|mean|/std 0.065-0.096 against 0.199-0.232), which points to the random walk. E2b tests
it directly rather than by inference.

## Primary metric — fixed now

Fit, per model, over rollout depths `k = 1..99`:

    log median_traj |C(z_k) - C(z_0)|  =  a + beta * log k

**`beta` is the registered statistic**, with a bootstrap CI over trajectories.

- **Random walk** if the CI contains 0.5 and excludes 1.0
- **Systematic bias** if the CI contains 1.0 and excludes 0.5
- **Neither** is reported as neither; no third mechanism is proposed in advance, and if `beta` lands
  outside [0.4, 1.1] the result is reported as unclassified rather than fitted to a story.

Registered prediction: **`beta` near 0.5**, following the |mean|/std evidence.

## Why this matters beyond bookkeeping

The two laws imply different things about the paper's claim. A systematic bias is a *defect of the
learned operator* that better training might remove. A random walk is closer to irreducible: it says
the transition is unbiased with respect to `C` but noisy, and no amount of training removes noise —
only an explicit projection does. The second is the stronger claim for the intervention, and it also
predicts the horizon scaling E6 measured.

It also supplies the mechanism E3's falsification left open: a small, near-zero-mean normal component
random-walking `C` away from its initial level set, while the larger tangent component moves the
state along the level set and costs phase accuracy but no energy.

## Controls

- **Random `C`** (20 norm-matched draws). Registered expectation: `beta` also near 0.5, since a
  random polynomial is not conserved and its value will also diffuse. **This control cannot
  discriminate**, and is run to confirm that `beta` alone is not being read as evidence of
  specificity — the specificity evidence is `rho_obs` and the intervention arms, not `beta`.
- **Damped models.** True energy decays systematically, so the damped arm is the positive control for
  `beta` near 1.0. If damped models do not show `beta` above the conservative models', the estimator
  is not measuring what it claims.

## Splits and freezing

Analysis split `204:`, `WARMUP = 10`, depth 100, `C` frozen exactly as in E1/E2/E3. Conservative
seeds 3/4/5 and damped seeds 0/1/2 at step 6,500. No new fitting of any kind.

## Falsifier for the amended C3

If `beta` is near 1.0 on the conservative models, the violation is a coherent systematic bias, the
"unbiased but noisy" reading is wrong, and the paper should say the transition has a directional
defect that training might fix — which would make E8's saturation sweep decisive rather than
supporting.
