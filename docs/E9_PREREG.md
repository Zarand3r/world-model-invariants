# Pre-registration — E9, truly disjoint causal evaluation

**Written 2026-08-27, before any E9 quantity was computed.** Governed by `docs/ROADMAP.md` Phase II.
Claims addressed: **C2 (physical validity)** and **C4-adjacent** — that the repair is not an artefact
of scoring on the trajectories the invariant was fitted to.

## The gap

Everything so far fits `C` on the analysis split `204:` of `runs/pendulum_pixels.npz` and scores the
intervention on those same 52 trajectories. The published paper discloses this ("the absolute effect
is in-sample with respect to invariant fitting") and does not close it. The direction-matched null
makes the *comparison* sound, since every arm shares the trajectories, but the absolute effect size
is still in-sample.

## Design

`runs/pendulum_pixels_eval.npz` — 512 trajectories x 200 frames, generator seed 777, never used for
training, fitting, calibration, or any analysis to date. Disjoint from both the Dreamer training
split (0:204) and the analysis split (204:).

    fit    C, h_mean, U, R   on the analysis split of pendulum_pixels.npz     <- unchanged, frozen
    score  the intervention  on pendulum_pixels_eval.npz                      <- never seen

The whole coordinate system is frozen, not just the coefficients: `h_mean`, the PCA basis `U`, and
the rank basis `R` were all derived from the fit data and are part of `C` as a function of `h`.
Re-deriving any of them on the eval set would leak.

## Primary metric — unchanged from E1

Median absolute `D_sec` at matched step magnitude, recovered direction versus 20 magnitude-matched
random directions and 5 equal-norm tangent controls, on the frozen `eps` grid
(0, 0.0025, 0.005, 0.01, 0.02). Statistic is the change from `eps = 0`, and the specificity count is
the number of random directions beating the recovered one.

**No metric, grid, or arm is changed for E9.** Only the trajectories being scored change.

## Horizons

`H = 100` for direct comparison with everything already run, and `H = 200` — which the 120-frame
analysis split cannot support and this 200-frame set can. E6 registers the prediction that correction
benefit grows with horizon; H = 200 is the first point that can test it beyond H = 100.

## Registered predictions

- **Primary:** the recovered direction still reduces median abs `D_sec` on the eval set, and no
  random direction beats it. Effect size may be smaller than in-sample; that is expected and is the
  point of running it.
- **Falsifier:** the effect vanishes or reverses out-of-sample. Then the in-sample effect was
  overfitting of the 8-dimensional `C` to the 52 analysis trajectories, and every absolute number
  reported so far must be restated as in-sample only.
- **E6 secondary:** benefit at H = 200 exceeds benefit at H = 100. Registered as a prediction, not a
  gate.

## Exclusions

The eval set was generated with the same rejection rule (no trajectory reaching the |thetadot| = 8
clip). No further exclusion. Trajectories are excluded only if the geometric readout fails on their
rendered frames, determined before any rollout is decoded.
