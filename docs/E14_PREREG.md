# Pre-registration — E14, out-of-distribution energies

**Written 2026-08-27, before any E14 data was generated.** `docs/ROADMAP.md` Phase III.
Claims addressed: **C2, C6**.

## The objection this answers

Kang et al. (ICML 2025, arXiv:2411.02385) find video models perform **case-based retrieval** — they
mimic the nearest training example and fail completely out of distribution. Under that reading, a
recovered "invariant" is a training-manifold artefact rather than a law, and a repair that works
in-distribution says nothing.

E9 does **not** answer this. E9 scored on 512 unseen *trajectories* drawn from the **same**
initial-condition distribution. It tests trajectory disjointness, not distribution shift, and the
log has said so since E9 ran.

## Design

Generate trajectories whose **energies lie outside the training band**, with everything else
identical: same integrator, same renderer, same rejection rule, same trajectory length.

- Pendulum: sample `theta_0`, `thetadot_0` from ranges giving energies **above and below** the
  training band, staying safely under the `|thetadot| = 8` clip so conservation is intact.
- 2-DoF non-central: same, via the initial-condition scale, keeping `|q| < 0.95 * HALF` so
  trajectories stay in frame.

Two bands per system, reported separately and never pooled:

- **LOW**: mean energy below the 5th percentile of the training distribution
- **HIGH**: mean energy above the 95th percentile

`C`, `h_mean`, `U`, `R` frozen from the analysis split, exactly as in E9. **No refitting of any
kind.**

## Primary metric — unchanged

Median absolute `D_sec`, recovered direction versus 20 magnitude-matched random directions, frozen
`eps` grid, H = 100. Statistic is the change from `eps = 0` and the count of random directions
beating the recovered one.

## Registered predictions

1. **Recovery survives**: `|rho_E|` on the OOD band stays above 0.7 (against 0.97-0.99
   in-distribution). A drop is expected; a collapse is not.
2. **Repair survives**: the recovered direction still reduces `D_sec`, with 0 of 20 random
   directions beating it, on both bands and both systems.

**Falsifier for the case-based-retrieval reading:** if predictions 1 and 2 hold, the recovered
scalar is not a training-manifold artefact.

**Falsifier for our own claim:** if the effect vanishes or reverses on either band, the repair is
confined to the training distribution and every claim in the project must be qualified as
in-distribution. That would be a significant negative result and will be reported as one.

## Confound to measure and report

An OOD trajectory is also a **harder** trajectory: the world model reconstructs it worse, so the
readout floor rises. The floor will be measured **on the OOD band at each checkpoint** before any
effect is quoted, exactly as was done for the in-distribution floors — and the 2026-08-27 correction
applies, that the floor must be measured at the horizon in use, not carried over from H = 50.

## Scope

Pendulum seeds 3/4/5 at step 6,500 and the 2-DoF non-central arm at step 30,000 — the checkpoints
where the in-distribution effect is best characterised, so any change is attributable to the band
rather than to training.
