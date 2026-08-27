# Pre-registration — E17b, is the flow-alignment fit ill-posed with two invariants?

**Written 2026-08-27, before any E17b quantity was computed.**

## Why a new test is needed

The degeneracy claim has been restated once already, from "the joint fit never resolves" to "the
joint fit is unstable". The evidence for instability is currently **confounded**: it compares
different checkpoints, different seeds, and one accidental retraining, so training differences and
data differences are entangled.

`fit_hamiltonian_pair` is **deterministic** given `(traj, flow)` — it initialises at `a[0] = 1.0`
with no random component — so the observed spread cannot come from the optimiser. It must come from
the data. That makes a controlled test possible.

## Design

At a **single fixed checkpoint**, bootstrap-resample the analysis trajectories with replacement,
refit the whole pipeline on each resample, and measure the spread of `|rho_E|` across resamples.
Everything except the trajectory sample is held constant: same model, same training, same
hyperparameters, same code path.

- **non-central** (one invariant) — `osc2d_nc_s3_step30000.pt`
- **central** (two invariants) — `osc2d_ce_s0_step30000.pt`

20 bootstrap resamples per arm, seeded and recorded.

## Primary metric

**Interquartile range of `|rho_E|` across resamples**, per arm.

- **Registered prediction:** the central arm's IQR is **at least 3x** the non-central arm's. An
  ill-posed criterion should be far more sensitive to which trajectories it sees.
- **Falsifier:** the IQRs are comparable (ratio < 2). Then the fit is no more data-sensitive with two
  invariants than with one, "instability" is the wrong description, and the degeneracy claim needs a
  third restatement or withdrawal.

## Reported alongside, not decisive

Median `|rho_E|` per arm, the full spread, and the same statistics for `|rho_L|`. The **subspace**
statistics (`best rho_E`, `best rho_L` per resample) are reported to check the claim that the
subspace is stable while the selected direction is not — that pairing is the substance of the
degeneracy account and has never been tested on matched data.

## Known limit

Bootstrap resampling of trajectories is not the same as re-running the optimiser from a different
start, which the deterministic implementation makes impossible without modifying the method. This
measures **data sensitivity**, which is what an ill-posed problem exhibits, but it is not a direct
measurement of solution multiplicity. No claim about the number of optima will be made from it.
