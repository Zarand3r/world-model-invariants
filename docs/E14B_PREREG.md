# Pre-registration — E14b, is energy still *conserved* out of distribution, or merely decodable?

**Written 2026-08-27, before any E14b quantity was computed.** Follows directly from E14.

## What E14 left open

E14 found that the frozen `C` collapses out of distribution (`|rho_E|` 0.03-0.31) while a **fresh
probe** on the same OOD latents recovers energy at **0.999**. So the latent encodes OOD energy and
the frozen polynomial simply fails to extrapolate.

But a probe measures **decodability**. It says nothing about whether the model's own transition
**preserves** energy on those states. That distinction is the paper's central thesis, and E14 has
placed it in a new setting where it can be tested directly.

## The question

Refit the whole pipeline — PCA basis, rank basis, `fit_hamiltonian_pair` — **on the OOD latents**,
at frozen hyperparameters (LD = 12, degree 4, `n_basis` = 8, WARMUP = 10), and ask two things:

1. **Identification:** does the refitted `C_ood` track true energy? (`|rho_E|`)
2. **Conservation:** does the model's transition preserve it? (`rho_obs`, the operator statistic
   from E2 — one autonomous step from every real encoded OOD state, normalised by the
   across-trajectory std of `C_ood`)

## Registered predictions

- **Identification succeeds:** `|rho_E| > 0.8` for the refitted `C_ood`. The fresh probe reached
  0.999, so the information is present; if the constrained fit cannot find it, that is a fact about
  the fit rather than the latent.
- **Conservation is the open question, and no direction is predicted.** Both outcomes are
  informative and are registered as such:
  - `rho_obs` comparable to in-distribution (6.3e-03 to 8.7e-03): the transition preserves energy
    even on states it was not trained on — a **stronger** generality claim than anything currently
    in the project.
  - `rho_obs` far worse (approaching the untrained/damped range, ~4e-01): energy is **decodable but
    not conserved** out of distribution. That is the decodability-vs-dynamics dissociation appearing
    in a new place, and it would sharpen the paper's thesis rather than weaken it.

Registering "no prediction" here is deliberate: the honest state is that I do not know, and
committing to a direction would invite reading the result to match it.

## Controls

- **In-distribution `rho_obs`** recomputed in the same run, so the comparison is within-run rather
  than across log entries.
- **Random `C`** on the OOD latents (10 draws), for the same reason it exists everywhere else.

## Scope and limits

Pendulum seeds 3/4/5 at step 6,500, OOD LOW band (`runs/pendulum_ood_low.npz`, 128 trajectories).

**This is not a repair test.** E14 established the repair arm is unresolvable on this band — the
model's reconstruction floor (6.54e-03) exceeds typical baseline drift. E14b measures recovery and
conservation only, both of which use the latent and true states rather than the pixel readout, and
are therefore unaffected by that floor.

The OOD band has 128 trajectories against the analysis split's 52, so `rho_obs` is estimated on more
data, not less.
