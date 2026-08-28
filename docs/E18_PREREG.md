# Pre-registration — E18, the supervised-energy baseline

**Written 2026-08-28, before any E18 quantity was computed.**

## The objection this answers

Nothing in this project has compared the recovered `C` against the obvious alternative: **fit a
readout to true energy, supervised, and project on that instead.** A reviewer will ask why the
label-free search is needed at all. Right now there is no answer.

## Design

Same latent, same coordinate frame, same degree-4 polynomial family, same analysis split. Only how
the coefficients are chosen differs:

- **unsupervised `C`** — `fit_hamiltonian_pair`, the existing pipeline, never sees energy
- **supervised `C_sup`** — ridge least squares of the degree-4 monomials onto **true energy**

Both then drive the identical direction-matched repair: fixed `eps` along the constraint's normal,
frozen `eps` grid, H = 100, `0/20` magnitude-matched random directions as the shared null.

## Registered predictions, with reasoning

The supervised probe is fitted to **track energy**; the unsupervised search is fitted to **be
conserved by the transition**. Those are different objectives, and E10b found that repair magnitude
tracks conservation quality, not decodability. Therefore:

1. **`|rho_E|`: supervised > unsupervised.** Near 1.0 by construction against ~0.93-0.97.
2. **`rho_obs`: supervised WORSE (larger) than unsupervised.** Nothing in a supervised fit asks the
   transition to preserve the result.
3. **Repair: supervised WORSE than unsupervised**, despite tracking energy better.

**If all three hold**, the paper gains a sharp statement: *searching for what the model conserves
beats fitting the physical quantity it is supposed to conserve* — which is the probe-versus-dynamics
thesis in its most direct form, and an argument for the label-free method rather than a concession.

**Falsifier:** the supervised probe repairs as well as or better than the unsupervised `C`. Then the
label-free search buys nothing over a supervised probe, its main practical justification is gone,
and the paper must say so.

## Controls and scope

The existing 20 magnitude-matched random directions on the same checkpoint, as a shared reference.
Pendulum seeds 3/4/5 at step 6,500 — where the unsupervised baseline is most thoroughly
characterised (n = 3 across recovery, repair, disjoint evaluation, E4 and E8).

## Known limit

A supervised probe fitted on the **same** trajectories it is scored on is optimistically biased in
its `|rho_E|`. That bias works **in the supervised arm's favour**, so it cannot manufacture the
registered prediction; it can only make the prediction harder to confirm.
