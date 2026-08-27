# Pre-registration — E10b, conservation at matched decodability

**Written 2026-08-28, before any E10b quantity was computed.**

## Why this is possible now and was not before

E10 required candidates whose energy-decodability matches the recovered `C` while their conservation
varies. On the pendulum the eigenfamily is bimodal — of 250 candidates exactly **one** had
`|rho_E| > 0.3` — so the band could not be populated at any pool size, and E10 was recorded as **not
constructible**.

The **central 2-DoF arm** changes this. Its jointly-fitted `C` lands at `|rho_E| = 0.064` (the
two-invariant degeneracy), and **147 of 150** candidates fall within +-0.10 of it, with invariance
ratios spanning **4.12e-06 to 9.33e-01** — five orders of magnitude at effectively fixed
decodability.

## What this tests, and what it does not

The band sits at **low** decodability, not high. So E10b does **not** ask "among equally
energy-correlated candidates, does conservation help?" It asks the complementary question:

> **Among candidates that are equally (and poorly) energy-correlated, does better conservation alone
> produce a repair?**

- If **yes**: conservation is *sufficient* for repair without energy-correlation, and the paper's
  mechanism is about dynamical structure rather than about energy specifically.
- If **no**: conservation is *not sufficient*; a candidate must also track the physical quantity.
  Combined with the two-factor account, that would say repair needs **both** conservation and
  correct identification — which is what the account already claims, tested here on a new axis.

Both outcomes are informative. **No direction is predicted**, for the same reason as E14b: the honest
state is not knowing.

## Design

From the 147 in-band candidates, select **20 stratified by invariance ratio** across the full range
(quantiles of `log10(ratio)`), so conservation varies by construction and decodability does not.

Intervene with each using the **direction-matched** protocol — fixed `eps` along the candidate's
normal — `eps = 0.02`, H = 100, analysis split, everything else frozen.

## Primary metric

**Spearman(invariance ratio, repair magnitude) across the 20 stratified candidates.**

- **Registered as informative in either direction**, with the falsifier for a *relationship* being a
  CI containing zero.
- Reported alongside: Spearman(`|rho_E|`, repair) within the band, to confirm decodability really is
  held fixed and is not secretly driving the result.

## Controls

The recovered `C` itself and 10 magnitude-matched random directions, both already characterised on
this checkpoint, as reference points on the same axis.

## Scope

Central seed 0 at step 60,000, one model. The evidence-base guard will flag it n = 1 and it will be
reported as provisional until central seed 2 finishes training.
