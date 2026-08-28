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

---

## Amendment, 2026-08-28 -- pool size, approved by Richard before running

**Written before any 400-pool quantity was computed.**

### Why

At the registered pool of 150, the band populates 147/150 on central s0 and 148/150 on s1, but only
**19/150** on s2, whose reference `|rho_E|` is 0.111 rather than 0.05-0.06 and therefore sits in a
sparse region of the eigenfamily. s2's 19 candidates span invariance ratios of `1.8e-06` to `7.3e-04`
-- under three orders of magnitude, against **five** on the other two seeds.

The design exists to vary conservation widely at fixed decodability. On s2 it barely varies
conservation, so s2's near-zero Spearman is a **weak test**, not evidence against the hypothesis.

Pool size was always a design lever rather than a result here: the original `E10_PREREG` recorded E10
as unconstructible on the pendulum "at any pool size", which presupposes that enlarging the pool is a
legitimate move to populate a band.

### The change

`N_CANDIDATES` 150 -> **400**, applied to **all three seeds**, not only to s2. Running s2 alone at a
larger pool would give one seed a bespoke design and would be indistinguishable from tuning. Every
other parameter is unchanged: band `|rho_E - ref| <= 0.10`, 20 strata by quantiles of
`log10(ratio)`, direction-matched, `eps = 0.02`, `H = 100`, same checkpoints.

### Registered predictions

**A1.** s2's band populates to **at least 100 of 400** and its selected candidates span **at least
four orders of magnitude** of invariance ratio. *If A1 fails, s2 is not constructible at this pool
size either, and that is reported as a property of the seed rather than retried at a larger pool.*

**A2.** The primary metric is re-read on all three seeds at the matched 400-pool design. The
registered falsifier is unchanged: **a CI containing zero** means no relationship is established on
that seed.

**No direction is predicted**, exactly as in the original registration. Enlarging the pool is
expected to change s2 the most and s0/s1 the least, since their bands were already near-saturated;
if s0 or s1 move substantially, that itself is evidence the metric is unstable to pool size and will
be reported as such.

### Reporting rule, fixed now

The 400-pool results become primary because the design is matched across seeds. The 150-pool results
are **kept in `runs/e10b_matched_band.json` and reported alongside**, not replaced. If the two pools
disagree on whether a CI excludes zero, both are stated and the claim is written to the weaker.
