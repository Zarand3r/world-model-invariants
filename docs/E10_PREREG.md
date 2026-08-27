# Pre-registration — E10, decodability-matched but conservation-mismatched null

**Written 2026-08-27, before any E10 quantity was computed.** `docs/ROADMAP.md` Phase III, and the
roadmap calls it "one of the most important controls". Claims addressed: **C2, C4** — and it is the
direct test of the paper's probe-vs-dynamics thesis.

## The gap this closes

Every null run so far answers *"does an arbitrary latent constraint help?"* — random degree-4
polynomials, magnitude-matched random directions, equal-norm tangent steps. All say no.

None answers the question the paper's thesis actually rests on: **is conservation specifically what
matters, or merely decodability?** A random polynomial is both non-conserved *and* weakly
energy-correlated, so it cannot separate the two. Liao & Cao (arXiv:2607.03728) argue this
conflation is the central methodological failure of the subfield, and the paper's own untrained
baseline (rho_E up to 0.908) shows decodability is nearly free on a 1-DoF conservative system.

## Candidate pool — existing machinery, no new fitting

`latent_noether.polynomial.polynomial_invariants(Z, degree=4, max_results=N)` returns the ranked
eigenfamily of the generalized problem `W a = lambda T a`, each candidate carrying its **invariance
ratio** `lambda` (0 = perfectly conserved, ~1 = not). Taking candidates across the spectrum gives a
range of `lambda` at no extra cost.

For each candidate `Q_i` we then measure `|rho(Q_i, E_true)|` — its **energy decodability** — using
the frozen geometric readout. Ground truth enters only at this scoring step, never in selection.

## The registered comparison

Restrict to candidates whose decodability is **matched to the recovered `C`**:

    | |rho(Q_i, E)| - |rho(C, E)| |  <=  0.05

and among those, correlate intervention benefit with the invariance ratio.

    improvement_i = -( D_sec(eps=0.02) - D_sec(eps=0) ) / D_sec(eps=0)      (positive = better)

**PRIMARY STATISTIC: Spearman(lambda_i, improvement_i) among decodability-matched candidates.**

- **Registered prediction:** **negative**, CI excluding 0. Better-conserved candidates (lower
  `lambda`) give more benefit, *at matched decodability*.
- **Falsifier:** the correlation is 0 or positive, or `|rho(Q_i, E)|` predicts improvement at least
  as well as `lambda` does. Then decodability, not conservation, is what the intervention is
  exploiting, and the paper's probe-vs-dynamics claim is unsupported.

Secondary, reported regardless: Spearman(`|rho|`, improvement) over the **whole** pool, so the two
predictors can be compared on equal footing rather than only inside the matched band.

## Guard against an empty or degenerate band

If fewer than 8 candidates fall inside the +-0.05 decodability band, the band is widened to +-0.10
and the change is recorded; if fewer than 8 remain at +-0.10, E10 is reported as **inconclusive for
lack of matched candidates**, not re-tuned further. Fixed now so band width cannot be chosen to
produce a result.

## Intervention protocol

The **direction-matched** protocol (`run_e1_direction_matched_null.py`): fixed step `eps` along each
candidate's normal, so candidates differ only in direction and in conservation quality, never in
edit magnitude. `eps = 0.02`, H = 100, analysis split, `C` and the coordinate frame frozen as in
E1/E2/E3.

## Seeds

Conservative seeds 3/4/5 at step 6,500. The registered statistic is computed per seed; the claim
requires the sign to agree on all three.

## Known limitation

`lambda` and `|rho|` are not independent in the population — on a 1-DoF conservative system the
best-conserved scalars *are* the energy-like ones, so the matched band is narrow by construction.
E10 tests whether conservation carries information **beyond** decodability inside that band; it
cannot fully decorrelate two quantities the physics ties together. That limit is stated here in
advance and will be stated in any writeup.
