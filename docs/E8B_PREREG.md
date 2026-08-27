# Pre-registration — E8b, the two-factor account of when the repair works

**Written 2026-08-27, before the predicted quantities were computed.** Arises from the E8 curve; the
account it tests was formed *after* seeing that curve, so E8b exists to give it an **out-of-sample**
test rather than to re-describe the data that generated it.

## The account

The repair effect requires **two** things simultaneously:

1. **residual invariant drift** — something to correct
2. **a well-identified invariant** — a correct direction to correct along

Evidence that generated it (2-DoF seed 3): at step 15,000 drift is largest (2.94e-02) but `|rho_E|`
is 0.909 and the effect is only -18.2%; at 30,000 drift is 7.71e-03, `|rho_E|` 0.966, effect -57.6%;
at 60,000 drift is 1.01e-03, `|rho_E|` 0.987, effect +4.1%. Non-monotone in training, peaking in the
middle.

**This is a hypothesis fitted to observed data.** It is recorded as such, and the predictions below
are what make it falsifiable.

## Prediction A — the central arm should FAIL to repair despite having drift

The strongest available test, because it decouples the two factors.

The central arm at step 30,000 has a **broken joint fit**: `|rho_E| = 0.146`, `|rho_L| = 0.015`,
even though its conserved subspace cleanly contains angular momentum at 0.967. The matched
non-central arm at the *same* step has `|rho_E| = 0.966` and repairs at **-57.6%**.

The two-factor account therefore predicts:

> **The central arm's repair at step 30,000 is small or absent — registered as `> -20%`, i.e. it does
> not reach even a third of the non-central arm's -57.6% — despite the central model having
> comparable residual drift.**

- **Falsifier:** the central arm repairs at `<= -40%`. Then a poorly-identified `C` repairs about as
  well as a well-identified one, factor 2 is not required, and the account is wrong.
- **Confound to check and report:** the central arm's baseline drift must be within a factor of ~3 of
  the non-central arm's at the same step, or the comparison is about drift rather than identification
  and the test is void. Reported either way.

## Prediction B — 2-DoF seed 4 at step 60,000 should show no repair

Seed 4 tracked seed 3 closely at 15,000 and 30,000 (baselines 2.74e-02 vs 2.94e-02, effects -21.0%
vs -18.2% and -54.4% vs -57.6%). Seed 3 at 60,000 gave +4.1%.

> **Registered: seed 4 at step 60,000 gives `|effect| < 15%`.**

- **Falsifier:** `|effect| >= 15%` in either direction, which would mean the step-60,000 vanishing on
  seed 3 was seed-specific rather than a property of the converged model.

## Protocol

Direction-matched, `eps` grid unchanged, H = 100, analysis split, `C` frozen. Identical to every
other intervention run this session. No new fitting, no tuning.

## What a pass would and would not establish

A pass on both makes the two-factor account **predictive rather than descriptive**, which is the
minimum for it to appear in a paper as an explanation rather than a narrative.

It would **not** establish the mechanism. Why the pendulum's drift stays flat under training while
the 2-DoF model's collapses 29x is a separate question that neither prediction addresses.
