# F14 — Does the probe-vs-operator gap survive cross-fitting?

**Registered 2026-09-08, before running. No training.**

## The defect this tests

E18's headline — a probe fitted to true energy is `6.7x` less preserved by the model's transition
than a label-free scalar — is measured with **`C` fitted and scored on the same 52 trajectories**
(`run_e18_supervised_baseline.py`: `cached_fit` on `fr[ANALYSIS]`, `rho_obs` scored on the same `Z`).

Both arms are in-sample, which is symmetric. The asymmetry that matters is what each is fitted
*for*: the label-free arm is fitted to **minimise conservation error** and is then **graded on
conservation error**, on the same trajectories. That is grading on the training set, and it favours
the label-free arm specifically.

E9 already holds out trajectories for the **repair** result, so that half is clean. This gap is not.

## Design

Fit on `frames[0:204]`, freeze **coefficients, `h_mean`, the PCA subspace and the rank basis**, then
score `rho_obs` on `frames[204:256]` — the exact slice the in-sample number uses, so the comparison
is like-for-like. The E9 machinery already supports this freeze-and-score pattern.

Both arms are re-fitted the same way: the label-free `C` via `cached_fit`, the supervised probe via
ridge onto true energy. Applied to the three reference models, and separately to the conv-GRU family
for the `767x` gap and to the OOD band for `155--267x`.

## Registered predictions

- **P1 (primary).** The cross-fit gap is `>= 3x` on at least **2 of 3** seeds, against `6.7x`
  in-sample. Some shrinkage is expected and is not itself a failure.
- **P2.** The conv-GRU separation stays `>= 100x` cross-fit, against `767x` in-sample.
- **P3.** The label-free `C` still correlates with true energy at `|rho_E| >= 0.9` cross-fit, i.e.
  it is the same quantity and not a different one refitted.

## Falsifier

If the gap falls below `2x` on 2 of 3 seeds, a material part of the headline was in-sample fitting.
The claim would then have to be restated as an in-sample one, or dropped. **That is the outcome to
report if it happens**, and finding it now rather than in review is the entire point of running this
before the write-up.

## Gate

- **G1 — the frozen readout must remain meaningful.** If the frozen `C` fails to correlate with
  anything on the held-out set (`|rho_E| < 0.5`), the freeze itself broke and neither arm is
  readable — an uninformative outcome, not a negative one.

## Direction

None stated.
