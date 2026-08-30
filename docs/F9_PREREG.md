# F9 --- Can these models resolve the scheme difference at all?

**Registered 2026-08-30, before writing the measurement. No training.**

## Why

Two instruments are now closed. The one-step statistic is determined by the evaluation dataset
(F7 amendment 2); the imagined-rollout statistic has no power (F8, argmin spanning the full grid
within a single model). Both failures are about *measurement*, and both left F6 and E19 "unresolved",
which is an uncomfortable place to leave a paper's headline claim.

This asks a prior question that neither instrument addressed, and that does not need a new instrument
at all: **is the difference between the two integrators even large enough for these models to
represent?**

Semi-implicit Euler and velocity Verlet differ in one step by exactly

    theta_SI' - theta_VV' = 0.5 * a(theta) * dt^2 = 0.01875 * sin(theta) rad at dt = 0.05

--- about one degree at the top of the training range. If a model's own one-step prediction error is
**larger** than that gap, it cannot encode which scheme produced its data, whatever statistic is used
to interrogate it. That would make F6's and E19's model-side claims unsupportable **in principle**
rather than merely unmeasured, and would close the question rather than leaving it open.

## Measurement

For semi-implicit-trained models (F6 `dt = 0.05`, seeds 3/4/5) on their own data, at each analysis
step `t`:

| quantity | definition |
|---|---|
| `D_model` | median &#124;`theta` decoded from `m.transition(h_t)` − true `theta_{t+1}`&#124; |
| `D_scheme` | median &#124;true `theta_{t+1}` − the `theta_{t+1}` velocity Verlet would give from the same `(theta_t, thetadot_t)`&#124; |
| `D_decoder` | median &#124;`theta` decoded from the *encoded real* frame at `t+1` − true `theta_{t+1}`&#124; |

`theta` is read from a single frame by the existing geometric readout, so no finite differencing and
no velocity estimate is involved.

`D_decoder` is a **required floor control**: without it, readout noise would be attributed to the
model's dynamics. The model's dynamical error is only interpretable relative to it.

## Registered predictions

- **P1 (cannot resolve).** `D_model / D_scheme >= 1.0` on at least **2 of 3** models -> the model's
  one-step error exceeds the entire difference between the schemes. F6 and E19's model-side claims
  are then unsupportable in principle, and the search for a better instrument should stop.
- **P2 (can resolve).** `D_model / D_scheme <= 0.3` on at least **2 of 3** -> the model is accurate
  enough to carry the distinction, the failure is in the measurement, and a better instrument is
  worth looking for.
- **Between 0.3 and 1.0** is neither; it is to be reported as marginal, not rounded to whichever
  conclusion is convenient.
- **P3 (floor).** Report `D_decoder / D_scheme` alongside. If `D_decoder >= D_scheme`, the readout
  itself cannot see the gap and **P1 must not be read as a fact about the model** --- the experiment
  would then be measuring the decoder, and would be inconclusive rather than decisive.

P3 exists because F8 taught me that an instrument can be alive and still blind, and I did not
register that possibility in advance. Registering it here.

## Direction

None stated.
