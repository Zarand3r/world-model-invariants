# Execution log — ROADMAP.md

Append-only. Newest entries at the bottom. Every entry records what was run, what came out, and what
it changed. Derived summaries live here; **immutable raw rows live in `runs/`**, and figures must
regenerate from those, never from this file.

Conventions:

- times UTC
- every experiment cites its preregistration file and the git commit it ran at
- negative and ambiguous results are recorded with the same weight as positive ones
- anything that deviates from `docs/ROADMAP.md` is flagged `DEVIATION` and requires Richard's approval

---

## 2026-08-26 — Session 1: adoption and Stage 1 bootstrap

### Roadmap adopted

`docs/ROADMAP.md` written and adopted as the governing plan (707 lines). Six-claim hierarchy
C1–C6; Phase I gate is **E1**; execution order Stage 1 → 5; decision tree fixed in advance.

### Environment: the repo does not carry its own artefacts

Discovered at start of execution, and material to reproducibility:

| needed | present at session start |
|---|---|
| `runs/*.pt` model checkpoints | **absent** (~54 MB each, deliberately uncommitted) |
| `runs/*.npz` datasets | **absent** |
| `external/dreamerv3-torch` | **absent** (vendored third-party checkout) |
| torch / project env | **absent** |

So Phase I could not begin "on existing checkpoints" as the roadmap assumes. Everything was rebuilt
from `docs/REPRODUCE.md`.

**Consequence for every downstream number, and it must be restated in the paper:** these are newly
trained checkpoints, not the ones behind arXiv:2608.23526. Training is capped by wall clock rather
than step count, so a re-run lands *near but not exactly on* the committed run logs. Any comparison
between a Session-1 number and a published number is a comparison across model populations, not a
reproduction of the same model.

### Repo defect found and fixed

`pyproject.toml` was broken by commit `009cdc9` ("Rename to world-model-invariants"): a
project-wide rename also renamed the **PyTorch package index**, so `[tool.uv.sources]` referenced
index `pytorch-cu128` while `[[tool.uv.index]]` declared `world-model-invariants`. `uv sync` could
not resolve torch. Fixed by restoring the index name; the `[project] name` is unchanged from HEAD.

Diff is one line. Verified: `torch 2.11.0+cu128`, CUDA available on an RTX PRO 6000 Blackwell
(96 GB); `pytest tests/ --ignore=tests/test_timing_convention.py` → **37 passed**.

### Vendored reference implementation

`external/dreamerv3-torch` cloned and checked out at the pinned commit
`6ef8646d807cd10ce0c88e10a7e943211e7fc44c`. Adapter agreement with the reference `img_step`
(`tests/test_timing_convention.py`) not yet run — **outstanding**, and REPRODUCE.md is explicit that
two adapter bugs previously masqueraded as model failure. To be run before any E1 metric is trusted.

### E1 preregistered

`docs/E1_PREREG.md` written **before** any E1 result. Primary metric fixed as the normalised secular
drift of decoded energy, `D_sec`, scored by its slope across the frozen alpha grid (D9 discipline,
never best-alpha). Readout is a non-learned geometric estimator whose noise floor is measured on real
frames and on Dreamer reconstructions before any intervention number is computed. Gate conditions and
the failure response are written down in advance.

### Stage 1 bootstrap launched

`scripts/run_stage1_bootstrap.sh` — new, idempotent, per-step logs under `runs/logs/`. Runs the
REPRODUCE.md commands in order: three datasets (conservative seed 0, eval seed 777, damped seed 11),
then 3 conservative models (seeds 3/4/5) and 3 damped models (seeds 0/1/2), each capped at 0.5 h.
Estimated ~3.5 h.

**Status: running.**

Next actions, in order: verify the adapter against the reference; implement and validate the E1
geometric readout on real frames; run E1 arms A–D.

---

## 2026-08-26 20:30–21:05 UTC — Session 1 cont.: adapter verified, E1 readout built and characterised

Git at `20fa8b4`. Loop `3a66bcaa` scheduled (`7,37 * * * *`, session-only, 7-day auto-expiry).

### Adapter verification — REPRODUCE.md's required check had no test

REPRODUCE.md states the adapter's `transition` must agree with the reference RSSM's `img_step` to
~1e-08 and calls it the check to run before trusting any result. That number lived only in a
docstring in `scripts/run_dreamer_extraction.py`; **nothing executed it**. Two adapter bugs
previously presented as model failure and were caught only by this comparison.

Added `tests/test_adapter_matches_reference.py`. Asserts the correspondence the adapter actually
claims — `img_step({"deter": h, "stoch": prior_stoch(h)}, 0)["deter"] == transition(h)` — plus
determinism of `transition`. **Result: max abs difference 0.000e+00** on a randomly initialised
model, batch 1 and 8.

Honest caveat: exact zero, rather than the documented 5.1e-08, because the reference is handed our
own `prior_stoch` output. The test therefore verifies the GRU / `_img_in_layers` / deter-update path
shares no divergence with the reference, but does **not** independently verify `prior_stoch` itself.
A full rollout-level comparison against the reference `imagine` would be strictly stronger and is
not yet written.

`tests/test_timing_convention.py::test_dreamer_encode_is_deterministic` still skips — it needs
`runs/dreamer_ref_s0.pt`, which the bootstrap does not train (bootstrap trains conservative s3/s4/s5
and damped s0/s1/s2). Noted for E13: `run_dreamer_extraction.py --untrained` seeds from
`sha256(ckpt_path)` and only needs the file to *exist*, so reproducing the paper's six untrained
draws requires files at those exact paths.

### Dataset regenerates faithfully

`runs/pendulum_pixels.npz` (256x120, seed 0) and `runs/pendulum_pixels_eval.npz` (512x200, seed 777)
built; damped (seed 11) in progress.

| statistic | this run | paper |
|---|---|---|
| rotating trajectories | 18.4% (47/256) | 17% |
| textbook-E relative oscillation (median) | 0.123 | ~12% |
| secular E slope / across-traj E std (median) | +3.5e-04 | "no secular drift" |
| max abs thetadot | 7.99 (11/256 above 7.9) | rejects at the 8.0 clip |

Nothing reaches the 8.0 clip, so conservation in the data is intact — but 140/256 trajectories
exceed 7.0, i.e. the dataset runs closer to the clip than the prose suggests. Recorded, not acted on.

### E1 readout: `latent_noether/pixel_readout.py` (new)

Geometric, not learned. Ink-weighted centroid about a pivot, `theta = -atan2(cx - p, -(cy - p))`,
backward-difference `thetadot`, textbook `E`.

**One fitted scalar, declared.** `downsample()` crops the *top-left* 448x448 of the 500x500 render,
so the pivot sits near 250/7 = 35.71 in 64-space, not at the image centre. Pure geometry at 250/7
gives 3.63 deg median angle error, because the ink centroid includes the axle glyph. Fitting the
single scalar `pivot` gives 0.11 deg. Fitted **on the Dreamer training trajectories (0:204)**, which
E1 never scores on, and frozen to `runs/pixel_readout_calibration.json` → `pivot = 35.20`. Sign and
angular offset were searched and came back exactly -1 and +0.0003 rad, so they are not fitted.

### The convention bug that would have hidden the E1 effect

First validation gave `thetadot` error 0.287 rad/s and decoded-energy error **0.155** across-traj
std. Diagnosis: not readout noise. Gymnasium integrates semi-implicitly —

    thdot_k = thdot_{k-1} + accel(th_{k-1}) dt ;   th_k = th_{k-1} + thdot_k dt

— so the stored `thdot_k` is *by construction* the backward difference. Measured against the stored
states: backward difference median error **0.0000** (exact); central difference **0.2905**.

`np.gradient` was therefore contributing 0.29 rad/s of pure convention error.

| metric, analysis split (52 traj, steps 10–60) | central diff | backward diff |
|---|---|---|
| median abs theta error | 0.110 deg | 0.110 deg |
| median abs thetadot error | 0.2866 | **0.0303** |
| median abs E error / across-traj std | 0.1549 | **0.0159** |
| p90 abs E error / across-traj std | 0.3401 | **0.0641** |
| `D_sec` noise floor (median abs) | 0.00133 | **0.000938** |

A 15% per-frame energy floor would plausibly have masked the effect E1 is looking for. Fixed, with
the reasoning in the function docstring and a regression test.

**Registered noise floor for E1: `D_sec` = 9.4e-04.** Any arm-A improvement smaller than this is
reported as "below readout resolution", per the preregistration.

### Artefacts

- `runs/e1_readout_validation.json` — raw rows, real-frames arm complete; reconstruction arm records
  all three checkpoints as `skipped` (not yet trained)
- `runs/pixel_readout_calibration.json` — the frozen pivot
- `tests/test_pixel_readout.py`, `tests/test_adapter_matches_reference.py` — **8 passed**

### Status

Stage 1 bootstrap still running (damped data, then 6 trainings, ~3 h). Nothing yet contradicts the
roadmap; no deviation requiring approval.

**Blocked on checkpoints:** E1 readout-validation arm (b) (reconstructions), then E1 arms A–D.

---

## 2026-08-26 20:44–21:00 UTC — Session 1 cont.: **DEVIATION, approved** — training contract

Git at `e0f239d`.

### What was found

`scripts/train_dreamer_pendulum.py` carries an M28 note on its own `--max-hours` argument:

> WALL CLOCK IS AN ENERGY BOUND, NOT A MATCHING VARIABLE (M28). Models being compared must receive
> equal OPTIMIZER STEPS; leaving wall clock to decide gave the conservative arm 37% more steps than
> the dissipative one because the machine was busier for the second batch. Set --max-hours loose
> enough that it never binds, and let --steps define the contract.

`docs/REPRODUCE.md` nonetheless invokes `--max-hours 0.5 --steps 40000`, and the loop is
`while step < steps AND elapsed < max_hours` — so **wall clock binds**, which is precisely the
failure M28 forbids. The module docstring also asserted the opposite of M28 ("Runs are capped by
WALL CLOCK, not step count"); corrected.

Measured on this box: **741 steps/min** (2000 steps in 2.7 min), so `--max-hours 0.5` yields
**~22,000 steps** against the paper's ~6,500 — a 3.4x difference, and a step count that is a
function of machine load.

Two consequences: models trained here would not be comparable to the published ones, and
conservative vs damped arms could silently receive different amounts of training, undermining the
cross-arm comparisons E1, E2 and E7 all rest on.

### Decision — put to Richard, approved

Training is now capped by **optimizer steps** (`--steps 60000`, `--max-hours 6` as a non-binding
energy bound), and each run saves the **E8 milestone grid** `{1k, 3k, 6.5k, 15k, 30k, 60k}` via a
new `--ckpt-at` argument. One run therefore yields the whole training trajectory on a single
optimisation path, giving both a paper-comparable 6,500-step checkpoint and a saturated one.

**E8 moves from Stage 3 to Stage 1.** `docs/ROADMAP.md` amended in place with the rationale.

Cost: ~81 min/model x 6 = ~8.5 h, against ~3 h for the original wall-clock plan.

### Actions

- `train_dreamer_pendulum.py`: `--ckpt-at`; dataset hash hoisted out of the loop; save routed
  through one helper so intermediate and final checkpoints carry identical M29 provenance
  (`steps`, `seed`, `data`, `data_sha256`, `argv`); `--max-hours` default 0.5 -> 6.0; docstring
  corrected.
- `run_stage1_bootstrap.sh`: step-capped contract, `STEPS`/`CKPT_AT` overridable from the
  environment.
- First bootstrap killed at ~10 min into seed 3; the partial `dreamer_ref_s3.pt` was deleted rather
  than kept, so no model trained under the old contract survives to be confused with a new one.
- Relaunched 20:54:34Z. Datasets were skipped as already present (the script is idempotent).

### Also written this session, ahead of the GPU

- `scripts/run_e1_physical_energy.py` — E1 arms A–D. The edit is copied verbatim from
  `run_dreamer_edit.py`; only the scoring differs. Raw per-trajectory `D_sec` rows are kept.
  Resumable per model and per random draw. Not yet run: no checkpoints.
- `docs/E2_PREREG.md` — preregistered while the GPU was busy and **before any E2 quantity existed**.
  Fixes the local conservation defect `r(z) = C(T(z)) - C(z)`, its normalisation, the three
  registered outcomes with numeric thresholds, a single preregistered off-support metric, and the
  result that would falsify the roadmap's target claim. Records explicitly that E2 can establish
  association between leaving support and defect growth, **not** causation.

### Status

Training seed 3 of 6, ~8.5 h remaining. No result yet bears on any claim C1–C6.

---

## 2026-08-26 21:10–21:30 UTC — E1 readout arm (b) complete; first E1 signal (PREVIEW, not the result)

Git at `afb99db`. Training: seed 3 of 6, past step 10,000.

### E1 readout validation arm (b) — reconstructions

Required by `docs/E1_PREREG.md` before any intervention number. Run on the s3 milestone sweep.

| source | theta err (deg) | E err / across-traj std | `D_sec` floor |
|---|---|---|---|
| real rendered frames | 0.110 | 0.0159 | 9.38e-04 |
| s3 @ step 1000 | 0.740 | 0.1304 | 1.23e-03 |
| s3 @ step 3000 | 0.415 | 0.0640 | 9.75e-04 |
| s3 @ step 6500 | 0.348 | 0.0409 | 9.97e-04 |

Two things fall out.

**The prereg's choice of a slope statistic is vindicated.** Per-frame decoded-energy error on
reconstructions is 2.6x worse than on rendered frames (0.041 vs 0.016), but the `D_sec` floor is
essentially unchanged (9.97e-04 vs 9.38e-04) — reconstruction error is close to white and averages
out of a slope. **Binding floor registered as 1.0e-03.** Blank-decode fraction 0.000 throughout.

**Bonus E8 signal.** Readout error on reconstructions falls monotonically with training. Not an E1
quantity; recorded because it comes free from the milestone grid.

Gap in my own implementation, found and fixed: `validate_recon` computed error statistics but not
`D_sec`, so the floor that actually binds E1 (reconstructions, not rendered frames) was unmeasured.
The prereg asked for the floor on both arms. Patched and re-run.

### E1 arm A, single model — PREVIEW

`runs/e1_preview_s3_step6500.json`, `runs/e1_preview_s3_step6500_H100.json`.
**Arm A only, one checkpoint, no arm B, no arm C.** Reported as a preview of the pipeline and an
early signal. It is not the E1 result and does not settle the gate.

| statistic | H = 50 | H = 100 |
|---|---|---|
| median abs `D_sec`, alpha 0 -> 0.4 | 2.396e-03 -> 2.189e-03 | 1.279e-03 -> **6.640e-04** |
| paired change, 95% CI (20k bootstrap over trajectories) | -2.07e-04 [-1.19e-03, **+4.78e-04**] | -6.15e-04 [-1.02e-03, **-1.92e-04**] |
| P(improve) | 0.747 | 0.998 |
| trajectories improved | 65% (34/52) | 69% |
| decoded E error vs true / std | 0.0611 -> 0.0560 | 0.0828 -> 0.0634 |
| pixel MSE change at max alpha | -1.71% | -10.95% |
| median abs `C` drift | 0.0462 -> 0.0130 | 0.0716 -> 0.0134 |

At H = 50 — the registered primary horizon, chosen to match the published intervention — the result
is **ambiguous**: monotone in the right direction, but the CI includes zero and the change is below
the per-trajectory floor. That is the prereg's registered "ambiguous outcome" case, and its written
response is to re-run at longer horizon. Followed, without reinterpreting anything.

At H = 100 the effect resolves: secular drift in **true decoded physical energy** falls 48%, the
paired CI excludes zero, and the pixel effect is 6x larger than at H = 50. This is E6's prediction
(correction benefit grows with horizon) arriving inside E1.

Note the published comparison is at H = 50 and reports -2.9% pixel; this checkpoint gives -1.71%.
Different models, and the roadmap already records that Session-1 checkpoints are not the paper's.

### Two flaws in my own preregistration, recorded rather than quietly fixed

**1. Signed vs magnitude, an ambiguity.** `E1_PREREG` says the gate needs "arm A's `D_sec` slope
versus alpha [to be] negative". `D_sec` has no preferred sign per trajectory, so the physically
meaningful statistic is `|D_sec|` — drift toward zero. Both are reported, and both agree here
(`mean|D_sec|` falls monotonically 3.197e-03 -> 2.859e-03 at H = 50), so nothing turns on the
choice. The wording should be fixed to say magnitude before the real run.

**2. Wrong granularity for the floor — NEEDS RICHARD'S RULING, NOT AMENDED.** The floor was
registered as a *per-trajectory* median absolute error (1.0e-03). The primary metric is a
*population median under a paired comparison*, in which systematic readout bias largely cancels and
resolution is better by roughly sqrt(n). Read literally, the registered rule rejects the H = 100
result (6.15e-04 < 1.0e-03) even though its paired bootstrap CI excludes zero.

I believe the paired CI is the correct resolution test. **I have not amended the prereg**, because
relaxing a resolution rule after seeing the result it would have blocked is the specific failure
preregistration exists to prevent. Flagged for Richard. Until ruled on, the H = 100 preview is
recorded as **promising but not gate-passing**.

### Status

No claim C1–C6 is yet supported. Arm B is the arm that can kill this: if projection helps regardless
of which `C` is enforced, the edit is a regulariser and says nothing about physics. Nothing decides
until arm B runs on all three conservative seeds.

Next: seeds 4 and 5, then damped 0–2 (~7 h), then E1 arms A/B/C at both H = 50 and H = 100.

---

## 2026-08-26 21:40–21:50 UTC — E1 x E8 cross on seed 3 (still arm A only)

Git at `0905409`. Training: seed 3 at step 24,000 of 60,000; `val_recon` 0.18 and still falling, so
**not yet saturated**. Each E1 run costs ~25 s, which makes arm B affordable.

Arm A at H = 100, across the E8 milestone grid. `runs/e1_milestone_s3_H100.json` (raw rows).

| step | median abs `D_sec` a=0 -> a=0.4 | paired change, 95% CI | P(improve) | pixel % | decoded E err / std |
|---|---|---|---|---|---|
| 1,000 | 3.345e-03 -> 2.392e-03 | -9.53e-04 [-2.17e-03, **+5.91e-04**] | 0.886 | -8.04 | 0.3146 -> 0.2444 |
| 3,000 | 1.271e-03 -> 1.180e-03 | -9.08e-05 [-6.70e-04, **+2.48e-04**] | 0.706 | -5.52 | 0.1055 -> 0.0857 |
| 6,500 | 1.279e-03 -> 6.640e-04 | -6.15e-04 [-1.02e-03, -1.90e-04] | 0.998 | -10.95 | 0.0828 -> 0.0634 |
| 15,000 | 1.989e-03 -> 1.549e-03 | -4.40e-04 [-9.22e-04, -1.91e-04] | 0.996 | -7.89 | 0.1108 -> 0.0929 |

### What this does and does not say

**Bearing on E8's critical question.** The effect is still present at step 15,000 — 2.3x the
published training budget — with a paired CI excluding zero. That is early evidence *against* the
"you found coarse physics in a half-trained model" reading, which is the strongest objection to the
current paper. It is one seed and not yet saturated, so it is a hint, not the E8 result.

**Decoded energy improves at every milestone**, including the two where `D_sec` does not resolve:
0.315->0.244, 0.106->0.086, 0.083->0.063, 0.111->0.093. Consistently 16-22%. Recorded as a secondary
metric per the prereg; not promoted, because promoting the metric that happens to look best is
exactly the failure the prereg exists to prevent.

**An anomaly, recorded because it is inconvenient.** Baseline drift is *worse* at step 15,000 than at
step 6,500 (1.989e-03 vs 1.279e-03), and baseline decoded-E error likewise (0.111 vs 0.083). More
training produced worse autonomous energy conservation at this milestone. With n = 1 seed this may
be noise; if it survives E7's seeds it is interesting in its own right and complicates any simple
"training fixes it" story. Flagged, not explained.

**step 3,000 is the weakest point** (P = 0.706, CI includes zero). No non-monotone story is offered.

### Status

Still arm A only. **Nothing here supports any claim C1-C6**, because a regulariser would reproduce
all of it. Arm B — 20 norm-matched random constraints on steps 6,500 and 15,000, H = 100 — launched
and running.

---

## 2026-08-26 21:50–22:00 UTC — **E1 arm B: the killer arm did not kill it** (seed 3 only)

`runs/e1_armB_s3_H100.json` — 40 rows, 20 norm-matched random degree-4 constraints on each of two
checkpoints, H = 100. Same 20 draws across checkpoints, so the comparison is paired.

| step 6,500 | recovered `C` | null median | draws beating recovered |
|---|---|---|---|
| `D_sec` paired change | -6.148e-04 | +3.858e-03 | **0 / 20** |
| decoded E error change | -0.0194 | +0.2210 | **0 / 20** |
| pixel MSE change | -10.95% | +88.42% | **0 / 20** |

| step 15,000 | recovered `C` | null median | draws beating recovered |
|---|---|---|---|
| `D_sec` paired change | -4.400e-04 | +2.147e-03 | **0 / 20** |
| decoded E error change | -0.0179 | +0.1387 | **0 / 20** |
| pixel MSE change | -7.89% | +62.96% | **0 / 20** |

**Zero of twenty random constraints improve any metric, at either checkpoint.** Every draw makes
secular drift worse, decoded energy error worse, and pixel error worse.

This is the arm the prereg names as the one that can kill the result — "if projection improves
rollouts regardless of which `C` is enforced, the edit is merely regularising the rollout and says
nothing about physics". It did not. It also directly answers Gruver et al. (ICLR 2022), whose
finding — that HNN generalisation comes from modelling acceptance directly rather than from
conservation structure — licenses exactly the "it's a generic regulariser" objection.

Stronger than the published result, which found specificity on only 2 of 3 seeds. That difference is
not yet meaningful: this is **one seed**, and the published failure was on a different one.

### Gate status — still not called

`E1_PREREG` requires both a CI excluding zero *and* an extreme percentile in the arm-B null.
At H = 100, step 6,500, both hold. But:

1. The **registered primary horizon is H = 50**, where arm A's CI includes zero. Arm B at H = 50 is
   now running, so the registered horizon can be reported completely rather than skipped.
2. **One model seed.** The roadmap makes model seed the independent unit for model-level claims.
3. The **floor-granularity question is still open and still Richard's**, not mine to resolve.

So: arm B is a genuinely strong result and the most encouraging thing so far, and it settles nothing
on its own. Seeds 4 and 5 decide it.
