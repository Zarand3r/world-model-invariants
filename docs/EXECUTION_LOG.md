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

---

## 2026-08-26 22:00–22:20 UTC — E1 arm B at the registered horizon; **E2 returns Outcome B**

Git at `8781b0c`. Training: seed 3 at step 36,000 of 60,000.

### E1 arm B at H = 50 — the registered primary horizon

| H = 50, step 6,500 | recovered `C` | null median | draws beating recovered |
|---|---|---|---|
| `D_sec` paired change | -2.07e-04 (CI includes 0) | +4.96e-03 | **0 / 20** |
| decoded E error change | -0.0051 | +0.1441 | **0 / 20** |
| pixel MSE change | -1.71% | **+152.07%** | **0 / 20** |

A distinction worth keeping sharp: at the registered horizon **specificity passes decisively**
(0/20, no random draw improves anything) while **magnitude does not resolve** (CI includes zero).
These are different claims and the paper should not blur them. On seed 3 the gate passes at H = 100
— the escalation the prereg itself registered — and is specific-but-unresolved at H = 50.

### E2 — and it goes against the roadmap's target mechanism

`runs/e2_s3.json` (registered depth 50), `runs/e2_s3_depth100.json` (secondary depth 100, declared).
Registered thresholds were fixed in `docs/E2_PREREG.md` before any value existed.

| step | rho_obs | rho(0) | rho(end) | ratio | slope, 95% CI | sinusoid var frac | **outcome** |
|---|---|---|---|---|---|---|---|
| 6,500 | 6.27e-03 | 1.011e-02 | 1.037e-02 | **1.65** | +1.20e-05 [-8.40e-06, +3.21e-05] **includes 0** | 0.104 | **B** |
| 30,000 | 1.48e-02 | 3.239e-02 | 1.757e-02 | **1.18** | -3.81e-05 [-7.26e-05, +4.08e-05] **includes 0** | 0.107 | **B** |

Thresholds: A needs ratio >= 3 *and* a slope CI excluding 0; C needs sinusoid variance >= 0.5.
Neither is met, at either checkpoint, at either depth. **Outcome B: systematic bias.**

The local conservation defect **does not grow with rollout depth**. It sits at a roughly constant
level from the first imagined step onward. The whitened nearest-neighbour distance does rise
(0.00 -> 0.15), so the state *does* drift off the observation-conditioned support — but that drift is
not accompanied by any deterioration in how well the transition conserves `C`.

**This falsifies, on this seed, the central mechanism the roadmap set out to establish**:

> recursive imagination drives the model into states where that structure is no longer respected

It is not what happens. The transition violates `C` by about the same amount everywhere tested, on
and off the manifold, and long-horizon error accumulates by *integrating a near-constant per-step
bias*. That is the roadmap's Outcome B, and its decision tree pre-authorises the response
("accumulated integrator bias rather than loss of physical structure under distribution shift.
Refocus accordingly").

### What E2 does support, strongly

The controls separate cleanly on `rho_obs` — how well the transition conserves the scalar at real
encoded states:

| arm | rho_obs | vs trained |
|---|---|---|
| trained conservative, recovered `C` | **6.27e-03** | — |
| untrained model, its own recovered `C` | 4.63e-01 | 74x worse |
| trained model, 5 random norm-matched `C` | 5.65e-01 – 1.65e+00 | 90–260x worse |

Two orders of magnitude. The trained model's recovered scalar is conserved by its own transition to
a degree neither a random constraint nor an untrained network comes close to. This is direct
evidence for C1 and for specificity, measured on the *operator* rather than on decodability — the
distinction the paper's probe-vs-dynamics thesis rests on.

### The anomaly is now confirmed by a second, independent measurement

`rho_obs` is **worse at step 30,000 than at step 6,500** (1.48e-02 vs 6.27e-03), matching the E1
finding that baseline drift and baseline decoded-E error are both worse at the later checkpoint.
Two independent statistics now say conservation *degrades* with further training on this seed. If
E7's seeds confirm it, that is a finding in its own right and it cuts against any simple "train
longer and it goes away" reading of E8.

### Status and what needs Richard

Seed 3 only, so none of this is yet a model-level claim. But the E2 outcome changes what the paper
argues, not merely how strongly, so it is surfaced now rather than at the end of Stage 1.
Recommended reframing is written up for Richard; **not acted on** beyond continuing Stage 1 as
planned.

---

## 2026-08-26 23:00–00:30 UTC — Audit of the significant results, and a confound that had to be removed

Prompted by Richard: check the headline results for bugs before building on them. Four audits and one
performance defect. Seed 3 finished training and passed all acceptance checks (raw KL 1.15,
1-step decode ratio 0.002, rollout finite, pixel std 0.070); seed 4 is at step 50,000.

### Audit 1 — **arm B was not a matched control.** Confirmed, and it threatened the headline

The E1 edit is a Newton step to the level set, so its size scales with how badly the constraint is
violated. Measured on seed 3 / step 6,500: median `||dz_edit||` per step is **2.80e-01 for random
draws against 9.57e-03 for the recovered `C` — a factor of 29.**

So "0/20 random constraints improve anything" was confounded: it could equally mean random
constraints perturb the latent 29x harder. Worse, the norm-matching that was supposed to control
this cannot: the Newton step is invariant under `C -> lambda C`, so **coefficient norm has no effect
on edit magnitude at all**. The published arXiv paper carries the same defect.

**Resolution — `scripts/run_e1_direction_matched_null.py`.** Move a fixed distance along each
constraint's normal, `z <- z - eps * sign(C - C0) * gradC/||gradC||`, so arms differ only in
direction. Added equal-norm **tangent** controls (a random direction with its normal component
projected out, which cannot change `C` to first order).

| eps | recovered | random median | random best | tangent median | random beats recovered |
|---|---|---|---|---|---|
| 0.0025 | 1.062e-03 | 1.218e-03 | 1.090e-03 | 1.278e-03 | 0/20 |
| 0.005 | 9.727e-04 | 1.351e-03 | 1.160e-03 | 1.259e-03 | 0/20 |
| 0.01 | 1.072e-03 | 1.467e-03 | 1.239e-03 | 1.394e-03 | 0/20 |
| 0.02 | **6.282e-04** | 1.495e-03 | 1.097e-03 | 1.379e-03 | 0/20 |

Baseline (eps = 0) is 1.279e-03 for every arm. At the matched magnitude the recovered direction
improves drift by up to **-51%** while random directions **worsen it by +17%** and equal-norm tangent
steps by **+8%**. Pixel MSE moves the same way: recovered 0.00250 -> 0.00235, random -> 0.00262,
tangent -> 0.00256.

**The specificity claim survives the audit and is now stronger than before.** It is a claim about
*direction*, with magnitude held fixed, and the tangent arm shows it is specifically the component
normal to the level set that carries the benefit. That also anticipates E3's registered prediction.

### Audit 2 — readout bias with alpha. Clean

No blank decodes at any alpha. Decoded theta error *improves* with alpha (0.154 -> 0.110), so the
readout is not degrading and cannot be manufacturing an apparent drift reduction.

### Audit 3 — is E2's `rho` inflated by a shrunken denominator? No, it is conservative

`rho = median|r| / std_traj(C)` is a ratio, so a small denominator could fake a large defect. Split:

| arm | median abs r | std_traj(C) | rho_obs |
|---|---|---|---|
| trained, recovered C | 6.34e-03 | 1.010 | 6.27e-03 |
| untrained | 4.42e-01 | 0.955 | 4.63e-01 |
| random C (median of 5) | 4.37e+01 | 4.13e+01 | ~7.9e-01 |

Numerator ratio random/trained **6904x**; denominator ratio only 41x. The normalisation *understates*
the raw difference. E2's specificity result is sound — and note it involves no intervention at all,
so it is untouched by the Audit-1 confound.

### Audit 4 — a real bug in my own code, caught before it produced a number

`_secular` used `np.arange(E.shape[-1], float)`, where `float` is read as `stop`, not `dtype`. It
crashed rather than returning a wrong answer, so no result was affected. Fixed.

### Performance defect — 95% of null-sweep runtime was a redundant eigendecomposition

The 1819x1819 generalized eigenproblem costs **21.2 s** and was recomputed for every random draw,
against **0.2 s** for the rollout it feeds. The fit depends only on `(Z, F)` and the fit parameters;
the random coefficients are drawn afterwards.

`latent_noether/fit_cache.py` memoises it, **keyed on a hash of the `Z` and `F` arrays**, not on file
paths — a path-keyed cache would silently serve a stale invariant after a dataset regeneration or a
retrain, which is the failure M29's provenance recording exists to prevent. Verified: exact match to
the uncached fit, 30,861x on a hit, and a 1e-9 perturbation of either `Z` or `F` misses. Tests in
`tests/test_fit_cache.py`. The direction-matched sweep went from ~8 min/draw to under 3 s/draw.

This matters for E7: 20 seeds x 20 draws would have cost ~2.3 GPU-days in redundant eigenproblems.

### Status

Still seed 3 for every analysis. Seed 4 trains now, seed 5 and damped 0-2 after it (~5 h).

---

## 2026-08-27 00:10–00:30 UTC — **Seed 4 replicates the two central findings; the anomaly does not**

Git at `77d64ec`. Seed 4 at step 56,000; its milestone grid is complete through step 30,000.
First model-level comparison, n = 2.

### Direction-matched null replicates (step 6,500, H = 100)

| seed | recovered | random median | tangent median | random beats recovered |
|---|---|---|---|---|
| 3 | **-50.9%** | +16.9% | +7.8% | 0/20 at eps 0.01 and 0.02 |
| 4 | **-42.2%** | -8.0% | -8.7% | 0/20 at eps 0.01 and 0.02 |

At matched step magnitude the recovered direction cuts secular drift in true decoded physical energy
by 42–51% on both seeds, and **no random direction beats it on either**. The null itself behaves
differently across seeds — random directions hurt on seed 3 (+16.9%) and help slightly on seed 4
(-8.0%) — which is exactly why the comparison is run per model rather than pooled. The recovered
direction is 5x better than the best random draw on seed 4 and beats it on seed 3 too.

Tangent controls (equal norm, no first-order change in `C`) track the random arm on both seeds, not
the recovered one. The benefit is in the normal component.

### E2 Outcome B replicates — now 4/4 across two seeds

| | rho_obs | ratio | sinusoid frac | outcome |
|---|---|---|---|---|
| s3 step 6,500 | 6.27e-03 | 1.65 | 0.104 | **B** |
| s3 step 30,000 | 1.49e-02 | 1.18 | 0.107 | **B** |
| s4 step 6,500 | 8.65e-03 | 1.47 | 0.101 | **B** |
| s4 step 30,000 | 6.61e-03 | 1.64 | 0.098 | **B** |

Registered thresholds are ratio >= 3 for A and sinusoid >= 0.5 for C. Neither is approached anywhere.
The amended C3 — a depth-independent per-step violation rather than progressive loss of support —
holds on both seeds.

### Operator-level specificity replicates

Seed 4: trained recovered `C` **8.65e-03**, untrained 4.96e-01 (57x), random `C` 9.12e-01 (105x).
Seed 3 gave 6.27e-03 / 4.63e-01 / ~7.9e-01. Same two-orders-of-magnitude separation on both.

### **Correction: the "conservation degrades with training" anomaly does not replicate**

Previously reported as "confirmed by two independent statistics". Both of those statistics were on
**seed 3**. Seed 4 goes the other way:

| | step 6,500 | step 30,000 | direction |
|---|---|---|---|
| seed 3 rho_obs | 6.27e-03 | 1.49e-02 | worse with training |
| seed 4 rho_obs | 8.65e-03 | 6.61e-03 | **better with training** |

So it is seed-specific, and most likely noise. It should not be described as a finding, and the
earlier framing — that it "cuts against any simple train-longer reading of E8" — is withdrawn. E8
will settle it properly; with n = 2 there is nothing to explain yet.

### Reproducibility: the pipeline is not bit-reproducible, and it does not matter

At `eps = 0` no edit is applied, so all 21 runs should be identical. They are not: max relative
deviation **9.0e-06**, from GPU floating-point nondeterminism (`transition` / `readout_from_h`).
In the reported statistic this is a spread of **3.2e-09** in median abs `D_sec`, against a measured
effect of **4.8e-04** — an SNR of about **1.5e5**. Recorded so that "0/20" is not mistaken for a
bit-exact claim, and so a future re-run landing on slightly different digits is not read as a bug.

### Status

n = 2 conservative seeds. Seed 5 and damped 0-2 still to train (~4 h). Arm C (damped) has not run at
all yet — it is the refusal control and remains outstanding.

---

## 2026-08-27 00:40–00:55 UTC — **Stage 1 conservative arm complete at n = 3**

Git at `cb0529d`. Seed 4 trained and passed acceptance (raw KL 1.33, decode ratio 0.003, rollout
finite). Seed 5 is at step 15,000; its step-6,500 checkpoint exists, so all three conservative seeds
can be compared at the paper-comparable milestone.

### Direction-matched null, all three seeds (step 6,500, H = 100, eps = 0.02)

| seed | baseline abs `D_sec` | recovered | recovered % | random % | tangent % | random beats recovered |
|---|---|---|---|---|---|---|
| 3 | 1.279e-03 | 6.282e-04 | **-50.9** | +16.9 | +7.8 | 0/20 |
| 4 | 1.130e-03 | 6.531e-04 | **-42.2** | -8.0 | -8.7 | 0/20 |
| 5 | 1.142e-03 | 7.740e-04 | **-32.2** | +10.2 | +2.4 | 0/20 |

- recovered: median **-42.2%**, range [-50.9, -32.2]
- random: median +10.2%, range [-8.0, +16.9]
- tangent: median +2.4%, range [-8.7, +7.8]
- **0 / 60 random directions beat the recovered one**, and the recovered and random ranges **do not
  overlap across seeds**

With step magnitude held fixed, correcting along the recovered invariant's normal reduces secular
drift in **true decoded physical energy** by 32–51% on every seed, while random directions of
identical size and equal-norm tangent steps do not help at all. Model seed is the independent unit
here, as the roadmap requires, and with n = 3 the honest summary is the range, not a CI.

For comparison, the published paper reports pixel-MSE specificity on **2 of 3** seeds using a
coefficient-norm-matched null that, as Audit 1 showed, does not control edit magnitude at all. This
is 3 of 3 on a physical metric with a control that does.

At the smaller step (eps = 0.01) seed 5 shows 3/20 random directions beating the recovered one, so
the separation is dose-dependent. Recorded rather than smoothed over: the clean claim is at
eps = 0.02, and the eps = 0.01 column is in the raw rows.

### E2 Outcome B, now 5/5 checkpoints across three seeds

| | rho_obs | ratio | sinusoid frac | outcome |
|---|---|---|---|---|
| s3 step 6,500 | 6.27e-03 | 1.65 | 0.104 | B |
| s3 step 30,000 | 1.49e-02 | 1.18 | 0.107 | B |
| s4 step 6,500 | 8.65e-03 | 1.47 | 0.101 | B |
| s4 step 30,000 | 6.61e-03 | 1.64 | 0.098 | B |
| s5 step 6,500 | 6.85e-03 | 0.92 | 0.103 | B |

Registered thresholds (ratio >= 3 for A, sinusoid >= 0.5 for C) are not approached at any
checkpoint on any seed. Seed 5's ratio of 0.92 means the defect is *smaller* at depth 100 than at
observation-conditioned states. The amended C3 stands.

### Operator-level specificity, all three seeds

| seed | trained recovered `C` | untrained | random `C` |
|---|---|---|---|
| 3 | 6.27e-03 | 4.63e-01 (74x) | ~7.9e-01 |
| 4 | 8.65e-03 | 4.96e-01 (57x) | 9.12e-01 (105x) |
| 5 | 6.85e-03 | 3.87e-01 (57x) | 1.08e+00 (158x) |

Two orders of magnitude on every seed, with no intervention involved — this measurement is immune to
the edit-magnitude confound that Audit 1 found in the intervention arms.

### Claim status

- **C1 emergence** — supported at n = 3. Every seed recovers a scalar its own transition preserves
  ~100x better than any matched random constraint or untrained network.
- **C2 physical validity** — supported at n = 3. Correcting it improves *true decoded physical
  energy*, not merely pixels, on every seed, with a magnitude-matched null.
- **C3 failure mechanism (amended)** — supported at 5/5 checkpoints. Depth-independent per-step
  violation, not loss of support.
- **C4 causal control** — untested. This is E4 and has not been run.
- **C5, C6** — untested.

### Outstanding

**Arm C, the damped refusal control, still has not run** — those models train last (~3 h). Until it
does, the specificity story is one-sided: we have shown the recovered direction beats random
directions *within* conservative models, but not that the pipeline declines to produce a useful
direction when the underlying dynamics have no conserved quantity.

---

## 2026-08-27 01:10–01:35 UTC — **E3 falsifies its own registered prediction, and the fallback explanation fails too**

Git at `cd687bb`. Seed 5 at step 36,000; damped models not yet started.

### E3 registered primary: FALSIFIED

`docs/E3_PREREG.md` predicted `f_perp > 1/12` with a CI excluding it, and named the falsifier
explicitly: "`f_perp <= 1/12`. Then one-step error is not preferentially normal, the geometric story
is wrong, and the projection's benefit needs a different explanation."

| arm | median `f_perp` | 95% CI | x null | mean |
|---|---|---|---|---|
| conservative s3 | 0.0203 | [0.0180, 0.0224] | 0.24 | 0.0536 |
| conservative s4 | 0.0218 | [0.0162, 0.0247] | 0.26 | 0.0544 |
| conservative s5 | 0.0237 | [0.0226, 0.0261] | 0.28 | 0.0595 |
| untrained | 0.0273 | [0.0249, 0.0306] | 0.33 | 0.0603 |
| random `C` (n = 60) | 0.0499 | [0.0352, 0.0692] | 0.60 | — |

One-step error is preferentially **tangent**, not normal — three to four times *less* aligned with
`grad C` than chance, on every seed, CI excluding the null. **The registered falsifier fired.**

### The registered null was also mis-specified, and I should say so plainly

`1/LD` assumes `dz` is isotropic in the extracted subspace. It is not: **every arm, including random
constraints, sits below 1/12.** A null that no arm can reach is not a null. The right comparison is
the empirical random-`C` arm, which the prereg did name as "the control that matters".

Against *that* null the direction of the effect is unchanged and clean: conservative seeds
[0.0203, 0.0237] against random [0.0352, 0.0692], with **60/60 random draws above the weakest
conservative seed**. The recovered constraint's gradient avoids the model's one-step error even more
strongly than a random polynomial's does.

**But this comparison is not independent of E2.** If `C` is well conserved then `C(T(z)) ~ C(z)`,
hence `grad C . dz ~ 0`, hence low `f_perp`. E3's conservative-vs-random contrast is close to a
restatement of E2's `rho_obs` contrast in different units, and must not be presented as
corroborating evidence from an independent measurement.

### The obvious fallback explanation also fails

Post-hoc, and labelled as such: if the normal component is small but **systematic** it would still
integrate into energy drift, while a larger **oscillatory** tangent component would not. Tested by
the ratio of |mean| to std of the signed normal projection along each trajectory:

| seed | normal \|mean\|/std | tangent \|mean\|/std |
|---|---|---|
| 3 | 0.096 | 0.199 |
| 4 | 0.076 | 0.232 |
| 5 | 0.065 | 0.200 |

The normal component is **less** systematic than the tangent one, on all three seeds. The fallback is
wrong too.

### Where this leaves the mechanism

The repair works and is highly specific — 3/3 seeds, 32-51% reduction in true decoded physical
energy drift, 0/60 magnitude-matched random directions beating it. **We now have no working
explanation of why.** The preregistered geometric account is falsified and its natural replacement
is falsified.

That is the honest state, and it is recorded as such rather than narrated into a story. The result
stands on its own evidence; the mechanism is open.

### A reporting bug in my own analysis, caught and fixed

The first E3 summary reported 0.052-0.059 and called it the median. It was the median of
*per-trajectory means*. `f_perp` is strongly right-skewed (median 0.020 against mean 0.054), so the
two are different statistics and the registered one is the median. Two independent implementations
of the underlying quantity were checked against each other and agree to 2.4e-07, and float32 vs
float64 gradients agree to a ratio of 1.000 — the maths was right, the summary was not. The script
now stores per-trajectory medians and means separately, under distinct keys. The conclusion is
unchanged and in fact stronger: 0.24x the null rather than 0.63x.

---

## 2026-08-27 01:40–02:05 UTC — **E9: the effect survives truly disjoint evaluation**

Git at `2432f15`. Seed 5 at step 50,000; damped models still not started.

Preregistered in `docs/E9_PREREG.md` before any E9 quantity existed. Only the trajectories being
scored change — metric, `eps` grid, arms and specificity count are all unchanged from E1.

### Design

`C`, `h_mean`, the PCA basis `U` and the rank basis `R` are all fitted on the analysis split
`204:` of `runs/pendulum_pixels.npz` and **frozen**. The whole coordinate frame is part of `C` as a
function of `h`, so re-deriving any of it on the eval set would leak.

Scoring moves to `runs/pendulum_pixels_eval.npz` — **512 trajectories x 200 frames, generator seed
777**, never used for training, fitting, calibration, or any analysis to this point.

### Result, seed 3 / step 6,500 / H = 100

| arm | out-of-sample (n = 512) | in-sample (n = 52) |
|---|---|---|
| baseline abs `D_sec` | 1.363e-03 | 1.279e-03 |
| **recovered** | 7.053e-04 (**-48.2%**) | 6.282e-04 (-50.9%) |
| random median | 1.496e-03 (+9.8%) | (+16.9%) |
| random best | 1.291e-03 | — |
| tangent median | 1.400e-03 (+2.8%) | (+7.8%) |
| random beating recovered | **0/20** | 0/20 |

**The effect does not shrink out-of-sample**: -48.2% against -50.9% in-sample, on ten times as many
trajectories, none of which the invariant or the coordinate frame ever saw. The registered falsifier
— "the effect vanishes or reverses out-of-sample, and every absolute number reported so far must be
restated as in-sample only" — did not fire.

This closes the weakness the published paper discloses and leaves open ("the absolute effect is
in-sample with respect to invariant fitting"). It is also worth noting against an unverified claim
in the review that prompted this roadmap, which asserted a disjoint evaluation gave -3.0% / +2.6% /
-4.1%. No such run exists anywhere in the repository, and this measurement — different models,
different metric, ten times the trajectories — does not reproduce that pattern. The claim should not
be relied on.

### Caveats, stated plainly

- **One seed and one checkpoint.** Seeds 4 and 5 have not been run out-of-sample yet.
- The eval set is drawn from the same generator and initial-condition distribution as the training
  data. E9 tests disjointness of *trajectories*, not distribution shift. Out-of-distribution energies
  are E14 and remain untested.
- H = 200 is running; it is E6's first test point beyond H = 100 and is not yet available.

### Status

Damped models still untrained, so arm C remains the one unrun arm of Stage 1.

---

## 2026-08-27 02:10–02:30 UTC — **Arm C, the damped refusal control: decisive on recovery, partial on intervention**

Git at `8fa62df`. Seed 5 trained; damped seed 0 at step 6,500. Last unrun arm of Stage 1.

### A metric correction that had to be made first

The conservative metric cannot be reused unmodified. On a damped pendulum energy **genuinely
decays**, so a nonzero `D_sec` is *correct physics* there:

| dataset | true median `D_sec` |
|---|---|
| conservative | -2.11e-04 |
| damped, zeta = 0.03 | **-3.93e-02** (186x larger) |

Driving abs `D_sec` toward zero on the damped arm would mean the model **fails to dissipate**. The
correct arm-C statistic is therefore deviation from correct physics,
`|D_sec_decoded - D_sec_true|`, and both are reported below. Applying the conservative metric here
unmodified would have produced a meaningless number.

### Recovery side — decisive refusal

| model | `rho_obs` |
|---|---|
| conservative s3 / s4 / s5 | 6.27e-03 / 8.65e-03 / 6.85e-03 |
| **damped s0** | **3.74e-01** |
| untrained | 3.87e-01 – 4.96e-01 |
| random `C` | 7.9e-01 – 1.08e+00 |

The damped model's best recovered scalar is conserved by its own transition **55x worse** than the
conservative models', and sits in the same range as an untrained network's. The extraction does not
manufacture a conserved quantity when the underlying dynamics have none. E2's depth ratio on the
damped model is 1.02 — flat, as everywhere else.

This is the arm that distinguishes "we found the physics" from "our pipeline always finds something",
and it comes down on the right side.

### Intervention side — partial refusal, weaker than registered

`E1_PREREG` registered arm C as "no improvement, or harm". Measured at eps = 0.02, H = 100:

| statistic | eps 0 -> 0.02 | conservative arm for comparison |
|---|---|---|
| pixel MSE | **+0.2%** | -8 to -11% |
| deviation from true physics | **-11.0%** | -32 to -51% |
| abs `D_sec` | 3.990e-02 -> 3.975e-02 | — |
| random draws, deviation | +2.4% | +9.8 to +16.9% |

Pixel MSE meets the registration (no improvement). Deviation from true physics does **not**: it
improves by 11%, where the registration expected none. The improvement is roughly a quarter of the
conservative arm's, and it still beats the damped model's own random draws (-11.0% vs +2.4%).

**Reported as a partially met expectation, not as a pass.** The honest reading is that the damped
model's recovered scalar carries some weak usable structure — unsurprising, since a damped pendulum
still has a well-defined decaying energy that a latent could track — but nothing resembling a
conserved quantity, and the intervention benefit is 4x smaller than on conservative models.

### Caveats

- **One damped model.** Seeds 1 and 2 are still training; `E1_PREREG` and `DISSIPATIVE_PREREG` both
  register the criterion over 3 seeds, so no damped claim is settled yet.
- 10 random draws and 3 tangent controls on this arm, against 20 and 5 elsewhere, to keep it inside
  one iteration. Raw rows are in `runs/e1_armC_damped_s0.json`.

### Stage 1 arm status

| arm | status |
|---|---|
| A conservative, recovered `C` | complete, n = 3, -32 to -51% |
| B magnitude-matched random | complete, 0/60 |
| B' equal-norm tangent | complete, +2.4 to +7.8% |
| **C damped, its own `C`** | **n = 1, partial** |
| D untrained | complete, `rho_obs` 57-74x worse |
| E9 disjoint evaluation | complete, seed 3, -48.2% on 512 unseen trajectories |

---

## 2026-08-27 02:40–03:05 UTC — **E6 confirmed: benefit grows with horizon, reaching -75.9% at H = 190**

Git at `1bd4f51`. Damped seed 0 at step 30,000; seeds 1 and 2 not started.

### A bug that crashed loudly rather than lying

E9 at H = 200 failed: a 200-frame eval set supplies only `T - WARMUP = 190` reference frames, and
`mse_loss` hit a 200-vs-190 shape mismatch. It crashed rather than silently truncating, which is the
right failure. The script now clamps the horizon to what the data supports, prints that it did, and
records `horizon_used` in every row so a requested horizon can never differ silently from the
reported one.

### E6: correction benefit versus horizon (seed 3, step 6,500, recovered arm, eps = 0.02)

| H | evaluation set | benefit |
|---|---|---|
| 50 | in-sample, n = 52 | CI included zero (ambiguous, per E1) |
| 100 | in-sample, n = 52 | -48.1% |
| 100 | disjoint, n = 512 | -48.2% |
| **190** | **disjoint, n = 512** | **-75.9%** |

Monotone in horizon, and the two H = 100 numbers agree to 0.1 percentage points across a tenfold
change in trajectory count and a complete change of trajectories.

### Dose-response at H = 190, on 512 never-seen trajectories

| eps | recovered | random median |
|---|---|---|
| 0.005 | -32.6% | +3.0% |
| 0.01 | -61.4% | +4.3% |
| 0.02 | **-75.9%** | +9.1% |

Monotone in step size, random directions worsen drift monotonically, and **0/12 random directions
beat the recovered one** (the random arm is still filling; 12 of 20 draws at time of writing).

### Why this matters to the argument

E6 was registered in the roadmap as the experiment that would turn a modest effect size into a
mechanism claim. It has. The published paper reports **-2.9% pixel MSE at H = 50**, a number small
enough that reviewers quote it back. The same intervention, measured on **true decoded physical
energy** rather than pixels, on trajectories the invariant never saw, reduces secular drift by
**76%** at H = 190.

The reading this supports: invariant violation is not a one-step defect but an accumulating one, so
the correction buys progressively more the longer the model imagines. That is consistent with E2's
Outcome B — a near-constant per-step violation that integrates — and it is the strongest available
answer to "the effect is too small to matter".

### Caveats

- **Seed 3 only** at long horizon. Seeds 4 and 5 have not been run at H = 190.
- One checkpoint (step 6,500).
- The random arm at H = 190 is 12/20 complete; the number will be restated when it finishes.
- H = 190 is the maximum the 200-frame eval set supports. Testing further would need a new dataset.

### Also this iteration

- `docs/E4_PREREG.md` written, before any E4 quantity exists. Fixes the offset grid, the primary
  metric (transfer correlation between intended change in `C` and realised change in decoded
  physical energy), the falsifier, the controls, and the rule that any `C`-to-`E` calibration is
  fitted only on the training split and frozen. Records that `C` is not assumed to be energy.
- E9 H = 100 completed on all arms: recovered **-48.2%**, random +9.8%, **tangent -0.4%**,
  0/20 and 0/5 beating. With all five tangent controls in, the tangent arm sits at essentially zero,
  which is what an equal-norm step along the level set should do.
- Arm C relaunched at 20 random draws for parity with the other arms.

---

## 2026-08-27 03:10–03:25 UTC — Final numbers, and a **correction to arm C**

Git at `30369ef`. Damped seed 0 at step 30,000; seeds 1 and 2 not started.

### E9 / E6 at H = 190, all arms complete

| arm | change in median abs `D_sec` |
|---|---|
| **recovered** | **-75.9%** |
| random median (n = 20) | +8.8% |
| random best of 20 | -16.3% |
| tangent median (n = 5) | +2.7% |
| random beating recovered | **0/20** |
| tangent beating recovered | **0/5** |

Unchanged from the 12-draw reading. The best of twenty random directions reaches -16.3%, still less
than a quarter of the recovered direction's effect.

### **Correction: arm C, with 20 draws instead of 10**

Last iteration reported arm C as "-11.0% recovered vs +2.4% random" on 10 draws and called it a
partially met expectation. With the full 20 draws the picture is different:

| damped s0, deviation from true physics | 10 draws (reported) | **20 draws (final)** |
|---|---|---|
| recovered | -11.0% | -11.0% |
| random median | +2.4% | **-3.5%** |
| tangent median | — | **-8.5%** |
| random beating recovered | — | **4/20** |
| pixel MSE | +0.2% | +0.2% |

The recovered number is unchanged; the **null** moved. With twenty draws, random directions improve
by 3.5%, equal-norm tangent steps improve by 8.5% — nearly as much as the recovered direction's
11.0% — and **4 of 20 random directions beat the recovered one**.

**This reads better as refusal, not worse.** The right axis is specificity, not absolute
improvement:

| | recovered | tangent | random beating recovered |
|---|---|---|---|
| conservative (n = 3 seeds) | -32 to -51% | ~0 to +7.8% | **0/60** |
| damped (n = 1 seed) | -11.0% | -8.5% | **4/20** |

On conservative models the recovered direction is sharply distinguished from every control. On the
damped model it is not: tangent steps do nearly as well and a fifth of random directions do better.
The small absolute improvement on the damped model is **non-specific**, which is exactly what should
happen when the extraction is run on dynamics with no conserved quantity to find. Combined with the
recovery-side result — `rho_obs` 55x worse than conservative, in the untrained range — arm C is a
pass on the axis that matters.

The earlier "partially met" framing was drawn from an underpowered null and is withdrawn. It is a
reminder that a 10-draw null was enough to make a control look decisive when it was not; the other
arms were run at 20 and should have been matched from the start.

### Launched

Seeds 4 and 5 at H = 190 on the disjoint eval set, to close the "seed 3 only at long horizon" caveat
on the -75.9% headline.

---

## 2026-08-27 03:40–03:50 UTC — **Headline replicates on all three seeds: -54.6 to -75.9% at H = 190, out of sample**

Git at `6dbfe54`. Damped seed 0 trained and passed acceptance (rollout finite, pixel std 0.0698);
damped seed 1 training.

`C`, `h_mean`, `U`, `R` frozen from the analysis split of `pendulum_pixels.npz`; scored on the 512
never-seen trajectories of `pendulum_pixels_eval.npz` (generator seed 777).

| seed | n random | baseline abs `D_sec` | recovered | recovered % | random % | random best | beats |
|---|---|---|---|---|---|---|---|
| 3 | 20 | 1.021e-03 | 2.459e-04 | **-75.9** | +8.8 | -16.3 | 0/20 |
| 4 | 20 | 8.116e-04 | 3.688e-04 | **-54.6** | +9.7 | -14.4 | 0/20 |
| 5 | 15 | 8.462e-04 | 3.485e-04 | **-58.8** | +7.4 | -24.7 | 0/15 |

- recovered: median **-58.8%**, range [-75.9, -54.6]
- random: +7.4 to +9.7% on every seed
- **0 / 55 random directions beat the recovered one**
- the best single random draw across all 55 (-24.7%) does not reach the **weakest** recovered
  seed (-54.6%)

This closes the last major caveat on the project's strongest number. The claim is now: on three
independently trained models, with step magnitude held fixed so arms differ only in direction, and
scored on 512 trajectories none of them ever saw, projecting the latent along the recovered
invariant's normal reduces secular drift in **true decoded physical energy** by 55-76%, while
identical-magnitude random and tangent directions do not.

Seed 5's tangent arm has not run yet (0/5); its `beats` count is over 15 random draws rather than 20.
Both will be restated when the run finishes.

### Stage 1 status

| item | status |
|---|---|
| adapter verification | complete |
| E1 readout validation, arms (a) and (b) | complete |
| E1 arms A / B / B' / D | complete, n = 3 |
| E1 arm C (damped) | n = 1; seeds 1-2 training |
| E2 | complete, Outcome B at 6/6 checkpoints |
| E3 | complete, registered prediction **falsified** |
| E6 | complete, monotone in horizon |
| E9 | complete, n = 3, effect does not shrink out of sample |
| E7 seeds | 3 conservative + 1 damped of 3 |

Stage 1 is substantively complete apart from damped seeds 1 and 2. Next in the roadmap's order is
Stage 2: **E4** (preregistered in `docs/E4_PREREG.md`), then E5, then E13.

---

## 2026-08-27 04:10–04:35 UTC — **E4 passes: the invariant is a control variable** (seed 3)

Git at `c56407f`. Damped seed 1 training. H = 190 finalised on all three seeds and all arms:
recovered -75.9 / -54.6 / -58.8%, random +7.5 to +9.7%, tangent -3.6 to +2.7%,
**0/60 random and 0/15 tangent** beating the recovered direction.

`scripts/run_e4_dialing.py` written and run per `docs/E4_PREREG.md`. The edit is applied **once** at
the start of the rollout, which then runs free — so the test includes whether the changed regime
persists without further forcing.

### E4a, donor-level — the registered primary

Target is `C(z_donor)` from an independent trajectory, read off the model's own latent. **The
intervention never sees true energy**; ground truth enters only when the decoded rollout is compared
with the donor's.

| statistic | value |
|---|---|
| trajectories kept | 51/52 (registered exclusion: edit norm <= 5x median) |
| **Spearman(intended dC, realised dE)** | **+0.916**, 95% CI [+0.819, +0.960] |
| verdict | **PASS** — CI excludes 0 |
| Spearman(realised dE, **true** donor dE) | **+0.914** |
| Spearman(intended dC, true donor dE) | +0.995 |
| random controls (n = 8) | median **-0.050**, range [-0.667, +0.335], 0/8 beating |
| tangent controls (n = 3) | median **-0.065**, range [-0.119, +0.058], 0/3 beating |

Setting the latent scalar to another trajectory's value moves the imagined world's **true decoded
physical energy** toward that trajectory's **actual** energy, with rank correlation 0.914. Random
constraints dialed by the same protocol produce no coherent energy change, and equal-norm tangent
edits produce none either.

### E4b, synthetic sweep — the registered offset grid

| offset (units of std_traj `C`) | -1.00 | -0.50 | -0.25 | 0 | +0.25 | +0.50 | +1.00 |
|---|---|---|---|---|---|---|---|
| median realised dE | -0.398 | -0.146 | -0.074 | 0 | +0.044 | +0.175 | +0.344 |

**Spearman(offset, realised dE) = +1.000.** Perfectly monotone across the preregistered grid, in both
directions.

### Directional asymmetry, registered as secondary

Lowering `C` moves energy slightly further than raising it (-0.398 at -1.0 sigma against +0.344 at
+1.0). The prereg registered this comparison in advance because Spies et al. (arXiv:2412.11867)
report activation is easier than suppression in world models; here the asymmetry runs the other way
and is mild. Reported, not interpreted.

### What this does and does not establish

C4 as registered is **supported on seed 3**: the recovered scalar is not merely worth restoring, it
is a control variable, and the imagined physics follows it quantitatively.

The prereg's interpretation rule applies and is not being set aside: moving along the `C`-normal
changes the latent microstate, not only `C`. This is evidence that the recovered **subspace** is
causally deployed for physical energy, **not** proof that the model maintains an internal energy
register.

**One seed.** Seeds 4 and 5 launched. Controls are 8 random and 3 tangent draws against 20 and 5
elsewhere — chosen to fit the iteration, and to be brought to parity before this is reported
anywhere, given that the arm-C correction two iterations ago was caused by exactly that shortcut.

---

## 2026-08-27 04:40–04:55 UTC — **E4 replicates on all three seeds: C4 supported at n = 3**

Git at `ec87029`. Damped seed 1 still training.

### E4a donor-level, registered primary

| seed | kept | Spearman(intended dC, realised dE) | 95% CI | Spearman(realised dE, **true** donor dE) | random median | tangent median | controls beating |
|---|---|---|---|---|---|---|---|
| 3 | 51/52 | **+0.916** | [+0.817, +0.959] | +0.914 | -0.050 | -0.065 | 0/11 |
| 4 | 50/52 | **+0.838** | [+0.699, +0.919] | +0.805 | -0.034 | -0.059 | 0/11 |
| 5 | 49/52 | **+0.808** | [+0.647, +0.904] | +0.802 | -0.030 | -0.270 | 0/11 |

Transfer correlation median **+0.838**, range [+0.808, +0.916]. Every CI excludes zero; **0/33
controls beat the recovered direction on any seed**.

### E4b synthetic sweep, registered offset grid

| seed | -1.00 | -0.50 | -0.25 | 0 | +0.25 | +0.50 | +1.00 | Spearman |
|---|---|---|---|---|---|---|---|---|
| 3 | -0.398 | -0.146 | -0.074 | 0 | +0.044 | +0.175 | +0.344 | **+1.000** |
| 4 | -0.060 | -0.014 | -0.000 | 0 | +0.007 | +0.052 | +0.130 | **+1.000** |
| 5 | -0.211 | -0.116 | -0.033 | 0 | +0.027 | +0.067 | +0.209 | **+1.000** |

Perfectly monotone on every seed, in both directions.

### The prereg's metric choice was load-bearing

Sweep **magnitudes** differ ~3x across seeds (seed 3 spans +-0.4, seed 4 only +-0.13) while
**monotonicity is exact on all three**. `E4_PREREG` registered monotonicity and transfer correlation
as the evidence, explicitly *not* slope-1 agreement, on the grounds that `C` may be a monotone
nonlinear function of energy. Had slope been the registered statistic, seed 4 would have looked like
a failure and seed 3 like a success, and the honest reading is that all three behave identically in
the property that matters.

### Claim status

| claim | status |
|---|---|
| **C1 emergence** | supported, n = 3 |
| **C2 physical validity** | supported, n = 3, out of sample, magnitude-matched |
| **C3 failure mechanism (amended)** | supported, Outcome B at 6/6 checkpoints |
| **C4 causal control** | **supported, n = 3** |
| C5 predictive utility | untested |
| C6 generality | untested |

The prereg's interpretation rule continues to apply: this is evidence the recovered **subspace** is
causally deployed for physical energy, not proof of an internal energy register.

### Follow-through on the control-parity flag

Last iteration ran 8 random and 3 tangent draws against 20 and 5 elsewhere, and flagged that as a
shortcut to be corrected before reporting — the same shortcut that produced the arm-C correction.
Relaunched at 20 and 5 on all three seeds; numbers above will be restated when it completes.

---

## 2026-08-27 05:10–05:20 UTC — E4 controls at parity; arm C launched at n = 3

Git at `7c64dfd`. Damped seed 1 trained; damped seed 2 training and already past step 6,500, so all
three damped models have the checkpoint arm C needs.

### E4a with controls at parity (20 random, 5 tangent per seed)

| seed | recovered rho | random median | **random max** | tangent median | **tangent max** | controls beating |
|---|---|---|---|---|---|---|
| 3 | **+0.916** | -0.078 | +0.582 | -0.065 | +0.058 | 0/25 |
| 4 | **+0.838** | -0.116 | +0.266 | -0.059 | +0.338 | 0/25 |
| 5 | **+0.808** | +0.005 | +0.538 | +0.052 | +0.165 | 0/25 |

**0/75 controls across all three seeds.** The transfer correlations are unchanged from the 8-draw
reading, so unlike arm C the E4 conclusion did not move when the null was properly powered.

**The maxima are worth stating and were not visible at 8 draws.** Random constraints are not at
zero: the best random draw reaches **+0.582** on seed 3 and +0.538 on seed 5. Medians are near zero
(-0.116 to +0.005), but the tail is real. The honest separation is recovered [+0.808, +0.916] against
best-of-twenty random [+0.266, +0.582] — clear, and not the "controls do nothing" picture the
medians alone would suggest. Any writeup should quote the maxima alongside the medians.

That a random degree-4 polynomial of the latent sometimes correlates with energy is unsurprising and
is the same phenomenon as the untrained-model result the paper already reports (rho_E up to 0.908):
on a 1-DoF conservative system, energy is close to the only trajectory-constant scalar, so anything
tracking the latent state at all will partly track it. E4's claim is comparative, not absolute.

### Launched

- Arm C on damped seeds 1 and 2, 20 random / 5 tangent, matching seed 0.
- E2 on damped seeds 1 and 2.

Together these complete the three-seed refusal criterion that `E1_PREREG` and the repo's existing
`docs/DISSIPATIVE_PREREG.md` both require, and which has been the outstanding gap in Stage 1 since
the start.

---

## 2026-08-27 05:40–05:55 UTC — **STAGE 1 COMPLETE: arm C is a decisive refusal at n = 3**

Git at `9f60dea`. Damped seed 2 still training past step 6,500; all three damped models have the
checkpoint arm C requires.

### Arm C, intervention side, n = 3

Deviation from **true decaying physics**, `|D_sec - D_sec_true|`, at eps = 0.02, H = 100:

| damped seed | recovered | random median | tangent median | random beating recovered | pixel MSE |
|---|---|---|---|---|---|
| 0 | -11.0% | -3.5% | -8.5% | 4/20 | +0.2% |
| 1 | **+20.7%** | +27.5% | +2.9% | 8/20 | +0.6% |
| 2 | +3.3% | +0.3% | +8.7% | 12/20 | -1.5% |
| **conservative, n = 3** | **-32 to -51%** | +7.4 to +9.7% | -3.6 to +7.8% | **0/60** | -8 to -11% |

Across the three damped models the recovered direction gives -11.0%, **+20.7%**, +3.3% — no
consistent sign, harm on two of three seeds — and **24 of 60 random directions beat it**, against
**0 of 60** on the conservative models.

This is the registered refusal, and it is unambiguous: when the extraction is run on dynamics with
no conserved quantity, the direction it returns is not distinguished from a random one.

### Arm C, recovery side, n = 3

| arm | `rho_obs` |
|---|---|
| conservative s3 / s4 / s5 | 6.274e-03 / 8.649e-03 / 6.851e-03 |
| damped s0 / s1 / s2 | 3.741e-01 / 3.411e-01 / 3.663e-01 |

**39x separation, no overlap between any conservative and any damped seed.** E2's depth ratio on the
damped models is 0.89-1.02, flat as everywhere else.

This reproduces, on the *operator* statistic, the non-overlap the published paper reports on its
recovery statistics ("no conservative seed overlaps any damped seed on any row").

### Correcting my own earlier characterisations of arm C, twice over

At n = 1 with 10 draws I called arm C "partially met" — the intervention appeared to help by 11%
where the registration expected none. At n = 1 with 20 draws I revised that to a pass on specificity
grounds. At n = 3 the picture is cleaner than either: the seed-0 result was **not representative**.
Seeds 1 and 2 show the recovered direction actively harming, and the fraction of random draws
beating it rises to 8/20 and 12/20. Both earlier readings were drawn from a single damped model, and
the roadmap's own instruction — model seed is the independent unit — was the thing that should have
prevented me from characterising the arm at all until three existed.

### STAGE 1 — final status

| item | result |
|---|---|
| adapter verification | agrees with reference `img_step` to 0.0e+00 |
| E1 readout validation (a) and (b) | theta 0.11 deg, decoded E 1.6% of spread, `D_sec` floor 1.0e-03 |
| E1 arm A, conservative | **-32 to -51%** (H = 100), **-54.6 to -75.9%** (H = 190, out of sample) |
| E1 arm B, magnitude-matched random | **0/60** beat recovered |
| E1 arm B', equal-norm tangent | **0/15** beat recovered |
| E1 arm C, damped | **refusal: 24/60 random beat recovered, harm on 2/3 seeds** |
| E1 arm D, untrained | `rho_obs` 57-74x worse |
| E2 | Outcome **B** at 9/9 checkpoints (6 conservative, 3 damped) |
| E3 | registered prediction **FALSIFIED**; mechanism unexplained |
| E6 | monotone in horizon, -48% at H = 100 -> -76% at H = 190 |
| E9 | effect does not shrink out of sample, n = 3 |
| E4 (Stage 2) | transfer correlation **+0.808 to +0.916**, 0/75 controls |

### Claim status

| claim | status |
|---|---|
| **C1 emergence** | supported, n = 3, with a matched dissipative control at n = 3 |
| **C2 physical validity** | supported, n = 3, out of sample, magnitude-matched |
| **C3 failure mechanism (amended)** | supported, Outcome B at 9/9 |
| **C4 causal control** | supported, n = 3 |
| C5 predictive utility | untested |
| C6 generality | untested |

Stage 1 is complete. Stage 2 has E4 done; E5 and E13 remain. The mechanism question raised by E3's
falsification is still open and is the most interesting unresolved item in the project.

---

## 2026-08-27 06:10–06:35 UTC — **E2b: violation accumulates sub-diffusively, not as a systematic bias**

Git at `6bc01a0`. Damped seed 2 at step 56,000; all other jobs complete.

Preregistered in `docs/E2B_PREREG.md`, under the roadmap's own E2-Outcome-B branch ("refocus
accordingly"). Tests whether `|C(z_k) - C(z_0)|` grows like `k^1` (systematic bias) or `k^0.5`
(random walk), by fitting `beta` in `log median|dC| = a + beta log k`.

### The registered positive control failed — and it was my prereg that was wrong

`E2B_PREREG` named damped models as the positive control, expecting `beta` near 1.0 because damped
energy decays systematically, and registered the falsifier: "If damped models do not show `beta`
above the conservative models', the estimator is not measuring what it claims."

Damped models returned `beta` = **-0.011, -0.031, -0.052**. The falsifier fired.

**Estimator validated independently on signals with known exponents:**

| signal | measured `beta` | truth |
|---|---|---|
| synthetic random walk | 0.473 | 0.50 |
| synthetic systematic drift | 1.000 | 1.00 |
| synthetic saturated noise | 0.006 | 0.00 |
| **true damped energy** | **0.947** | decays systematically |
| true conservative energy | -0.037 | shadow-H oscillation, no drift |

The estimator is correct. **The damped *model* was the wrong control.** Its recovered `C` is not
energy-correlated at all — that is precisely the refusal result — so `|C(z_k) - C(z_0)|` saturates at
k = 1 and `beta` measures time-to-saturation rather than an accumulation law. Random `C` behaves the
same way (`beta` = 0.011). The right positive control is **true damped energy**, which the estimator
handles correctly at 0.947.

**This is the second control I have mis-specified by plausible-sounding reasoning** — E3's isotropic
`1/LD` null was the first, and no arm could reach it. Both were caught by the arm that was supposed
to be easy. The lesson recorded for the rest of the project: validate an estimator on a signal with a
known answer *before* registering a threshold on it, not after the threshold fails.

### Result, with the estimator validated

| seed | `beta` | 95% CI |
|---|---|---|
| conservative s3 | **0.506** | [0.413, 0.618] |
| conservative s4 | **0.358** | [0.253, 0.465] |
| conservative s5 | **0.210** | [0.121, 0.309] |

All three are far below 1.0, and two of three are below 0.5. Violation accumulates **diffusively or
sub-diffusively — decisively not as a coherent systematic bias.** Seed 3 sits exactly on the random
walk (CI contains 0.5, excludes 1.0); seeds 4 and 5 are sub-diffusive, i.e. the violation accumulates
*more slowly* than an unbiased random walk, which implies a partially anti-correlated normal
component — the transition mildly self-corrects.

### What this settles, and what it does not

It supplies the mechanism E3's falsification left open, and it is consistent with everything measured:

- E2: per-step defect constant with depth
- E3: normal component small (2% of error energy) and *less* systematic than the tangent component
- E2b: accumulation diffusive or slower, not linear
- E6: benefit grows with horizon, as accumulated deviation grows

The reading: the transition is **near-unbiased but noisy with respect to `C`**. Each step nudges `C`
by a small, near-zero-mean amount normal to the level set; those nudges accumulate diffusively into
physical energy error, while the much larger tangent component moves the state along the level set
and costs phase accuracy but no energy. Projection removes the accumulated normal displacement and
leaves the tangent error alone.

**The stronger implication, stated carefully.** A systematic bias would be an operator defect that
better training might remove. Diffusive accumulation is closer to irreducible — training reduces
noise but does not eliminate it, and nothing in the objective rewards cancelling it. That predicts
the effect should *persist* at saturation, which is E8's question and is not yet answered. It is a
prediction, not a result.

Sub-diffusive `beta` on two of three seeds is not explained and is not being explained here.

---

## 2026-08-27 06:35–07:00 UTC — **E10 inconclusive as registered; the reason is itself informative**

Git at `f4bce31`. Richard approved the order E10 -> E11 -> E12 -> 2-DoF system.

Preregistered in `docs/E10_PREREG.md` before any E10 quantity existed. Candidates come from the
existing eigenfamily (`polynomial_invariants`), each carrying its invariance ratio; no new fitting.

### The registered primary could not be computed

`E10_PREREG` requires >= 8 candidates whose `|rho_E|` is within +-0.05 of the recovered `C`'s,
widening once to +-0.10, and otherwise reporting **inconclusive for lack of matched candidates,
not re-tuned further**.

With a 40-candidate pool, **exactly 1 candidate** falls inside +-0.10 on every seed. The guard
fires. Reported inconclusive, band not widened again.

### Why the band is empty — not the reason the prereg predicted

The prereg anticipated a narrow band on the grounds that "the best-conserved scalars *are* the
energy-like ones". The measured structure is different and more interesting:

| seed | candidates with `|rho_E|` > 0.8 | > 0.5 | ratio range |
|---|---|---|---|
| 3 | 1/40 | 1/40 | 5.3e-05 – 4.1e-01 |
| 4 | 1/40 | 1/40 | 1.0e-04 – 4.2e-01 |
| 5 | 1/40 | 1/40 | 8.0e-05 – 4.5e-01 |

The family contains **many** well-conserved candidates but only **one** that is linearly
energy-correlated. The likely reason is that any function of `E` is also conserved — `E^2`, `E^3`,
mixtures — and those decorrelate *linearly* with `E` while remaining perfectly conserved. So the pool
is bimodal in `|rho_E|`: one high member and a long low tail, with nothing in between to match
against.

### Registered secondary — reported regardless, and it points the right way

| seed | Spearman(ratio, improvement) | Spearman(\|rho_E\|, improvement) | Spearman(ratio, \|rho_E\|) | best eigen improvement | recovered |
|---|---|---|---|---|---|
| 3 | **-0.223** | +0.283 | -0.052 | +33.3% | +50.9% |
| 4 | **-0.482** | +0.396 | -0.310 | +41.4% | +42.2% |
| 5 | **-0.187** | +0.135 | -0.330 | +35.7% | +32.2% |

Both predictors move as expected — better conservation (lower ratio) and higher decodability each
associate with more benefit — and **the two are only weakly correlated with each other**
(-0.05 to -0.33). That is the opposite of the tight confound the prereg feared, and it means a
properly powered matched-band test is feasible in principle; it just needs a pool containing more
high-`rho` members.

Also worth recording: the recovered `C` beats **every** one of the 40 eigenfamily candidates on two
of three seeds (+50.9 vs +33.3, +32.2 vs +35.7 on seed 5 where one candidate edges it, +42.2 vs
+41.4 on seed 4). The jointly-fitted `C` is not simply the best eigenvector — consistent with
`fit_hamiltonian_pair`'s own docstring, which says the answer is "a direction inside the conserved
subspace that no single eigenvector isolates".

### What this changes

E10 cannot be settled on a 1-DoF pendulum with a 40-candidate pool. Two follow-ups, in order:

1. **Enlarge the pool, not the band.** The prereg fixes the band width; pool size is unconstrained
   and a larger pool is strictly more information. Running seed 3 at 250 candidates to see whether
   the matched band populates.
2. **If it does not, E10 is a 2-DoF experiment.** A system with 4-dimensional state has a far richer
   family of scalars at varying (conservation, decodability) combinations, which is exactly what the
   matched band needs. This is an independent argument for the 2-DoF work beyond the "energy is the
   only thing to find" objection — the control that most directly tests the paper's central thesis
   may simply not be constructible on a pendulum.

---

## 2026-08-27 06:40–07:15 UTC — E11 passes; **E12's registered test was mis-specified — the third such case**

Git at `b07799d`. Damped seed 2 trained and passed acceptance: **all six models complete**, Stage 1
training finished.

### E10, final: structurally impossible on this system

Enlarging the candidate pool from 40 to **250** (the prereg fixes band width, not pool size) does not
help. On seed 3, of 250 eigenfamily candidates, **exactly 1 has `|rho_E|` > 0.3.**

| pool | candidates with \|rho_E\| > 0.9 | > 0.5 | > 0.3 | in +-0.10 band |
|---|---|---|---|---|
| 250 | 1 | 1 | 1 | 1 |

E10 requires variation in decodability at matched conservation. This system provides **essentially
no variation in decodability at all** within the conserved family. E10 is not merely underpowered
here — it is not constructible. Recorded as such; it is now a 2-DoF experiment.

### E11 step 1 — `D_sec` is phase-invariant, as registered

Time-shifting real rendered trajectories by 1, 2, 5, 10 steps moves median abs `D_sec` by at most
**1.63e-04**, against the registered readout floor of 1.0e-03. Energy does not depend on phase, and
the metric behaves accordingly.

### E11 step 2 — the intervention repairs energy, not timing

| seed | energy drift | phase error | lag |
|---|---|---|---|
| 3 | 1.28e-03 -> 6.28e-04 (**-50.9%**) | 0.066 -> 0.057 (-13.4%) | 0.00 -> 0.00 |
| 4 | 1.13e-03 -> 6.53e-04 (**-42.2%**) | 0.068 -> 0.059 (-13.0%) | 1.00 -> 0.00 |
| 5 | 1.14e-03 -> 7.74e-04 (**-32.2%**) | 0.068 -> 0.054 (-20.5%) | 0.00 -> 0.00 |

Registered prediction met on all three seeds: energy drift falls two to four times more, in relative
terms, than phase error. **The Samanta & Behera phase rival is answered** — both structurally (step 1)
and empirically (step 2).

### E12 — registered test FAILED, and the test was the problem

| seed | \|Spearman(C, E)\| on unedited rollouts | registered threshold |
|---|---|---|
| 3 | 0.194 | > 0.5 |
| 4 | 0.121 | > 0.5 |
| 5 | 0.189 | > 0.5 |
| random `C` (n = 20) | 0.140 | — |

The recovered `C` scores **0.194 against random `C` at 0.140**. A test in which the treatment barely
separates from its own null is a test without power, not evidence of a dormant pathway.

**Diagnosis (exploratory, labelled as such).** Within one rollout, `C`'s standard deviation is
**3-4% of its across-trajectory standard deviation** — which is exactly what "conserved" means. The
registered test correlates two near-constant quantities within a trajectory, so it correlates noise.

Exploratory diagnostics, reported including the uncomfortable one:

| seed | across-traj Spearman(C_0, E_0) | across-traj Spearman(dC, dE) |
|---|---|---|
| 3 | **+0.963** | +0.356 |
| 4 | **+0.850** | -0.124 |
| 5 | **+0.978** | +0.167 |

`C` encodes energy very strongly across trajectories. But **how much `C` drifts does not strongly
predict how much decoded energy drifts** (+0.356, -0.124, +0.167). That is not what a simple
"C-drift causes E-drift" story predicts, and it is recorded rather than explained. A cleaner version
would compare secular slopes rather than endpoint differences, since endpoint `dE` mixes drift with
shadow-Hamiltonian oscillation — but that is a **new test and will be preregistered as E12b**, not
substituted for the one that failed.

**The Makelov dormant-pathway objection is therefore NOT resolved.** It remains open.

### The pattern, stated plainly: three mis-specified controls

| experiment | registered statistic | why it failed |
|---|---|---|
| **E3** | isotropic null `1/LD` | assumed `dz` isotropic; every arm fell below it, so no arm could reach the null |
| **E2b** | damped models as the `beta ~ 1` positive control | damped recovered `C` is not energy-correlated, so `|dC|` saturates at k=1 |
| **E12** | within-trajectory Spearman(C, E) | both quantities near-constant within a trajectory, so it correlates noise |

All three share one cause: **a threshold was registered on a statistic without first checking that
the statistic had the dynamic range to detect anything.** In each case the diagnosis came from the
arm that was supposed to be easy.

Standing rule adopted for the remainder of the project, and it should be in the paper's methods:
**before registering a threshold, validate the statistic on a signal with a known answer** — as was
eventually done for E2b (synthetic random walk 0.473, drift 1.000, saturated noise 0.006) and should
have been done first.

This costs little when followed and has now cost three experiments when not.

---

## 2026-08-27 07:15–07:45 UTC — **E12b: invariant drift does predict energy drift** (2/3 at the registered bar)

Git at `2508459`. All six models trained. Preregistered in `docs/E12B_PREREG.md`, which for the first
time **validated the statistic on known-answer signals before registering a threshold** — the rule
adopted after E3, E2b and E12.

### The validation, and a prediction of mine it refuted

`Spearman(D_sec(A), D_sec(B))` on synthetic series: **+0.987** (strong coupling), **+0.417**
(coupling plus large independent drift), **-0.025** (independent), **+0.980** (coupling plus strong
oscillation). The statistic has range and returns ~0 for independence.

I had predicted the endpoint-difference version would be degraded by oscillation. On the synthetic
series it was **not** (+0.982), and I recorded that in the prereg as making E12b a genuine test with
a live negative outcome rather than a rescue.

### Result

| arm | rho | 95% CI |
|---|---|---|
| conservative s3 | **+0.740** | [+0.542, +0.848] |
| conservative s4 | +0.454 | [+0.157, +0.685] |
| conservative s5 | **+0.740** | [+0.551, +0.848] |
| **random `C`** (n = 20) | **+0.022** | range [-0.323, +0.360] |
| **damped** (n = 3) | +0.073, +0.095, +0.002 | descriptive |

The registered bar was `rho > 0.5` with CI excluding 0 **on all three seeds**. Met on **two of
three**; seed 4 is +0.454, positive with a CI excluding 0 but not clearing 0.5. The registered
falsifier — "near 0 or inconsistent in sign" — **did not fire**: all three are positive, all three
CIs exclude 0, and both control arms sit at zero.

Reported as a **partial pass**: the coupling is real and specific, and the registered threshold was
not met on every seed.

### This resolves the uncomfortable diagnostic from last iteration

The exploratory endpoint-difference numbers were +0.356, -0.124, +0.167, and I recorded them as
possibly real because the synthetic validation said the endpoint statistic was not degraded by
oscillation. On real data it clearly is: the slope statistic gives +0.740 / +0.454 / +0.740 on the
same rollouts.

**Why the validation missed it, recorded because it is the same class of error a fourth time.** The
synthetic coupling was `B = 2.5A` — one gain, shared by every trajectory. Real trajectories have
*different* local `dE/dC` gains, which endpoint differences absorb badly and a 100-point slope
absorbs well. A validation is only as good as the structure it reproduces, and mine omitted the
per-trajectory gain variation that E4 had already demonstrated exists across seeds.

The rule stands and should be sharpened in the paper's methods: **validate the statistic on a
known-answer signal that reproduces the structure of the real data**, not merely on a signal with a
known answer.

### Bearing on the dormant-pathway objection

Makelov et al. warn that a subspace edit can work through a pathway the model does not normally use.
E12b shows the `C` direction carries a **real, specific relationship to physical energy in unedited
rollouts** — random constraints and damped models both give ~0. That is on-pathway evidence, and it
is the evidence the failed E12 was reaching for.

It does not fully close the objection: a correlation shows the direction is informative about energy
in the model's own dynamics, not that the forward pass reads it. Combined with E4's transfer
correlation (+0.808 to +0.916), the pair is substantially stronger than either alone. The objection
is now **substantially addressed but not eliminated**, and should be stated that way.

### Status

Stage 1 complete. Stage 3 controls: E10 **not constructible** on this system, E11 **passed**, E12
**failed as registered**, E12b **partial pass**. Next per Richard's approved order: the 2-DoF system,
which now carries E10 as well as the generality question.

---

## 2026-08-27 07:45–08:20 UTC — **E17 chaos gate: it changed the design twice, before any data was generated**

Git at `dd2a5a2`. Preregistered in `docs/E17_PREREG.md`, approved by Richard. The gate is the whole
point of running simulation before rendering, and it earned its place.

### Failure 1 — weak coupling leaves extra invariants

At `w1 = 1.0, w2 = 1.3, a = b = 0.20` the **per-mode energies are conserved almost as well as the
total** (invariance ratio 0.052 against 0.033). The "non-central" arm would have had **three**
approximate invariants, not one, silently defeating the design. KAM tori survive a ~10% perturbation.

Strengthening to `a = b = 0.50` broke the mode energies but **failed the chaos gate** (frac 0.832
against a registered 0.95).

### Failure 2 — symmetry-breaking trades off directly against chaos

Holding `a = 0.30` and raising `b` to break the rotational symmetry: `b = 0.60` already fails
(lambda*T = 1.369), and it gets worse from there. Sixteen parameter sets recorded in
`runs/e17_chaos_gate_search.json`, per the prereg's requirement to record every set tried.

### The design correction that fell out of it

At `w1 = w2` the central potential expands **exactly**:

    1/4 a r^4  =  1/4 a (q1^4 + q2^4) + 1/2 a q1^2 q2^2

So **central is just `b = a`**. The two arms collapse into one system with **one parameter**
changed — strictly cleaner than the two separately-specified potentials in the original prereg, and
a matched pair in the same sense as conservative/damped. The prereg was amended in place, before any
data existed.

### Frozen parameters

    w1 = w2 = 1.0     a = 0.05     dt = 0.05
    non-central b = 0.40      central b = 0.05

| arm | H=100 | H=190 | H=200 | E invariance | L invariance |
|---|---|---|---|---|---|
| non-central | 1.000 PASS | 0.977 PASS | 0.980 PASS | 0.0290 | **0.1006** broken |
| central | 1.000 PASS | 1.000 PASS | 1.000 PASS | 0.0283 | **0.0000** exact |

Total energy is equally well conserved in both arms; angular momentum is exact in one and 3.5x worse
in the other. That is the contrast the experiment turns on, and it now exists by construction.

Integrator is **semi-implicit (symplectic) Euler**, deliberately matching gymnasium's pendulum
convention so `p_k = (q_k - q_{k-1})/dt` holds exactly and the geometric readout's
backward-difference convention transfers unchanged. Choosing a "better" integrator here would have
silently broken a readout that already cost one iteration to get right.

### What this cost and saved

Roughly half an hour of pure simulation. It caught a design that would have produced a
three-invariant "one-invariant" arm, and would not have been detectable from the pixel results
without a great deal of confusion. Nothing was rendered or trained under the broken design.

### Next

Renderer, then dataset, then training under the frozen step-capped contract. The extraction runs at
**frozen hyperparameters first** (LD = 12, degree 4, n_basis = 8) with any adaptation recorded as a
separate experiment, per the roadmap.

---

## 2026-08-27 08:20–08:45 UTC — E17 renderer, datasets, and training launched

Git at `46294d3`. Frozen parameters from the chaos gate: `w1 = w2 = 1.0, a = 0.05, dt = 0.05`,
non-central `b = 0.40`, central `b = 0.05`.

### Renderer

Single anti-aliased disk at `(q1, q2)` on white, 64x64, frame covering `[-2, 2]` — measured
`|q|` max over the initial-condition distribution is 1.80, so trajectories stay in frame with
margin. Ink colour `(204, 77, 77)`, the same as the pendulum rod, so the readout's ink threshold
carries over unchanged.

Anti-aliasing is deliberate. The readout recovers position from the ink-weighted centroid, and a
hard-edged disk quantises that to whole pixels. The smooth edge buys sub-pixel accuracy — the
property the pendulum got for free from its 500 -> 64 block-average.

**Readout validated before any dataset was generated** (the standing rule). On five known
positions spanning the frame: **max position error 0.0014 units = 0.022 px.** Essentially exact,
and far better than the pendulum's 0.11 deg angular error, because a disk's centroid *is* its
position while a rod's centroid is offset by the axle glyph.

### Datasets

| file | trajectories x steps | b | seed | E across-traj std | **L invariance ratio** |
|---|---|---|---|---|---|
| `osc2d_noncentral.npz` | 256 x 120 | 0.40 | 0 | 0.3269 | **0.0638** |
| `osc2d_central.npz` | 256 x 120 | 0.05 | 11 | 0.3128 | **0.0000** |
| `osc2d_noncentral_eval.npz` | 512 x 200 | 0.40 | 777 | 0.3057 | **0.1040** |

Zero trajectories rejected for leaving the frame in any set. Angular momentum is **exactly**
conserved in the central arm and clearly broken in the non-central one, while total energy is
equally well conserved in both — the contrast the experiment turns on, confirmed in the data as
generated rather than assumed.

`states` is saved as the 4-D phase-space state `(q1, q2, p1, p2)`, named to match the pendulum
dataset so `train_dreamer_pendulum.py` loads these files **unchanged**. Also saved: `q`, `p`,
`energy`, `angmom`, and the parameter vector, so every downstream number is traceable to the
generating physics without re-simulation.

### Training launched

Six models under the **frozen step-capped contract**, identical to the pendulum arm: 60,000 steps,
E8 milestone grid `{1k, 3k, 6.5k, 15k, 30k, 60k}`, `--max-hours 6` as a non-binding energy bound.

- non-central seeds 3/4/5 -> `runs/osc2d_nc_s{3,4,5}.pt`
- central seeds 0/1/2 -> `runs/osc2d_ce_s{0,1,2}.pt`

Estimated ~8.5 h. The step-6,500 checkpoints — enough to begin analysis — arrive roughly 10 minutes
into each run.

### Registered next step, restated so it cannot drift

Extraction runs at **frozen hyperparameters first**: LD = 12, degree 4, `n_basis` = 8, WARMUP = 10.
LD = 12 was chosen for a **2-dimensional** physical state; this state is **4-dimensional**. If
recovery fails at those settings that is a **result about generality and will be reported as one**,
and any adapted setting becomes a separate experiment with its own log entry. No silent tuning.

---

## 2026-08-27 08:45–09:00 UTC — **E17 recovery FAILS at frozen hyperparameters** (preliminary: 1 seed, 1 checkpoint)

Git at `fcaa500`. Non-central seed 3 at step 6,500 — the first 2-DoF checkpoint. Run at **frozen**
LD = 12, degree 4, `n_basis` = 8, WARMUP = 10, with **no tuning**, as `docs/E17_PREREG.md` requires.

### Registered prediction 1: FAILED

| | invariance ratio | \|rho_E\| | \|rho_L\| |
|---|---|---|---|
| **recovered `C`** (jointly fitted) | 2.62e-04 | **0.158** | 0.109 |
| best eigenvector (rank 0) | 8.16e-05 | **0.696** | **0.818** |
| pendulum reference | ~1e-04 | 0.967-0.975 | — |

Registered bar was `|rho_E| > 0.8`. Measured **0.158**. **FAIL.**

### What did and did not fail

**Conservation transferred.** The empirical invariance ratio is 2.6e-04, the same order as the
pendulum's. The method finds a genuinely well-conserved latent scalar on a 4-dimensional system at
frozen settings. Retained rank is the full 12/12.

**Identification did not.** The scalar it converges on is not energy. And the jointly-fitted `C` is
**four times worse at energy correlation than the best raw eigenvector** (0.158 against 0.696) — so
the failure is not "there is nothing to find", it is the **selection step choosing the wrong
direction inside the conserved subspace**.

That points at flow alignment specifically. `fit_hamiltonian_pair`'s docstring argues the answer is
"a direction inside the conserved subspace that no single eigenvector isolates", and on the pendulum
the flow criterion found it. Here it appears to move *away* from energy. The pairing residual is
0.805, in the same range as the pendulum's 0.829-0.865, so the fit believes it succeeded.

**A finding worth its own line.** The best-conserved eigenvector correlates **0.818 with angular
momentum** and 0.696 with energy — in the arm where `L` is supposed to be broken. `L`'s invariance
ratio in the generated data is 0.0638 against energy's 0.029, so `L` is *partially* conserved here,
not absent. The latent's best-conserved direction may be tracking a mixture, or tracking `L`
preferentially. Recorded, not explained, and not yet trustworthy on one seed.

### Registered prediction 3: also failed, but improved

E10 needs >= 8 candidates within +-0.10 of the recovered `C`'s `|rho_E|`. Found **3**.

| pool | \|rho_E\| > 0.8 | > 0.5 | > 0.3 |
|---|---|---|---|
| pendulum, 250 candidates | 1 | 1 | 1 |
| **2-DoF, 60 candidates** | 0 | **2** | **3** |

Richer than the pendulum at a quarter of the pool size, and still not enough. E10 remains
unconstructible for now.

### Caveats that must be attached to every number above

- **One seed, one checkpoint, and an early one.** Step 6,500 of 60,000. `val_recon` was still
  falling when this was measured.
- LD = 12 was fixed for a **2-dimensional** physical state; this one is **4-dimensional**. The
  prereg flagged this in advance as the most likely failure point.
- The central arm has not been trained yet — it queues behind the three non-central seeds.

### What is NOT being done

**No hyperparameters are being tuned.** The roadmap and `E17_PREREG` both require the frozen-setting
failure to stand as a recorded result before any adaptation, and any adapted setting to be a separate
experiment with its own entry. That order is being kept. The next evidence is later checkpoints and
further seeds at the same frozen settings — not a larger LD.

---

## 2026-08-27 09:10–09:25 UTC — **E17 recovery PASSES at step 15,000: the step-6,500 failure was undertraining**

Git at `f482d2e`. Non-central seed 3, frozen hyperparameters throughout — **nothing was tuned
between the failure and this result.**

### The milestone sweep

| step | recovered `C` ratio | **\|rho_E\|** | \|rho_L\| | best eigenvector \|rho_E\| | \|rho_E\| in pool > 0.5 |
|---|---|---|---|---|---|
| 1,000 | 9.89e-01 | 0.007 | 0.027 | 0.016 | 0 |
| 3,000 | 3.55e-04 | 0.383 | 0.005 | 0.323 | 2 |
| 6,500 | 2.62e-04 | **0.158** | 0.109 | 0.696 | 2 |
| **15,000** | **6.62e-05** | **0.909** | 0.138 | 0.708 | 2 |

**Registered prediction 1 (`|rho_E| > 0.8`): PASSES at step 15,000.**

The invariance ratio improves monotonically across training (9.9e-01 -> 6.6e-05), and the recovered
scalar's energy correlation reaches 0.909 — within reach of the pendulum's 0.967-0.975.

### The previous entry's headline is superseded, and the discipline is why

Last iteration reported `|rho_E| = 0.158` at step 6,500 as a **failure of generality at frozen
hyperparameters**, with the caveat that it was one early checkpoint. It was an **undertraining
artefact**. The 2-DoF system needs roughly **2.3x the pendulum's training** before the invariant
becomes recoverable — which is unsurprising for a 4-dimensional state and is itself a finding worth
reporting.

The reason this resolved cleanly is that the prereg forbade tuning on the failure. Had LD been raised
at step 6,500 — the obvious move, and the one the prereg flagged in advance as most tempting — the
result would have been "generality requires a larger extraction dimension", which is **false**, and
it would have been extremely hard to detect afterwards.

### Flow alignment: broken at 6,500, working at 15,000

| step | recovered `C` | best eigenvector | which is better |
|---|---|---|---|
| 6,500 | 0.158 | 0.696 | **eigenvector, by 4x** |
| 15,000 | **0.909** | 0.708 | **joint fit, by 1.3x** |

At step 6,500 the joint fit was actively worse than taking the top eigenvector; at 15,000 it is
clearly better. This is direct evidence for `fit_hamiltonian_pair`'s own claim — that the answer is
"a direction inside the conserved subspace that no single eigenvector isolates" — **and** for the
condition under which that claim holds: the conserved subspace has to be good enough first. The
pairing criterion cannot find a direction that is not yet there.

Recorded because it is a mechanism for the earlier failure, not a restatement of it.

### `L` is not what it converges on

At step 15,000 the recovered `C` has `|rho_E| = 0.909` against `|rho_L| = 0.138`. The step-6,500
observation that the best eigenvector correlated 0.818 with angular momentum does **not** persist:
with adequate training the method picks energy specifically, in the arm where energy is the
better-conserved quantity. The earlier observation is retained in the log as what it was — a reading
from an undertrained model.

### Registered prediction 3 still fails

E10 needs >= 8 candidates within +-0.10 of the recovered `C`'s `|rho_E|`. The pool still has only
**2** candidates above 0.5 at any checkpoint. Richer than the pendulum, still not enough. E10 remains
unconstructible, and it is now clear that 2 degrees of freedom is not by itself the fix.

### Caveats

One seed. Steps 30,000 and 60,000 not yet reached. The central arm — the two-invariant test, and the
sharper half of the matched pair — queues behind the remaining non-central seeds.

---

## 2026-08-27 09:40–10:00 UTC — **E17: recovery AND repair both transfer to two degrees of freedom**

Git at `3a405e1`. Non-central seed 3. All at **frozen hyperparameters**, nothing tuned at any point.

### 2-DoF physical readout, validated before any intervention number

`decode_physics_osc2d` added to `latent_noether/pixel_readout.py`, reusing
`centroids_from_frames`. Backward differences for momentum, for the same reason as the pendulum:
`make_oscillator2d` integrates semi-implicitly so `p_k` is exactly `(q_k - q_{k-1})/dt`.

| dataset | q error | p error | E error / across-traj std | L error / \|L\| | blank |
|---|---|---|---|---|---|
| non-central | 0.00102 | 0.0262 | **0.0316** | 0.0270 | 0.0000 |
| central | 0.00105 | 0.0272 | **0.0408** | 0.0253 | 0.0000 |

Comparable to the pendulum readout's 1.6%. No calibrated offset was needed — a disk's ink-weighted
centroid **is** its centre, unlike the pendulum's rod-plus-axle.

### Prediction 1, recovery: PASSES, and matches the pendulum by step 30,000

| step | invariance ratio | **\|rho_E\|** | \|rho_L\| | verdict |
|---|---|---|---|---|
| 1,000 | 9.89e-01 | 0.007 | 0.027 | fail |
| 3,000 | 3.55e-04 | 0.383 | 0.005 | fail |
| 6,500 | 2.62e-04 | 0.158 | 0.109 | fail |
| 15,000 | 6.62e-05 | **0.909** | 0.138 | PASS |
| **30,000** | **1.64e-05** | **0.966** | 0.355 | **PASS** |

0.966 against the pendulum's 0.967-0.975. The invariance ratio improves monotonically by nearly five
orders of magnitude. The 2-DoF system needs roughly **4.6x the pendulum's training** to reach parity
— a real and reportable difference, not a failure.

### Prediction 3, repair: PASSES

Direction-matched, H = 100, step 30,000:

| eps | recovered | random median | tangent median | random beating recovered |
|---|---|---|---|---|
| 0.005 | **-32.5%** | -5.2% | -2.5% | 0/20 |
| 0.01 | **-46.3%** | -7.1% | -1.3% | 0/20 |
| 0.02 | **-57.6%** | -17.2% | +0.7% | **0/20** |

Pendulum reference at the same horizon: -32 to -51%, random +7 to +10%, 0/60.

Monotone dose-response, tangent controls at zero, and no random direction beats the recovered one at
any step size. **The effect is if anything larger than on the pendulum.**

One difference worth recording: random directions **help** here (-17.2%) where they hurt on the
pendulum (+7 to +10%). With a 4-dimensional state and a richer conserved family, a random degree-4
polynomial is more likely to have some overlap with a genuinely conserved direction. The recovered
constraint is still 3.3x better and beaten 0/20, so the comparison holds — but "random constraints
hurt" is a pendulum-specific statement and should not be carried over.

### Claim C6 (generality)

**Supported on the non-central arm, one seed.** Both halves of the central result — that the method
recovers a physically meaningful invariant, and that enforcing it repairs imagined physics — survive
a move from 2-dimensional to 4-dimensional state at frozen hyperparameters.

### Still open

- **Prediction 2, the two-invariant test**, is the sharper half of the matched pair and has not run:
  the central models queue behind the remaining non-central seeds.
- **Prediction 4, E10**, still fails — only 2 pool candidates above `rho_E` 0.5. Two degrees of
  freedom is not by itself enough to make the decodability-matched null constructible.
- One seed; seeds 4 and 5 pending.

---

## 2026-08-27 10:10 UTC — Central arm launched concurrently (execution change, not a protocol change)

Git at `75bac91`. Non-central seed 3 at step 46,000.

The central arm is the **two-invariant test** — the sharpest thing the 2-DoF design offers, and the
only experiment in the project that asks whether the method recovers an invariant *manifold* rather
than a lucky scalar. In the queued order it sat behind non-central seeds 4 and 5, roughly five hours
out.

**Launched central seed 0 concurrently** rather than reordering or interrupting.

This is an execution change with **no effect on the scientific protocol**, and specifically because
of the training-contract decision made on 2026-08-26: training is capped by **optimizer steps**, not
wall clock, so GPU contention changes how long a run takes and not what it produces. Under the
original wall-clock contract this would have silently given the concurrent models less training than
the sequential ones — exactly the M28 failure that motivated the change. Recorded here because it is
the first time that decision has paid off in a way that was not the reason for making it.

Memory is not a constraint (15.8 GB of 96 GB before the second job). Nothing was interrupted and no
partial checkpoint was discarded.

---

## 2026-08-27 10:10–10:45 UTC — **2-DoF result survives disjoint evaluation**

Git at `56e28a4`. Non-central seed 3 finishing; central seed 0 training concurrently.

`C`, `h_mean`, `U` and `R` frozen from the analysis split of `osc2d_noncentral.npz`; scored on
`osc2d_noncentral_eval.npz` — **512 trajectories x 200 frames, generator seed 777**, never used for
training, fitting or any prior analysis. Horizon clamped to 190 by the recorded rule.

| eps | recovered | random median | random beating recovered |
|---|---|---|---|
| 0.005 | -13.5% | -2.5% | 0/12 |
| 0.01 | -32.1% | -4.1% | 0/12 |
| 0.02 | **-62.6%** | -10.0% | **0/12** |

Monotone dose-response, and the effect is **larger out of sample at H = 190 (-62.6%) than in sample
at H = 100 (-57.6%)** — the same horizon scaling E6 found on the pendulum, now reproduced on a
different system.

Side-by-side at H = 190 on disjoint data:

| system | recovered | random |
|---|---|---|
| pendulum, 3 seeds | -54.6 to -75.9% | +7.4 to +9.7% |
| **2-DoF non-central, 1 seed** | **-62.6%** | -10.0% |

The random arm is still filling (12 of 20); the number will be restated when it completes.

### What C6 now rests on

Claim C6 (generality) is supported by three independent transfers, all at frozen hyperparameters:

1. **Recovery** — `|rho_E|` 0.966 at step 30,000, matching the pendulum's 0.967-0.975
2. **Repair** — -57.6% in sample, 0/20 random
3. **Out-of-sample repair** — -62.6% on 512 unseen trajectories, 0/12 random, with the same
   horizon scaling

One seed, non-central arm only. The two-invariant central test remains the sharpest thing untested.
