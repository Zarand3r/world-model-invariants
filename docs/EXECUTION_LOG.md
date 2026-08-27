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
