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
