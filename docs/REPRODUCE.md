# Reproducing paper 1.2

## 0. From nothing

```bash
git clone --recurse-submodules -b paper1.2 https://github.com/Zarand3r/world-model-invariants
cd world-model-invariants
uv sync
uv run python scripts/fetch_assets.py          # 1.98 GB, every file sha256-checked
uv run pytest tests/ -q                        # 58 tests
uv run python scripts/make_results_summary.py  # regenerates docs/RESULTS.md byte-for-byte
```

`fetch_assets.py --what-backs` lists every artifact against the claim it supports, so a partial
download is possible: `--only e18` or `--only osc2d` pulls just what one result needs. On a machine
that already has the artifacts in a sibling checkout, `--from-local` copies instead of downloading
and checks the same hashes.

The results in `docs/RESULTS.md` need no GPU and no artifacts at all — they are regenerated from the
committed run records, which is the cheapest way to check that the reported numbers are the ones the
experiments produced.

To re-derive a headline rather than read it, E18 is the central claim and the cheapest to run:

```bash
uv run python scripts/fetch_assets.py --only dreamer_ref_s3_step6500 pendulum_pixels
uv run python scripts/run_e18_supervised_baseline.py --ckpts runs/dreamer_ref_s3_step6500.pt \
    --out runs/e18_check.json
```

Expected on seed 3: the supervised probe reaches `rho_E` 0.9999 but `rho_obs` 0.0458 and makes the
rollout **worse** by 26.8%; the label-free scalar reaches `rho_E` 0.9730 with `rho_obs` 0.0063 and
improves it by 50.9%. Verified 2026-09-03: effect sizes reproduce to 3e-05 relative, the small
residual being the GPU nondeterminism recorded in the audit.

## 1. The reference implementation

A git submodule pinned at one upstream commit, rather than a copy in this tree: copying would
silently fork it, and a floating branch would let it move underneath us. `--recurse-submodules`
above fetches it; for an existing checkout:

```bash
git submodule update --init external/dreamerv3-torch
```

That lands `external/dreamerv3-torch` at `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`.

`latent_noether/dreamer_adapter.py` adds `external/dreamerv3-torch` to `sys.path` and imports
`networks` from it. The adapter is ~30 lines of interface — `encode`, `transition`,
`readout_from_h` — over the reference `RSSM`, `MultiEncoder` and `MultiDecoder`.

**Verify the adapter before trusting any result**: our `transition` must agree with the reference's
own `img_step` to ~1e-08. Two adapter bugs (a reversed `get_feat` concatenation, and feeding soft
probabilities to networks trained on one-hot samples) both look like *model* failure and were caught
only by that check.

## 2. Data

```bash
# conservative (the main substrate)
uv run python scripts/make_pendulum_pixels.py --n-traj 256 --n-steps 120 --seed 0 \
    --out runs/pendulum_pixels.npz
# a disjoint evaluation set — the across-trajectory statistics have ONE SAMPLE PER TRAJECTORY,
# so n is the binding constraint, and 51 held-out trajectories was not enough
uv run python scripts/make_pendulum_pixels.py --n-traj 512 --n-steps 200 --seed 777 \
    --out runs/pendulum_pixels_eval.npz
# dissipative control. zeta = 0.03 is SELECTED BY A GATE, not chosen: see docs/DISSIPATIVE_PREREG.md
uv run python scripts/make_pendulum_pixels.py --zeta 0.03 --n-traj 256 --n-steps 120 --seed 11 \
    --out runs/pendulum_pixels_damped.npz
```

## 3. Training

```bash
for s in 3 4 5; do
  uv run python scripts/train_dreamer_pendulum.py --seed $s --max-hours 0.5 --steps 40000 \
      --out runs/dreamer_ref_s$s.pt
done
```

World model only; offline on the fixed dataset; actions identically zero. The wall-clock cap is the
energy bound, and it yields ~6.5k gradient steps — a small fraction of a standard DreamerV3 run.
**Free bits are 0, not the reference 1.0** (see `train_dreamer_pendulum.py` for why).

Each run prints registered acceptance checks. **A model failing them is not evidence about the
method** and its extraction results must be discarded, not reported.

## 4. Analysis

```bash
uv run python scripts/run_dreamer_extraction.py --ld 12 \
    --ckpts runs/dreamer_ref_s{3,4,5}.pt              # recover
uv run python scripts/run_dreamer_extraction.py --ld 12 --untrained \
    --ckpts runs/dreamer_ref_s{0,1,2,3,4,5}.pt        # the null — never omit this arm
uv run python scripts/run_dreamer_refusal.py          # refuse (resumable, one model per call)
uv run python scripts/run_dreamer_edit.py             # intervene (resumable)
```

## 5. Two practical notes that cost real time

**Thread oversubscription.** Each fit solves a 1819×1819 generalized eigenproblem (degree 4 in 12
dimensions). Running several concurrently drove load average to **50 on a 32-core machine**. Memory
is not the constraint (~7 GB). Cap BLAS threads and run sequentially:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 uv run python ...
```

**Long runs.** `run_dreamer_refusal.py` and `run_dreamer_edit.py` checkpoint to JSON after every
model and skip completed ones, so repeated invocations make monotonic progress. Use `--max-models N`
to bound a single invocation.
