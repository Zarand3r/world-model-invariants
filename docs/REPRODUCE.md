# Reproducing Phase 3 (DreamerV3)

## 0. From scratch on a new machine

```bash
git clone --recurse-submodules https://github.com/Zarand3r/world-model-invariants
cd world-model-invariants
uv sync                                          # pinned by uv.lock
uv run python scripts/fetch_assets.py            # 359 MB, sha256-checked against docs/ASSETS.json
uv run pytest tests/ -q                          # 40 tests, ~6 s
uv run python scripts/run_dreamer_extraction.py --ld 12 --ckpts runs/dreamer_ref_s{3,4,5}.pt
```

That last command should print `|rho|_E` of 0.973, 0.967 and 0.975. If it does, the recovery result
in the paper is reproduced on your machine, and sections 1-3 below are only needed if you want to
rebuild the inputs rather than download them.

To rebuild the figures from the committed run logs instead — no GPU, seconds:

```bash
uv run python paper/make_figures.py
```

## 1. The world-model implementation

The reference implementation is a **git submodule** pinned at the exact upstream commit, rather than
a copy in this tree: copying it would silently fork it, and a floating branch would let it move
under us. `--recurse-submodules` above fetches it. For an existing checkout:

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

`scripts/fetch_assets.py` downloads these. They are regenerable instead, deterministically — the
`.npz` produced by these commands are byte-identical to the hosted ones:

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

**Write reproductions somewhere else.** Every script's default `--out` is the committed log it
originally produced, and `run_dreamer_refusal.py` / `run_dreamer_edit.py` resume by skipping arms
already present in that file. Run them bare and they will overwrite the paper's evidence, or worse,
skip every arm and print a verdict computed from the committed numbers as if it had just recomputed
them. Send reproductions to `runs/repro/` and diff:

```bash
mkdir -p runs/repro
uv run python scripts/run_dreamer_extraction.py --ld 12 \
    --ckpts runs/dreamer_ref_s{3,4,5}.pt --out runs/repro/extraction_ld12.json   # recover
uv run python scripts/run_dreamer_extraction.py --ld 12 --untrained \
    --ckpts runs/dreamer_ref_s{0,1,2,3,4,5}.pt --out runs/repro/untrained_null.json
uv run python scripts/run_dreamer_refusal.py --out runs/repro/refusal.json       # refuse
uv run python scripts/run_dreamer_edit.py --out runs/repro/edit.json             # intervene
```

The untrained arm's six paths include three checkpoints that no longer exist. That is fine and
deliberate: it never opens them. Its randomly-initialised model is seeded from the checkpoint *path
string*, so the arm reproduces exactly from the filenames alone — see the comment in
`run_dreamer_extraction.py`.

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
