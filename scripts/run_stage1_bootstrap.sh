#!/usr/bin/env bash
# Stage 1 bootstrap for docs/ROADMAP.md.
#
# The repo commits run logs, not artefacts: no datasets, no checkpoints, no vendored DreamerV3.
# This script regenerates everything Phase I needs, running the exact commands in docs/REPRODUCE.md
# in order, with timestamps and per-step logs under runs/logs/.
#
# Idempotent: every step is skipped if its output already exists, so re-invoking makes monotonic
# progress after an interruption.
#
# TRAINING CONTRACT (approved by Richard 2026-08-26, see docs/EXECUTION_LOG.md).
# Capped by OPTIMIZER STEPS, not wall clock. REPRODUCE.md's `--max-hours 0.5 --steps 40000` let wall
# clock bind (741 steps/min on this GPU => ~22k steps, against the paper's ~6.5k), which is exactly
# what M28 in train_dreamer_pendulum.py forbids for models that are going to be compared.
#
# Each run saves the whole E8 milestone grid, so one training run yields both a paper-comparable
# 6,500-step model and a saturated one, on a single optimisation path.
#
# Wall clock: ~20 min data + 6 x ~81 min training = ~8.5 h on one GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8   # REPRODUCE.md §5

STEPS="${STEPS:-60000}"
CKPT_AT="${CKPT_AT:-1000,3000,6500,15000,30000,60000}"   # E8 grid; 6500 ~ the published models

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
step() {                       # step <output> <logname> <cmd...>
  local out="$1" log="runs/logs/$2.log"; shift 2
  if [[ -e "$out" ]]; then echo "$(ts)  SKIP  $out exists"; return 0; fi
  echo "$(ts)  START $out  -> $log"
  if "$@" >"$log" 2>&1; then
    echo "$(ts)  DONE  $out  ($(du -h "$out" 2>/dev/null | cut -f1))"
  else
    echo "$(ts)  FAIL  $out  (exit $?) — see $log"; return 1
  fi
}

echo "$(ts)  stage1 bootstrap begins; git $(git rev-parse --short HEAD)"

# ---- 1. data -------------------------------------------------------------
step runs/pendulum_pixels.npz       data_conservative \
  uv run python scripts/make_pendulum_pixels.py --n-traj 256 --n-steps 120 --seed 0 \
      --out runs/pendulum_pixels.npz
step runs/pendulum_pixels_eval.npz  data_eval \
  uv run python scripts/make_pendulum_pixels.py --n-traj 512 --n-steps 200 --seed 777 \
      --out runs/pendulum_pixels_eval.npz
step runs/pendulum_pixels_damped.npz data_damped \
  uv run python scripts/make_pendulum_pixels.py --zeta 0.03 --n-traj 256 --n-steps 120 --seed 11 \
      --out runs/pendulum_pixels_damped.npz

# ---- 2. conservative models (the published seeds) ------------------------
for s in 3 4 5; do
  step "runs/dreamer_ref_s$s.pt" "train_ref_s$s" \
    uv run python scripts/train_dreamer_pendulum.py --seed "$s" --max-hours 6 --steps "$STEPS" \
        --ckpt-at "$CKPT_AT" --data runs/pendulum_pixels.npz --out "runs/dreamer_ref_s$s.pt"
done

# ---- 3. damped models (arm C / the refusal control) ----------------------
for s in 0 1 2; do
  step "runs/dreamer_damped_s$s.pt" "train_damped_s$s" \
    uv run python scripts/train_dreamer_pendulum.py --seed "$s" --max-hours 6 --steps "$STEPS" \
        --ckpt-at "$CKPT_AT" --data runs/pendulum_pixels_damped.npz --out "runs/dreamer_damped_s$s.pt"
done

echo "$(ts)  stage1 bootstrap complete"
grep -hE "raw KL|one-step|rollout|OK|FAIL" runs/logs/train_*.log 2>/dev/null | tail -40
