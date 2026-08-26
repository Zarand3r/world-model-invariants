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
# Wall clock: ~20 min data + 6 x 30 min training = ~3.5 h on one GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8   # REPRODUCE.md §5

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
    uv run python scripts/train_dreamer_pendulum.py --seed "$s" --max-hours 0.5 --steps 40000 \
        --data runs/pendulum_pixels.npz --out "runs/dreamer_ref_s$s.pt"
done

# ---- 3. damped models (arm C / the refusal control) ----------------------
for s in 0 1 2; do
  step "runs/dreamer_damped_s$s.pt" "train_damped_s$s" \
    uv run python scripts/train_dreamer_pendulum.py --seed "$s" --max-hours 0.5 --steps 40000 \
        --data runs/pendulum_pixels_damped.npz --out "runs/dreamer_damped_s$s.pt"
done

echo "$(ts)  stage1 bootstrap complete"
grep -hE "raw KL|one-step|rollout|OK|FAIL" runs/logs/train_*.log 2>/dev/null | tail -40
