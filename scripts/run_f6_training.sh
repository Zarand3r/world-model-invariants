#!/usr/bin/env bash
# F6: 4 timesteps x 3 seeds, analysed at the paper's primary step-6,500 checkpoint.
# Guarded step() so a re-run is a no-op rather than a silent retrain (2026-08-27 incident).
set -uo pipefail
cd "$(dirname "$0")/.."
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
step() {
  local out="$1" log="runs/logs/$2.log"; shift 2
  if [[ -e "$out" ]]; then echo "$(ts)  SKIP  $out exists"; return 0; fi
  echo "$(ts)  START $out"
  if "$@" >"$log" 2>&1; then echo "$(ts)  DONE  $out"; else echo "$(ts)  FAIL  $out — see $log"; return 1; fi
}
for dt in 0.02 0.035 0.05 0.08; do
  for s in 3 4 5; do
    step "runs/f6_dt${dt}_s${s}_step6500.pt" "f6_dt${dt}_s${s}" \
      env OMP_NUM_THREADS=8 uv run python scripts/train_dreamer_pendulum.py \
        --seed "$s" --steps 6500 --max-hours 2 --ckpt-at 6500 \
        --data "runs/pend_dt${dt}.npz" --out "runs/f6_dt${dt}_s${s}.pt"
  done
done
echo "$(ts)  all F6 models complete"
