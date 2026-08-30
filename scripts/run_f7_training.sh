#!/usr/bin/env bash
# F7 (docs/F7_PREREG.md): 3 seeds trained on VELOCITY VERLET data at dt=0.05.
# Identical to F6's dt=0.05 arm in every respect except the integrator, so "the model merely
# learned dt" cannot explain a difference. Guarded step() so a re-run is a no-op, not a silent
# retrain (2026-08-27 incident).
set -uo pipefail
cd "$(dirname "$0")/.."
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
step() {
  local out="$1" log="runs/logs/$2.log"; shift 2
  if [[ -e "$out" ]]; then echo "$(ts)  SKIP  $out exists"; return 0; fi
  echo "$(ts)  START $out"
  if "$@" >"$log" 2>&1; then echo "$(ts)  DONE  $out"; else echo "$(ts)  FAIL  $out — see $log"; return 1; fi
}
for s in 3 4 5; do
  step "runs/f7_verlet_s${s}_step6500.pt" "f7_verlet_s${s}" \
    env OMP_NUM_THREADS=8 uv run python scripts/train_dreamer_pendulum.py \
      --seed "$s" --steps 6500 --max-hours 2 --ckpt-at 6500 \
      --data "runs/pend_verlet_dt0.05.npz" --out "runs/f7_verlet_s${s}.pt"
done
echo "$(ts)  all F7 models complete"
