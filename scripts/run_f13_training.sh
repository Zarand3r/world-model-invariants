#!/usr/bin/env bash
# F13 (docs/F13_PREREG.md): 3 seeds on MIXED-timestep data with dt in the conditioning channel.
# More steps than F6 (15000 vs 6500) because the data is 4x larger and the task is harder -- the
# model must learn a dt-parameterised family rather than one dynamics. Checkpoint at 6500 too, so
# the F6-matched budget is available for comparison.
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
  step "runs/f13_mixed_s${s}_step15000.pt" "f13_mixed_s${s}" \
    env OMP_NUM_THREADS=8 uv run python scripts/train_dreamer_pendulum.py \
      --seed "$s" --steps 15000 --max-hours 3 --ckpt-at 6500 --ckpt-at 15000 \
      --data "runs/pend_mixed_dt.npz" --out "runs/f13_mixed_s${s}.pt"
done
echo "$(ts)  all F13 models complete"
