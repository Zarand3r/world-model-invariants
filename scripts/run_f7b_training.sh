#!/usr/bin/env bash
# F7b (docs/F7_PREREG.md): 3 seeds on REVERSED semi-implicit Euler at dt=0.05.
# Same guarded step() as F6/F7 so a re-run is a no-op, not a silent retrain.
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
  step "runs/f7b_reversed_s${s}_step6500.pt" "f7b_reversed_s${s}" \
    env OMP_NUM_THREADS=8 uv run python scripts/train_dreamer_pendulum.py \
      --seed "$s" --steps 6500 --max-hours 2 --ckpt-at 6500 \
      --data "runs/pend_reversed_dt0.05.npz" --out "runs/f7b_reversed_s${s}.pt"
done
echo "$(ts)  all F7b models complete"
