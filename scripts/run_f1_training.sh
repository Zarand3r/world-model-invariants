#!/usr/bin/env bash
# F1: train three action-conditioned seeds on the actuated dataset.
#
# Uses the guarded step() pattern from run_stage1_bootstrap.sh. An inline loop without this guard is
# what silently retrained over already-analysed ce_s0 checkpoints on 2026-08-27; the `-e` check makes
# a re-run a no-op instead of a rewrite.
set -uo pipefail
cd "$(dirname "$0")/.."
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
DATA=runs/pendulum_actuated.npz
for s in 3 4 5; do
  step "runs/f1_act_s${s}.pt" "f1_train_s${s}" \
    env OMP_NUM_THREADS=8 uv run python scripts/train_dreamer_pendulum.py \
      --seed "$s" --steps 60000 --max-hours 6 \
      --ckpt-at 1000,3000,6500,15000,30000,60000 \
      --data "$DATA" --out "runs/f1_act_s${s}.pt"
done
echo "$(ts)  all seeds complete"
