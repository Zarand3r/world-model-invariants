#!/usr/bin/env python3
"""F7 cross-evaluation control (docs/F7_PREREG.md, amendment 2).

Crosses checkpoint against evaluation dataset. If the recovered argmin follows the CHECKPOINT, the
sweep is measuring the model's learned dynamics. If it follows the DATA, the sweep is measuring the
readout construction and F7's interpretation collapses.

No training -- existing checkpoints only.

Usage:  uv run python scripts/run_f7_cross.py --out runs/f7_cross.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_analysis import run

DT = 0.05
CKPT = {"semi-implicit": [f"runs/f6_dt0.05_s{s}_step6500.pt" for s in (3, 4, 5)],
        "verlet":        [f"runs/f7_verlet_s{s}_step6500.pt" for s in (3, 4, 5)]}
DATA = {"semi-implicit": "runs/pend_dt0.05.npz", "verlet": "runs/pend_verlet_dt0.05.npz"}
EXPECT = {"semi-implicit": 1.0, "verlet": 0.0}   # what the MODEL should give, whatever the data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f7_cross.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"cells": []}
    done = {(c["ckpt"], c["eval_data"]) for c in out["cells"]}

    for trained_on, ckpts in CKPT.items():
        for eval_on, data in DATA.items():
            for ck in ckpts:
                if (ck, data) in done or not pathlib.Path(ck).exists():
                    continue
                r = run(ck, data, DT)
                cell = {"ckpt": ck, "trained_on": trained_on, "eval_data": data,
                        "eval_on": eval_on, "argmin_r": r["argmin_r"],
                        "rho_obs_at_r0": r["rho_obs_at_r0"], "rho_obs_at_r1": r["rho_obs_at_r1"],
                        "follows_model": bool(r["argmin_r"] == EXPECT[trained_on]),
                        "diagonal": trained_on == eval_on}
                print(f"  train={trained_on:14s} eval={eval_on:14s} {pathlib.Path(ck).name:28s} "
                      f"argmin r={r['argmin_r']:+.2f}  (model predicts {EXPECT[trained_on]:+.0f})",
                      flush=True)
                out["cells"].append(cell); op.write_text(json.dumps(out, indent=1) + "\n")

    off = [c for c in out["cells"] if not c["diagonal"]]
    if off:
        verdict = {}
        for trained_on in CKPT:
            cells = [c for c in off if c["trained_on"] == trained_on]
            n = sum(c["follows_model"] for c in cells)
            verdict[f"{trained_on}_off_diagonal_follows_model"] = f"{n}/{len(cells)}"
            verdict[f"{trained_on}_pass"] = bool(n >= 2)
        verdict["P1_pass"] = bool(all(v for k, v in verdict.items() if k.endswith("_pass")))
        out["verdict"] = verdict
        print("\n  " + json.dumps(verdict))
        print("  argmin follows the MODEL" if verdict["P1_pass"]
              else "  argmin follows the DATA -- F7's interpretation does not survive")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
