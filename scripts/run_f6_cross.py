#!/usr/bin/env python3
"""F6 cross-timestep control (docs/F7_PREREG.md, amendment 3).

Amendment 2 showed F7's argmin follows the evaluation dataset, not the checkpoint. F6 uses the same
measurement and never evaluated a model off its own timestep. This crosses dt=0.02 and dt=0.08
checkpoints against both datasets to find out whether F6's model arm has the same confound.

No training -- existing checkpoints only.

Usage:  uv run python scripts/run_f6_cross.py --out runs/f6_cross.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_analysis import run
from scripts.run_f6_timestep import c_star

DTS = (0.02, 0.08)
SEEDS = (3, 4, 5)
GRID_MAX = 3.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f6_cross.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"cells": []}
    done = {(c["ckpt"], c["eval_dt"]) for c in out["cells"]}

    for dt_ck in DTS:
        for dt_ev in DTS:
            for s in SEEDS:
                ck = f"runs/f6_dt{dt_ck}_s{s}_step6500.pt"
                if (ck, dt_ev) in done or not pathlib.Path(ck).exists():
                    continue
                # sweep is in units of c*(dt_eval), so r=1 always means "the eval data's own shadow"
                r = run(ck, f"runs/pend_dt{dt_ev}.npz", dt_ev)
                pred_model = c_star(dt_ck) / c_star(dt_ev)
                follows_model = (abs(r["argmin_r"] - pred_model) <= 0.15 if pred_model <= GRID_MAX
                                 else r["argmin_r"] >= 2.0)
                cell = {"ckpt": ck, "ckpt_dt": dt_ck, "eval_dt": dt_ev,
                        "argmin_r": r["argmin_r"],
                        "predicted_r_if_model": pred_model, "predicted_r_if_data": 1.0,
                        "follows_model": bool(follows_model),
                        "follows_data": bool(abs(r["argmin_r"] - 1.0) <= 0.15),
                        "diagonal": dt_ck == dt_ev}
                print(f"  ckpt dt={dt_ck:<6} eval dt={dt_ev:<6} s{s}  argmin r={r['argmin_r']:+.2f}  "
                      f"(model->{pred_model:.2f}, data->1.00)  "
                      f"{'diag' if cell['diagonal'] else 'CROSS'}", flush=True)
                out["cells"].append(cell); op.write_text(json.dumps(out, indent=1) + "\n")

    off = [c for c in out["cells"] if not c["diagonal"]]
    if len(off) >= 6:
        v = {}
        for dt_ck in DTS:
            cells = [c for c in off if c["ckpt_dt"] == dt_ck]
            v[f"dt{dt_ck}_follows_model"] = f"{sum(c['follows_model'] for c in cells)}/{len(cells)}"
            v[f"dt{dt_ck}_follows_data"] = f"{sum(c['follows_data'] for c in cells)}/{len(cells)}"
        v["P1_pass"] = bool(all(sum(c["follows_model"] for c in off if c["ckpt_dt"] == d) >= 2
                                for d in DTS))
        # A three-way outcome, not two. "Not following the model" is NOT the same as "following the
        # data": the sweep can simply degenerate off-diagonal, which is what happened. An earlier
        # version of this line hard-coded the else-branch to "follows the data" and printed that
        # while follows_data was 0/3 -- a verdict contradicted by the rows directly above it.
        n_data = sum(c["follows_data"] for c in off)
        at_edge = sum(abs(c["argmin_r"]) >= 2.0 for c in off)
        v["off_diagonal_following_data"] = f"{n_data}/{len(off)}"
        v["off_diagonal_pinned_at_grid_edge"] = f"{at_edge}/{len(off)}"
        v["outcome"] = ("model" if v["P1_pass"] else
                        "data" if n_data >= len(off) - 1 else "neither -- degenerate")
        out["verdict"] = v
        print("\n  " + json.dumps(v))
        print({"model": "  F6's model arm is sound",
               "data": "  F6's model arm has the same confound as F7",
               "neither -- degenerate": ("  INCONCLUSIVE: off-diagonal sweeps are degenerate "
                                        "(pinned at grid edges), matching neither hypothesis. "
                                        "Cross-timestep evaluation is out-of-distribution, so this "
                                        "control cannot adjudicate F6.")}[v["outcome"]])
    attach(out, op, inputs=sorted({f'runs/f6_dt{d}_s{s}_step6500.pt' for d in DTS for s in SEEDS}
                              | {f'runs/pend_dt{d}.npz' for d in DTS}))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
