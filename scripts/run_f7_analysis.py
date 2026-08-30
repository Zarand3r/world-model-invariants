#!/usr/bin/env python3
"""F7: does a model trained on velocity Verlet recover Verlet's coefficient, or Euler's?

Preregistered in docs/F7_PREREG.md. The timestep is held at 0.05 for both arms, so a difference
cannot be explained by "the model learned dt".

Deliberately imports `run` and `summarize` from run_f6_analysis rather than reimplementing them:
the whole experiment is a contrast against F6's dt=0.05 models, and it is only readable if both arms
are measured by the same code.

Usage:  uv run python scripts/run_f7_analysis.py --out runs/f7_models.json
"""
from __future__ import annotations

import argparse, json, pathlib, re, statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_analysis import run

DT = 0.05
DATA = "runs/pend_verlet_dt0.05.npz"


def training_acceptance(seed):
    """P3: parse the same acceptance checks F6's models were held to, from the training log."""
    log = pathlib.Path(f"runs/logs/f7_verlet_s{seed}.log")
    if not log.exists():
        return {"found": False}
    t = log.read_text()
    mse = re.search(r"1-step decode MSE [\d.]+ vs predict-the-mean [\d.]+\s+ratio ([\d.]+)", t)
    roll = re.search(r"rollout finite: (\w+)\s+rollout pixel std ([\d.]+)", t)
    return {"found": True,
            "decode_ratio": float(mse.group(1)) if mse else None,
            "rollout_finite": (roll.group(1) == "True") if roll else None,
            "pixel_std": float(roll.group(2)) if roll else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*",
                   default=[f"runs/f7_verlet_s{s}_step6500.pt" for s in (3, 4, 5)])
    p.add_argument("--out", default="runs/f7_models.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}

    for ck in a.ckpts:
        if ck in done:
            continue
        if not pathlib.Path(ck).exists():
            print(f"  PENDING {ck}", flush=True)
            continue
        r = run(ck, DATA, DT)
        r["seed"] = int(re.search(r"_s(\d+)_", ck).group(1))
        r["acceptance"] = training_acceptance(r["seed"])
        print(f"  {pathlib.Path(ck).name:32s} argmin r={r['argmin_r']:+.2f}  "
              f"c_rec={r['c_recovered']:.4f}  "
              f"rho_obs r=0 {r['rho_obs_at_r0']:.5f} vs r=1 {r['rho_obs_at_r1']:.5f}", flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")

    verlet = out["models"]
    if len(verlet) < 3:
        print(f"\n  {len(verlet)}/3 Verlet models analysed; predictions evaluated at n=3.")
    else:
        # F6's semi-implicit models at the SAME timestep are the contrast arm.
        f6 = [m for m in json.loads(pathlib.Path("runs/f6_models.json").read_text())["models"]
              if abs(m["dt"] - DT) < 1e-9]
        v_arg = [m["argmin_r"] for m in verlet]
        s_arg = [m["argmin_r"] for m in f6]
        p1 = sum(abs(x) <= 0.5 for x in v_arg)
        p3 = all(m["acceptance"].get("decode_ratio") is not None
                 and m["acceptance"]["decode_ratio"] < 0.05
                 and m["acceptance"]["rollout_finite"] for m in verlet)
        summary = {
            "verlet_argmin_r": v_arg, "semi_implicit_argmin_r": s_arg,
            "verlet_median_r": st.median(v_arg), "semi_implicit_median_r": st.median(s_arg),
            "P1_seeds_near_zero": p1, "P1_pass": bool(p1 >= 2),
            "P2_median_gap": st.median(s_arg) - st.median(v_arg),
            "P2_pass": bool(st.median(s_arg) - st.median(v_arg) >= 0.5),
            "P3_models_acceptable": bool(p3),
        }
        out["summary"] = summary
        print(f"\n  Verlet argmin r        {v_arg}   median {summary['verlet_median_r']:+.2f}")
        print(f"  semi-implicit argmin r {s_arg}   median {summary['semi_implicit_median_r']:+.2f}")
        print(f"  P1 {summary['P1_pass']} ({p1}/3 within |r|<=0.5)   "
              f"P2 {summary['P2_pass']} (gap {summary['P2_median_gap']:+.2f})   "
              f"P3 {summary['P3_models_acceptable']}")
        if not p3:
            print("  P3 FAILED -- models did not train acceptably; do not read P1/P2 as a scheme effect.")

    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
