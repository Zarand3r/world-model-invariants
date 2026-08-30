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

# One measurement path for every arm. The whole experiment is a contrast between arms, so they must
# not diverge in how they are analysed -- hence a table rather than a second script.
ARMS = {
    "verlet":   {"data": "runs/pend_verlet_dt0.05.npz",   "tag": "f7_verlet",
                 "target_r": 0.0,  "out": "runs/f7_models.json"},
    "reversed": {"data": "runs/pend_reversed_dt0.05.npz", "tag": "f7b_reversed",
                 "target_r": -1.0, "out": "runs/f7b_models.json"},
}


def training_acceptance(tag, seed):
    """P3: parse the same acceptance checks F6's models were held to, from the training log."""
    log = pathlib.Path(f"runs/logs/{tag}_s{seed}.log")
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
    p.add_argument("--arm", choices=sorted(ARMS), default="verlet")
    p.add_argument("--ckpts", nargs="*", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    arm = ARMS[a.arm]
    if a.ckpts is None:
        a.ckpts = [f"runs/{arm['tag']}_s{s}_step6500.pt" for s in (3, 4, 5)]
    op = pathlib.Path(a.out or arm["out"])
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}

    for ck in a.ckpts:
        if ck in done:
            continue
        if not pathlib.Path(ck).exists():
            print(f"  PENDING {ck}", flush=True)
            continue
        r = run(ck, arm["data"], DT)
        r["seed"] = int(re.search(r"_s(\d+)_", ck).group(1))
        r["acceptance"] = training_acceptance(arm["tag"], r["seed"])
        print(f"  {pathlib.Path(ck).name:32s} argmin r={r['argmin_r']:+.2f}  "
              f"c_rec={r['c_recovered']:.4f}  "
              f"rho_obs r=0 {r['rho_obs_at_r0']:.5f} vs r=1 {r['rho_obs_at_r1']:.5f}", flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")

    verlet = out["models"]
    tgt = arm["target_r"]
    if len(verlet) < 3:
        print(f"\n  {len(verlet)}/3 {a.arm} models analysed; predictions evaluated at n=3.")
    else:
        # F6's semi-implicit models at the SAME timestep are the contrast arm.
        f6 = [m for m in json.loads(pathlib.Path("runs/f6_models.json").read_text())["models"]
              if abs(m["dt"] - DT) < 1e-9]
        v_arg = [m["argmin_r"] for m in verlet]
        s_arg = [m["argmin_r"] for m in f6]
        p1 = sum(abs(x - tgt) <= 0.5 for x in v_arg)
        # The prereg named THREE acceptance criteria; an earlier version of this script checked
        # only two, which would have let a registered criterion pass silently. Report each.
        F6_STD = (0.0694, 0.0699)   # observed across all 12 F6 models
        p3_decode = all(m["acceptance"].get("decode_ratio") is not None
                        and m["acceptance"]["decode_ratio"] < 0.05 for m in verlet)
        p3_finite = all(m["acceptance"]["rollout_finite"] for m in verlet)
        p3_std = [F6_STD[0] <= m["acceptance"]["pixel_std"] <= F6_STD[1] for m in verlet]
        p3 = p3_decode and p3_finite and all(p3_std)
        summary = {
            "arm": a.arm, "target_r": tgt,
            "arm_argmin_r": v_arg, "semi_implicit_argmin_r": s_arg,
            "arm_median_r": st.median(v_arg), "semi_implicit_median_r": st.median(s_arg),
            "P1_seeds_near_target": p1, "P1_pass": bool(p1 >= 2),
            "P2_median_gap": st.median(s_arg) - st.median(v_arg),
            "P2_pass": bool(st.median(s_arg) - st.median(v_arg) >= 0.5),
            "P3_models_acceptable": bool(p3),
            "P3_decode_ratio_ok": bool(p3_decode), "P3_rollout_finite_ok": bool(p3_finite),
            "P3_pixel_std_in_F6_range": p3_std,
            "P3_pixel_std": [m["acceptance"]["pixel_std"] for m in verlet],
            "F6_pixel_std_range": list(F6_STD),
        }
        out["summary"] = summary

        if a.arm == "reversed":
            vp = pathlib.Path(ARMS["verlet"]["out"])
            v_med = (st.median(m["argmin_r"] for m in json.loads(vp.read_text())["models"])
                     if vp.exists() else None)
            ordered = (v_med is not None
                       and st.median(v_arg) < v_med < st.median(s_arg)
                       and st.median(v_arg) <= -0.5 and st.median(s_arg) >= 0.5)
            summary["P2_verlet_median_r"] = v_med
            summary["P2_ordering_pass"] = bool(ordered)
            summary["P2_pass"] = bool(ordered)
            print(f"  P2 ordering reversed {st.median(v_arg):+.2f} < verlet {v_med:+.2f} "
                  f"< semi-implicit {st.median(s_arg):+.2f}  -> {ordered}")
        print(f"\n  {a.arm:14s} argmin r {v_arg}   median {summary['arm_median_r']:+.2f}"
              f"   (predicted r={tgt:+.0f})")
        print(f"  semi-implicit argmin r {s_arg}   median {summary['semi_implicit_median_r']:+.2f}")
        print(f"  P1 {summary['P1_pass']} ({p1}/3 within 0.5 of r={tgt:+.0f})   "
              f"P2 {summary['P2_pass']} (gap {summary['P2_median_gap']:+.2f})   "
              f"P3 {summary['P3_models_acceptable']}")
        if not p3:
            print(f"  P3 partial: decode {p3_decode}, finite {p3_finite}, "
                  f"pixel std in F6 range {sum(p3_std)}/3 "
                  f"({summary['P3_pixel_std']} vs {list(F6_STD)})")
            if not (p3_decode and p3_finite):
                print("  P3 FAILED on a substantive criterion -- do not read P1/P2 as a scheme effect.")

    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
