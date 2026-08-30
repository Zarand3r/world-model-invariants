#!/usr/bin/env python3
"""F7 Gate 0: at a FIXED timestep, do different integrators want different `c*`?

Preregistered in docs/F7_PREREG.md. Pure ground truth -- no world model, no training.
The paper's title says the model learns its simulator's *integrator*, but F6 varied only the
*timestep* at one fixed scheme. This gate asks whether the `c*` measurement can distinguish schemes
at all. If it cannot, the title is not earned and the paper narrows.

Usage:  uv run python scripts/run_f7_gate0.py --out runs/f7_gate0.json
"""
from __future__ import annotations

import argparse, json, pathlib
import numpy as np

from latent_noether.provenance import attach, inputs_from_args

G, M, L = 10.0, 1.0, 1.0
I = M * L ** 2 / 3.0
MGL2 = M * G * L / 2.0
THD_CLIP = 8.0
# Fine relative grid: Verlet's optimum is expected near zero, which F6's coarse R_GRID
# would only bracket, not locate.
R_GRID = np.round(np.arange(-1.5, 3.0001, 0.05), 4)


def c_star(dt):
    """Semi-implicit Euler's own prediction: (dt/2) * m g (l/2)."""
    return 0.5 * dt * MGL2


def accel(th):
    return (3 * G / (2 * L)) * np.sin(th)


def energy(th, thd):
    return 0.5 * I * thd ** 2 + MGL2 * np.cos(th)


def rollout(scheme, n_traj, n_steps, dt, seed):
    """Same initial conditions for every scheme, so the comparison is like-for-like."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(-1.8, 1.8, n_traj)
    thd = rng.uniform(-2.2, 2.2, n_traj)
    TH = np.zeros((n_traj, n_steps)); THD = np.zeros((n_traj, n_steps))
    for t in range(n_steps):
        TH[:, t], THD[:, t] = th, thd
        if scheme == "SI":        # velocity from the OLD angle, position from the NEW velocity
            thd = np.clip(thd + accel(th) * dt, -THD_CLIP, THD_CLIP)
            th = th + thd * dt
        elif scheme == "EE":      # both from the OLD state
            th_new = th + thd * dt
            thd = np.clip(thd + accel(th) * dt, -THD_CLIP, THD_CLIP)
            th = th_new
        elif scheme == "VV":      # velocity Verlet
            a = accel(th)
            th_new = th + thd * dt + 0.5 * a * dt ** 2
            thd = np.clip(thd + 0.5 * (a + accel(th_new)) * dt, -THD_CLIP, THD_CLIP)
            th = th_new
        else:
            raise ValueError(scheme)
    ok = ~(np.abs(THD) >= THD_CLIP - 1e-9).any(1)
    return TH[ok], THD[ok], int(ok.sum())


def invariance_ratio(q):
    q = np.asarray(q, float)
    return float(np.mean(np.var(q, axis=-1)) / max(np.var(q), 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dts", nargs="+", type=float, default=[0.02, 0.035, 0.05, 0.08])
    p.add_argument("--n-traj", type=int, default=400)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=pathlib.Path, required=True)
    a = p.parse_args()

    res = {}
    for dt in a.dts:
        cs = c_star(dt)
        per = {}
        for scheme in ("SI", "VV", "EE"):
            TH, THD, kept = rollout(scheme, a.n_traj, a.n_steps, dt, a.seed)
            if kept == 0:
                per[scheme] = {"kept": 0, "note": "every trajectory clipped"}
                print(f"  dt={dt:<6} {scheme}: all {a.n_traj} trajectories clipped", flush=True)
                continue
            grid = [{"r": float(r),
                     "ratio": invariance_ratio(energy(TH, THD) + r * cs * THD * np.sin(TH))}
                    for r in R_GRID]
            best = min(grid, key=lambda x: x["ratio"])
            per[scheme] = {"kept": kept, "c_star_SI": cs, "argmin_r": best["r"],
                           "argmin_c": best["r"] * cs, "best_ratio": best["ratio"],
                           "ratio_at_r0": next(g for g in grid if g["r"] == 0.0)["ratio"],
                           "grid": grid}
            print(f"  dt={dt:<6} {scheme}: argmin r={best['r']:+.2f} "
                  f"(c={best['r']*cs:+.4f})  best ratio={best['ratio']:.3e}  kept {kept}/{a.n_traj}",
                  flush=True)
        res[str(dt)] = per

    # --- registered predictions -------------------------------------------------
    g0 = {str(dt): abs(res[str(dt)]["SI"]["argmin_r"] - 1.0) <= 0.05 for dt in a.dts}
    g1 = {}
    for dt in a.dts:
        vv, si = res[str(dt)].get("VV"), res[str(dt)]["SI"]
        g1[str(dt)] = bool(vv and vv.get("kept") and
                           abs(vv["argmin_c"]) <= 0.25 * abs(si["argmin_c"]))
    ee5, si5 = res["0.05"].get("EE"), res["0.05"]["SI"]
    # "Diverged, so the ratio is undefined" is NOT the same evidence as "the ratio exceeds 5".
    # An earlier version set this to inf on divergence, which recorded G2_pass=true off a
    # measurement that never happened. Report the two outcomes separately.
    g2_evaluable = bool(ee5 and ee5.get("kept"))
    g2_ratio = (ee5["best_ratio"] / si5["best_ratio"]) if g2_evaluable else None
    verdict = {
        "G0_SI_argmin_at_r1": g0, "G0_pass": all(g0.values()),
        "G1_VV_separated": g1, "G1_pass": all(g1.values()),
        "G2_EE_over_SI_at_dt0.05": g2_ratio,
        "G2_evaluable_at_dt0.05": g2_evaluable,
        "G2_pass": bool(g2_evaluable and g2_ratio >= 5.0),
        "G2_note": (None if g2_evaluable else
                    "explicit Euler diverged at dt=0.05 (every trajectory clipped), so the "
                    "registered 5x comparison could not be evaluated at this timestep"),
    }
    # The finite comparison exists only where explicit Euler survives.
    for _dt, _per in res.items():
        if _per.get("EE", {}).get("kept"):
            verdict["G2_finite_ratio_at_smallest_surviving_dt"] = {
                "dt": _dt, "ratio": _per["EE"]["best_ratio"] / _per["SI"]["best_ratio"]}
            break
    verdict["gate_pass"] = bool(verdict["G0_pass"] and verdict["G1_pass"] and verdict["G2_pass"])
    if not g2_evaluable:
        print("  G2 NOT EVALUABLE at dt=0.05: " + verdict["G2_note"])
    print("\n  " + json.dumps({k: v for k, v in verdict.items() if k.endswith("pass")}))
    if not verdict["G0_pass"]:
        print("  G0 FAILED -- positive control broken, do not read G1/G2")

    out = {"params": vars(a) | {"out": str(a.out)}, "results": res, "verdict": verdict}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    attach(out, inputs_from_args(a))
    a.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
