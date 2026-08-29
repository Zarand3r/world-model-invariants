"""F6: does a pixel-trained world model encode the SIMULATOR'S TIMESTEP?

Pre-registered in `docs/F6_PREREG.md`. E19 showed the recovered invariant is the integrator's shadow
Hamiltonian with coefficient `c* = (dt/2) m g (l/2)`. That is a property of the discretisation, so it
predicts a parameter-free line: `c*(dt) = 2.5 dt`.

The sweep is RELATIVE, `r = c / (2.5 dt)`, so the prediction is a single value of `r` at every
timestep rather than four separate claims.

`--physics` runs the P1 gate on ground-truth states with no model, and must pass before any model is
analysed.
"""
import argparse, json, pathlib
import numpy as np

from latent_noether.provenance import attach, inputs_from_args

G, M, L = 10.0, 1.0, 1.0
I = M * L ** 2 / 3.0
MGL2 = M * G * L / 2.0
THD_CLIP = 8.0
R_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def c_star(dt):
    """The integrator's own prediction: (dt/2) * m g (l/2)."""
    return 0.5 * dt * MGL2


def energy(th, thd):
    return 0.5 * I * thd ** 2 + MGL2 * np.cos(th)


def simulate(n_traj, n_steps, dt, seed, reject_clipped=True):
    """gymnasium's semi-implicit Euler: velocity from the OLD angle, position from the NEW velocity."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(-1.8, 1.8, n_traj)
    thd = rng.uniform(-2.2, 2.2, n_traj)
    TH = np.zeros((n_traj, n_steps)); THD = np.zeros((n_traj, n_steps))
    for t in range(n_steps):
        TH[:, t], THD[:, t] = th, thd
        thd = np.clip(thd + (3 * G / (2 * L)) * np.sin(th) * dt, -THD_CLIP, THD_CLIP)
        th = th + thd * dt
    if reject_clipped:
        ok = ~(np.abs(THD) >= THD_CLIP - 1e-9).any(1)
        TH, THD = TH[ok], THD[ok]
    return TH, THD


def invariance_ratio(q):
    q = np.asarray(q, float)
    return float(np.mean(np.var(q, axis=-1)) / max(np.var(q), 1e-30))


def physics(dts, n_traj, n_steps, seed):
    out = {}
    for dt in dts:
        TH, THD = simulate(n_traj, n_steps, dt, seed)
        cs = c_star(dt)
        rows = [{"r": float(r),
                 "ratio": invariance_ratio(energy(TH, THD) + r * cs * THD * np.sin(TH))}
                for r in R_GRID]
        best = min(rows, key=lambda x: x["ratio"])
        zero = next(x for x in rows if x["r"] == 0.0)["ratio"]
        out[str(dt)] = {"c_star": cs, "grid": rows, "argmin_r": best["r"],
                        "improvement_over_r0": zero / max(best["ratio"], 1e-30),
                        "n_kept": int(TH.shape[0]),
                        "P1_pass": bool(best["r"] == 1.0 and zero / max(best["ratio"], 1e-30) >= 2.0)}
        print(f"  dt={dt:<6} c*={cs:.4f}  argmin r={best['r']:+.2f}  "
              f"improvement {zero / max(best['ratio'], 1e-30):8.1f}x  kept {TH.shape[0]}/{n_traj}  "
              f"P1 {out[str(dt)]['P1_pass']}", flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dts", nargs="+", type=float, default=[0.02, 0.035, 0.05, 0.08])
    p.add_argument("--n-traj", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--physics", action="store_true")
    p.add_argument("--out", default="runs/f6_physics.json")
    a = p.parse_args()
    if a.physics:
        res = physics(a.dts, a.n_traj, a.n_steps, a.seed)
        allpass = all(v["P1_pass"] for v in res.values())
        print(f"  P1 gate: {'PASS' if allpass else 'FAIL'} on {sum(v['P1_pass'] for v in res.values())}/{len(res)} timesteps")
        # M29: this script wrote its record directly and so carried no provenance stamp, unlike
        # every other producer here. Stamped now.
        out = {"physics": res, "r_grid": list(R_GRID)}
        op = pathlib.Path(a.out)
        attach(out, op, inputs=inputs_from_args(a))
        op.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {a.out}")
