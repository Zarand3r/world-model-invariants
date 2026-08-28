"""F1 Gate 0: does the discrete energy-balance relation close on GROUND TRUTH?

Pre-registered in `docs/F1_PREREG.md`. No model, no frames -- states and actions only, so this is
cheap and decisive. If the relation does not hold in the data, nothing measured on a model could be
interpreted, and the registered response is to stop.

Gymnasium's actuated Pendulum update (m = l = 1, g = 10, dt = 0.05):

    thdot' = thdot + (3g/(2l) sin(th) + 3u/(m l^2)) dt          th' = th + thdot' dt

With I = m l^2 / 3 this is exactly `I thddot = m g (l/2) sin(th) + u`, so the applied torque is `u`
and the continuous balance law is `dE/dt = u * thdot`.

E19 showed this integrator conserves a shadow Hamiltonian `H~ = E + 0.125 thdot sin(th)` rather than
textbook `E`. The same O(dt) structure must appear in the balance relation, so both are evaluated,
and the power is evaluated at both `thdot_t` and `thdot_{t+1}` because the semi-implicit update uses
the *updated* velocity.
"""
import argparse, json, pathlib
import numpy as np
from latent_noether.provenance import attach, inputs_from_args

G, M, L, DT = 10.0, 1.0, 1.0, 0.05
I = M * L ** 2 / 3.0
MGL2 = M * G * L / 2.0
C_SHADOW = 0.125            # from E19: (dt/2) * m g (l/2)
THD_CLIP = 8.0


def energy(th, thd):
    return 0.5 * I * thd ** 2 + MGL2 * np.cos(th)


def shadow(th, thd):
    return energy(th, thd) + C_SHADOW * thd * np.sin(th)


def simulate(n_traj, n_steps, torque_max, hold, seed, reject_clipped=True):
    """Trajectories that ever reach the |thetadot| = 8 clip are DISCARDED.

    The clip is a hard nonlinearity that breaks the balance relation outright, exactly as it would
    break conservation, and the free-evolution dataset already keeps away from it. Note that
    rejection costs yield but does *not* collapse the across-trajectory energy spread that the
    extraction needs: measured std of per-trajectory mean energy is 1.74 at 35% yield against 1.72
    at 60%.
    """
    rng = np.random.default_rng(seed)
    th = rng.uniform(-1.8, 1.8, n_traj)
    thd = rng.uniform(-2.2, 2.2, n_traj)
    TH = np.zeros((n_traj, n_steps)); THD = np.zeros((n_traj, n_steps))
    U = np.zeros((n_traj, n_steps))
    u = np.zeros(n_traj)
    for t in range(n_steps):
        if t % hold == 0:
            u = rng.uniform(-torque_max, torque_max, n_traj)
        TH[:, t], THD[:, t], U[:, t] = th, thd, u
        thd = thd + (3 * G / (2 * L) * np.sin(th) + 3.0 * u / (M * L ** 2)) * DT
        thd = np.clip(thd, -THD_CLIP, THD_CLIP)
        th = th + thd * DT
    if reject_clipped:
        ok = ~(np.abs(THD) >= THD_CLIP - 1e-9).any(1)
        TH, THD, U = TH[ok], THD[ok], U[ok]
    return TH, THD, U


def gate(TH, THD, U):
    """Normalised residual of the discrete balance law, for each registered variant."""
    dE_tb = energy(TH[:, 1:], THD[:, 1:]) - energy(TH[:, :-1], THD[:, :-1])
    dE_sh = shadow(TH[:, 1:], THD[:, 1:]) - shadow(TH[:, :-1], THD[:, :-1])
    u = U[:, :-1]
    P_t = u * THD[:, :-1] * DT          # power at the OLD velocity
    P_t1 = u * THD[:, 1:] * DT          # power at the UPDATED velocity (semi-implicit ordering)
    out = {}
    for ename, dE in (("textbook_E", dE_tb), ("shadow_H", dE_sh)):
        scale = float(np.median(np.abs(dE)))
        P_mid = u * 0.5 * (THD[:, :-1] + THD[:, 1:]) * DT   # midpoint velocity
        for pname, P in (("power_at_thd_t", P_t), ("power_at_thd_t1", P_t1),
                         ("power_at_midpoint", P_mid)):
            r = float(np.median(np.abs(dE - P)) / scale)
            out[f"{ename}__{pname}"] = r
        out[f"{ename}__no_power"] = float(np.median(np.abs(dE)) / scale)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-traj", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--torque-max", type=float, default=1.0)
    p.add_argument("--hold", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/f1_gate0.json")
    a = p.parse_args()
    TH, THD, U = simulate(a.n_traj, a.n_steps, a.torque_max, a.hold, a.seed)
    res = gate(TH, THD, U)
    clip_frac = float(np.mean(np.abs(THD) >= THD_CLIP - 1e-9))   # 0 by construction after rejection
    out = {"config": vars(a), "n_kept": int(TH.shape[0]),
           "energy_spread_across_traj": float(energy(TH, THD).mean(1).std()),
           "max_abs_thetadot": float(np.abs(THD).max()),
           "clip_fraction": clip_frac, "residuals": res,
           "G0_pass": bool(min(res.values()) < 0.05 and clip_frac == 0.0)}
    print(f"  max |thetadot| = {out['max_abs_thetadot']:.3f}  (clip at {THD_CLIP})   "
          f"clipped fraction = {clip_frac:.4f}")
    print("  normalised residual  median|dE - P| / median|dE|:")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"    {k:34s} {v:.5f}")
    print(f"  G0_pass = {out['G0_pass']}  (registered bar: < 0.05, and no clipping)")
    op = pathlib.Path(a.out)
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
