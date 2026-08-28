"""F1 data: the pendulum under nonzero torque, rendered to pixels, with the actions stored.

Pre-registered in `docs/F1_PREREG.md`. Kept SEPARATE from `make_pendulum_pixels.py` deliberately:
that script documents that its RNG consumption is bit-exact so the free-evolution `data_sha256`
provenance chain the checkpoints record stays valid, and editing it to add actions would break that.

Actions are piecewise-constant random torques held for `--hold` steps. Gate 0 established that the
balance relation is only measurable once actuation dominates the integrator's own O(dt^2) energy
error, which fixes `--torque-max` at 1.0 rather than leaving it to taste; see the execution log entry
of 2026-08-28.

Trajectories reaching the |thetadot| = 8 speed clip are discarded, matching the free generator --
the clip breaks the balance relation exactly as it breaks conservation.
"""
import argparse
import pathlib

import gymnasium as gym
import numpy as np

from latent_noether.provenance import stamp
from scripts.make_pendulum_pixels import RES, TH_MAX, THD_MAX, downsample, energy

DT = 0.05


def main(n_traj, n_steps, seed, out, torque_max, hold):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    u = env.unwrapped
    rng = np.random.default_rng(seed)
    frames = np.zeros((n_traj, n_steps, RES, RES, 3), dtype=np.uint8)
    states = np.zeros((n_traj, n_steps, 2), dtype=np.float64)
    actions = np.zeros((n_traj, n_steps, 1), dtype=np.float32)

    kept, attempts = 0, 0
    while kept < n_traj and attempts < n_traj * 8:
        attempts += 1
        env.reset(seed=int(rng.integers(0, 2 ** 31)))
        u.state = np.array([rng.uniform(-TH_MAX, TH_MAX), rng.uniform(-THD_MAX, THD_MAX)])
        st, fr, ac = [], [], []
        tau = np.zeros(1, dtype=np.float32)
        clipped = False
        for t in range(n_steps):
            if t % hold == 0:
                tau = np.array([rng.uniform(-torque_max, torque_max)], dtype=np.float32)
            st.append(u.state.copy()); fr.append(downsample(env.render())); ac.append(tau.copy())
            env.step(tau)
            if abs(u.state[1]) >= u.max_speed - 1e-6:
                clipped = True
                break
        if clipped or len(st) < n_steps:
            continue
        states[kept], frames[kept], actions[kept] = np.stack(st), np.stack(fr), np.stack(ac)
        kept += 1
        if kept % 32 == 0:
            print(f"  {kept}/{n_traj} trajectories (attempts {attempts})", flush=True)

    if kept < n_traj:
        raise SystemExit(f"only kept {kept}/{n_traj} after {attempts} attempts -- lower --torque-max")
    th, thd = states[..., 0], states[..., 1]
    E = energy(th, thd)
    tau = actions[..., 0]
    # The balance quantity the model will have to account for, recorded for evaluation only.
    power_mid = tau[:, :-1] * 0.5 * (thd[:, :-1] + thd[:, 1:]) * DT
    dE = E[:, 1:] - E[:, :-1]
    print(f"\n  kept {kept}/{n_traj} (rejected {attempts - kept} for the speed clip)")
    print(f"  |tau| <= {torque_max}, held {hold} steps;  max |thetadot| {np.abs(thd).max():.3f}")
    print(f"  across-trajectory energy spread (std of per-traj mean E): {E.mean(1).std():.3f}")
    print(f"  textbook balance residual median|dE - P|/median|dE| = "
          f"{np.median(np.abs(dE - power_mid)) / np.median(np.abs(dE)):.4f}  (expected ~0.9: see G0)")
    print(f"  frames {frames.shape} {frames.dtype}  ~{frames.nbytes / 1e9:.2f} GB")
    p = pathlib.Path(out)
    np.savez_compressed(p, frames=frames, states=states, actions=actions, energy=E)
    (p.parent / (p.stem + ".prov.json")).write_text(
        __import__("json").dumps({"runs": [stamp(torque_max=torque_max, hold=hold, seed=seed,
                                                 n_traj=n_traj, n_steps=n_steps)]}, indent=1) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--n-traj", type=int, default=256)
    a.add_argument("--n-steps", type=int, default=120)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--torque-max", type=float, default=1.0)
    a.add_argument("--hold", type=int, default=5)
    a.add_argument("--out", default="runs/pendulum_actuated.npz")
    g = a.parse_args()
    main(g.n_traj, g.n_steps, g.seed, g.out, g.torque_max, g.hold)
