"""Build a PIXEL dataset from a real third-party simulator (gymnasium Pendulum-v1).

Everything in this project so far has used our own RK4 integrator on 2-D coordinates, with a
64-unit GRU reading positions directly. Three separate objections follow from that, and this
dataset removes all three:

  1. **the simulator is ours** -> gymnasium Pendulum-v1, a standard third-party environment
  2. **observations are coordinates** -> 64x64 RGB frames; the model must LEARN the state
     representation rather than integrate a state it was handed
  3. **the architecture is a toy GRU** -> a conv-encoder RSSM is trained on this (separate script)

**What is conserved here, and it is not the textbook energy.** Gymnasium integrates with
semi-implicit (symplectic) Euler:

    thdot' = thdot + (3g/2l) sin(th) dt        th' = th + thdot' dt

A symplectic integrator exactly conserves a *shadow* Hamiltonian H~ = H + O(dt), not H itself. We
measure ~8% oscillation in the textbook E = ½(ml²/3)θ̇² + mg(l/2)cos θ across every amplitude
regime, with no secular drift and no dependence on the velocity clip. That is a real property of a
real simulator, and it sharpens the question: **the model learns gymnasium's map, so the quantity
it should conserve is the shadow Hamiltonian.** Our recovery method never sees either.

Free evolution only (action = 0), so the system is conservative. Initial conditions are kept away
from the |thdot| = 8 clip, which would break conservation outright.

Frames are rendered at 500x500 by gymnasium and downsampled to 64x64 by block-averaging (no
external image dependency). Stored uint8; ~0.4 GB for the default 256 trajectories.
"""
import argparse
import pathlib

import gymnasium as gym
import numpy as np

TH_MAX, THD_MAX = 1.8, 2.2          # away from the |thdot|=8 clip
RES = 64


def downsample(frame, res=RES):
    """500x500x3 -> 64x64x3 by block mean. Crop to a multiple of the block size first."""
    h = frame.shape[0] // res * res
    f = frame[:h, :h].astype(np.float32)
    b = h // res
    return f.reshape(res, b, res, b, 3).mean(axis=(1, 3)).astype(np.uint8)


def energy(th, thd, g=10.0, m=1.0, l=1.0):
    """Textbook rod energy. The simulator conserves a shadow Hamiltonian close to this."""
    return 0.5 * (m * l ** 2 / 3) * thd ** 2 + m * g * (l / 2) * np.cos(th)


def damped_step(state, zeta: float, dt: float = 0.05, g: float = 10.0, l: float = 1.0):
    """Gymnasium's semi-implicit Euler with a linear damping term added.

    Gymnasium's Pendulum has no damping, so the update is done here and written straight into
    `u.state`; the renderer only reads that, so frames stay pixel-identical in construction to the
    conservative dataset. Damping enters exactly as in the GRU dissipative experiments:

        thdot' = thdot + (3g/2l sin(th) - 2 zeta omega0 thdot) dt,   omega0 = sqrt(3g/2l)
        th'    = th + thdot' dt

    Only the dynamics differ from the conservative generator. Everything else -- resolution,
    initial-condition ranges, horizon, rejection of speed-clipped trajectories -- is unchanged, so
    the comparison isolates conservativeness.
    """
    th, thd = float(state[0]), float(state[1])
    w0 = np.sqrt(3 * g / (2 * l))
    thd = thd + ((3 * g / (2 * l)) * np.sin(th) - 2 * zeta * w0 * thd) * dt
    return np.array([th + thd * dt, thd])


def main(n_traj: int, n_steps: int, seed: int, out: str, zeta: float = 0.0,
         th_lo: float = 0.0, th_hi: float = TH_MAX,
         thd_lo: float = 0.0, thd_hi: float = THD_MAX):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    u = env.unwrapped
    rng = np.random.default_rng(seed)
    frames = np.zeros((n_traj, n_steps, RES, RES, 3), dtype=np.uint8)
    states = np.zeros((n_traj, n_steps, 2), dtype=np.float64)
    zero = np.array([0.0], dtype=np.float32)

    kept = 0
    attempts = 0
    while kept < n_traj and attempts < n_traj * 4:
        attempts += 1
        env.reset(seed=int(rng.integers(0, 2 ** 31)))
        # E14 (docs/E14_PREREG.md) needs initial conditions OUTSIDE the training band. The
        # defaults reproduce the training distribution exactly; the --th-lo/--thd-lo arguments let
        # an out-of-distribution energy band be generated with everything else identical.
        if th_lo == 0.0 and thd_lo == 0.0:
            # EXACT original sampling. `uniform(0, hi) * choice(+-1)` is distributionally identical
            # to `uniform(-hi, hi)` but consumes the RNG differently, so regenerating the training
            # dataset through the new branch would produce DIFFERENT trajectories and break the
            # data_sha256 provenance chain the checkpoints record (M29). Defaults stay bit-exact.
            th, thd = rng.uniform(-th_hi, th_hi), rng.uniform(-thd_hi, thd_hi)
        else:
            th = rng.uniform(th_lo, th_hi) * rng.choice([-1.0, 1.0])
            thd = rng.uniform(thd_lo, thd_hi) * rng.choice([-1.0, 1.0])
        u.state = np.array([th, thd])
        st, fr = [], []
        clipped = False
        for _ in range(n_steps):
            st.append(u.state.copy())
            fr.append(downsample(env.render()))
            if zeta > 0:
                u.state = damped_step(u.state, zeta)
            else:
                env.step(zero)
            if abs(u.state[1]) >= u.max_speed - 1e-6:
                clipped = True
                break
        if clipped or len(st) < n_steps:
            continue                                  # clipped trajectories are not conservative
        states[kept] = np.stack(st)
        frames[kept] = np.stack(fr)
        kept += 1
        if kept % 32 == 0:
            print(f"  {kept}/{n_traj} trajectories", flush=True)

    th, thd = states[..., 0], states[..., 1]
    E = energy(th, thd)
    rel = np.std(E, axis=1) / np.abs(np.mean(E, axis=1)).clip(1e-12)
    print(f"\n  kept {kept}/{n_traj} (rejected {attempts - kept} for hitting the speed clip)")
    if zeta > 0:
        decay = np.median(E[:, -1] - E[:, 0])
        print(f"  DAMPED (zeta={zeta}): median energy change over the horizon {decay:+.3f}")
    else:
        print(f"  textbook-E relative oscillation: median {np.median(rel):.3f}  "
              f"(shadow-Hamiltonian artifact of symplectic Euler, not secular drift)")
    print(f"  frames {frames.shape} {frames.dtype}  ~{frames.nbytes/1e9:.2f} GB")

    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, frames=frames[:kept], states=states[:kept], energy=E[:kept])
    print(f"  wrote {p}")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--th-lo", type=float, default=0.0)
    a.add_argument("--th-hi", type=float, default=TH_MAX)
    a.add_argument("--thd-lo", type=float, default=0.0)
    a.add_argument("--thd-hi", type=float, default=THD_MAX)
    a.add_argument("--n-traj", type=int, default=256)
    a.add_argument("--n-steps", type=int, default=120)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="runs/pendulum_pixels.npz")
    a.add_argument("--zeta", type=float, default=0.0,
                   help="linear damping; 0 is the conservative dataset, 0.15 the GRU control value")
    a = a.parse_args()
    print("Building a PIXEL dataset from gymnasium Pendulum-v1 (real third-party simulator).")
    main(a.n_traj, a.n_steps, a.seed, a.out, zeta=a.zeta,
         th_lo=a.th_lo, th_hi=a.th_hi, thd_lo=a.thd_lo, thd_hi=a.thd_hi)
