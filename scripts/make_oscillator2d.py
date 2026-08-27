"""2-DoF anharmonic oscillator: chaos gate, simulation, and pixel dataset.

Pre-registered in `docs/E17_PREREG.md`. Two arms differing only in whether the potential is central:

    non-central   V = 1/2(w1^2 q1^2 + w2^2 q2^2) + 1/4 a(q1^4 + q2^4) + 1/2 b q1^2 q2^2
    central       V = 1/2 w^2 r^2 + 1/4 a r^4

The non-central arm conserves energy alone; the central arm also conserves angular momentum. That
makes them a matched pair in the same sense as the conservative/damped pendulum pair.

**Semi-implicit (symplectic) Euler**, matching gymnasium's pendulum convention exactly, so
`p_k = (q_k - q_{k-1})/dt` holds and the geometric readout's backward-difference convention
transfers unchanged. A different integrator would silently break a readout that took an iteration
to get right.
"""
import argparse
import numpy as np

DT = 0.05


def accel(q, central, w1, w2, a, b):
    q1, q2 = q[..., 0], q[..., 1]
    if central:
        r2 = q1 ** 2 + q2 ** 2
        f = -(w1 ** 2) * q - a * r2[..., None] * q
    else:
        f1 = -(w1 ** 2) * q1 - a * q1 ** 3 - b * q1 * q2 ** 2
        f2 = -(w2 ** 2) * q2 - a * q2 ** 3 - b * q2 * q1 ** 2
        f = np.stack([f1, f2], -1)
    return f


def step(q, p, central, w1, w2, a, b, dt=DT):
    p = p + accel(q, central, w1, w2, a, b) * dt      # semi-implicit: momentum first
    q = q + p * dt
    return q, p


def simulate(q0, p0, n_steps, central, w1, w2, a, b, dt=DT):
    q, p = q0.copy(), p0.copy()
    Q, P = [q.copy()], [p.copy()]
    for _ in range(n_steps - 1):
        q, p = step(q, p, central, w1, w2, a, b, dt)
        Q.append(q.copy()); P.append(p.copy())
    return np.stack(Q, 1), np.stack(P, 1)


def energy(q, p, central, w1, w2, a, b):
    q1, q2 = q[..., 0], q[..., 1]
    kin = 0.5 * (p ** 2).sum(-1)
    if central:
        r2 = q1 ** 2 + q2 ** 2
        pot = 0.5 * w1 ** 2 * r2 + 0.25 * a * r2 ** 2
    else:
        pot = (0.5 * (w1 ** 2 * q1 ** 2 + w2 ** 2 * q2 ** 2)
               + 0.25 * a * (q1 ** 4 + q2 ** 4) + 0.5 * b * q1 ** 2 * q2 ** 2)
    return kin + pot


def angular_momentum(q, p):
    return q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0]


def lyapunov(n_traj, n_steps, central, w1, w2, a, b, seed=0, d0=1e-8, dt=DT):
    """Maximal Lyapunov exponent by two-trajectory renormalisation. The E17 chaos gate."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, (n_traj, 2)); p = rng.uniform(-1.0, 1.0, (n_traj, 2))
    d = rng.normal(size=(n_traj, 2)); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    q2_, p2_ = q + d0 * d, p.copy()
    total = np.zeros(n_traj)
    for _ in range(n_steps):
        q, p = step(q, p, central, w1, w2, a, b, dt)
        q2_, p2_ = step(q2_, p2_, central, w1, w2, a, b, dt)
        sep = np.linalg.norm(np.concatenate([q2_ - q, p2_ - p], -1), axis=-1)
        sep = np.maximum(sep, 1e-30)
        total += np.log(sep / d0)
        scale = (d0 / sep)[:, None]
        q2_ = q + (q2_ - q) * scale
        p2_ = p + (p2_ - p) * scale
    return total / (n_steps * dt)


if __name__ == "__main__":
    p_ = argparse.ArgumentParser()
    p_.add_argument("--gate", action="store_true")
    p_.add_argument("--n-steps", type=int, default=200)
    a_ = p_.parse_args()
    if a_.gate:
        H = a_.n_steps * DT
        print(f"E17 CHAOS GATE   horizon {a_.n_steps} steps = {H:.1f} time units")
        print(f"PASS if lambda_max * T < 1 on >= 95% of initial conditions\n")
        print(f"{'arm':>12s} {'w1':>5s} {'w2':>5s} {'a':>6s} {'b':>6s} {'med lam':>9s} {'p95 lam':>9s} {'lam*T p95':>10s} {'frac<1':>8s} {'gate':>6s}")
        for central in (False, True):
            for (w1, w2, aa, bb) in [(1.0, 1.3, 0.5, 0.5), (1.0, 1.3, 0.2, 0.2),
                                     (1.0, 1.3, 0.1, 0.1), (1.0, 1.3, 0.05, 0.05)]:
                lam = lyapunov(256, a_.n_steps, central, w1, w2, aa, bb)
                frac = float((lam * H < 1).mean())
                print(f"{'central' if central else 'non-central':>12s} {w1:5.1f} {w2:5.1f} {aa:6.2f} {bb:6.2f} "
                      f"{np.median(lam):9.4f} {np.percentile(lam,95):9.4f} {np.percentile(lam,95)*H:10.3f} "
                      f"{frac:8.3f} {'PASS' if frac>=0.95 else 'FAIL':>6s}")
