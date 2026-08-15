"""Ground-truth oscillator simulators.

Used only to generate training data and to evaluate extractions. Simulator state is
never visible to the extraction pipeline (see docs/PROPOSAL.md section 7).
"""
from dataclasses import dataclass

import numpy as np

_RK4_SUBSTEPS = 20  # dt/20 substeps keep global RK4 error below 1e-9 for the tested regimes


@dataclass(frozen=True)
class OscillatorParams:
    omega: float = 1.0       # natural frequency (rad / unit time)
    zeta: float = 0.0        # damping ratio; > 0 is the anti-overclaim control
    drive_amp: float = 0.0   # sinusoidal forcing amplitude; > 0 is the driven control
    drive_freq: float = 0.0
    dt: float = 0.05         # observation interval


def _accel(params: OscillatorParams, q, p, t):
    drive = params.drive_amp * np.cos(params.drive_freq * t)
    return -params.omega ** 2 * q - 2.0 * params.zeta * params.omega * p + drive


def simulate(params: OscillatorParams, init, n_steps: int) -> np.ndarray:
    """Integrate (q, p) from init (n, 2). Returns (n, n_steps + 1, 2)."""
    init = np.asarray(init, dtype=np.float64)
    if init.ndim != 2 or init.shape[1] != 2:
        raise ValueError(f"init must have shape (n, 2), got {init.shape}")
    out = np.empty((init.shape[0], n_steps + 1, 2))
    out[:, 0] = init
    q, p = init[:, 0].copy(), init[:, 1].copy()
    h = params.dt / _RK4_SUBSTEPS
    t = 0.0
    for step in range(1, n_steps + 1):
        for _ in range(_RK4_SUBSTEPS):
            k1q, k1p = p, _accel(params, q, p, t)
            k2q, k2p = p + 0.5 * h * k1p, _accel(params, q + 0.5 * h * k1q, p + 0.5 * h * k1p, t + 0.5 * h)
            k3q, k3p = p + 0.5 * h * k2p, _accel(params, q + 0.5 * h * k2q, p + 0.5 * h * k2p, t + 0.5 * h)
            k4q, k4p = p + h * k3p, _accel(params, q + h * k3q, p + h * k3p, t + h)
            q = q + (h / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            p = p + (h / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
            t += h
        out[:, step, 0] = q
        out[:, step, 1] = p
    return out


def energy(params: OscillatorParams, traj: np.ndarray) -> np.ndarray:
    """H = p²/2 + ω²q²/2 (unit mass). Exact invariant only for zeta = drive_amp = 0."""
    q, p = traj[..., 0], traj[..., 1]
    return 0.5 * p ** 2 + 0.5 * params.omega ** 2 * q ** 2


def sample_initial_conditions(n: int, rng: np.random.Generator,
                              q_range=(-1.5, 1.5), p_range=(-1.5, 1.5)) -> np.ndarray:
    return rng.uniform((q_range[0], p_range[0]), (q_range[1], p_range[1]), size=(n, 2))


# ---------------------------------------------------------------------------
# Pendulum: the classic NONLINEAR Newton test.
#   theta'' = -(g/L) sin(theta)
# Linear systems cannot distinguish a recovered law from its local linearization, so
# the sin term is what makes symbolic recovery falsifiable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendulumParams:
    g_over_L: float = 4.0    # omega^2 for small oscillations
    damping: float = 0.0     # control: adds a -damping*omega term
    dt: float = 0.05


def _pendulum_accel(params: PendulumParams, theta, omega):
    return -params.g_over_L * np.sin(theta) - params.damping * omega


def simulate_pendulum(params: PendulumParams, init, n_steps: int) -> np.ndarray:
    """Integrate (theta, omega) from init (n, 2). Returns (n, n_steps + 1, 2)."""
    init = np.asarray(init, dtype=np.float64)
    if init.ndim != 2 or init.shape[1] != 2:
        raise ValueError(f"init must have shape (n, 2), got {init.shape}")
    out = np.empty((init.shape[0], n_steps + 1, 2))
    out[:, 0] = init
    th, om = init[:, 0].copy(), init[:, 1].copy()
    h = params.dt / _RK4_SUBSTEPS
    for step in range(1, n_steps + 1):
        for _ in range(_RK4_SUBSTEPS):
            k1t, k1o = om, _pendulum_accel(params, th, om)
            k2t, k2o = om + 0.5 * h * k1o, _pendulum_accel(params, th + 0.5 * h * k1t, om + 0.5 * h * k1o)
            k3t, k3o = om + 0.5 * h * k2o, _pendulum_accel(params, th + 0.5 * h * k2t, om + 0.5 * h * k2o)
            k4t, k4o = om + h * k3o, _pendulum_accel(params, th + h * k3t, om + h * k3o)
            th = th + (h / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
            om = om + (h / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
        out[:, step, 0] = th
        out[:, step, 1] = om
    return out


def pendulum_energy(params: PendulumParams, traj: np.ndarray) -> np.ndarray:
    """E = omega^2/2 + (g/L)(1 - cos theta) — exact invariant only when undamped."""
    return 0.5 * traj[..., 1] ** 2 + params.g_over_L * (1.0 - np.cos(traj[..., 0]))


def sample_pendulum_initial_conditions(n: int, rng: np.random.Generator,
                                       theta_max: float = 1.6) -> np.ndarray:
    """Wide angles on purpose: near theta = 0 the pendulum is indistinguishable from a
    linear oscillator, so the nonlinearity must be excited to be recoverable."""
    return np.stack([rng.uniform(-theta_max, theta_max, n),
                     rng.uniform(-1.0, 1.0, n)], axis=1)


# ---------------------------------------------------------------------------
# 2-D central force: the symmetry testbed.
#
# accel = -(1 + eps·r²)·(x, y)  — rotationally symmetric, and NONLINEAR for eps > 0.
# Nonlinearity is the whole point: on linear dynamics every linear field commutes with
# the flow, so a symmetry search there is vacuous (see RESEARCH_NOTES Stage E/F). Here
# rotation is a genuine symmetry and angular momentum L = x·vy − y·vx a genuine
# Noether invariant, so recovery is falsifiable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CentralForceParams:
    eps: float = 0.6   # anharmonicity; 0 collapses to the (linear) isotropic oscillator
    dt: float = 0.05
    kepler: bool = False   # inverse-square instead: accel = -mu * pos / r^3
    mu: float = 1.0


def _central_accel(params: CentralForceParams, pos):
    r2 = (pos ** 2).sum(axis=-1, keepdims=True)
    if params.kepler:
        # a genuinely different nonlinearity (singular, not polynomial) — the real test
        # of whether "degree 2 suffices" is a property of the method or of one testbed
        return -params.mu * pos / np.maximum(r2, 1e-8) ** 1.5
    return -(1.0 + params.eps * r2) * pos


def simulate_central(params: CentralForceParams, init, n_steps: int) -> np.ndarray:
    """Integrate (x, y, vx, vy) from init (n, 4). Returns (n, n_steps + 1, 4)."""
    init = np.asarray(init, dtype=np.float64)
    if init.ndim != 2 or init.shape[1] != 4:
        raise ValueError(f"init must have shape (n, 4), got {init.shape}")
    out = np.empty((init.shape[0], n_steps + 1, 4))
    out[:, 0] = init
    pos, vel = init[:, :2].copy(), init[:, 2:].copy()
    h = params.dt / _RK4_SUBSTEPS
    for step in range(1, n_steps + 1):
        for _ in range(_RK4_SUBSTEPS):
            k1x, k1v = vel, _central_accel(params, pos)
            k2x, k2v = vel + 0.5 * h * k1v, _central_accel(params, pos + 0.5 * h * k1x)
            k3x, k3v = vel + 0.5 * h * k2v, _central_accel(params, pos + 0.5 * h * k2x)
            k4x, k4v = vel + h * k3v, _central_accel(params, pos + h * k3x)
            pos = pos + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            vel = vel + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[:, step, :2] = pos
        out[:, step, 2:] = vel
    return out


def angular_momentum(traj: np.ndarray) -> np.ndarray:
    """L = x·vy − y·vx — conserved by any central force (the Noether charge of rotation)."""
    return traj[..., 0] * traj[..., 3] - traj[..., 1] * traj[..., 2]


def central_energy(params: CentralForceParams, traj: np.ndarray) -> np.ndarray:
    r2 = (traj[..., :2] ** 2).sum(-1)
    return 0.5 * (traj[..., 2:] ** 2).sum(-1) + 0.5 * r2 + 0.25 * params.eps * r2 ** 2


def sample_kepler_initial_conditions(n: int, rng: np.random.Generator,
                                     mu: float = 1.0) -> np.ndarray:
    """Bound, non-degenerate Kepler orbits: speed a fraction of escape so the orbit stays
    bound, and velocity mostly tangential so angular momentum is well away from zero
    (a near-radial orbit dives into the singularity)."""
    r = rng.uniform(0.9, 1.3, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    v_circ = np.sqrt(mu / r)
    speed = v_circ * rng.uniform(0.80, 1.15, size=n)
    tilt = rng.uniform(-0.35, 0.35, size=n)          # radians off tangential
    ang = theta + np.pi / 2 + tilt
    return np.stack([r * np.cos(theta), r * np.sin(theta),
                     speed * np.cos(ang), speed * np.sin(ang)], axis=1)


def sample_central_initial_conditions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample states on annular orbits with a spread of angular momenta (both signs),
    so L is a genuinely varying quantity across trajectories rather than a constant."""
    r = rng.uniform(0.6, 1.3, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    v_mag = rng.uniform(0.5, 1.1, size=n)
    v_ang = rng.uniform(0, 2 * np.pi, size=n)
    return np.stack([r * np.cos(theta), r * np.sin(theta),
                     v_mag * np.cos(v_ang), v_mag * np.sin(v_ang)], axis=1)


@dataclass(frozen=True)
class CoupledParams:
    """Two anharmonic oscillators with nonlinear coupling.

        H = ½(p1² + p2²) + ½(q1² + q2²) + ¼κ(q1⁴ + q2⁴) + λ q1² q2²

    A second 2-DOF conservative substrate, structurally unlike the central force: it has **no
    rotational symmetry**, so energy is the ONLY conserved quantity. That makes it a cleaner
    test of the Hamiltonian pairing — on the central force, energy competes with angular
    momentum (and their products) inside a large conserved family.

    Nonlinear by construction, as the pairing requires (DECISIONS D7).
    """

    kappa: float = 1.0      # self-anharmonicity
    lam: float = 0.5        # nonlinear coupling
    zeta: float = 0.0       # damping; 0 = conservative
    dt: float = 0.05


def _coupled_accel(p: CoupledParams, q, v):
    q1, q2 = q[..., 0], q[..., 1]
    a1 = -(q1 + p.kappa * q1 ** 3 + 2.0 * p.lam * q1 * q2 ** 2)
    a2 = -(q2 + p.kappa * q2 ** 3 + 2.0 * p.lam * q2 * q1 ** 2)
    return np.stack([a1, a2], axis=-1) - 2.0 * p.zeta * v


def simulate_coupled(params: CoupledParams, init, n_steps: int) -> np.ndarray:
    """Integrate (q1, q2, v1, v2) from init (n, 4). Returns (n, n_steps + 1, 4)."""
    init = np.asarray(init, dtype=np.float64)
    if init.ndim != 2 or init.shape[1] != 4:
        raise ValueError(f"init must have shape (n, 4), got {init.shape}")
    out = np.empty((init.shape[0], n_steps + 1, 4))
    out[:, 0] = init
    q, v = init[:, :2].copy(), init[:, 2:].copy()
    h = params.dt / _RK4_SUBSTEPS
    for step in range(1, n_steps + 1):
        for _ in range(_RK4_SUBSTEPS):
            k1q, k1v = v, _coupled_accel(params, q, v)
            k2q, k2v = v + 0.5 * h * k1v, _coupled_accel(params, q + 0.5 * h * k1q, v + 0.5 * h * k1v)
            k3q, k3v = v + 0.5 * h * k2v, _coupled_accel(params, q + 0.5 * h * k2q, v + 0.5 * h * k2v)
            k4q, k4v = v + h * k3v, _coupled_accel(params, q + h * k3q, v + h * k3v)
            q = q + (h / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[:, step, :2] = q
        out[:, step, 2:] = v
    return out


def coupled_energy(params: CoupledParams, traj: np.ndarray) -> np.ndarray:
    q1, q2 = traj[..., 0], traj[..., 1]
    v1, v2 = traj[..., 2], traj[..., 3]
    return (0.5 * (v1 ** 2 + v2 ** 2) + 0.5 * (q1 ** 2 + q2 ** 2)
            + 0.25 * params.kappa * (q1 ** 4 + q2 ** 4) + params.lam * q1 ** 2 * q2 ** 2)
