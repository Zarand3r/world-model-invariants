"""Geometric image -> (theta, thetadot, E) readout for decoded pendulum frames.

Used by E1 (`docs/E1_PREREG.md`) to ask whether correcting the latent invariant improves the
*physics* of an imagined rollout, not merely its pixel error. The published intervention is scored
with pixel MSE alone, which cannot separate "the imagined world became more physical" from "a latent
regulariser lowered reconstruction error".

**Deliberately not a learned probe.** A trained readout would refit the very quantity under test, and
would give the intervention a second chance to look good. This is image moments plus arithmetic.

**The one calibrated number.** Gymnasium renders at 500x500 and `make_pendulum_pixels.downsample`
crops the *top-left* 448x448 before block-averaging by 7, so the pivot sits near 250/7 = 35.71 in
64-space rather than at the image centre. Using that geometric value directly gives 3.63 deg median
angle error, because the ink centroid is the centroid of rod *and* axle glyph, not of the rod alone.
Fitting the single scalar `pivot` (shared by both axes) absorbs the glyph and gives 0.25 deg.

That scalar is fitted **once**, on the Dreamer *training* trajectories (0..203), which E1 never
scores on, and then frozen -- see `runs/pixel_readout_calibration.json`. Sign and angular offset are
not fitted: the search returned exactly -1 and +0.0003 rad, i.e. plain geometry.

Conventions match `scripts/make_pendulum_pixels.py`: `theta` measured from upright, `dt = 0.05`,
`E = 0.5 (m l^2 / 3) thetadot^2 + m g (l/2) cos theta` with g=10, m=1, l=1.
"""
import json
import pathlib

import numpy as np

RES = 64
DT = 0.05
INK_FLOOR = 15.0          # drop antialias haze; the rod is (204, 77, 77) on white
CALIB_PATH = pathlib.Path(__file__).resolve().parent.parent / "runs" / "pixel_readout_calibration.json"

_YS, _XS = np.mgrid[0:RES, 0:RES]


def _ink(frames):
    """(..., 64, 64, 3) uint8 or float in [0, 255] -> (..., 64, 64) non-negative weight."""
    f = np.asarray(frames, dtype=np.float32)
    if f.max() <= 1.001:                      # tolerate [0,1] or [-0.5,0.5] scaled frames
        f = (f + 0.5) * 255.0 if f.min() < -0.001 else f * 255.0
    return np.clip((255.0 - f).sum(-1) - INK_FLOOR, 0.0, None)


def load_pivot():
    """The frozen calibration. Fails loudly rather than silently defaulting."""
    if not CALIB_PATH.exists():
        raise FileNotFoundError(
            f"{CALIB_PATH} missing. Run `python -m latent_noether.pixel_readout --calibrate` "
            "to fit the pivot on the training split before using the readout."
        )
    return json.loads(CALIB_PATH.read_text())["pivot"]


def centroids_from_frames(frames):
    """(..., 64, 64, 3) -> (cy, cx, mass). Independent of `pivot`, so calibration hoists it out."""
    w = _ink(frames)
    m = w.sum((-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        cy = (w * _YS).sum((-2, -1)) / m
        cx = (w * _XS).sum((-2, -1)) / m
    return cy, cx, m


def theta_from_centroids(cy, cx, mass, pivot):
    """The pivot-dependent half of the readout: one atan2, no image work."""
    with np.errstate(invalid="ignore"):
        th = -np.arctan2(cx - pivot, -(cy - pivot))
    return np.where(mass > 0, th, np.nan)


def theta_from_frames(frames, pivot=None):
    """(..., 64, 64, 3) -> (...) wrapped theta in radians, measured from upright.

    Returns NaN where a frame carries no ink at all (a fully blank decode), so callers can exclude
    it rather than silently reading atan2(0, 0).
    """
    pivot = load_pivot() if pivot is None else pivot
    cy, cx, m = centroids_from_frames(frames)
    return theta_from_centroids(cy, cx, m, pivot)


def unwrap_theta(theta, axis=-1):
    """Undo the 2*pi wrap along time. Rotating trajectories cross the branch cut repeatedly."""
    return np.unwrap(np.asarray(theta, dtype=np.float64), axis=axis)


def thetadot_from_theta(theta_unwrapped, dt=DT, axis=-1):
    """BACKWARD differences -- which is exact here, and central differences are not.

    Gymnasium integrates the pendulum semi-implicitly:

        thdot_k = thdot_{k-1} + accel(th_{k-1}) dt
        th_k    = th_{k-1} + thdot_k dt

    so the stored `thdot_k` is *by construction* the backward difference (th_k - th_{k-1}) / dt,
    not a centred derivative at step k. Measured on the analysis split against the stored states:

        backward difference   median |error| = 0.0000      (exact)
        central difference    median |error| = 0.2905

    Using `np.gradient` here cost 0.29 rad/s of pure convention error, which propagated into a
    median decoded-energy error of 0.155 across-trajectory standard deviations -- large enough to
    have masked the E1 effect entirely. Matching the integrator's own convention removes it.

    Step 0 has no predecessor and is filled by the forward difference, which is *not* exact; callers
    that care should drop it. E1 does: its window starts after WARMUP.
    """
    th = np.asarray(theta_unwrapped, dtype=np.float64)
    th = np.moveaxis(th, axis, -1)
    out = np.empty_like(th)
    out[..., 1:] = np.diff(th, axis=-1) / dt
    out[..., 0] = out[..., 1]                   # no predecessor at k=0; flagged in the docstring
    return np.moveaxis(out, -1, axis)


def energy(theta, thetadot, g=10.0, m=1.0, l=1.0):
    """Textbook rod energy -- identical to `scripts/make_pendulum_pixels.energy`."""
    return 0.5 * (m * l ** 2 / 3) * np.asarray(thetadot) ** 2 + m * g * (l / 2) * np.cos(theta)


def decode_physics(frames, pivot=None, dt=DT):
    """(..., T, 64, 64, 3) -> dict of theta (unwrapped), thetadot, E. The full E1 readout."""
    th = unwrap_theta(theta_from_frames(frames, pivot))
    thd = thetadot_from_theta(th, dt)
    return {"theta": th, "thetadot": thd, "energy": energy(th, thd)}


def _calibrate(data="runs/pendulum_pixels.npz", train=slice(0, 204), out=CALIB_PATH):
    """Fit the single pivot scalar on the TRAINING trajectories and freeze it.

    E1 scores on the analysis split (204:), which this never touches.
    """
    d = np.load(data)
    fr, st = d["frames"][train], d["states"][train]
    cy, cx, m = centroids_from_frames(fr)          # the expensive part, done once
    grid = np.arange(33.0, 38.001, 0.05)
    errs = []
    for p in grid:
        e = np.abs(np.angle(np.exp(1j * (theta_from_centroids(cy, cx, m, p) - st[..., 0]))))
        errs.append(np.nanmedian(e))
    best = float(grid[int(np.argmin(errs))])
    rec = {
        "pivot": best,
        "fitted_on": {"data": data, "trajectories": f"{train.start}:{train.stop}"},
        "median_abs_theta_error_rad": float(np.min(errs)),
        "median_abs_theta_error_deg": float(np.degrees(np.min(errs))),
        "geometric_prediction_250_over_7": 250 / 7,
        "note": "sign and angular offset are NOT fitted; the search returned -1 and +0.0003 rad.",
    }
    pathlib.Path(out).write_text(json.dumps(rec, indent=2) + "\n")
    return rec


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true")
    a = p.parse_args()
    if a.calibrate:
        print(json.dumps(_calibrate(), indent=2))


# ---------------------------------------------------------------------------
# 2-DoF anharmonic oscillator readout (E17). The pendulum functions above stay
# untouched; this reuses `centroids_from_frames` and adds the oscillator's own
# geometry and energy.
# ---------------------------------------------------------------------------

OSC_HALF = 2.0          # must match scripts/make_oscillator2d.HALF


def position_from_frames(frames, half=OSC_HALF):
    """(..., 64, 64, 3) -> (..., 2) position (q1, q2).

    A disk's ink-weighted centroid IS its centre, so this needs no calibrated offset -- unlike the
    pendulum, whose centroid is displaced by the axle glyph. Validated against known positions at
    0.0014 units = 0.022 px before any dataset was generated.
    """
    cy, cx, mass = centroids_from_frames(frames)
    q1 = (cx / (RES - 1) - 0.5) * 2 * half
    q2 = -(cy / (RES - 1) - 0.5) * 2 * half          # +q2 is up, matching the renderer
    bad = ~(mass > 0)
    return np.stack([np.where(bad, np.nan, q1), np.where(bad, np.nan, q2)], -1)


def momentum_from_position(q, dt=DT, axis=-2):
    """BACKWARD differences, for the same reason as the pendulum.

    `make_oscillator2d` integrates with semi-implicit Euler -- deliberately, to match gymnasium --
    so `q_k = q_{k-1} + p_k dt` and the stored `p_k` is exactly the backward difference. Central
    differences would inject the same convention error that cost 0.29 rad/s on the pendulum.
    """
    q = np.asarray(q, dtype=np.float64)
    q = np.moveaxis(q, axis, -2)
    out = np.empty_like(q)
    out[..., 1:, :] = np.diff(q, axis=-2) / dt
    out[..., 0, :] = out[..., 1, :]
    return np.moveaxis(out, -2, axis)


def osc2d_energy(q, p, w=1.0, a=0.05, b=0.40):
    q1, q2 = q[..., 0], q[..., 1]
    kin = 0.5 * (p ** 2).sum(-1)
    pot = (0.5 * w ** 2 * (q1 ** 2 + q2 ** 2) + 0.25 * a * (q1 ** 4 + q2 ** 4)
           + 0.5 * b * q1 ** 2 * q2 ** 2)
    return kin + pot


def osc2d_angmom(q, p):
    return q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0]


def decode_physics_osc2d(frames, w=1.0, a=0.05, b=0.40, dt=DT, half=OSC_HALF):
    """The E17 analogue of `decode_physics`: frames -> q, p, energy, angular momentum."""
    q = position_from_frames(frames, half)
    p = momentum_from_position(q, dt)
    return {"q": q, "p": p, "energy": osc2d_energy(q, p, w, a, b), "angmom": osc2d_angmom(q, p)}
