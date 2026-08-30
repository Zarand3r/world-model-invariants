"""Before comparing two data-generating schemes, check the difference is IN THE OBSERVATIONS.

Added 2026-08-30 after F7, F7b, F9 and F10 -- four preregistrations, six training runs and five
measurement designs -- were built on a difference that does not exist in the data the model consumes.
Semi-implicit Euler and velocity Verlet eliminate to the same three-term position recurrence and
differ only in which finite difference is called the velocity, a variable the pixels never render.

Every gate I wrote asked whether the instrument could see a difference. None asked whether the
difference was there. This is that check, and it is three lines and no GPU.
"""
import numpy as np

G, L = 10.0, 1.0
DT = 0.05


def accel(th):
    return (3 * G / (2 * L)) * np.sin(th)


def _positions(scheme, th0, thd0, n, dt=DT):
    th, thd, out = th0, thd0, [th0]
    for _ in range(n):
        if scheme == "semi-implicit":
            thd = thd + accel(th) * dt
            th = th + thd * dt
        elif scheme == "verlet":
            a = accel(th)
            tn = th + thd * dt + 0.5 * a * dt ** 2
            thd = thd + 0.5 * (a + accel(tn)) * dt
            th = tn
        else:
            raise ValueError(scheme)
        out.append(th)
    return np.array(out)


def test_semi_implicit_and_verlet_share_one_position_recurrence():
    """The documented reason F7/F7b/F10 are withdrawn. If this ever fails, that reasoning changes."""
    for scheme in ("semi-implicit", "verlet"):
        th = _positions(scheme, 0.7, 1.3, 60)
        residual = th[2:] - (2 * th[1:-1] - th[:-2] + accel(th[1:-1]) * DT ** 2)
        assert np.abs(residual).max() < 1e-12, (
            f"{scheme} no longer satisfies the shared position recurrence; the F7/F10 retraction "
            f"rests on it doing so")


def test_timestep_IS_observable_in_the_position_recurrence():
    """The contrast: dt enters the recurrence through a(th) dt^2, so F6's axis is well posed.

    This is what makes F6 recoverable where F7 was not, so it is pinned rather than assumed.
    """
    th = _positions("semi-implicit", 0.7, 1.3, 60, dt=0.05)
    # scoring the same positions under the WRONG timestep leaves a large residual
    wrong = th[2:] - (2 * th[1:-1] - th[:-2] + accel(th[1:-1]) * 0.08 ** 2)
    right = th[2:] - (2 * th[1:-1] - th[:-2] + accel(th[1:-1]) * 0.05 ** 2)
    assert np.abs(wrong).max() > 100 * max(np.abs(right).max(), 1e-15)
