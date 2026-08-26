"""Guards for the E1 geometric readout (`latent_noether/pixel_readout.py`).

The convention test is the important one. Using `np.gradient` (central differences) instead of the
backward difference cost 0.29 rad/s of pure convention error and inflated the decoded-energy error
from 1.6% to 15.5% of the across-trajectory spread -- large enough to have hidden the E1 effect.
Nothing about that failure is visible from the code; it only shows up against the stored states.
"""
import pathlib

import numpy as np
import pytest

from latent_noether.pixel_readout import (
    decode_physics, energy, theta_from_frames, thetadot_from_theta, unwrap_theta,
)

DATA = pathlib.Path("runs/pendulum_pixels.npz")
ANALYSIS = slice(204, None)          # never used to fit the pivot
WINDOW = slice(10, 60)

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="pendulum dataset not generated")


@pytest.fixture(scope="module")
def split():
    d = np.load(DATA)
    return d["frames"][ANALYSIS][:, WINDOW], d["states"][ANALYSIS][:, WINDOW]


def test_thetadot_is_the_backward_difference(split):
    """Gymnasium's semi-implicit Euler makes thdot_k EXACTLY (th_k - th_{k-1})/dt.

    Regression guard: central differences are wrong here by ~0.29 rad/s.
    """
    _, st = split
    th, thd = st[..., 0], st[..., 1]
    got = thetadot_from_theta(th)
    err = np.abs(got[:, 1:] - thd[:, 1:])            # k=0 has no predecessor
    assert np.median(err) < 1e-9, f"backward difference not exact: median {np.median(err):.3e}"


def test_theta_readout_accuracy_on_held_out_frames(split):
    fr, st = split
    th_hat = unwrap_theta(theta_from_frames(fr))
    off = np.round((th_hat - st[..., 0]).mean(-1, keepdims=True) / (2 * np.pi)) * 2 * np.pi
    err = np.abs(th_hat - off - st[..., 0])
    assert np.median(err) < 0.005, f"theta error {np.degrees(np.median(err)):.3f} deg"


def test_decoded_energy_error_is_small_relative_to_signal(split):
    """The decoded-energy floor must stay far below the spread E1 needs to resolve."""
    fr, st = split
    d = np.load(DATA)
    norm = energy(d["states"][..., 0], d["states"][..., 1]).mean(-1).std()
    got = decode_physics(fr)
    rel = np.abs(got["energy"] - energy(st[..., 0], st[..., 1])) / norm
    assert np.median(rel) < 0.05, f"decoded energy error {np.median(rel):.3f} of across-traj std"


def test_blank_frame_is_nan_not_zero():
    """A fully blank decode must be excludable, not silently read as atan2(0, 0)."""
    blank = np.full((1, 64, 64, 3), 255, dtype=np.uint8)
    assert np.isnan(theta_from_frames(blank)).all()


def test_readout_is_pure_and_repeatable(split):
    fr, _ = split
    a = theta_from_frames(fr[:4])
    b = theta_from_frames(fr[:4])
    assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))
