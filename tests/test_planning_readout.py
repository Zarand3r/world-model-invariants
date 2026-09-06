"""The GPU planning readout must agree with the validated numpy one, on real frames."""
import numpy as np
import pathlib

import pytest
import torch

from latent_noether.pixel_readout import decode_physics
from latent_noether.planning_readout import energy_from_frames


@pytest.mark.parametrize("data", ["runs/pendulum_pixels.npz", "runs/pendulum_actuated.npz"])
# Skip rather than fail when the dataset is absent, which is the state of a fresh clone
# before scripts/fetch_assets.py runs. Six other tests in this suite already skip that way;
# these two hard-failed, so a cold checkout read "2 failed" for a missing download.
def test_torch_readout_matches_numpy_reference(data):
    """A port, not a reimplementation: any divergence here invalidates every planning number.

    The numpy readout is the instrument the paper's decoded-energy results are measured with. The
    planner needs the same quantity thousands of times per control step, so it is ported to torch --
    and pinned here, because a silent divergence would be invisible in the planning results.
    """
    if not pathlib.Path(data).exists():
        pytest.skip(f"{data} not present; run scripts/fetch_assets.py")
    d = np.load(data)
    fr = d["frames"][:8]                                   # (8, T, 64, 64, 3) uint8
    ref = decode_physics(fr)
    got = energy_from_frames(torch.as_tensor(fr).float().div_(255.0).sub_(0.5))

    for key, tol in (("theta", 1e-4), ("thetadot", 1e-3), ("energy", 1e-3)):
        a = np.asarray(ref[key], dtype=np.float64)
        b = got[key].double().cpu().numpy()
        m = np.isfinite(a) & np.isfinite(b)
        # column 0 of thetadot/energy depends on the backward-difference seed; compare from t=1
        if key != "theta":
            m[:, 0] = False
        assert m.any()
        assert np.abs(a[m] - b[m]).max() == pytest.approx(0.0, abs=tol), (
            f"{key}: torch port diverges from the numpy reference by "
            f"{np.abs(a[m] - b[m]).max():.3e}"
        )
