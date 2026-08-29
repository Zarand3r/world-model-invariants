"""GPU readout for planning: decoded pendulum energy, batched, differentiable-free.

CEM evaluates thousands of imagined frames per control step. The validated readout in
`pixel_readout.py` is numpy and would dominate the runtime, so this is an exact torch port of the
same arithmetic -- same ink floor, same frozen pivot, same `atan2`, same BACKWARD differences.

It is a port, not a reimplementation: `tests/test_planning_readout.py` asserts agreement with
`pixel_readout.decode_physics` on real rendered frames. Nothing here may diverge from that reference.
"""
from __future__ import annotations

import numpy as np
import torch

from latent_noether.pixel_readout import DT, INK_FLOOR, RES, load_pivot

_G, _M, _L = 10.0, 1.0, 1.0


def energy_from_frames(frames: torch.Tensor, pivot: float | None = None, dt: float = DT):
    """(B, T, 64, 64, 3) in [-0.5, 0.5] -> dict of theta (unwrapped), thetadot, energy, all (B, T).

    `thetadot[:, 0]` is 0 by construction, exactly as the numpy backward-difference readout leaves it.
    """
    pivot = load_pivot() if pivot is None else pivot
    f = (frames + 0.5) * 255.0
    w = torch.clamp((255.0 - f).sum(-1) - INK_FLOOR, min=0.0)          # (B, T, 64, 64)
    ys = torch.arange(RES, device=frames.device, dtype=w.dtype).view(1, 1, RES, 1)
    xs = torch.arange(RES, device=frames.device, dtype=w.dtype).view(1, 1, 1, RES)
    m = w.sum((-2, -1))
    cy = (w * ys).sum((-2, -1)) / m
    cx = (w * xs).sum((-2, -1)) / m
    th = -torch.atan2(cx - pivot, -(cy - pivot))
    th = torch.where(m > 0, th, torch.full_like(th, float("nan")))

    # unwrap along time, matching np.unwrap
    d = th[:, 1:] - th[:, :-1]
    d = d - 2 * np.pi * torch.round(d / (2 * np.pi))
    thu = torch.cat([th[:, :1], th[:, :1] + torch.cumsum(d, dim=1)], dim=1)

    thd = torch.zeros_like(thu)
    thd[:, 1:] = (thu[:, 1:] - thu[:, :-1]) / dt                        # BACKWARD differences
    E = 0.5 * (_M * _L ** 2 / 3.0) * thd ** 2 + _M * _G * (_L / 2.0) * torch.cos(thu)
    return {"theta": thu, "thetadot": thd, "energy": E}
