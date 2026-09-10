"""Where the bench finds models and data, and what it is willing to say about them.

The released checkpoints land in `runs/` via `scripts/fetch_assets.py`. A training-step ladder
(`dreamer_ref_s*_step*.pt`) exists on the machine this study was run on but is not part of the
release, so it is picked up from `$WMI_EXTRA_CKPT_DIR` when present and simply absent otherwise —
the bench must degrade to the three released seeds on any other machine rather than fail.

That default is a sibling checkout on one machine and nothing anywhere else. It is a convenience,
not a dependency: when the directory was renamed the ladder silently dropped out of the model list,
which is the intended failure — eighteen models fewer, nothing broken.

Every model carries the dataset it was trained on. Pointing a damped model at conservative frames
produces a plausible-looking latent and a meaningless invariant, which is the single easiest way to
generate a wrong figure from this UI, so the pairing is made here and never chosen in the browser.
"""
import dataclasses
import functools
import os
import pathlib
import re

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
EXTRA = pathlib.Path(os.environ.get(
    "WMI_EXTRA_CKPT_DIR", str(ROOT.parent / "world-model-invariants-extension" / "runs")))

CONSERVATIVE_DATA = "pendulum_pixels.npz"
DAMPED_DATA = "pendulum_pixels_damped.npz"


@dataclasses.dataclass(frozen=True)
class Model:
    key: str
    path: str
    arm: str                # "conservative" | "damped" | "ladder"
    data: str
    seed: int | None
    steps: int | None
    hours: float | None
    released: bool
    label: str


def _meta(path: pathlib.Path) -> dict:
    """steps/hours/seed off a checkpoint without materialising the weights.

    `mmap` keeps this to a header read; there are two dozen 54 MB checkpoints to enumerate. A bare
    `except` here used to fall back to a full CPU load, which turned a corrupt checkpoint into a
    slow startup instead of an error, so only the narrow "this file cannot be mmapped" case falls
    back and anything else propagates.
    """
    try:
        ck = torch.load(path, map_location="meta", weights_only=False, mmap=True)
    except (RuntimeError, ValueError):          # not a zipfile-format checkpoint
        ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: ck.get(k) for k in ("steps", "hours", "seed")}


@functools.lru_cache(maxsize=1)
def models() -> tuple[Model, ...]:
    out = []
    for p in sorted(RUNS.glob("dreamer_ref_s?.pt")):
        m = _meta(p)
        out.append(Model(p.stem, str(p), "conservative", CONSERVATIVE_DATA, m["seed"], m["steps"],
                         m["hours"], True, f"conservative seed {m['seed']}"))
    for p in sorted(RUNS.glob("dreamer_damped_s?.pt")):
        m = _meta(p)
        out.append(Model(p.stem, str(p), "damped", DAMPED_DATA, m["seed"], m["steps"], m["hours"],
                         True, f"damped seed {m['seed']}"))
    if EXTRA.is_dir():
        rows = []
        for p in EXTRA.glob("dreamer_ref_s*_step*.pt"):
            n = re.search(r"_step(\d+)\.pt$", p.name)
            if n:
                rows.append((int(n.group(1)), p))
        for steps, p in sorted(rows):
            m = _meta(p)
            out.append(Model(p.stem, str(p), "ladder", CONSERVATIVE_DATA, m["seed"], steps,
                             m["hours"], False, f"seed {m['seed']} @ {steps:,} steps"))
    return tuple(out)


def model(key: str) -> Model:
    for m in models():
        if m.key == key:
            return m
    raise KeyError(f"unknown model {key!r}; have {[m.key for m in models()]}")


SPLIT_START = 204          # the analysis trajectories; everything before them trained the model


@functools.lru_cache(maxsize=3)
def dataset(name: str) -> dict:
    """The raw arrays, on the CPU. Nothing here reaches the GPU."""
    d = np.load(RUNS / name)
    return {"frames": d["frames"], "states": d["states"]}


@functools.lru_cache(maxsize=4)
def analysis_split(name: str, split_start: int = SPLIT_START) -> dict:
    """The analysis trajectories: scaled frames on the GPU, plus their states.

    **The only place the split is applied.** Every consumer — the extraction, the rollouts, the
    leverage sweep — asks for it here, so none of them can disagree about which trajectories are
    held out.

    Uploading the whole array instead cost 1.51 GB of VRAM per conservative dataset and 5.03 GB for
    the evaluation set, to use 0.31 GB: only 52 of 256 trajectories are ever touched.
    """
    d = dataset(name)
    frames = d["frames"][split_start:]
    return {"scaled": torch.as_tensor(frames).float().div_(255.).sub_(0.5).cuda(),
            "states": d["states"][split_start:], "split_start": split_start}
