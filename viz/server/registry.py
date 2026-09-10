"""One GPU, one lock, a few resident world models.

Each adapter is ~3.8 GB of VRAM and ~0.5 s to load, so models are kept resident and evicted least-
recently-used. The cap is set for headroom rather than necessity: this box has 96 GB, but a bench
that quietly grows to twenty resident models is a bench that fails on someone else's card.

**Everything that touches the GPU takes `GPU`.** The rollouts are 0.12 s and the fits are 13-22 s,
so contention is real but queueing is adequate — FastAPI runs sync endpoints in a threadpool, and
this lock is what keeps two of them off the device at once. A long fit therefore blocks rollouts
while it runs, which is why fits are jobs the UI polls rather than requests it waits on.
"""
import collections
import threading

import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from viz.server import assets

GPU = threading.Lock()
MAX_RESIDENT = 3

_models: collections.OrderedDict[str, DreamerV3Adapter] = collections.OrderedDict()
_lock = threading.Lock()
_loading: dict[str, threading.Lock] = {}


def get(key: str) -> DreamerV3Adapter:
    """The frozen adapter for `key`, loading and evicting as needed. Never trains anything."""
    with _lock:
        if key in _models:
            _models.move_to_end(key)
            return _models[key]
        # One loader per key. Without this, two concurrent misses each built a 3.8 GB adapter and
        # both landed on the device before either reached the eviction check.
        gate = _loading.setdefault(key, threading.Lock())
    with gate:
        with _lock:
            if key in _models:
                _models.move_to_end(key)
                return _models[key]
        return _load(key)


def _load(key: str) -> DreamerV3Adapter:
    spec = assets.model(key)
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(spec.path, map_location="cuda")["model"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    with _lock:
        _models[key] = m
        _models.move_to_end(key)
        while len(_models) > MAX_RESIDENT:
            _models.popitem(last=False)
        torch.cuda.empty_cache()
    return m


def resident() -> list[str]:
    with _lock:
        return list(_models)
