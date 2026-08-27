"""Extraction bundles: computed once per (model, dimension, degree), cached to disk forever.

The fit is 13-22 s depending on the extraction dimension, which is three orders of magnitude past
anything a slider can wait for. So it is a *job*: the UI asks for a bundle, gets either the cached
one or a job id, and watches progress. Everything interactive is then arithmetic on the result.

The cache key includes the checkpoint's size and mtime, so replacing a checkpoint invalidates its
bundles instead of silently serving an extraction of the previous weights.
"""
import dataclasses
import hashlib
import json
import pathlib
import threading
import time
import uuid

import numpy as np
import torch

from latent_noether.extraction import Bundle, build_bundle, drift_of_C, rho_energy
from viz.server import assets, registry

CACHE = pathlib.Path(__file__).resolve().parents[2] / "runs" / "viz_cache"
CACHE.mkdir(parents=True, exist_ok=True)

_ARRAYS = ("h_mean", "P", "P_pinv", "Z", "flow", "G", "basis_coeffs", "weights", "energy", "eigenvalues")
_SCALARS = ("ckpt", "ld", "degree", "warmup", "split_start", "participation_ratio", "residual",
            "rank_and_test_residual")

_mem: dict[str, Bundle] = {}
_mem_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def key(model_key: str, ld: int, degree: int) -> str:
    p = pathlib.Path(assets.model(model_key).path)
    st = p.stat()
    stamp = hashlib.sha256(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:8]
    return f"{model_key}__ld{ld}__deg{degree}__{stamp}"


def _path(k: str) -> pathlib.Path:
    return CACHE / f"{k}.npz"


def cached(k: str) -> bool:
    return _path(k).exists()


def load(k: str) -> Bundle:
    with _mem_lock:
        if k in _mem:
            return _mem[k]
    z = np.load(_path(k), allow_pickle=False)
    meta = json.loads(_path(k).with_suffix(".json").read_text())
    b = Bundle(**{n: z[n] for n in _ARRAYS}, **{n: meta[n] for n in _SCALARS})
    with _mem_lock:
        _mem[k] = b
    return b


def _save(k: str, b: Bundle) -> None:
    np.savez_compressed(_path(k), **{n: getattr(b, n) for n in _ARRAYS})
    _path(k).with_suffix(".json").write_text(json.dumps(
        {n: getattr(b, n) for n in _SCALARS} | summary(b), indent=2))


def summary(b: Bundle) -> dict:
    """The scalars a UI shows about a bundle, at its own fitted weights."""
    return {"rho_energy": rho_energy(b), "drift_of_C": drift_of_C(b),
            "n_traj": int(b.Z.shape[0]), "n_steps": int(b.Z.shape[1]),
            "retained_rank": int(b.Z.shape[-1]), "n_basis": int(b.basis_coeffs.shape[0]),
            "n_monomials": int(b.basis_coeffs.shape[1])}


def build(model_key: str, ld: int, degree: int, progress=lambda s: None) -> Bundle:
    """Compute and cache. Holds the GPU lock across the encode and the transition sweep."""
    k = key(model_key, ld, degree)
    if cached(k):
        progress("cached")
        return load(k)
    spec = assets.model(model_key)
    progress(f"loading {spec.key}")
    m = registry.get(model_key)
    progress(f"reading {spec.data}")
    d = assets.dataset(spec.data)
    progress(f"encoding, then a {degree}-degree fit in {ld} dimensions")
    t0 = time.time()
    with registry.GPU:
        b = build_bundle(m, d["scaled"], d["states"], ckpt=spec.path, ld=ld, degree=degree)
    progress(f"fitted in {time.time() - t0:.1f} s")
    _save(k, b)
    with _mem_lock:
        _mem[k] = b
    return b


def start_job(model_key: str, ld: int, degree: int) -> str:
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {"state": "queued", "messages": [], "key": key(model_key, ld, degree)}

    def run():
        j = _jobs[jid]
        j["state"] = "running"
        try:
            build(model_key, ld, degree, progress=lambda s: j["messages"].append(s))
            j["state"] = "done"
        except Exception as exc:                      # surfaced to the UI, not swallowed
            j["state"] = "failed"
            j["messages"].append(f"{type(exc).__name__}: {exc}")

    threading.Thread(target=run, daemon=True).start()
    return jid


def job(jid: str) -> dict:
    return _jobs[jid]
