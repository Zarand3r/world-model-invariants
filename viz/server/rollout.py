"""Autonomous imagination, with the projection edit in the loop, and frames on the wire.

The edit is the paper's, unchanged:

    z <- z - alpha (C(z) - C0) grad C(z) / ||grad C(z)||^2

mapped back through `pinv(P)` so the part of `h` outside the extracted subspace is untouched. It
runs at decode time; the model is never adapted and the decoder is never involved.

**One alpha per forward pass, and that is a correctness requirement rather than a style choice.**
Batching the five alphas together is the obvious optimisation and it changes the answer. Measured on
seed 3: rolling alpha = 0.1 alone over the 52 analysis trajectories gives a rollout MSE of
2.70458893e-03, matching `run_dreamer_edit.py`; rolling it inside a five-alpha batch gives
2.70815194e-03, off by 1.3e-03 relative. Five *identical* alphas batched together agree with each
other exactly, so nothing leaks between batch elements — the batch size selects different GEMM
kernels, and fifty steps of autonomous rollout amplify the last-bit difference. Most alphas are
insensitive (1e-7); this one is not.

The consequence for anything this bench displays: **rollout MSE differences below about 0.2% are
not meaningful**, because they are the size of the arithmetic. The intervention's effect is 2.9-3.5%,
comfortably above that, but a UI that invites you to compare two curves has to say where the floor
is. The published shape is (all trajectories, one alpha), so that is the shape used here.

**Frames leave as one PNG per track**, a vertical sprite sheet the browser animates by moving a
source rectangle; JSON-encoding pixel arrays was never on the table. `optimize=True` is deliberately
off: it cost 0.382 s per sheet against 0.037 s, three sheets per request, to save 5% of the bytes.
Decoded frames are noisy and do not deflate well, so the extra passes buy almost nothing.

**Encoded latents are cached per model.** A rollout needs one start state, but encoding a single
trajectory measured 0.297 s against 0.173 s for all fifty-two — a batch of one leaves the GPU idle.
So the whole split is encoded once and every later rollout indexes into it. The cache is keyed on
the model, not the bundle: the encoding depends only on the weights and the frames, so bundles that
differ merely in extraction dimension share one entry instead of each storing a copy.

**Indices are checked before anything reaches the device.** An out-of-range trajectory used to fire
a CUDA device-side assert, which is not a recoverable error: it poisons the process's CUDA context,
so every later request failed until the server was restarted while the page kept serving 200s.
"""
import threading
import base64
import collections
import io

import numpy as np
import torch
from PIL import Image

from latent_noether.extraction import Bundle
from latent_noether.polynomial import monomial_features


_STARTS: collections.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
_STARTS_LOCK = threading.Lock()
MAX_STARTS = 3                      # ~12 MB each; bounded so a long session cannot grow without end
MAX_ALPHA = 2.0                     # negative alpha is a legitimate anti-correction control; the
                                    # bound only keeps the projection from diverging outright


def analysis_frames(b: Bundle) -> torch.Tensor:
    """The scaled analysis frames this bundle was built from."""
    from viz.server import assets
    return assets.analysis_split(assets.model(b.model_key).data, b.split_start)["scaled"]


def start_states(model, b: Bundle) -> torch.Tensor:
    """Teacher-forced latents for the whole analysis split, encoded once. (n, T, D) on the GPU."""
    key = f"{b.model_key}@{b.split_start}"
    with _STARTS_LOCK:
        if key in _STARTS:
            _STARTS.move_to_end(key)
            return _STARTS[key]
    with torch.no_grad():
        hs = model.encode(analysis_frames(b)).detach()
    with _STARTS_LOCK:
        _STARTS[key] = hs
        _STARTS.move_to_end(key)
        while len(_STARTS) > MAX_STARTS:
            _STARTS.popitem(last=False)
    return hs


def _C_and_grad(z: torch.Tensor, coeffs: torch.Tensor, degree: int):
    """C and grad C at `z`. `enable_grad` is required: the rollout runs under no_grad, but the
    projection needs a gradient even though nothing is being trained."""
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        vals = monomial_features(zz, degree) @ coeffs
        g, = torch.autograd.grad(vals.sum(), zz)
    return vals.detach(), g.detach()


def max_horizon(b: Bundle) -> int:
    """Imagined steps that can be scored against real frames: everything after the warmup."""
    return int(b.Z.shape[1])


def check_request(b: Bundle, trajs, horizon: int, alphas) -> list[int]:
    """Validate and normalise a rollout request. Raises ValueError; never reaches the GPU."""
    n = int(b.Z.shape[0])
    idx = list(range(n)) if trajs is None else [int(t) for t in trajs]
    bad = [t for t in idx if not 0 <= t < n]
    if bad:
        raise ValueError(f"trajectory {bad[0]} is out of range; this bundle has {n} (0-{n - 1})")
    if not 1 <= horizon <= max_horizon(b):
        raise ValueError(f"horizon must be between 1 and {max_horizon(b)}, got {horizon}")
    a = np.asarray(alphas, dtype=np.float64).ravel()
    if a.size == 0:
        raise ValueError("no alphas given")
    if not np.all(np.isfinite(a)):
        raise ValueError("alphas must all be finite")
    if np.any(np.abs(a) > MAX_ALPHA):
        raise ValueError(f"|alpha| must be at most {MAX_ALPHA}")
    return idx


def imagine(model, b: Bundle, trajs, horizon: int, alphas, weights=None,
            keep_frames: bool = True) -> dict:
    """Roll each trajectory forward `horizon` steps once per alpha.

    The start state is the encoded latent at the end of the warmup — the same state the paper's
    intervention starts from — so `alpha = 0` is the model's own unaided imagination.

    `trajs` is a list of analysis-split indices; the UI passes one and the verification against
    `run_dreamer_edit.py` passes all 52. Both go through this function on purpose: a gate that
    exercised a second implementation would prove nothing about what the browser shows. The batch
    is laid out as (alpha, traj) and flattened, so one rollout covers the whole grid.
    """
    idx = check_request(b, trajs, horizon, alphas)
    frames = analysis_frames(b)
    dev = frames.device
    w = b.weights if weights is None else np.asarray(weights, dtype=np.float64)
    from viz.server.law import check_weights
    check_weights(b, w)
    coeffs = torch.as_tensor(w @ b.basis_coeffs, dtype=torch.float32, device=dev)

    h_mean = torch.as_tensor(b.h_mean, device=dev)
    P = torch.as_tensor(b.P, dtype=torch.float32, device=dev)
    P_pinv = torch.as_tensor(b.P_pinv, dtype=torch.float32, device=dev)
    al = np.asarray(alphas, dtype=np.float32).ravel()
    hs = start_states(model, b)[idx]
    ref = frames[idx][:, b.warmup: b.warmup + horizon]

    preds, Cs, C0s = [], [], []
    for alpha in al:
        h = hs[:, b.warmup].contiguous()
        a = float(alpha)
        z0 = (h - h_mean) @ P
        C0, _ = _C_and_grad(z0, coeffs, b.degree)
        C0s.append(C0)
        track, cs = [], []
        with torch.no_grad():
            for _ in range(horizon):
                track.append(model.readout_from_h(h))
                Cv, _ = _C_and_grad((h - h_mean) @ P, coeffs, b.degree)
                cs.append(Cv)
                h = model.transition(h)
                if a != 0.0:
                    z = (h - h_mean) @ P
                    Cv, gr = _C_and_grad(z, coeffs, b.degree)
                    step = a * ((Cv - C0) / gr.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * gr
                    h = h - (step @ P_pinv)
        preds.append(torch.stack(track, 1))
        Cs.append(torch.stack(cs, 1))

    pred = torch.stack(preds, 0)                                     # (na, nt, T, 64, 64, 3)
    mse = ((pred - ref.unsqueeze(0)) ** 2).mean(dim=(3, 4, 5))       # (na, nt, T)
    out = {
        "alphas": [float(x) for x in al],
        "trajs": idx,
        "mse": mse.cpu().numpy(),                                    # per alpha, per traj, per step
        "mse_by_alpha": mse.mean(dim=(1, 2)).cpu().numpy(),          # the script's scoring statistic
        "C": torch.stack(Cs, 0).cpu().numpy(),
        "C0": torch.stack(C0s, 0).cpu().numpy(),
    }
    if keep_frames:
        out["frames"] = pred.cpu().numpy()
        out["truth"] = ref.cpu().numpy()
    return out


def sheet_png(frames: np.ndarray) -> bytes:
    """(T, 64, 64, 3) in [-0.5, 0.5] -> one vertical sprite sheet, PNG bytes."""
    a = np.clip((np.asarray(frames) + 0.5) * 255.0, 0, 255).astype(np.uint8)
    t, h, w, _ = a.shape
    buf = io.BytesIO()
    Image.fromarray(a.reshape(t * h, w, 3)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def sheet_data_uri(frames: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(sheet_png(frames)).decode()
