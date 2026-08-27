"""The bench's HTTP surface.

Sync endpoints on purpose: FastAPI runs them in a threadpool, so the event loop stays free for the
job stream while a 13-second fit holds the GPU lock. The split that matters is which routes touch
the device — `/law` never does, which is why the mixing sliders are instant.
"""
import json
import pathlib
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from viz.server import assets, bundles, law, leverage, paper_refs, registry, rollout

app = FastAPI(title="Invariant Probe Bench")
DIST = pathlib.Path(__file__).resolve().parents[1] / "web" / "dist"

PAPER_ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.4)


def _bundle(key: str):
    if not bundles.cached(key):
        raise HTTPException(404, f"bundle {key} is not built; POST /api/bundles first")
    return bundles.load(key)


@app.get("/api/models")
def get_models():
    return {"models": [m.__dict__ for m in assets.models()],
            "resident": registry.resident(),
            "cached_bundles": sorted(p.stem for p in bundles.CACHE.glob("*.npz"))}


@app.get("/api/paper")
def get_paper():
    return paper_refs.refs()


class BundleReq(BaseModel):
    model: str
    ld: int = 12
    degree: int = 4


@app.post("/api/bundles")
def post_bundle(req: BundleReq):
    k = bundles.key(req.model, req.ld, req.degree)
    if bundles.cached(k):
        return {"key": k, "cached": True}
    return {"key": k, "cached": False, "job": bundles.start_job(req.model, req.ld, req.degree)}


@app.get("/api/jobs/{jid}/events")
def job_events(jid: str):
    def stream():
        sent = 0
        while True:
            j = bundles.job(jid)
            while sent < len(j["messages"]):
                yield f"data: {json.dumps({'message': j['messages'][sent]})}\n\n"
                sent += 1
            if j["state"] in ("done", "failed"):
                yield f"data: {json.dumps({'state': j['state'], 'key': j['key']})}\n\n"
                return
            time.sleep(0.25)
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/bundles/{key}")
def get_bundle(key: str):
    b = _bundle(key)
    law.basis_grads(b)          # 5.8 s once per bundle; paid here, not under the first slider drag
    return {"key": key, "ckpt": b.ckpt, "ld": b.ld, "degree": b.degree, "warmup": b.warmup,
            "eigenvalues": b.eigenvalues.tolist(), **bundles.summary(b),
            "weights": b.weights.tolist(), "pairing_residual": b.residual,
            "rank_and_test_residual": b.rank_and_test_residual,
            "scatter": law.scatter(b), "energy_range": [float(b.energy.min()), float(b.energy.max())]}


class LawReq(BaseModel):
    weights: list[float] | None = None
    draw: int | None = None


@app.post("/api/bundles/{key}/law")
def post_law(key: str, req: LawReq):
    b = _bundle(key)
    w = np.asarray(req.weights) if req.weights is not None else (
        law.random_weights(b, req.draw) if req.draw is not None else b.weights)
    s = law.scores(b, w)
    return {**s, "scatter": law.scatter(b, np.asarray(s["weights"]))}


class RollReq(BaseModel):
    traj: int = 0
    horizon: int = 50
    alpha: float = 0.2
    weights: list[float] | None = None


@app.post("/api/bundles/{key}/rollout")
def post_rollout(key: str, req: RollReq):
    b = _bundle(key)
    m = registry.get(pathlib.Path(b.ckpt).stem)
    with registry.GPU:
        r = rollout.imagine(m, b, [req.traj], req.horizon, (0.0, req.alpha),
                            weights=req.weights, cache_key=key)
    return {
        "alpha": req.alpha, "traj": req.traj, "horizon": req.horizon,
        "truth": rollout.sheet_data_uri(r["truth"][0]),
        "free": rollout.sheet_data_uri(r["frames"][0, 0]),
        "corrected": rollout.sheet_data_uri(r["frames"][1, 0]),
        "mse": {"free": r["mse"][0, 0].tolist(), "corrected": r["mse"][1, 0].tolist()},
        "C": {"free": r["C"][0, 0].tolist(), "corrected": r["C"][1, 0].tolist()},
        "C0": float(r["C0"][0, 0]),
    }


class DoseReq(BaseModel):
    horizon: int = 50
    weights: list[float] | None = None
    alphas: list[float] | None = None


@app.post("/api/bundles/{key}/dose")
def post_dose(key: str, req: DoseReq):
    b = _bundle(key)
    m = registry.get(pathlib.Path(b.ckpt).stem)
    al = tuple(req.alphas or PAPER_ALPHAS)
    with registry.GPU:
        r = rollout.imagine(m, b, None, req.horizon, al, weights=req.weights,
                            keep_frames=False, cache_key=key)
    e = np.asarray(r["mse_by_alpha"], dtype=np.float64)
    return {"alphas": list(al), "mse": e.tolist(),
            "normalised_slope": float(np.polyfit(np.asarray(al), e / e[0], 1)[0]),
            "relative_change_at_max_alpha": float((e[-1] - e[0]) / e[0]),
            "n_traj": len(r["trajs"])}


@app.get("/api/bundles/{key}/leverage")
def get_leverage(key: str, horizon: int = leverage.HORIZON):
    b = _bundle(key)
    cache = bundles.CACHE / f"{key}__leverage_h{horizon}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    m = registry.get(pathlib.Path(b.ckpt).stem)
    with registry.GPU:
        r = leverage.measure(m, b, horizon=horizon)
    cache.write_text(json.dumps(r))
    return r


if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")
