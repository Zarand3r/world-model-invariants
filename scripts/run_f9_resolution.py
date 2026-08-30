#!/usr/bin/env python3
"""F9: is the scheme difference even large enough for these models to represent?

Preregistered in docs/F9_PREREG.md. No training.

Semi-implicit Euler and velocity Verlet differ in one step by 0.5*a(theta)*dt^2 = 0.01875*sin(theta)
rad at dt=0.05. If a model's own one-step prediction error exceeds that, it cannot encode which
scheme produced its data under ANY statistic, and F6/E19's model-side claims are unsupportable in
principle rather than merely unmeasured.

Usage:  uv run python scripts/run_f9_resolution.py --out runs/f9_resolution.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.pixel_readout import theta_from_frames
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_analysis import ANALYSIS, W

DT, G, L = 0.05, 10.0, 1.0
CKPTS = [f"runs/f6_dt0.05_s{s}_step6500.pt" for s in (3, 4, 5)]
DATA = "runs/pend_dt0.05.npz"


def accel(th):
    return (3 * G / (2 * L)) * np.sin(th)


def to_frames(x):
    return ((x + 0.5) * 255.0).clamp(0, 255).cpu().numpy()


def run(ckpt, data):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr).detach()

    # states aligned to the encoded window; predict t -> t+1 within it
    th = st[:, W:, 0]; thd = st[:, W:, 1]
    H = hs[:, W:]
    Hc, th_t, thd_t = H[:, :-1], th[:, :-1], thd[:, :-1]
    th_next_true = th[:, 1:]

    with torch.no_grad():
        pred_img = m.readout_from_h(m.transition(Hc.reshape(-1, H.shape[-1])))
        recon_img = m.readout_from_h(H[:, 1:].reshape(-1, H.shape[-1]))
    shape = th_next_true.shape
    th_pred = theta_from_frames(to_frames(pred_img)).reshape(shape)
    th_recon = theta_from_frames(to_frames(recon_img)).reshape(shape)

    # what velocity Verlet would have given from the same (theta_t, thetadot_t)
    th_vv = th_t + thd_t * DT + 0.5 * accel(th_t) * DT ** 2

    def wrap(a):
        return np.abs(np.arctan2(np.sin(a), np.cos(a)))

    D_model = float(np.median(wrap(th_pred - th_next_true)))
    D_decoder = float(np.median(wrap(th_recon - th_next_true)))
    D_scheme = float(np.median(wrap(th_next_true - th_vv)))
    return {"ckpt": ckpt, "D_model": D_model, "D_decoder": D_decoder, "D_scheme": D_scheme,
            "model_over_scheme": D_model / max(D_scheme, 1e-12),
            "decoder_over_scheme": D_decoder / max(D_scheme, 1e-12)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f9_resolution.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {m["ckpt"] for m in out["models"]}
    for ck in CKPTS:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        r = run(ck, DATA)
        print(f"  {pathlib.Path(ck).name:28s} D_model {r['D_model']:.5f}  "
              f"D_decoder {r['D_decoder']:.5f}  D_scheme {r['D_scheme']:.5f}  "
              f"model/scheme {r['model_over_scheme']:.2f}  decoder/scheme "
              f"{r['decoder_over_scheme']:.2f}", flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")

    if len(out["models"]) >= 3:
        ms = [m["model_over_scheme"] for m in out["models"]]
        ds = [m["decoder_over_scheme"] for m in out["models"]]
        p1 = sum(x >= 1.0 for x in ms); p2 = sum(x <= 0.3 for x in ms)
        floor_blind = sum(x >= 1.0 for x in ds)
        summ = {"model_over_scheme": [round(x, 3) for x in ms],
                "decoder_over_scheme": [round(x, 3) for x in ds],
                "P1_cannot_resolve": f"{p1}/3", "P2_can_resolve": f"{p2}/3",
                "P3_decoder_blind": f"{floor_blind}/3",
                "verdict": ("decoder-limited -- INCONCLUSIVE" if floor_blind >= 2 else
                            "cannot resolve" if p1 >= 2 else
                            "can resolve" if p2 >= 2 else "marginal")}
        out["summary"] = summ
        print(f"\n  model/scheme   {summ['model_over_scheme']}")
        print(f"  decoder/scheme {summ['decoder_over_scheme']}  (floor control)")
        print(f"  verdict: {summ['verdict']}")
        if floor_blind >= 2:
            print("  -> the READOUT cannot see the scheme gap, so this says nothing about the model.")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
