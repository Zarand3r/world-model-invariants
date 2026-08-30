#!/usr/bin/env python3
"""F11: does the model apply ITS OWN timestep's dynamics, or the evaluation data's?

Preregistered in docs/F11_PREREG.md. No training.

Unlike F7/F10, this compares along an axis that genuinely exists in the observations: dt enters the
position recurrence as a(th) dt^2, so a 4x change in dt is a 16x change in the acceleration term.

Usage:  uv run python scripts/run_f11_timestep_choice.py --out runs/f11_timestep_choice.json
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.pixel_readout import theta_from_frames
from latent_noether.provenance import attach
from scripts.run_f6_analysis import ANALYSIS, W

G, L = 10.0, 1.0
DTS = (0.02, 0.08)
SEEDS = (3, 4, 5)


def accel(th):
    return (3 * G / (2 * L)) * np.sin(th)


def wrap(a):
    return np.abs(np.arctan2(np.sin(a), np.cos(a)))


def run(ckpt, data, dt_model, dt_data):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr).detach()
    th = st[:, W:, 0]
    H = hs[:, W:]
    # need (th_{t-1}, th_t) to form the recurrence, and predict th_{t+1}
    th_prev, th_cur = th[:, :-2], th[:, 1:-1]
    Hc = H[:, 1:-1]
    with torch.no_grad():
        img = m.readout_from_h(m.transition(Hc.reshape(-1, H.shape[-1])))
    th_pred = theta_from_frames(((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()).reshape(th_cur.shape)

    base = 2 * th_cur - th_prev
    cand_model = base + accel(th_cur) * dt_model ** 2
    cand_data = base + accel(th_cur) * dt_data ** 2
    sep = wrap(cand_model - cand_data)
    d_model = wrap(th_pred - cand_model)
    d_data = wrap(th_pred - cand_data)
    err = float(np.median(np.minimum(d_model, d_data)))
    sep_med = float(np.median(sep))
    diagonal = abs(dt_model - dt_data) < 1e-12
    hits = int((d_model < d_data).sum()); n = int(d_model.size)
    rate = hits / n
    se = math.sqrt(0.25 / n); z = (rate - 0.5) / se
    return {"ckpt": ckpt, "data": data, "dt_model": dt_model, "dt_data": dt_data,
            "diagonal": diagonal, "n": n, "rate_own_dt": rate, "z": z,
            "p_two_sided": math.erfc(abs(z) / math.sqrt(2)),
            "median_separation": sep_med, "median_pred_error": err,
            "G1_separated": bool(sep_med > err), "G2_readable": bool(err < sep_med)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f11_timestep_choice.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"cells": []}
    done = {(c["ckpt"], c["data"]) for c in out["cells"]}
    inputs = sorted({f"runs/f6_dt{d}_s{s}_step6500.pt" for d in DTS for s in SEEDS}
                    | {f"runs/pend_dt{d}.npz" for d in DTS})
    for dt_m in DTS:
        for dt_d in DTS:
            for s in SEEDS:
                ck = f"runs/f6_dt{dt_m}_s{s}_step6500.pt"; data = f"runs/pend_dt{dt_d}.npz"
                if (ck, data) in done or not pathlib.Path(ck).exists():
                    continue
                r = run(ck, data, dt_m, dt_d)
                tag = "diag" if r["diagonal"] else "CROSS"
                print(f"  model dt={dt_m:<5} data dt={dt_d:<5} s{s}  rate_own={r['rate_own_dt']:.3f}"
                      f"  sep={r['median_separation']:.4f} err={r['median_pred_error']:.4f}"
                      f"  readable={r['G2_readable']}  {tag}", flush=True)
                out["cells"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")

    off = [c for c in out["cells"] if not c["diagonal"]]
    if len(off) >= 6:
        readable = [c for c in off if c["G2_readable"]]
        summ = {"off_diagonal": len(off), "readable": len(readable)}
        if not readable:
            summ["verdict"] = "NOT READABLE -- crossed models are out of distribution; cannot test"
        else:
            per = {}
            for dt_m in DTS:
                cs = [c for c in readable if c["dt_model"] == dt_m]
                per[str(dt_m)] = {"rates": [round(c["rate_own_dt"], 3) for c in cs],
                                  "n_sig_above_half": sum(c["rate_own_dt"] > 0.5
                                                          and c["p_two_sided"] < 0.01 for c in cs),
                                  "n_cells": len(cs)}
            summ["per_model_dt"] = per
            summ["P1_pass"] = bool(all(v["n_sig_above_half"] >= 2 for v in per.values() if v["n_cells"]))
            summ["verdict"] = ("model applies its OWN timestep" if summ["P1_pass"]
                               else "model follows the DATA's timestep -- F6 model-side claim fails")
        out["summary"] = summ
        print(f"\n  off-diagonal cells {len(off)}, readable {len(readable)}")
        print(f"  {json.dumps(summ.get('per_model_dt', {}))}")
        print(f"  verdict: {summ['verdict']}")
    attach(out, op, inputs=inputs)
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
