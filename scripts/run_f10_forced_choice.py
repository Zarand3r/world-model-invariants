#!/usr/bin/env python3
"""F10: does the model's predicted next state land nearer ITS OWN scheme's next state?

Preregistered in docs/F10_PREREG.md (see amendment 1 for why the comparison is in theta). No training.

Two-alternative forced choice with an exact 50% null. For each state, compute what semi-implicit
Euler and velocity Verlet would each give as the next angle, decode the model's predicted next angle,
and score a hit when the model lands nearer its own training scheme.

The control carries the weight: a Verlet-trained model run on the SAME semi-implicit data should
prefer Verlet's answer, i.e. score BELOW 50%. That signed reversal cannot be produced by the
renderer, the encoder, or the counterfactual construction.

Usage:  uv run python scripts/run_f10_forced_choice.py --out runs/f10_forced_choice.json
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.pixel_readout import theta_from_frames
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_analysis import ANALYSIS, W

DT, G, L = 0.05, 10.0, 1.0
FAMILIES = {"semi-implicit": [f"runs/f6_dt0.05_s{s}_step6500.pt" for s in (3, 4, 5)],
            "verlet":        [f"runs/f7_verlet_s{s}_step6500.pt" for s in (3, 4, 5)]}
DATASETS = {"semi-implicit": "runs/pend_dt0.05.npz", "verlet": "runs/pend_verlet_dt0.05.npz"}


def accel(th):
    return (3 * G / (2 * L)) * np.sin(th)


def next_si(th, thd):
    thd2 = thd + accel(th) * DT
    return th + thd2 * DT


def next_vv(th, thd):
    return th + thd * DT + 0.5 * accel(th) * DT ** 2


def wrap(a):
    return np.abs(np.arctan2(np.sin(a), np.cos(a)))


def run(ckpt, data):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr).detach()
    th, thd = st[:, W:, 0], st[:, W:, 1]
    H = hs[:, W:]
    th_t, thd_t = th[:, :-1], thd[:, :-1]
    with torch.no_grad():
        img = m.readout_from_h(m.transition(H[:, :-1].reshape(-1, H.shape[-1])))
    th_pred = theta_from_frames(((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()).reshape(th_t.shape)

    d_si = wrap(th_pred - next_si(th_t, thd_t))
    d_vv = wrap(th_pred - next_vv(th_t, thd_t))
    sep = wrap(next_si(th_t, thd_t) - next_vv(th_t, thd_t))
    hits = int((d_si < d_vv).sum()); n = int(d_si.size)
    rate = hits / n
    se = math.sqrt(0.25 / n)
    z = (rate - 0.5) / se
    D_model = float(np.median(np.minimum(d_si, d_vv)))
    D_scheme = float(np.median(sep))
    return {"ckpt": ckpt, "data": data, "n": n, "hits_for_SI": hits, "rate_SI": rate,
            "z": z, "p_two_sided": math.erfc(abs(z) / math.sqrt(2)),
            "D_model": D_model, "D_scheme": D_scheme,
            "readable": bool(D_model < D_scheme)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f10_forced_choice.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"cells": []}
    done = {(c["ckpt"], c["data"]) for c in out["cells"]}
    for ev, data in DATASETS.items():
        for fam, ckpts in FAMILIES.items():
            for ck in ckpts:
                if (ck, data) in done or not pathlib.Path(ck).exists():
                    continue
                r = run(ck, data); r["family"] = fam; r["eval_data"] = ev
                print(f"  eval={ev:14s} {fam:14s} {pathlib.Path(ck).name:28s} "
                      f"rate_SI={r['rate_SI']:.3f}  z={r['z']:+7.1f}  p={r['p_two_sided']:.1e}  "
                      f"readable={r['readable']}", flush=True)
                out["cells"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")

    cells = out["cells"]
    if len(cells) >= 12:
        def grp(ev, fam):
            return [c for c in cells if c["eval_data"] == ev and c["family"] == fam]
        si_on_si = grp("semi-implicit", "semi-implicit")
        vv_on_si = grp("semi-implicit", "verlet")
        si_on_vv = grp("verlet", "semi-implicit")
        vv_on_vv = grp("verlet", "verlet")
        p1 = sum(c["rate_SI"] > 0.5 and c["p_two_sided"] < 0.01 for c in si_on_si)
        p2 = sum(c["rate_SI"] < 0.5 for c in vv_on_si)
        p2b = min(c["rate_SI"] for c in si_on_si) > max(c["rate_SI"] for c in vv_on_si)
        p3 = (sum(c["rate_SI"] > 0.5 for c in si_on_vv) >= 2
              and sum(c["rate_SI"] < 0.5 for c in vv_on_vv) >= 2)
        summ = {
            "rate_SI_semi_implicit_models_on_semi_implicit_data": [round(c["rate_SI"], 3) for c in si_on_si],
            "rate_SI_verlet_models_on_semi_implicit_data": [round(c["rate_SI"], 3) for c in vv_on_si],
            "rate_SI_semi_implicit_models_on_verlet_data": [round(c["rate_SI"], 3) for c in si_on_vv],
            "rate_SI_verlet_models_on_verlet_data": [round(c["rate_SI"], 3) for c in vv_on_vv],
            "all_readable": all(c["readable"] for c in cells),
            "P1_pass": bool(p1 >= 2), "P1": f"{p1}/3",
            "P2_pass": bool(p2 >= 2 and p2b), "P2": f"{p2}/3 below 0.5, families ordered {p2b}",
            "P3_symmetry_pass": bool(p3)}
        out["summary"] = summ
        print("\n  hit rate for SEMI-IMPLICIT's answer, by (model family, eval data):")
        print(f"    semi-implicit models on semi-implicit data  {summ['rate_SI_semi_implicit_models_on_semi_implicit_data']}")
        print(f"    verlet        models on semi-implicit data  {summ['rate_SI_verlet_models_on_semi_implicit_data']}")
        print(f"    semi-implicit models on verlet        data  {summ['rate_SI_semi_implicit_models_on_verlet_data']}")
        print(f"    verlet        models on verlet        data  {summ['rate_SI_verlet_models_on_verlet_data']}")
        print(f"  readable (D_model < D_scheme) on all cells: {summ['all_readable']}")
        print(f"  P1 {summ['P1_pass']} ({summ['P1']})   P2 {summ['P2_pass']} ({summ['P2']})   "
              f"P3 symmetry {summ['P3_symmetry_pass']}")
        if not summ["all_readable"]:
            print("  -> at least one cell is degenerate; its rate is not to be read.")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
