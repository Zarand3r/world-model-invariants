#!/usr/bin/env python3
"""F8 amendment 2: at which rollout horizon, if any, does the measurement have power?

Horizon 1 is the confounded one-step statistic; horizon 100 is noise (contrast 1.12x-1.45x). This
sweeps between them. One rollout per model, evaluated over prefixes, so nothing is recomputed.

Usage:  uv run python scripts/run_f8_horizon.py --out runs/f8_horizon.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_analysis import ANALYSIS, DEG, LD, W
from scripts.run_f6_timestep import R_GRID, c_star, energy
from scripts.run_f8_imagined import DT, FAMILIES

HORIZONS = (2, 3, 5, 10, 25, 50, 100)


def per_model(ckpt, data, dt, horizons):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    th, thd = st[:, W:, 0], st[:, W:, 1]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr).detach()
    H = hs[:, W:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad():                       # one rollout to the longest horizon
        h = hs[:, W].clone(); traj = [h]
        for _ in range(max(horizons) - 1):
            h = m.transition(h); traj.append(h)
        Himg = torch.stack(traj, 1)
    Zimg = ((Himg - hm) @ U) @ R

    X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    MZi = monomial_features(Zimg.reshape(-1, LD), DEG)
    Etrue = energy(th, thd); cs = c_star(dt)

    curves = {}
    for r in R_GRID:
        y = (Etrue + r * cs * thd * np.sin(th)).ravel()[:len(X)]
        w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        c = torch.as_tensor(w / (np.linalg.norm(w) + 1e-30), dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            Cimg = (MZi @ c).reshape(Zimg.shape[:2])
        curves[float(r)] = Cimg.cpu().numpy()

    out = {}
    for T in horizons:
        sweep = []
        for r, C in curves.items():
            Ct = C[:, :T]
            dC = np.abs(np.diff(Ct, axis=1))
            s = float(np.median(Ct.std(-1)))
            sweep.append({"r": r, "rho_img": float(np.median(dC) / max(s, 1e-12))})
        vals = [s["rho_img"] for s in sweep]
        best = min(sweep, key=lambda x: x["rho_img"])
        out[str(T)] = {"argmin_r": best["r"], "contrast": max(vals) / max(min(vals), 1e-30),
                       "sweep": sweep}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f8_horizon.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {m["ckpt"] for m in out["models"]}
    for fam, ckpts in FAMILIES.items():
        for ck in ckpts:
            if ck in done or not pathlib.Path(ck).exists():
                continue
            res = per_model(ck, "runs/pend_dt0.05.npz", DT, HORIZONS)
            rec = {"ckpt": ck, "family": fam, "by_horizon": res}
            print(f"  {fam:14s} {pathlib.Path(ck).name:28s} " +
                  "  ".join(f"T={T}:r={res[str(T)]['argmin_r']:+.2f}/{res[str(T)]['contrast']:.2f}x"
                            for T in HORIZONS), flush=True)
            out["models"].append(rec); op.write_text(json.dumps(out, indent=1) + "\n")

    if len(out["models"]) >= 6:
        summ = {}
        for T in HORIZONS:
            cs_ = [m["by_horizon"][str(T)]["contrast"] for m in out["models"]]
            si = [m["by_horizon"][str(T)]["argmin_r"] for m in out["models"]
                  if m["family"] == "semi-implicit"]
            vv = [m["by_horizon"][str(T)]["argmin_r"] for m in out["models"]
                  if m["family"] == "verlet"]
            summ[str(T)] = {"contrast": [round(c, 2) for c in cs_],
                            "readable_P4": bool(sum(c >= 2.0 for c in cs_) >= 4),
                            "semi_implicit_argmin": si, "verlet_argmin": vv,
                            "separates_P5": bool(sum(x >= 0.5 for x in si) >= 2
                                                 and sum(x <= 0.5 for x in vv) >= 2)}
        out["summary"] = summ
        print(f"\n  {'T':>5}{'contrast (6 models)':>34}{'readable':>10}{'separates':>11}")
        for T in HORIZONS:
            s = summ[str(T)]
            print(f"  {T:>5}{str(s['contrast']):>34}{str(s['readable_P4']):>10}"
                  f"{str(s['separates_P5']):>11}")
        readable = [T for T in HORIZONS if summ[str(T)]["readable_P4"]]
        print(f"\n  readable horizons (P4): {readable or 'NONE'}")
        if not readable:
            print("  -> the imagined-rollout approach cannot adjudicate F6/E19 at any horizon tested.")
    attach(out, op, inputs=sorted({c for v in FAMILIES.values() for c in v} | {'runs/pend_dt0.05.npz'}))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
