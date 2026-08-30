#!/usr/bin/env python3
"""F8: does the MODEL carry the integrator? Conservation along its own imagined rollout.

Preregistered in docs/F8_PREREG.md.

F6/E19/F7 all measure `rho_obs` from a ONE-STEP prediction on encoded real frames. F7's
cross-evaluation control showed that statistic is determined by the evaluation dataset, because a
good model reproduces the next state of whatever trajectory it is shown. This removes the trajectory:
the data supplies only the initial latent, and every subsequent state comes from `m.transition`.

Everything else -- latent, PCA frame, degree-4 basis, target family -- is inherited unchanged from
run_f6_analysis, so a difference here is attributable to the dynamics being imagined rather than to
the measurement.

Usage:  uv run python scripts/run_f8_imagined.py --out runs/f8_imagined.json
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_analysis import ANALYSIS, DEG, LD, W
from scripts.run_f6_timestep import R_GRID, c_star, energy

DT = 0.05
DATA = "runs/pend_dt0.05.npz"          # same real frames for every model: only the model differs
FAMILIES = {"semi-implicit": [f"runs/f6_dt0.05_s{s}_step6500.pt" for s in (3, 4, 5)],
            "verlet":        [f"runs/f7_verlet_s{s}_step6500.pt" for s in (3, 4, 5)]}


def run(ckpt, data, dt, horizon):
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

    # --- the imagined trajectory: data supplies h_0 only, every later state is the model's ---
    with torch.no_grad():
        h = hs[:, W].clone()
        traj = [h]
        for _ in range(horizon - 1):
            h = m.transition(h)
            traj.append(h)
        Himg = torch.stack(traj, 1)
    finite = bool(torch.isfinite(Himg).all())
    Zimg = ((Himg - hm) @ U) @ R

    X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    MZ = monomial_features(Z.reshape(-1, LD), DEG)
    MZi = monomial_features(Zimg.reshape(-1, LD), DEG)

    Etrue = energy(th, thd); cs = c_star(dt)
    sweep = []
    for r in R_GRID:
        y = (Etrue + r * cs * thd * np.sin(th)).ravel()[:len(X)]
        w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        c = torch.as_tensor(w / (np.linalg.norm(w) + 1e-30), dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            Creal = (MZ @ c).reshape(Z.shape[:2])
            Cimg = (MZi @ c).reshape(Zimg.shape[:2])
        dC = (Cimg[:, 1:] - Cimg[:, :-1]).cpu().numpy()
        s_img = float(Cimg.std(-1).median().cpu())      # per-trajectory spread, imagined
        s_real = float(Creal.std(-1).median().cpu())    # per-trajectory spread, real
        sweep.append({"r": float(r),
                      "rho_img": float(np.median(np.abs(dC)) / max(s_img, 1e-12)),
                      "std_ratio_img_over_real": s_img / max(s_real, 1e-12)})
    best = min(sweep, key=lambda x: x["rho_img"])
    alive = float(np.median([s["std_ratio_img_over_real"] for s in sweep]))
    return {"ckpt": ckpt, "dt": dt, "horizon": horizon, "sweep": sweep,
            "argmin_r": best["r"], "rho_img_at_best": best["rho_img"],
            "rho_img_at_r0": next(x["rho_img"] for x in sweep if x["r"] == 0.0),
            "rho_img_at_r1": next(x["rho_img"] for x in sweep if x["r"] == 1.0),
            "rollout_finite": finite, "alive_std_ratio": alive,
            "P2_alive": bool(finite and alive >= 0.20)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--out", default="runs/f8_imagined.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {m["ckpt"] for m in out["models"]}

    for fam, ckpts in FAMILIES.items():
        for ck in ckpts:
            if ck in done or not pathlib.Path(ck).exists():
                continue
            r = run(ck, DATA, DT, a.horizon)
            r["family"] = fam
            print(f"  {fam:14s} {pathlib.Path(ck).name:28s} argmin r={r['argmin_r']:+.2f}  "
                  f"rho_img r=0 {r['rho_img_at_r0']:.5f} r=1 {r['rho_img_at_r1']:.5f}  "
                  f"alive {r['alive_std_ratio']:.2f}  P2 {r['P2_alive']}", flush=True)
            out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")

    if len(out["models"]) >= 6:
        import statistics as st
        by = {f: [m for m in out["models"] if m["family"] == f] for f in FAMILIES}
        p2 = all(m["P2_alive"] for m in out["models"])
        n_si = sum(m["argmin_r"] >= 0.5 for m in by["semi-implicit"])
        n_vv = sum(m["argmin_r"] <= 0.5 for m in by["verlet"])
        # P3 (POST-HOC -- see docs/F8_PREREG.md amendment 1). P2 was registered to catch a rollout
        # that COLLAPSES. It did not anticipate an alive rollout whose sweep is nearly FLAT, which is
        # what happened: contrast across the whole r grid is 1.12x-1.45x, against ~5.7x for the
        # one-step measure between r=0 and r=1 alone. A flat sweep makes the argmin noise, so "the
        # families do not separate" would be a statement about the instrument, not the models.
        # Added after seeing the data. It can only WITHHOLD a conclusion, never create one.
        contrasts = [max(s["rho_img"] for s in m["sweep"]) / min(s["rho_img"] for s in m["sweep"])
                     for m in out["models"]]
        p3 = all(c >= 2.0 for c in contrasts)
        summ = {"P2_all_alive": bool(p2),
                "P3_sweep_contrast": [round(c, 3) for c in contrasts],
                "P3_discriminating": bool(p3), "P3_note": "post-hoc gate, added 2026-08-30",
                "alive_std_ratio": {f: [m["alive_std_ratio"] for m in by[f]] for f in by},
                "semi_implicit_argmin_r": [m["argmin_r"] for m in by["semi-implicit"]],
                "verlet_argmin_r": [m["argmin_r"] for m in by["verlet"]],
                "P1_semi_implicit_ge_0.5": f"{n_si}/3", "P1_verlet_le_0.5": f"{n_vv}/3",
                "P1_pass": bool(n_si >= 2 and n_vv >= 2)}
        out["summary"] = summ
        print(f"\n  semi-implicit argmin {summ['semi_implicit_argmin_r']}   "
              f"verlet argmin {summ['verlet_argmin_r']}")
        if not p2:
            print("  P2 FAILED -- imagined rollouts collapsed; P1 is NOT to be read.")
        elif not p3:
            print(f"  P2 pass (alive) but P3 FAILS: sweep contrast {[round(c,2) for c in contrasts]} "
                  f"vs 5.7x for the one-step measure.")
            print("  -> UNINFORMATIVE. The sweep barely varies with r, so the argmin is noise and")
            print("     P1 is NOT to be read. This neither supports nor refutes F6/E19.")
        else:
            print(f"  P2 pass (rollouts alive).  P1 {summ['P1_pass']} "
                  f"(semi-implicit {n_si}/3 >= 0.5, verlet {n_vv}/3 <= 0.5)")
            print("  -> the model carries the integrator" if summ["P1_pass"]
                  else "  -> the families do NOT separate: F6/E19 model-side claims fail as F7's did")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
