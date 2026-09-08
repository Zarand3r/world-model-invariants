#!/usr/bin/env python3
"""F14: does the probe-vs-operator gap survive fitting and scoring on DISJOINT trajectories?

Preregistered in docs/F14_PREREG.md. No training.

E18 fits C on frames[204:] and scores rho_obs on the same 52 trajectories. The label-free arm is
fitted to MINIMISE conservation error and then graded on conservation error, on the same data.
Here: fit on frames[0:204], freeze coefficients / h_mean / PCA subspace / rank basis, score on
frames[204:256] -- the exact slice the in-sample number uses, so it is like-for-like.

Usage:  uv run python scripts/run_f14_crossfit.py --out runs/f14_crossfit.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach
from scripts.run_f6_analysis import DEG, LD, W
from scripts.run_f6_timestep import energy

DATA = "runs/pendulum_pixels.npz"
FIT = slice(0, 204)          # disjoint from SCORE
SCORE = slice(204, None)     # the exact slice E18's in-sample number uses
CKPTS = [f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3, 4, 5)]


def latents(m, frames):
    with torch.no_grad():
        hs = m.encode(frames).detach()
    return hs[:, W:]


def score_with(coeff, m, H, hm, U, R):
    """rho_obs and rho_E for a FROZEN readout on a held-out set."""
    Z = ((H - hm) @ U) @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    Zn = ((nxt - hm) @ U) @ R
    c = torch.as_tensor(np.asarray(coeff), dtype=Z.dtype, device=Z.device)
    with torch.no_grad():
        Cv = (monomial_features(Z.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
        Cn = (monomial_features(Zn.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
    r = (Cn - Cv).cpu().numpy()
    return Cv, float(np.median(np.abs(r)) / abs(float(Cv.mean(-1).std().cpu())))


def run(ckpt, data):
    d = np.load(data)
    fr_fit = torch.as_tensor(d["frames"][FIT]).float().div_(255.).sub_(0.5).cuda()
    fr_sc = torch.as_tensor(d["frames"][SCORE]).float().div_(255.).sub_(0.5).cuda()
    st_fit, st_sc = d["states"][FIT], d["states"][SCORE]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()

    # --- fit everything on FIT, including the frame, then freeze ---
    Hf = latents(m, fr_fit)
    hm = Hf.reshape(-1, Hf.shape[-1]).mean(0)
    U = pca_subspace(Hf, LD); Zf = (Hf - hm) @ U; R = effective_rank_basis(Zf); Zf = Zf @ R
    with torch.no_grad():
        nxt_f = m.transition(Hf.reshape(-1, Hf.shape[-1])).reshape(Hf.shape)
    Ff = (((nxt_f - hm) @ U) @ R) - Zf
    unsup = np.asarray(cached_fit(Zf.double().cpu(), Ff.double().cpu(), DEG, 8)["coeffs"])
    Ef = energy(st_fit[:, W:, 0], st_fit[:, W:, 1])
    Xf = monomial_features(Zf.reshape(-1, LD), DEG).double().cpu().numpy()
    yf = Ef.ravel()[:len(Xf)]
    sup = np.linalg.lstsq(Xf.T @ Xf + 1e-6 * np.eye(Xf.shape[1]), Xf.T @ yf, rcond=None)[0]

    # --- score both, frozen, on the disjoint SCORE set ---
    Hs = latents(m, fr_sc)
    Es = energy(st_sc[:, W:, 0], st_sc[:, W:, 1])
    out = {"ckpt": ckpt}
    for tag, coeff in (("unsupervised", unsup), ("supervised", sup)):
        Cv, rho = score_with(coeff / (np.linalg.norm(coeff) + 1e-30), m, Hs, hm, U, R)
        q = Cv.cpu().numpy(); k = min(q.shape[1], Es.shape[1])
        out[tag] = {"rho_obs": rho,
                    "rho_E": float(abs(np.corrcoef(q[:, :k].ravel(), Es[:, :k].ravel())[0, 1]))}
    out["gap_crossfit"] = out["supervised"]["rho_obs"] / max(out["unsupervised"]["rho_obs"], 1e-30)
    out["G1_readout_meaningful"] = bool(out["unsupervised"]["rho_E"] >= 0.5)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f14_crossfit.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {m["ckpt"] for m in out["models"]}
    for ck in CKPTS:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        r = run(ck, DATA)
        print(f"  {pathlib.Path(ck).name:28s} unsup rho_obs {r['unsupervised']['rho_obs']:.5f} "
              f"(rho_E {r['unsupervised']['rho_E']:.3f})   sup {r['supervised']['rho_obs']:.5f} "
              f"(rho_E {r['supervised']['rho_E']:.3f})   gap {r['gap_crossfit']:.2f}x", flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")

    ms = out["models"]
    if len(ms) >= 3:
        import statistics as st
        gaps = [m["gap_crossfit"] for m in ms]
        summ = {"gaps_crossfit": [round(g, 2) for g in gaps],
                "median_gap_crossfit": round(st.median(gaps), 2), "in_sample_gap": 6.7,
                "G1_pass": bool(all(m["G1_readout_meaningful"] for m in ms)),
                "P1_seeds_ge_3x": sum(g >= 3.0 for g in gaps),
                "P1_pass": bool(sum(g >= 3.0 for g in gaps) >= 2),
                "falsifier_fired": bool(sum(g < 2.0 for g in gaps) >= 2),
                "rho_E_crossfit": [round(m["unsupervised"]["rho_E"], 3) for m in ms]}
        out["summary"] = summ
        print(f"\n  cross-fit gaps {summ['gaps_crossfit']}  median {summ['median_gap_crossfit']}x "
              f"(in-sample 6.7x)")
        print(f"  label-free rho_E cross-fit {summ['rho_E_crossfit']}   G1 {summ['G1_pass']}")
        print(f"  P1 {summ['P1_pass']} ({summ['P1_seeds_ge_3x']}/3 at >=3x)")
        if summ["falsifier_fired"]:
            print("  -> FALSIFIER FIRED: a material part of the headline was in-sample fitting.")
        elif not summ["G1_pass"]:
            print("  -> G1 failed: the frozen readout broke. Uninformative, not negative.")
    attach(out, op, inputs=sorted({DATA} | {m["ckpt"] for m in out["models"]}))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
