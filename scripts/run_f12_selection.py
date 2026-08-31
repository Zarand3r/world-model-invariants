#!/usr/bin/env python3
"""F12: does rho_obs pick better checkpoints than validation loss?

Preregistered in docs/F12_PREREG.md. No training -- existing checkpoints only.

Target is open-loop PIXEL fidelity, deliberately not energy drift: rho_obs is a conservation
statistic, so scoring it against a conservation target would be near-circular.

Usage:  uv run python scripts/run_f12_selection.py --out runs/f12_selection.json
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
from scripts.run_f6_analysis import ANALYSIS, DEG, LD, W
from scripts.run_f6_timestep import energy

DATA = "runs/pendulum_pixels.npz"
SEEDS = (3, 4, 5)
STEPS = (1000, 3000, 6500, 15000, 30000, 60000)
HORIZON = 100


def val_recon_at(seed, step):
    h = json.loads(pathlib.Path(f"runs/dreamer_ref_s{seed}_hist.json").read_text())
    best = min((r for r in h if r.get("val_recon") is not None),
               key=lambda r: abs(r["step"] - step), default=None)
    return (float(best["val_recon"]), int(best["step"])) if best else (None, None)


def run(ckpt, data):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    E = energy(st[:, W:, 0], st[:, W:, 1])
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr).detach()
    H = hs[:, W:]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    Zn = (((nxt - hm) @ U) @ R); F = Zn - Z
    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)
    c = torch.as_tensor(np.asarray(fit["coeffs"]), dtype=Z.dtype, device=Z.device)
    with torch.no_grad():
        Cv = (monomial_features(Z.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
        Cn = (monomial_features(Zn.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
    q = Cv.cpu().numpy(); r = (Cn - Cv).cpu().numpy()
    k = min(q.shape[1], E.shape[1])
    rho_E = float(abs(np.corrcoef(q[:, :k].ravel(), E[:, :k].ravel())[0, 1]))
    rho_obs = float(np.median(np.abs(r)) / abs(float(Cv.mean(-1).std().cpu())))

    # --- the target: open-loop pixel fidelity over HORIZON steps ---
    with torch.no_grad():
        h = hs[:, W].clone(); preds = []
        for _ in range(HORIZON):
            preds.append(m.readout_from_h(h)); h = m.transition(h)
        img = torch.stack(preds, 1)
        ref = fr[:, W:W + HORIZON]
        fidelity = float(torch.nn.functional.mse_loss(img, ref))
    return {"rho_obs": rho_obs, "rho_E": rho_E, "fidelity_pixel_mse": fidelity}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-30))


def partial_spearman(a, b, ctrl):
    """Spearman(a,b) with ctrl partialled out -- G2: both rankers improve with training step."""
    a, b, c = (np.argsort(np.argsort(np.asarray(x, float))).astype(float) for x in (a, b, ctrl))
    def resid(y):
        A = np.stack([c, np.ones_like(c)], 1)
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    ra, rb = resid(a), resid(b)
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f12_selection.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {m["ckpt"] for m in out["models"]}
    for s in SEEDS:
        for step in STEPS:
            ck = f"runs/dreamer_ref_s{s}_step{step}.pt"
            if ck in done or not pathlib.Path(ck).exists():
                continue
            rec = run(ck, DATA)
            vr, vstep = val_recon_at(s, step)
            rec |= {"ckpt": ck, "seed": s, "step": step, "val_recon": vr, "val_step": vstep}
            print(f"  s{s} step{step:<6} rho_obs {rec['rho_obs']:.5f}  rho_E {rec['rho_E']:.4f}  "
                  f"val_recon {vr if vr is None else round(vr,5)}  fidelity {rec['fidelity_pixel_mse']:.6f}",
                  flush=True)
            out["models"].append(rec); op.write_text(json.dumps(out, indent=1) + "\n")

    ms = [m for m in out["models"] if m["val_recon"] is not None]
    if len(ms) >= 12:
        fid = [m["fidelity_pixel_mse"] for m in ms]
        g1 = max(fid) / max(min(fid), 1e-30)
        summ = {"n": len(ms), "G1_fidelity_spread": g1, "G1_pass": bool(g1 >= 2.0)}
        for name, key in (("rho_obs", "rho_obs"), ("val_recon", "val_recon"), ("rho_E", "rho_E")):
            v = [m[key] for m in ms]
            summ[f"spearman_{name}"] = spearman(v, fid)
            summ[f"partial_{name}"] = partial_spearman(v, fid, [m["step"] for m in ms])
        summ["P1_pass"] = bool(abs(summ["spearman_rho_obs"]) > abs(summ["spearman_val_recon"]))
        summ["P2_pass"] = bool(abs(summ["spearman_rho_obs"]) > abs(summ["spearman_rho_E"]))
        summ["G2_pass"] = bool(abs(summ["partial_rho_obs"]) > abs(summ["partial_val_recon"]))
        per = {}
        for s in SEEDS:
            g = [m for m in ms if m["seed"] == s]
            if len(g) >= 4:
                f_ = [x["fidelity_pixel_mse"] for x in g]
                per[str(s)] = {"rho_obs": spearman([x["rho_obs"] for x in g], f_),
                               "val_recon": spearman([x["val_recon"] for x in g], f_)}
        summ["P3_per_seed"] = per
        summ["P3_pass"] = bool(sum(abs(v["rho_obs"]) > abs(v["val_recon"]) for v in per.values()) >= 2)
        out["summary"] = summ
        print(f"\n  G1 fidelity spread {g1:.1f}x  -> {'pass' if summ['G1_pass'] else 'FAIL (nothing to rank)'}")
        print(f"  Spearman vs fidelity:  rho_obs {summ['spearman_rho_obs']:+.3f}   "
              f"val_recon {summ['spearman_val_recon']:+.3f}   rho_E {summ['spearman_rho_E']:+.3f}")
        print(f"  partialling out step:  rho_obs {summ['partial_rho_obs']:+.3f}   "
              f"val_recon {summ['partial_val_recon']:+.3f}   rho_E {summ['partial_rho_E']:+.3f}")
        print(f"  P1 {summ['P1_pass']}   P2 {summ['P2_pass']}   P3 {summ['P3_pass']}   G2 {summ['G2_pass']}")
        if summ["P1_pass"] and not summ["G2_pass"]:
            print("  -> rho_obs wins on the RAW ranking only. G2 says that is a training-duration")
            print("     proxy, not a selection result. Do not claim one.")
    attach(out, op, inputs=sorted({DATA} | {m["ckpt"] for m in out["models"]}))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
