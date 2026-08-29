"""F6 analysis: sweep r = c/(2.5 dt) on each trained model and test the scaling law.

Pre-registered in `docs/F6_PREREG.md`. The sweep is relative, so the prediction is a SINGLE value of
`r` at every timestep. Everything else -- latent, PCA frame, degree-4 basis, rho_obs -- is inherited
unchanged from E19, so a difference here is attributable to the timestep and not to the measurement.
"""
import argparse, json, pathlib, re
import numpy as np, torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args
from scripts.run_f6_timestep import R_GRID, c_star, energy

DEG, LD, W = 4, 12, 10
ANALYSIS = slice(204, None)


def run(ckpt, data, dt):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    th, thd = st[:, W:, 0], st[:, W:, 1]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr).detach()
    H = hs[:, W:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    Zn = ((nxt - hm) @ U) @ R
    X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    MZ = monomial_features(Z.reshape(-1, LD), DEG)
    MZn = monomial_features(Zn.reshape(-1, LD), DEG)
    Etrue = energy(th, thd); cs = c_star(dt)
    sweep = []
    for r in R_GRID:
        y = (Etrue + r * cs * thd * np.sin(th)).ravel()[:len(X)]
        w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        c = torch.as_tensor(w / (np.linalg.norm(w) + 1e-30), dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            Cv = (MZ @ c).reshape(Z.shape[:2]); Cn = (MZn @ c).reshape(Z.shape[:2])
        res = (Cn - Cv).cpu().numpy()
        sweep.append({"r": float(r),
                      "rho_obs": float(np.median(np.abs(res)) / abs(float(Cv.mean(-1).std().cpu())))})
    best = min(sweep, key=lambda x: x["rho_obs"])
    return {"ckpt": ckpt, "dt": dt, "c_star_predicted": cs, "sweep": sweep,
            "argmin_r": best["r"], "c_recovered": best["r"] * cs,
            "rho_obs_at_r1": next(x["rho_obs"] for x in sweep if x["r"] == 1.0),
            "rho_obs_at_rm1": next(x["rho_obs"] for x in sweep if x["r"] == -1.0),
            "rho_obs_at_r0": next(x["rho_obs"] for x in sweep if x["r"] == 0.0)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", default="runs/f6_models.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists(): continue
        dt = float(re.search(r"dt([0-9.]+)_s", ck).group(1))
        r = run(ck, f"runs/pend_dt{dt}.npz", dt)
        print(f"  {pathlib.Path(ck).name:30s} dt={dt:<6} argmin r={r['argmin_r']:+.2f}  "
              f"c_rec={r['c_recovered']:.4f} (pred {r['c_star_predicted']:.4f})  "
              f"rho_obs r=1 {r['rho_obs_at_r1']:.5f} vs r=0 {r['rho_obs_at_r0']:.5f} vs r=-1 {r['rho_obs_at_rm1']:.5f}",
              flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
