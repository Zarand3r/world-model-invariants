"""E14b: out of distribution, is energy still decodable, and is it still conserved?

Pre-registered in `docs/E14B_PREREG.md`. Refits the whole pipeline on OOD latents at frozen
hyperparameters and reports identification (`rho_E`) and conservation (`rho_obs`) separately.

Written as a script because the original run was done inline and its numbers never reached a run
record -- caught by the evidence-base guard in `make_results_summary.py`, which counts per-seed
files. Every headline number in this project must regenerate from `runs/*.json`; this one could not.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import energy
from latent_noether.polynomial import monomial_features

DEG, LD, W = 4, 12, 10
ANALYSIS = slice(204, None)


def _fit_and_score(m, fr, E_true, rand=False, draw=0):
    with torch.no_grad():
        H = m.encode(fr).detach()[:, W:]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    Zn = (((nxt - hm) @ U) @ R); F = Zn - Z
    c = torch.as_tensor(np.asarray(cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)["coeffs"]),
                        dtype=Z.dtype, device=Z.device)
    if rand:
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)
        rc = torch.randn(c.shape[0], generator=g, dtype=torch.float64)
        c = (rc / rc.norm() * c.norm().cpu()).to(Z.dtype).to(Z.device)
    with torch.no_grad():
        Cv = (monomial_features(Z.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
        Cn = (monomial_features(Zn.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
    r = (Cn - Cv).cpu().numpy(); norm = float(Cv.mean(-1).std().cpu())
    q = Cv.cpu().numpy(); k = min(q.shape[1], E_true.shape[1])
    return {"rho_E": float(abs(np.corrcoef(q[:, :k].ravel(), E_true[:, :k].ravel())[0, 1])),
            "rho_obs": float(np.median(np.abs(r)) / abs(norm))}


def _free_probe(m, fr, E_true):
    """Is the information present at all? A free least-squares readout, no conservation constraint."""
    with torch.no_grad():
        H = m.encode(fr).detach()[:, W:]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    y = np.asarray(E_true).ravel()[:len(X)]
    w = np.linalg.lstsq(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y, rcond=None)[0]
    return float(abs(np.corrcoef(X @ w, y)[0, 1]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3, 4, 5)])
    p.add_argument("--train-data", default="runs/pendulum_pixels.npz")
    p.add_argument("--ood-data", default="runs/pendulum_ood_low.npz")
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--out", default="runs/e14b_ood_conservation.json")
    a = p.parse_args()
    tr = np.load(a.train_data); ood = np.load(a.ood_data)
    frt = torch.as_tensor(tr["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    fro = torch.as_tensor(ood["frames"]).float().div_(255.).sub_(0.5).cuda()
    Ei = energy(tr["states"][ANALYSIS][:, W:, 0], tr["states"][ANALYSIS][:, W:, 1])
    Eo = energy(ood["states"][:, W:, 0], ood["states"][:, W:, 1])
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        print(f"[E14b] {ck}", flush=True)
        m = DreamerV3Adapter(device="cuda").cuda()
        m.load_state_dict(torch.load(ck, map_location="cuda")["model"]); m.eval()
        rec = {"ckpt": ck, "ood_data": a.ood_data,
               "in_dist": _fit_and_score(m, frt, Ei),
               "ood": _fit_and_score(m, fro, Eo),
               "free_probe_in_dist": _free_probe(m, frt, Ei),
               "free_probe_ood": _free_probe(m, fro, Eo),
               "ood_random": [_fit_and_score(m, fro, Eo, rand=True, draw=d)["rho_obs"]
                              for d in range(a.n_random)]}
        rec["degradation"] = rec["ood"]["rho_obs"] / rec["in_dist"]["rho_obs"]
        out["models"].append(rec)
        op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
