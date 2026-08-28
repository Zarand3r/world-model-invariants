"""F4: does the extraction recover the invariant from a second world-model family?

Pre-registered in `docs/F4_PREREG.md`. Identical pipeline to `run_e17_recovery.py` — frozen LD = 12,
degree 4, n_basis = 8, WARMUP = 10 — with only the model class swapped, so any difference is
attributable to the architecture rather than to the measurement.
"""
import argparse, datetime, hashlib, json, pathlib, subprocess
import numpy as np, torch
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.gru_world_model import ConvGRUWorldModel
from latent_noether.pixel_readout import energy
from latent_noether.polynomial import monomial_features, polynomial_invariants

DEG, LD, W = 4, 12, 10
ANALYSIS = slice(204, None)


def run(ckpt, data, n_candidates=40):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    E = energy(st[:, W:, 0], st[:, W:, 1])
    m = ConvGRUWorldModel(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        H = m.encode(fr).detach()[:, W:]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    Zn = (((nxt - hm) @ U) @ R); F = Zn - Z
    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)

    def score(coef):
        c = torch.as_tensor(np.asarray(coef), dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            Cv = (monomial_features(Z.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
            Cn = (monomial_features(Zn.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
        q = Cv.cpu().numpy(); r = (Cn - Cv).cpu().numpy()
        k = min(q.shape[1], E.shape[1])
        return {"rho_E": float(abs(np.corrcoef(q[:, :k].ravel(), E[:, :k].ravel())[0, 1])),
                "ratio_empirical": float(np.mean(np.var(q, axis=1)) / max(np.var(q), 1e-30)),
                "rho_obs": float(np.median(np.abs(r)) / abs(float(Cv.mean(-1).std().cpu())))}

    out = {"ckpt": ckpt, "arch": "ConvGRUWorldModel", "retained_rank": int(Z.shape[-1]),
           "recovered": {**score(fit["coeffs"]), "pairing_residual": float(fit["residual"])}}
    cands = polynomial_invariants(Z.double().cpu(), degree=DEG, max_results=n_candidates)
    out["candidates"] = [{"rank": i, "ratio": float(c["ratio"]), **score(c["coeffs"])}
                         for i, c in enumerate(cands)]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--out", default="runs/f4_recovery.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    # M29 provenance. Recorded per invocation; `runs` accumulates because this script is
    # resumable, so a record written across several sessions keeps one entry per session.
    prov = {"data": a.data, "data_sha256": hashlib.sha256(pathlib.Path(a.data).read_bytes()).hexdigest(),
            "deg": DEG, "ld": LD, "warmup": W, "n_basis": 8,
            "analysis_slice": [ANALYSIS.start, ANALYSIS.stop],
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "ckpts_requested": list(a.ckpts)}
    out.setdefault("provenance", {}).setdefault("runs", []).append(prov)
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        print(f"[F4] {ck}", flush=True)
        out["models"].append(run(ck, a.data)); op.write_text(json.dumps(out, indent=1) + "\n")
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
