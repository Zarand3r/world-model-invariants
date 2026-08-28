"""E17: does the frozen extraction recover invariants from a 2-DoF system?

Pre-registered in `docs/E17_PREREG.md`. Claim C6 (generality), plus E10, which turned out not to be
constructible on the pendulum.

**Frozen hyperparameters, no tuning**: LD = 12, degree 4, n_basis = 8, WARMUP = 10 -- exactly the
settings fixed for the pendulum, whose physical state is 2-dimensional. This state is 4-dimensional.
Failure at these settings is a result about generality and is reported as one; any adapted setting
becomes a separate experiment.

Registered predictions:
  1. non-central: one scalar with |rho_E| > 0.8, invariance ratio within an order of magnitude of
     the pendulum's
  2. central: a TWO-dimensional conserved subspace, second direction with |rho_L| > 0.8
  3. E10 becomes constructible: >= 8 candidates within +-0.10 of the recovered C's |rho_E|
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features, polynomial_invariants
from latent_noether.provenance import attach, inputs_from_args

DEGREE, LD, WARMUP = 4, 12, 10
ANALYSIS = slice(204, None)


def _corr(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0: return 0.0
    return float(abs(np.corrcoef(a[ok], b[ok])[0, 1]))


def run(ckpt, data, n_candidates=60):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    E = d["energy"][ANALYSIS][:, WARMUP:]; L = d["angmom"][ANALYSIS][:, WARMUP:]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    H = hs[:, WARMUP:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - hm) @ U) @ R) - Z

    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEGREE, 8)
    def evaluate(coef):
        c = torch.as_tensor(np.asarray(coef), dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            q = (monomial_features(Z.reshape(-1, LD), DEGREE) @ c).reshape(Z.shape[:2]).cpu().numpy()
        n = min(q.shape[1], E.shape[1])
        within = float(np.mean(np.var(q, axis=1))); total = float(np.var(q))
        return {"rho_E": _corr(q[:, :n], E[:, :n]), "rho_L": _corr(q[:, :n], L[:, :n]),
                "ratio_empirical": within / max(total, 1e-30)}

    out = {"ckpt": ckpt, "data": data, "retained_rank": int(Z.shape[-1]),
           "recovered": {**evaluate(fit["coeffs"]), "pairing_residual": float(fit["residual"])}}
    cands = polynomial_invariants(Z.double().cpu(), degree=DEGREE, max_results=n_candidates)
    rows = []
    for i, cd in enumerate(cands):
        rows.append({"rank": i, "ratio": float(cd["ratio"]), **evaluate(cd["coeffs"])})
    out["candidates"] = rows
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n-candidates", type=int, default=60)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists(): continue
        print(f"[E17] {ck}", flush=True)
        out["models"].append(run(ck, a.data, a.n_candidates))
        op.write_text(json.dumps(out, indent=1) + "\n")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
