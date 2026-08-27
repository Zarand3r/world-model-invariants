"""E10: at matched energy-decodability, does CONSERVATION predict intervention benefit?

Pre-registered in `docs/E10_PREREG.md`. The control the roadmap calls "one of the most important",
and the direct test of the paper's probe-vs-dynamics thesis.

Every null so far compares the recovered `C` against constraints that are BOTH non-conserved and
weakly energy-correlated, so none can separate "conservation matters" from "decodability matters".
This one holds decodability fixed and varies conservation.

Candidates come from the existing eigenfamily (`polynomial_invariants`), each carrying its invariance
ratio. No new fitting.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features, polynomial_invariants

DEGREE, LD, WARMUP, HORIZON, EPS = 4, 12, 10, 100, 0.02
ANALYSIS = slice(204, None)


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def _secular(E, norm):
    E = np.asarray(E, float); k = np.arange(E.shape[-1], dtype=float); kc = k - k.mean()
    return ((E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)) / norm


def run(ckpt, data, n_candidates=40):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"]; norm = float(energy(st[..., 0], st[..., 1]).mean(-1).std())
    E_true = energy(st[ANALYSIS][:, WARMUP:, 0], st[ANALYSIS][:, WARMUP:, 1])

    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    H = hs[:, WARMUP:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - hm) @ U) @ R) - Z
    P = U @ R; Ppinv = torch.linalg.pinv(P)

    recovered = cached_fit(Z.double().cpu(), F.double().cpu(), DEGREE, 8)
    cands = polynomial_invariants(Z.double().cpu(), degree=DEGREE, max_results=n_candidates)

    def score(coef):
        c = torch.as_tensor(np.asarray(coef), dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            q = (monomial_features(Z.reshape(-1, LD), DEGREE) @ c).reshape(Z.shape[:2]).cpu().numpy()
        n = min(q.shape[1], E_true.shape[1])
        a = q[:, :n].ravel(); b = E_true[:, :n].ravel()
        rho = float(abs(np.corrcoef(a, b)[0, 1])) if np.std(a) > 0 else 0.0
        out = {}
        for eps in (0.0, EPS):
            with torch.no_grad():
                h = hs[:, WARMUP].clone(); C0, _ = _C_and_grad((h - hm) @ P, c); preds = []
                for _ in range(HORIZON):
                    preds.append(m.readout_from_h(h)); h = m.transition(h)
                    if eps > 0:
                        z = (h - hm) @ P; Cv, g = _C_and_grad(z, c)
                        u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        h = h - ((eps * torch.sign(Cv - C0).unsqueeze(-1) * u) @ Ppinv)
                img = torch.stack(preds, 1)
                frames = ((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()
            out[eps] = float(np.nanmedian(np.abs(_secular(decode_physics(frames)["energy"], norm))))
        imp = -(out[EPS] - out[0.0]) / max(out[0.0], 1e-30)
        return rho, out[0.0], out[EPS], imp

    rows = []
    rho_c, b_c, a_c, imp_c = score(recovered["coeffs"])
    rows.append({"kind": "recovered", "ratio": None, "rho_E": rho_c,
                 "D0": b_c, "D1": a_c, "improvement": imp_c})
    for i, cd in enumerate(cands):
        rho, b, a, imp = score(cd["coeffs"])
        rows.append({"kind": "eigen", "rank": i, "ratio": float(cd["ratio"]), "rho_E": rho,
                     "D0": b, "D1": a, "improvement": imp})
    return {"ckpt": ckpt, "rows": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3,4,5)])
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--n-candidates", type=int, default=40)
    p.add_argument("--out", default="runs/e10_decodability_matched.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists(): continue
        print(f"[E10] {ck}", flush=True)
        out["models"].append(run(ck, a.data, a.n_candidates))
        op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
