"""E17 prediction 3: does the repair transfer to two degrees of freedom?

Pre-registered in `docs/E17_PREREG.md`. Uses the **direction-matched** protocol throughout — fixed
step `eps` along each constraint's normal, so arms differ only in direction and never in edit
magnitude, the confound the 2026-08-26 audit found in the original norm-matched null.

Readout is `decode_physics_osc2d`, validated on real rendered frames before any intervention number:
position error 0.001 units, energy error 3.2% of the across-trajectory spread.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics_osc2d
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP = 4, 12, 10
ANALYSIS = slice(204, None)
EPS_GRID = (0.0, 0.005, 0.01, 0.02)


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def _secular(E, norm):
    E = np.asarray(E, float); k = np.arange(E.shape[-1], dtype=float); kc = k - k.mean()
    return ((E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)) / norm


def run(ckpt, data, horizon, b, random_law=False, draw=0, tangent=False):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    norm = float(d["energy"].mean(-1).std())
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    H = hs[:, WARMUP:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - hm) @ U) @ R) - Z
    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEGREE, 8)
    coeffs = torch.as_tensor(np.asarray(fit["coeffs"]), dtype=Z.dtype, device=Z.device)
    if random_law:
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)
        rc = torch.randn(coeffs.shape[0], generator=g, dtype=torch.float64)
        coeffs = (rc / rc.norm() * coeffs.norm().cpu()).to(Z.dtype).to(Z.device)
    P = U @ R; Ppinv = torch.linalg.pinv(P)
    ref = fr[ANALYSIS][:, WARMUP:WARMUP + horizon]

    by_eps = {}
    for eps in EPS_GRID:
        with torch.no_grad():
            h = hs[:, WARMUP].clone(); C0, _ = _C_and_grad((h - hm) @ P, coeffs); preds = []
            for _ in range(horizon):
                preds.append(m.readout_from_h(h)); h = m.transition(h)
                if eps > 0:
                    z = (h - hm) @ P; Cv, g = _C_and_grad(z, coeffs)
                    u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    if tangent:
                        rnd = torch.randn_like(z)
                        rnd = rnd - (rnd * u).sum(-1, keepdim=True) * u
                        u = rnd / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        step = eps * u
                    else:
                        step = eps * torch.sign(Cv - C0).unsqueeze(-1) * u
                    h = h - (step @ Ppinv)
            img = torch.stack(preds, 1)
            pmse = float(torch.nn.functional.mse_loss(img, ref))
            frames = ((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()
        ph = decode_physics_osc2d(frames, b=b)
        ds = _secular(ph["energy"], norm)
        by_eps[float(eps)] = {"pixel_mse": pmse,
                              "D_sec_per_traj": np.where(np.isfinite(ds), ds, np.nan).tolist(),
                              "D_sec_median_abs": float(np.nanmedian(np.abs(ds)))}
    return {"ckpt": ckpt, "random_law": random_law, "tangent": tangent,
            "draw": draw if random_law else None, "horizon": horizon, "by_eps": by_eps}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--b", type=float, default=0.40)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--n-tangent", type=int, default=5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"recovered": [], "random": [], "tangent": []}
    save = lambda: op.write_text(json.dumps(out, indent=1) + "\n")
    if not out["recovered"]:
        out["recovered"].append(run(a.ckpt, a.data, a.horizon, a.b)); save()
    for dr in range(len(out["random"]), a.n_random):
        out["random"].append(run(a.ckpt, a.data, a.horizon, a.b, True, dr)); save()
    for dr in range(len(out["tangent"]), a.n_tangent):
        out["tangent"].append(run(a.ckpt, a.data, a.horizon, a.b, tangent=True, draw=dr)); save()
    print(f"wrote {op}")
