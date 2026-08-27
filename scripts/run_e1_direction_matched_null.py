"""E1 control: separate the DIRECTION of the correction from its MAGNITUDE.

Found during the 2026-08-26 audit. The E1 edit

    z <- z - alpha (C(z) - C0) grad C / ||grad C||^2

is a Newton step to the level set, so its size scales with how badly the constraint is violated.
A random constraint is violated far more than the recovered one, so it takes far larger steps:
measured on seed 3 / step 6500, median ||dz_edit|| is 2.80e-01 for random draws against 9.57e-03 for
the recovered `C` -- a factor of **29**.

That makes the headline arm-B comparison unmatched. "0/20 random constraints improve anything" may
mean "random constraints perturb the latent 29x harder", not "random constraints point the wrong
way". Norm-matching the coefficients does not fix this and never did: the Newton step is invariant
under C -> lambda*C, so coefficient norm has no effect on edit magnitude whatsoever.

This script removes the confound by moving a FIXED distance along each constraint's normal:

    z <- z - eps * sign(C(z) - C0) * grad C / ||grad C||

Both arms then take identical-size steps and differ only in direction. Sweeping `eps` over a grid
that brackets arm A's natural step size gives a dose-response in which magnitude is held fixed.

Registered prediction, written before running: if the recovered `C` is a genuinely privileged
direction, it still beats the null at matched `eps`. If the null catches up once magnitude is
equalised, the E1 specificity claim is an artefact of step size and must be withdrawn.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.fit_cache import cached_fit
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP = 4, 12, 10
ANALYSIS = slice(204, None)
EPS_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02)     # brackets arm A's measured 9.57e-03


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        vals = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(vals.sum(), zz)
    return vals.detach(), g.detach()


def _secular(E, norm):
    E = np.asarray(E, float); k = np.arange(E.shape[-1], dtype=float); kc = k - k.mean()
    return ((E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)) / norm


def run(ckpt, data, horizon, random_law=False, draw=0, tangent=False):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"]; norm = float(energy(st[..., 0], st[..., 1]).mean(-1).std())
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
                preds.append(m.readout_from_h(h))
                h = m.transition(h)
                if eps > 0.0:
                    z = (h - hm) @ P
                    Cv, gr = _C_and_grad(z, coeffs)
                    u = gr / gr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    if tangent:
                        # equal-norm TANGENT control: a random direction with its normal
                        # component removed, so it cannot change C to first order
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
        ph = decode_physics(frames)
        ds = _secular(ph["energy"], norm)
        by_eps[float(eps)] = {"pixel_mse": pmse, "D_sec_per_traj": np.where(np.isfinite(ds), ds, np.nan).tolist(),
                              "D_sec_median_abs": float(np.nanmedian(np.abs(ds)))}
    return {"ckpt": ckpt, "random_law": random_law, "tangent": tangent,
            "draw": draw if random_law else None, "horizon": horizon, "by_eps": by_eps}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dreamer_ref_s3_step6500.pt")
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--n-tangent", type=int, default=5)
    p.add_argument("--out", default="runs/e1_direction_matched.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"recovered": [], "random": [], "tangent": []}
    save = lambda: op.write_text(json.dumps(out, indent=1) + "\n")
    if not out["recovered"]:
        print("[recovered]", flush=True); out["recovered"].append(run(a.ckpt, a.data, a.horizon)); save()
    done = {r["draw"] for r in out["random"]}
    for dr in range(a.n_random):
        if dr in done: continue
        print(f"[random {dr}]", flush=True); out["random"].append(run(a.ckpt, a.data, a.horizon, True, dr)); save()
    doneT = {r["draw"] for r in out["tangent"]}
    for dr in range(a.n_tangent):
        if dr in doneT: continue
        print(f"[tangent {dr}]", flush=True)
        out["tangent"].append(run(a.ckpt, a.data, a.horizon, False, dr, tangent=True)); save()
    print(f"wrote {op}")
