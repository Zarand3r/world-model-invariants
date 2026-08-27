"""E4: does SETTING the invariant change the imagined physics in the predicted direction?

Pre-registered in `docs/E4_PREREG.md`. Claim C4 in `docs/ROADMAP.md`, Stage 2.

Everything measured so far is restorative: put `C` back to `C_0` and the rollout gets more physical.
That shows violating `C` costs accuracy. It does not show `C` is a *control variable*. E4 sets `C` to
values it was never at and asks whether the decoded world moves as predicted.

Two protocols, both editing ONCE at the start of the rollout and then running free -- so the test
includes whether the changed regime PERSISTS without further forcing.

  E4a  donor-level (primary). Target C(z_donor) from an independent trajectory. The intervention
       never sees true energy; ground truth enters only when the decoded rollout is compared with
       the donor's true energy. Staying inside the distribution of C values the model actually
       produces means the edit cannot be dismissed as pushing the latent somewhere meaningless.

  E4b  synthetic sweep over a PREREGISTERED offset grid in units of std_traj(C), fixed before
       running so that which offsets get reported cannot be chosen afterwards.

PRIMARY METRIC: Spearman rho between intended change in `C` and realised change in decoded physical
energy. Registered prediction rho > 0 with a bootstrap CI excluding 0; falsifier rho <= 0 or CI
including 0, in which case `C` is not a control variable and C4 is unsupported.

`C` is NOT assumed to be energy -- it may be a monotone nonlinear function of it -- so the registered
evidence is monotonicity and transfer correlation, never slope-1 agreement.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP = 4, 12, 10
ANALYSIS = slice(204, None)
OFFSET_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)   # registered, in units of std_traj(C)
NEWTON_STEPS = 25
MAX_EDIT_RATIO = 5.0                                      # registered exclusion


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def _dial_to(z, target, coeffs):
    """Newton iteration along the local C-normal until C(z) = target. Returns (z_edited, ||dz||)."""
    z0 = z.clone()
    z = z.clone()
    for _ in range(NEWTON_STEPS):
        Cv, g = _C_and_grad(z, coeffs)
        step = ((Cv - target) / g.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * g
        z = z - step
    return z, (z - z0).norm(dim=-1)


def run(ckpt, data, horizon=100, random_law=False, draw=0, tangent=False, seed=0):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    E_true = energy(st[..., 0], st[..., 1])[:, WARMUP:WARMUP + horizon].mean(-1)

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

    with torch.no_grad():
        C_traj = (monomial_features(Z.reshape(-1, LD), DEGREE) @ coeffs).reshape(Z.shape[:2])
    C0_all = C_traj[:, 0]
    std_C = float(C_traj.mean(-1).std().cpu())

    def rollout(h0):
        with torch.no_grad():
            h = h0.clone(); preds = []
            for _ in range(horizon):
                preds.append(m.readout_from_h(h)); h = m.transition(h)
            img = torch.stack(preds, 1)
            return ((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()

    h0 = hs[:, WARMUP].clone()
    base_E = np.nanmedian(decode_physics(rollout(h0))["energy"], axis=-1)

    out = {"ckpt": ckpt, "random_law": random_law, "tangent": tangent, "horizon": horizon,
           "draw": draw if random_law else None, "std_C": std_C,
           "E_true_mean": E_true.tolist(), "base_decoded_E": base_E.tolist()}

    # ---- E4a: donor targets ----
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(h0.shape[0], generator=g)
    z0 = (h0 - hm) @ P
    target = C0_all[perm]
    if tangent:
        rnd = torch.randn(z0.shape, generator=torch.Generator().manual_seed(seed), device="cpu").to(z0.device)
        _, gg = _C_and_grad(z0, coeffs)
        u = gg / gg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rnd = rnd - (rnd * u).sum(-1, keepdim=True) * u
        _, dn = _dial_to(z0, target, coeffs)          # match the NORM of the real edit
        z_ed = z0 + dn.unsqueeze(-1) * rnd / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        edit_norm = dn
    else:
        z_ed, edit_norm = _dial_to(z0, target, coeffs)
    med = float(edit_norm.median())
    keep = (edit_norm <= MAX_EDIT_RATIO * med).cpu().numpy()          # registered exclusion
    h_ed = h0 + (z_ed - z0) @ Ppinv
    E_ed = np.nanmedian(decode_physics(rollout(h_ed))["energy"], axis=-1)
    out["a_intended_dC"] = (target - C0_all).cpu().numpy().tolist()
    out["a_realised_dE"] = (E_ed - base_E).tolist()
    out["a_true_dE"] = (E_true[perm.numpy()] - E_true).tolist()
    out["a_keep"] = keep.tolist()
    out["a_edit_norm"] = edit_norm.cpu().numpy().tolist()

    # ---- E4b: synthetic offset sweep ----
    sweep = {}
    for off in OFFSET_GRID:
        if off == 0.0:
            sweep[str(off)] = {"realised_dE": [0.0] * len(base_E)}; continue
        tgt = C0_all + off * std_C
        z_e, en = _dial_to(z0, tgt, coeffs)
        h_e = h0 + (z_e - z0) @ Ppinv
        Ee = np.nanmedian(decode_physics(rollout(h_e))["energy"], axis=-1)
        sweep[str(off)] = {"realised_dE": (Ee - base_E).tolist(),
                           "edit_norm_median": float(en.median())}
    out["b_sweep"] = sweep
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/dreamer_ref_s3_step6500.pt")
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--n-tangent", type=int, default=5)
    p.add_argument("--out", default="runs/e4_dialing.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"recovered": [], "random": [], "tangent": []}
    save = lambda: op.write_text(json.dumps(out, indent=1) + "\n")
    if not out["recovered"]:
        print("[recovered]", flush=True); out["recovered"].append(run(a.ckpt, a.data, a.horizon)); save()
    done = {r["draw"] for r in out["random"]}
    for dr in range(a.n_random):
        if dr in done: continue
        out["random"].append(run(a.ckpt, a.data, a.horizon, True, dr)); save()
    for dr in range(len(out["tangent"]), a.n_tangent):
        out["tangent"].append(run(a.ckpt, a.data, a.horizon, tangent=True, seed=dr)); save()
    print(f"wrote {op}")
