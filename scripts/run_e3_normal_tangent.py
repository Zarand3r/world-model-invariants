"""E3: is one-step error preferentially NORMAL to the level set of C?

Pre-registered in `docs/E3_PREREG.md`. Promoted into Stage 1 after E2 returned Outcome B: a
depth-independent conservation defect should show up as a persistent error component transverse to
the level set, which is the geometric statement of which the E1 projection is the remedy
(Hairer, Geometric Numerical Integration, section IV.4).

At each observation-conditioned state, compare the autonomous step with the true next encoded state:

    dz_t = T(z_t^obs) - z_{t+1}^obs

and with g = grad C(z_t), split it into normal and tangent parts:

    dz_perp = ((g . dz) / ||g||^2) g          f_perp = ||dz_perp||^2 / ||dz||^2

REGISTERED PRIMARY: median f_perp, against the isotropic null 1/LD = 1/12 = 0.08333. An error with
no preferential orientation relative to grad C puts that fraction of its energy in the normal
direction by chance.

    prediction  f_perp > 1/12 with a CI excluding it
    falsifier   f_perp <= 1/12  -- one-step error is not preferentially normal, and the projection's
                benefit needs a different explanation

The random-C arm is the control that matters: a random constraint's gradient bears no relation to
the transition, so its f_perp should sit at 1/12. If random constraints ALSO show f_perp >> 1/12,
the effect is a property of the latent geometry rather than of the recovered invariant.

KNOWN LIMITATION, stated in the prereg before running: `dz` compares an autonomous step against the
ENCODER's next state, and encoder and transition are different maps, so `dz` carries representation
mismatch as well as transition error. E3 measures the orientation of the combined discrepancy --
the right object for explaining the projection, but not a clean measurement of transition error
alone. No claim about the transition in isolation is made from it.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP = 4, 12, 10
ANALYSIS = slice(204, None)
ISOTROPIC_NULL = 1.0 / LD


def _grad_C(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(v.sum(), zz)
    return g.detach()


def run(ckpt, data, random_law=False, draw=0, untrained=False):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    if untrained:
        import hashlib
        torch.manual_seed(int(hashlib.sha256(ckpt.encode()).hexdigest()[:8], 16))
        m = DreamerV3Adapter(device="cuda").cuda()
    else:
        m = DreamerV3Adapter(device="cuda").cuda()
        m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
    m.eval()

    with torch.no_grad():
        hs = m.encode(fr[ANALYSIS]).detach()
    H = hs[:, WARMUP:]
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD)
    Z = (H - h_mean) @ U
    R = effective_rank_basis(Z)
    Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    Znext_auto = ((nxt - h_mean) @ U) @ R
    F = Znext_auto - Z

    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEGREE, 8)
    coeffs = torch.as_tensor(np.asarray(fit["coeffs"]), dtype=Z.dtype, device=Z.device)
    if random_law:
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)
        rc = torch.randn(coeffs.shape[0], generator=g, dtype=torch.float64)
        coeffs = (rc / rc.norm() * coeffs.norm().cpu()).to(Z.dtype).to(Z.device)

    # dz = autonomous step from z_t, minus the encoder's actual z_{t+1}
    dz = (Znext_auto[:, :-1] - Z[:, 1:]).reshape(-1, LD)
    z_t = Z[:, :-1].reshape(-1, LD)
    g = _grad_C(z_t, coeffs)
    gn2 = g.pow(2).sum(-1).clamp_min(1e-30)
    proj = (g * dz).sum(-1) / gn2
    dz_perp = proj.unsqueeze(-1) * g
    n_perp = dz_perp.pow(2).sum(-1)
    n_tot = dz.pow(2).sum(-1).clamp_min(1e-30)
    f_perp = (n_perp / n_tot).cpu().numpy()

    # Per-trajectory MEDIANS, not means. f_perp is strongly right-skewed (median 0.020 against a
    # mean of 0.054 on seed 3), so a per-trajectory mean is a different statistic from the
    # registered median and must not be summarised as one.
    ntraj = Z.shape[0]
    f_by_traj = np.median(f_perp.reshape(ntraj, -1), axis=-1)
    return {
        "ckpt": ckpt, "data": data, "random_law": random_law, "untrained": untrained,
        "draw": draw if random_law else None,
        "isotropic_null": ISOTROPIC_NULL,
        "f_perp_median": float(np.median(f_perp)),
        "f_perp_mean": float(np.mean(f_perp)),
        "f_perp_per_traj_median": f_by_traj.tolist(),   # RAW ROW: bootstrap unit is the trajectory
        "f_perp_per_traj_mean": f_perp.reshape(ntraj, -1).mean(-1).tolist(),
        "dz_norm_median": float(dz.norm(dim=-1).median().cpu()),
        "n_states": int(f_perp.size),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conservative", nargs="+", default=["runs/dreamer_ref_s3_step6500.pt"])
    p.add_argument("--damped", nargs="*", default=[])
    p.add_argument("--untrained", nargs="*", default=[])
    p.add_argument("--conservative-data", default="runs/pendulum_pixels.npz")
    p.add_argument("--damped-data", default="runs/pendulum_pixels_damped.npz")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--out", default="runs/e3_normal_tangent.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {}
    for k in ("A_conservative", "B_random", "C_damped", "D_untrained"):
        out.setdefault(k, [])
    save = lambda: op.write_text(json.dumps(out, indent=1) + "\n")

    for ck in a.conservative:
        if ck in {r["ckpt"] for r in out["A_conservative"]} or not pathlib.Path(ck).exists(): continue
        print(f"[A] {ck}", flush=True); out["A_conservative"].append(run(ck, a.conservative_data)); save()
    for ck in a.damped:
        if ck in {r["ckpt"] for r in out["C_damped"]} or not pathlib.Path(ck).exists(): continue
        print(f"[C] {ck}", flush=True); out["C_damped"].append(run(ck, a.damped_data)); save()
    for ck in a.untrained:
        if ck in {r["ckpt"] for r in out["D_untrained"]} or not pathlib.Path(ck).exists(): continue
        print(f"[D] {ck}", flush=True)
        out["D_untrained"].append(run(ck, a.conservative_data, untrained=True)); save()
    doneB = {(r["ckpt"], r["draw"]) for r in out["B_random"]}
    for ck in a.conservative:
        if not pathlib.Path(ck).exists(): continue
        for dr in range(a.n_random):
            if (ck, dr) in doneB: continue
            out["B_random"].append(run(ck, a.conservative_data, True, dr)); save()
    print(f"wrote {op}")
