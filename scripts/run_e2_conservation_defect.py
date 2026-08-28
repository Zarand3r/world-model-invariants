"""E2: where does conservation fail -- on the data manifold, or off it?

Pre-registered in `docs/E2_PREREG.md`. Claim C3 (failure mechanism) in `docs/ROADMAP.md`.

The paper shows `C` is nearly constant on observation-conditioned trajectories and drifts under
autonomous imagination. Two very different mechanisms give that same summary: a small systematic
per-step bias that accumulates, or a transition that is faithful to `C` only on states the
observation distribution supports. E2 separates them with a LOCAL statistic -- one autonomous step
applied at a given state -- which the existing drift measure cannot do.

    r(z) = C(T(z)) - C(z)

measured at real encoded states (`r_obs`) and at rollout depth k (`r_auto(k)`), normalised by the
across-trajectory standard deviation of `C`:

    rho(k) = median_traj |r_auto(k)| / std_traj(C)

Registered outcomes, thresholds fixed before any value was computed:
    A  support loss     rho(49)/rho_obs >= 3 and slope CI excludes 0
    B  systematic bias  rho(49)/rho_obs <  3 and slope CI includes 0
    C  phase coupling   a single sinusoid at the libration frequency explains >= 0.5 of var(r_auto)

Off-support is measured by ONE preregistered metric: whitened nearest-neighbour distance to the
observation-conditioned latents. E2 establishes ASSOCIATION between leaving support and defect
growth, never causation -- that needs E3/E12.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.fit_cache import cached_fit
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args

DEGREE, LD, WARMUP = 4, 12, 10
DEPTH = 50                     # registered: k = 0..49, identical to E1's horizon
ANALYSIS = slice(204, None)


def _C(z, coeffs):
    return monomial_features(z, DEGREE) @ coeffs


def run(ckpt, data, random_law=False, draw=0, untrained=False, depth=DEPTH):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    m = DreamerV3Adapter(device="cuda").cuda()
    if not untrained:
        m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
    else:
        import hashlib
        torch.manual_seed(int(hashlib.sha256(ckpt.encode()).hexdigest()[:8], 16))
        m = DreamerV3Adapter(device="cuda").cuda()
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
    F = (((nxt - h_mean) @ U) @ R) - Z

    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEGREE, 8)
    coeffs = torch.as_tensor(np.asarray(fit["coeffs"]), dtype=Z.dtype, device=Z.device)
    if random_law:
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)
        rc = torch.randn(coeffs.shape[0], generator=g, dtype=torch.float64)
        coeffs = (rc / rc.norm() * coeffs.norm().cpu()).to(Z.dtype).to(Z.device)

    P = U @ R
    with torch.no_grad():
        # --- normaliser: across-trajectory std of C on observation-conditioned states ---
        C_obs = _C(Z.reshape(-1, Z.shape[-1]), coeffs).reshape(Z.shape[:2])
        norm = float(C_obs.mean(-1).std().cpu())

        # --- r_obs: one autonomous step from every real encoded state ---
        z_next_obs = (((nxt - h_mean) @ U) @ R)
        r_obs = (_C(z_next_obs.reshape(-1, LD), coeffs).reshape(Z.shape[:2]) - C_obs).cpu().numpy()

        # --- r_auto(k): the same local statistic at rollout depth k ---
        Zobs_flat = Z.reshape(-1, Z.shape[-1]).double()
        cov = torch.cov(Zobs_flat.T) + 1e-6 * torch.eye(Zobs_flat.shape[-1], device=Z.device, dtype=torch.float64)
        Lw = torch.linalg.cholesky(torch.linalg.inv(cov))
        ref_w = Zobs_flat @ Lw

        h = hs[:, WARMUP].clone()
        r_auto, nnd = [], []
        for _ in range(depth):
            z = (h - h_mean) @ P
            h_next = m.transition(h)
            z_next = (h_next - h_mean) @ P
            r_auto.append((_C(z_next, coeffs) - _C(z, coeffs)).cpu().numpy())
            zw = z.double() @ Lw
            nnd.append(torch.cdist(zw, ref_w).min(-1).values.cpu().numpy())
            h = h_next
    r_auto = np.asarray(r_auto).T          # (traj, depth)
    nnd = np.asarray(nnd).T

    # --- registered Outcome C test: does ONE sinusoid explain >= 0.5 of var(r_auto(k))? ---
    # rho(k) oscillates rather than growing, so the phase-coupling outcome must be evaluated on its
    # own terms and not defaulted past. Best single frequency per trajectory via the FFT peak
    # (excluding DC), variance explained by that component alone.
    ra = r_auto - r_auto.mean(-1, keepdims=True)
    spec = np.abs(np.fft.rfft(ra, axis=-1)) ** 2
    spec[:, 0] = 0.0
    peak = spec.max(-1)
    total = spec.sum(-1)
    frac = np.divide(peak, total, out=np.zeros_like(peak), where=total > 0)
    # rfft power is one-sided: a real sinusoid puts its energy in one bin, so peak/total is the
    # variance-explained fraction directly for interior bins.
    outcome_c_frac = float(np.median(frac))
    peak_bin = np.argmax(spec, axis=-1)

    rho_obs = float(np.median(np.abs(r_obs)) / abs(norm))
    rho_k = np.median(np.abs(r_auto), axis=0) / abs(norm)
    k = np.arange(depth, dtype=np.float64)
    return {
        "ckpt": ckpt, "data": data, "random_law": random_law, "untrained": untrained,
        "draw": draw if random_law else None, "depth": depth,
        "pairing_residual": fit["residual"], "C_across_traj_std": norm,
        "rho_obs": rho_obs,
        "rho_k": rho_k.tolist(),                                   # RAW ROW
        "r_auto_per_traj_last": r_auto[:, -1].tolist(),             # RAW ROW
        "r_obs_median_abs": float(np.median(np.abs(r_obs))),
        "nn_dist_median_k": np.median(nnd, axis=0).tolist(),        # RAW ROW
        "rho_last": float(rho_k[-1]),
        "rho_ratio_last_over_obs": float(rho_k[-1] / rho_obs) if rho_obs > 0 else float("inf"),
        "rho_slope": float(np.polyfit(k, rho_k, 1)[0]),
        "outcome_c_sinusoid_var_frac_median": outcome_c_frac,
        "outcome_c_peak_bin_median": float(np.median(peak_bin)),
        "outcome_c_period_steps_median": float(depth / max(np.median(peak_bin), 1e-9)),
        "r_auto_per_traj": r_auto.tolist(),          # RAW ROW: full series, for re-analysis
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conservative", nargs="+", default=["runs/dreamer_ref_s3_step6500.pt"])
    p.add_argument("--damped", nargs="*", default=[])
    p.add_argument("--untrained", nargs="*", default=[])
    p.add_argument("--conservative-data", default="runs/pendulum_pixels.npz")
    p.add_argument("--damped-data", default="runs/pendulum_pixels_damped.npz")
    p.add_argument("--n-random", type=int, default=0)
    p.add_argument("--depth", type=int, default=DEPTH)
    p.add_argument("--out", default="runs/e2_conservation_defect.json")
    a = p.parse_args()

    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {}
    for key in ("A_conservative", "B_random", "C_damped", "D_untrained"):
        out.setdefault(key, [])
    save = lambda: op.write_text(json.dumps(out, indent=1) + "\n")

    for ck in a.conservative:
        if ck in {r["ckpt"] for r in out["A_conservative"]} or not pathlib.Path(ck).exists():
            continue
        print(f"[A] {ck}", flush=True)
        out["A_conservative"].append(run(ck, a.conservative_data, depth=a.depth)); save()
    for ck in a.damped:
        if ck in {r["ckpt"] for r in out["C_damped"]} or not pathlib.Path(ck).exists():
            continue
        print(f"[C] {ck}", flush=True)
        out["C_damped"].append(run(ck, a.damped_data, depth=a.depth)); save()
    for ck in a.untrained:
        if ck in {r["ckpt"] for r in out["D_untrained"]} or not pathlib.Path(ck).exists():
            continue
        print(f"[D-untrained] {ck}", flush=True)
        out["D_untrained"].append(run(ck, a.conservative_data, untrained=True, depth=a.depth)); save()
    done_B = {(r["ckpt"], r["draw"]) for r in out["B_random"]}
    for ck in a.conservative:
        if not pathlib.Path(ck).exists():
            continue
        for dr in range(a.n_random):
            if (ck, dr) in done_B:
                continue
            print(f"[B] {ck} draw {dr}", flush=True)
            out["B_random"].append(run(ck, a.conservative_data, True, dr, depth=a.depth)); save()
    attach(out, op, inputs=inputs_from_args(a))
    save()
    print(f"\nwrote {op}  A={len(out['A_conservative'])} B={len(out['B_random'])} "
          f"C={len(out['C_damped'])} D={len(out['D_untrained'])}")
