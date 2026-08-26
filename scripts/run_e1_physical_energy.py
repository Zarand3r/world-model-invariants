"""E1: does correcting the latent invariant improve the PHYSICS of an imagined rollout?

Pre-registered in `docs/E1_PREREG.md`. The gate experiment of `docs/ROADMAP.md` Phase I, and the
test of claim C2 (physical validity).

`scripts/run_dreamer_edit.py` scores the same intervention with pixel MSE alone, which cannot tell
"the imagined world became more physical" from "a latent regulariser lowered reconstruction error".
Wang (arXiv:2606.24945) states the objection directly: a model can conserve a learned latent scalar
while drifting in true energy. This script decodes the imagined frames and measures the physics.

**The edit is copied verbatim from run_dreamer_edit.py.** Nothing about the intervention is
re-derived here; only the scoring is added. If the two scripts ever disagree on the edit, that is a
bug in this one.

PRIMARY METRIC, fixed before any result (D9 discipline, and E1_PREREG "Primary metric"):

    D_sec = OLS slope of Ehat_k against k, k = 0..H-1,
            divided by the across-trajectory std of true E in the dataset

reported as the median over trajectories. The statistic is the SLOPE OF D_sec ACROSS THE ALPHA GRID,
never the best alpha -- D9 exists because best-over-alpha once reported -5.9% for an arm whose curve
rose monotonically.

Registered noise floor from runs/e1_readout_validation.json: D_sec = 9.4e-04. An arm-A improvement
smaller than that is reported as "below readout resolution", not as an effect.

ARMS
    A  conservative + its own recovered C     registered: D_sec DECREASES with alpha
    B  conservative + 20 norm-matched random  registered: no improvement, or harm
    C  damped + its own recovered C           registered: no improvement, or harm
    D  alpha = 0, shared baseline within every arm
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP = 4, 12, 10
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.4)      # identical for every arm (D9)
HORIZON = 50
N_RANDOM_LAWS = 20
ANALYSIS = slice(204, None)


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        vals = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(vals.sum(), zz)
    return vals.detach(), g.detach()


def _secular_drift(E, norm):
    """OLS slope of E against step index, per trajectory, normalised by `norm`."""
    E = np.asarray(E, dtype=np.float64)
    k = np.arange(E.shape[-1], dtype=np.float64)
    kc = k - k.mean()
    return ((E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)) / norm


def run(ckpt, data, random_law=False, draw=0):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"]
    norm = float(energy(st[..., 0], st[..., 1]).mean(-1).std())
    E_true_window = energy(st[ANALYSIS][:, WARMUP:WARMUP + HORIZON, 0],
                           st[ANALYSIS][:, WARMUP:WARMUP + HORIZON, 1])

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
    F = (((nxt - h_mean) @ U) @ R) - Z

    fit = fit_hamiltonian_pair(Z.double().cpu(), F.double().cpu(), degree=DEGREE, n_basis=8)
    coeffs = torch.as_tensor(fit["coeffs"], dtype=Z.dtype, device=Z.device)
    if random_law:
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)      # same 20 across models
        rc = torch.randn(coeffs.shape[0], generator=g, dtype=torch.float64)
        coeffs = (rc / rc.norm() * coeffs.norm().cpu()).to(Z.dtype).to(Z.device)

    P = U @ R
    P_pinv = torch.linalg.pinv(P)
    ref = fr[ANALYSIS][:, WARMUP: WARMUP + HORIZON]

    per_alpha = {}
    for alpha in ALPHAS:
        with torch.no_grad():
            h = hs[:, WARMUP].clone()
            z0 = (h - h_mean) @ P
            C0, _ = _C_and_grad(z0, coeffs)
            preds, Cs = [], []
            for _ in range(HORIZON):
                preds.append(m.readout_from_h(h))
                Cs.append(_C_and_grad((h - h_mean) @ P, coeffs)[0].cpu().numpy())
                h = m.transition(h)
                if alpha > 0.0:
                    z = (h - h_mean) @ P
                    Cv, gr = _C_and_grad(z, coeffs)
                    step = alpha * ((Cv - C0) / gr.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * gr
                    h = h - (step @ P_pinv)
            img = torch.stack(preds, 1)
            pixel_mse = float(torch.nn.functional.mse_loss(img, ref))
            frames = ((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()

        phys = decode_physics(frames)
        Ehat = phys["energy"]
        dsec = _secular_drift(Ehat, norm)
        blank = float(np.mean(~np.isfinite(phys["theta"])))
        finite = np.isfinite(dsec)
        per_alpha[float(alpha)] = {
            "pixel_mse": pixel_mse,
            "D_sec_per_traj": np.where(finite, dsec, np.nan).tolist(),   # RAW ROW
            "D_sec_median": float(np.nanmedian(dsec)),
            "D_sec_mean_abs": float(np.nanmean(np.abs(dsec))),
            "E_abs_err_vs_true_median": float(np.nanmedian(np.abs(Ehat - E_true_window)) / norm),
            "theta_abs_err_median": float(np.nanmedian(np.abs(
                phys["theta"] - st[ANALYSIS][:, WARMUP:WARMUP + HORIZON, 0]))),
            "C_drift_median": float(np.median(np.abs(np.asarray(Cs)[-1] - np.asarray(Cs)[0]))),
            "blank_decode_fraction": blank,
        }

    a = np.array(ALPHAS, dtype=np.float64)
    dm = np.array([per_alpha[float(x)]["D_sec_median"] for x in ALPHAS])
    pm = np.array([per_alpha[float(x)]["pixel_mse"] for x in ALPHAS])
    return {
        "ckpt": ckpt, "data": data, "random_law": random_law,
        "draw": draw if random_law else None,
        "pairing_residual": fit["residual"],
        "across_traj_E_std": norm,
        "by_alpha": per_alpha,
        # primary statistic: slope of |D_sec| across the frozen grid, and of D_sec itself
        "D_sec_abs_slope": float(np.polyfit(a, np.abs(dm), 1)[0]),
        "D_sec_slope": float(np.polyfit(a, dm, 1)[0]),
        "pixel_normalised_slope": float(np.polyfit(a, pm / max(pm[0], 1e-30), 1)[0]),
        "pixel_relative_change_at_max_alpha": float((pm[-1] - pm[0]) / max(pm[0], 1e-30)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conservative", nargs="+",
                   default=[f"runs/dreamer_ref_s{s}.pt" for s in (3, 4, 5)])
    p.add_argument("--damped", nargs="+",
                   default=[f"runs/dreamer_damped_s{s}.pt" for s in (0, 1, 2)])
    p.add_argument("--conservative-data", default="runs/pendulum_pixels.npz")
    p.add_argument("--damped-data", default="runs/pendulum_pixels_damped.npz")
    p.add_argument("--n-random", type=int, default=N_RANDOM_LAWS)
    p.add_argument("--max-models", type=int, default=999)
    p.add_argument("--out", default="runs/e1_physical_energy.json")
    a = p.parse_args()

    out_path = pathlib.Path(a.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.setdefault("A_conservative_own", [])
    out.setdefault("B_conservative_random", [])
    out.setdefault("C_damped_own", [])

    def save():
        out_path.write_text(json.dumps(out, indent=1) + "\n")

    done_A = {r["ckpt"] for r in out["A_conservative_own"]}
    for ck in a.conservative:
        if ck in done_A or not pathlib.Path(ck).exists():
            continue
        print(f"[A] {ck}", flush=True)
        out["A_conservative_own"].append(run(ck, a.conservative_data)); save()

    done_C = {r["ckpt"] for r in out["C_damped_own"]}
    for ck in a.damped:
        if ck in done_C or not pathlib.Path(ck).exists():
            continue
        print(f"[C] {ck}", flush=True)
        out["C_damped_own"].append(run(ck, a.damped_data)); save()

    done_B = {(r["ckpt"], r["draw"]) for r in out["B_conservative_random"]}
    for ck in a.conservative:
        if not pathlib.Path(ck).exists():
            continue
        for dr in range(a.n_random):
            if (ck, dr) in done_B:
                continue
            print(f"[B] {ck} draw {dr}", flush=True)
            out["B_conservative_random"].append(run(ck, a.conservative_data, True, dr)); save()

    print(f"\nwrote {out_path}  "
          f"A={len(out['A_conservative_own'])} B={len(out['B_conservative_random'])} "
          f"C={len(out['C_damped_own'])}")
