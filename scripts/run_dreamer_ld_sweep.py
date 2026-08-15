"""Was LD=6 ever right for a DreamerV3 latent? Sweep it, and report the whole curve.

**Why this is not tuning-to-rescue.** LD=6 was chosen on a 64-unit GRU whose latent has a
participation ratio of 2.13 of 6 — sharply anisotropic, two dominant directions. The DreamerV3
latent is 512-dimensional with a participation ratio of 5.04-5.61 of 6, i.e. NEARLY ISOTROPIC over
whatever we keep. N7 already established that extraction dimension is load-bearing in BOTH
directions (over-weighting low-variance directions collapses recovery to 0.30; discarding them
collapses it to 0.54). Inheriting a hyperparameter across substrates with different geometry is a
methodological error, not a conservative choice.

**Discipline.** The FULL curve is reported, not the best point. If recovery is flat in LD, the
S2 result stands as measured and LD was never the issue. If it peaks sharply, that is reported
WITH the caveat that LD was selected post hoc on this substrate — which weakens the number
relative to a pre-registered one, and must be said in the paper.

Also reported per LD: participation ratio and the fraction of latent variance retained, so the
reader can see what is being kept rather than trusting a scalar.
"""
import argparse, json
import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features

DEGREE = 4
G, M, L = 10.0, 1.0, 1.0
LDS = (2, 3, 4, 6, 8, 12, 16)


def true_energy(states):
    th, thd = states[..., 0], states[..., 1]
    return 0.5 * (M * L ** 2 / 3) * thd ** 2 + M * G * (L / 2) * np.cos(th)


def run(ckpt, data, warmup=10):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    E_all = true_energy(d["states"])
    ck = torch.load(ckpt, map_location="cuda")
    m = DreamerV3Adapter(device="cuda").cuda(); m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        H = m.encode(fr[204:])[:, warmup:].detach()
        nxt_full = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    E = torch.as_tensor(E_all[204:][:, warmup:], dtype=torch.float64)

    flat = H.reshape(-1, H.shape[-1])
    tot_var = float((flat - flat.mean(0)).pow(2).sum())
    h_mean = flat.mean(0)
    rows = []
    for ld in LDS:
        U = pca_subspace(H, ld)
        Z = (H - h_mean) @ U
        R = effective_rank_basis(Z)
        Z = Z @ R
        F = (((nxt_full - h_mean) @ U) @ R) - Z
        kept = float(((H - h_mean) @ U).pow(2).sum()) / tot_var
        cov = torch.cov(Z.reshape(-1, Z.shape[-1]).T)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0); p = ev / ev.sum().clamp_min(1e-30)
        pr = float(1.0 / (p ** 2).sum().clamp_min(1e-30))
        Zc, Fc = Z.double().cpu(), F.double().cpu()
        fit = fit_hamiltonian_pair(Zc, Fc, degree=DEGREE, n_basis=8)
        c = torch.as_tensor(fit["coeffs"], dtype=Zc.dtype)
        C = (monomial_features(Zc.reshape(-1, Zc.shape[-1]), DEGREE) @ c).reshape(Zc.shape[:2])
        n = min(C.shape[1], E.shape[1])
        rho = abs(float(torch.corrcoef(torch.stack(
            [C[:, :n].reshape(-1), E[:, :n].reshape(-1)]))[0, 1]))
        rows.append({"ckpt": ckpt, "ld": ld, "eff_dim": int(Z.shape[-1]), "rho_energy": rho,
                     "pairing_residual": fit["residual"], "participation_ratio": pr,
                     "variance_kept": kept})
        print(f"    LD {ld:2d} (eff {Z.shape[-1]:2d})  |rho|_E {rho:.3f}  residual "
              f"{fit['residual']:.3f}  PR {pr:.2f}  var kept {kept:.1%}", flush=True)
    return rows


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpts", nargs="+",
                   default=[f"runs/dreamer_ref_s{i}.pt" for i in range(3)])
    a.add_argument("--out", default="runs/dreamer_ld_sweep.json")
    a = a.parse_args()
    print("Was LD=6 right for a DreamerV3 latent? Full curve, not the best point.\n")
    rows = []
    for ck in a.ckpts:
        print(f"  {ck}")
        rows += run(ck, "runs/pendulum_pixels.npz")
        json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\n--- medians across seeds")
    print(f"    {'LD':>4}{'|rho|_E':>10}{'residual':>11}{'PR':>7}{'var kept':>10}{'pass>0.7':>10}")
    for ld in LDS:
        v = [r for r in rows if r["ld"] == ld]
        rh = [r["rho_energy"] for r in v]
        print(f"    {ld:>4}{np.median(rh):>10.3f}"
              f"{np.median([r['pairing_residual'] for r in v]):>11.3f}"
              f"{np.median([r['participation_ratio'] for r in v]):>7.2f}"
              f"{np.median([r['variance_kept'] for r in v]):>9.1%}"
              f"{sum(1 for x in rh if x>0.7)}/{len(rh):>8}")
