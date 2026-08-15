"""WHY do recovery and pairing residual move in opposite directions on DreamerV3?

Measured (D36): LD 6 -> 12 takes |rho|_E from 0.588 to 0.972 while the pairing residual WORSENS
from 0.756 to 0.879. That breaks the residual as a confidence statistic and is unexplained.

**The hypothesis, from external review.** The physical invariant may be represented more broadly
than the approximately-Hamiltonian part of the latent flow. Adding directions exposes enough
information to recover E, while ALSO exposing representational/world-model dynamics that
`f = B grad C` cannot approximate. That would produce the anti-correlation mechanically.

**Note on a wrong explanation, so it is not reused.** "The residual is bad because the latent is
512-D" is insufficient: the fit happens AFTER projection to LD, so the fit never sees 512
dimensions. What differs is which information the projection keeps.

TEST. Split the LD=12 subspace into the first 6 PCA directions (the LD=6 subspace, nested by
construction) and the added directions 7-12. Then measure, per block:

  - energy information: R^2 of a probe from that block to true E
  - residual contribution: the share of ||f - B grad C||^2 that lands in that block

PREDICTION if the hypothesis holds: the ADDED directions carry substantial energy information
AND a disproportionate share of the residual — i.e. they are where E lives and where the
Hamiltonian approximation fails, simultaneously.

If instead the added directions carry energy information and a PROPORTIONATE share of the
residual, the anti-correlation is not explained by this decomposition and the hypothesis dies.
"""
import argparse, json
import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import _antisym_basis, fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features

DEGREE, G, M, L = 4, 10.0, 1.0, 1.0


def true_energy(states):
    th, thd = states[..., 0], states[..., 1]
    return 0.5 * (M * L ** 2 / 3) * thd ** 2 + M * G * (L / 2) * np.cos(th)


def probe_r2(X, y):
    """Held-out R^2 of a ridge probe from X to y, split by trajectory."""
    n = X.shape[0] // 2
    Xtr, ytr = X[:n].reshape(-1, X.shape[-1]), y[:n].reshape(-1)
    Xte, yte = X[n:].reshape(-1, X.shape[-1]), y[n:].reshape(-1)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    A = (Xtr - mu) / sd
    w = torch.linalg.solve(A.T @ A + 1e-6 * torch.eye(A.shape[1], dtype=A.dtype),
                           A.T @ (ytr - ytr.mean()))
    pred = ((Xte - mu) / sd) @ w + ytr.mean()
    return float(1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())


def run(ckpt, data, ld=12, split=6, warmup=10):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    ck = torch.load(ckpt, map_location="cuda")
    m = DreamerV3Adapter(device="cuda").cuda(); m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        H = m.encode(fr[204:])[:, warmup:].detach()
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    E = torch.as_tensor(true_energy(d["states"])[204:][:, warmup:], dtype=torch.float64)

    U = pca_subspace(H, ld)                      # PCA dirs are ordered, so [:split] IS the LD=6 space
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)
    Z = ((H - h_mean) @ U).double().cpu()
    F = ((((nxt - h_mean) @ U)).double().cpu()) - Z

    fit = fit_hamiltonian_pair(Z, F, degree=DEGREE, n_basis=8)
    coeffs = torch.as_tensor(fit["coeffs"], dtype=Z.dtype)
    zf, ff = Z.reshape(-1, ld), F.reshape(-1, ld)
    zz = zf.detach().requires_grad_(True)
    v = monomial_features(zz, DEGREE) @ coeffs
    gC, = torch.autograd.grad(v.sum(), zz)
    basis = _antisym_basis(ld, Z.dtype)
    A = torch.stack([gC @ b.T for b in basis], -1)
    beta = torch.linalg.lstsq(A.reshape(-1, A.shape[-1]), ff.reshape(-1)).solution
    resid = ff - (A @ beta)                                  # (n, ld) per-direction residual

    first, added = slice(0, split), slice(split, ld)
    out = {"ckpt": ckpt, "ld": ld, "split": split,
           "rho_full": None, "overall_residual": fit["residual"]}
    C = (monomial_features(zf, DEGREE) @ coeffs).reshape(Z.shape[:2])
    n = min(C.shape[1], E.shape[1])
    out["rho_full"] = abs(float(torch.corrcoef(
        torch.stack([C[:, :n].reshape(-1), E[:, :n].reshape(-1)]))[0, 1]))
    for name, sl in (("first6", first), ("added7_12", added)):
        out[f"{name}_energy_r2"] = probe_r2(Z[:, :n, sl], E[:, :n])
        out[f"{name}_resid_share"] = float(resid[:, sl].pow(2).sum() / resid.pow(2).sum())
        out[f"{name}_flow_share"] = float(ff[:, sl].pow(2).sum() / ff.pow(2).sum())
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{i}.pt" for i in range(3)])
    a.add_argument("--out", default="runs/dreamer_residual_decomp.json")
    a = a.parse_args()
    print("Why are recovery and residual anti-correlated on DreamerV3?")
    print("PREDICT: the ADDED dims (7-12) carry energy info AND a disproportionate residual share.\n")
    rows = []
    for ck in a.ckpts:
        r = run(ck, "runs/pendulum_pixels.npz")
        rows.append(r)
        print(f"  {ck}  |rho|_E(LD12) {r['rho_full']:.3f}")
        for nm in ("first6", "added7_12"):
            print(f"    {nm:10s} energy R2 {r[nm+'_energy_r2']:+.3f}   "
                  f"residual share {r[nm+'_resid_share']:.1%}   flow share {r[nm+'_flow_share']:.1%}")
        json.dump(rows, open(a.out, "w"), indent=2)
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"\n--- medians")
    for nm in ("first6", "added7_12"):
        print(f"    {nm:10s} energy R2 {med(nm+'_energy_r2'):+.3f}   "
              f"residual share {med(nm+'_resid_share'):.1%}   flow share {med(nm+'_flow_share'):.1%}")
    print("\n--- VERDICT")
    rs, fs = med("added7_12_resid_share"), med("added7_12_flow_share")
    er = med("added7_12_energy_r2")
    if er > 0.3 and rs > 1.5 * fs:
        print(f"  HYPOTHESIS SUPPORTED. The added directions carry energy information "
              f"(R2 {er:+.2f}) AND")
        print(f"  a disproportionate residual share ({rs:.0%} of residual vs {fs:.0%} of flow).")
        print("  E lives partly where the constant-B Hamiltonian approximation fails, which")
        print("  mechanically produces the anti-correlation.")
    elif er > 0.3:
        print(f"  PARTIAL: added dims carry energy info (R2 {er:+.2f}) but their residual share")
        print(f"  ({rs:.0%}) is proportionate to their flow share ({fs:.0%}). The decomposition")
        print("  does not explain the anti-correlation; the hypothesis is not supported.")
    else:
        print(f"  HYPOTHESIS DEAD: the added dims carry little energy information (R2 {er:+.2f}).")
