"""F1 positive control: is the balance extraction adequate at the MODEL's operating point?

Registered in `docs/F1_PREREG.md`, amendment 3, with both outcomes interpreted before running.

Ground-truth states in periodic coordinates, where the balance law is known and exactly
representable, embedded in 12 dimensions by a random linear map, then run through the IDENTICAL
pipeline the model gets: PCA to LD=12, effective-rank basis, degree-4 C basis, power-degree sweep,
held-out split, and the same trajectory and step counts as the analysis set.

The embedding preserves information, so a method adequate at LD=12 with this sample size must still
recover the law. If it does not, F1 is inconclusive rather than negative.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.balance import fit_balance_pair
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args
import scripts.run_f1_gate0 as G
from scripts.run_f1_balance import DEG, LD, POWER_DEGREES, _resid

N_TRAJ, N_STEPS = 52, 110      # matches the F1 analysis set exactly
LAGS = 4                       # 4 lags x 3 coords = 12, matching the model's retained rank


def main(seed, embed_dim, out, n_traj=N_TRAJ):
    rng = np.random.default_rng(seed)
    TH, THD, U = G.simulate(n_traj * 3, N_STEPS + LAGS, 1.0, 5, seed)
    TH, THD, U = TH[:n_traj], THD[:n_traj], U[:n_traj]
    assert TH.shape[0] >= n_traj, 'clip rejection left too few trajectories'
    coords = np.stack([np.cos(TH), np.sin(TH), THD], -1)            # (traj, T, 3)
    # TIME-LAGGED embedding (amendment 4). A random linear map cannot manufacture rank: it collapsed
    # straight back to 3, because the system has 3 degrees of freedom. Lagged coordinates of a
    # NONLINEAR system are not linearly dependent -- cos(th_{t-1}) is a nonlinear function of the
    # state at t -- so the linear rank is genuinely higher while the information is unchanged and the
    # balance law stays exactly representable through the t block. It is also the closer analogue of
    # an RSSM, whose deterministic state encodes history rather than the instantaneous state.
    lags = [coords[:, LAGS - 1 - k: coords.shape[1] - k] for k in range(LAGS)]
    coords = np.concatenate(lags, axis=-1)                           # (traj, T-LAGS+1, 3*LAGS)
    TH, THD, U = (x[:, LAGS - 1:] for x in (TH, THD, U))
    W = rng.standard_normal((coords.shape[-1], embed_dim))
    Hh = torch.tensor(coords @ W).float()
    hm = Hh.reshape(-1, Hh.shape[-1]).mean(0)
    Uu = pca_subspace(Hh, LD); Z = (Hh - hm) @ Uu
    R = effective_rank_basis(Z); Z = Z @ R
    Zt, Zn = Z[:, :-1], Z[:, 1:]
    A = torch.tensor(U[:, :-1]).float()
    half = n_traj // 2
    prep = lambda sl: (Zt[sl].reshape(-1, Zt.shape[-1]).double().numpy(),
                       Zn[sl].reshape(-1, Zt.shape[-1]).double().numpy(),
                       A[sl].reshape(-1).double().numpy())
    zt, zn, at = prep(slice(0, half)); zt_e, zn_e, at_e = prep(slice(half, None))
    MZ = monomial_features(torch.tensor(zt), DEG).numpy()
    MZn = monomial_features(torch.tensor(zn), DEG).numpy()
    MZ_e = monomial_features(torch.tensor(zt_e), DEG).numpy()
    MZn_e = monomial_features(torch.tensor(zn_e), DEG).numpy()
    th_e = TH[half:, :-1].reshape(-1); thd_e = THD[half:, :-1].reshape(-1)
    E = G.energy(th_e, thd_e); H = G.shadow(th_e, thd_e)
    cor = lambda x, y: float(abs(np.corrcoef(x, y)[0, 1]))
    print(f"  latent dim {Z.shape[-1]}, C terms {MZ.shape[1]}, train samples {len(MZ)}, "
          f"held-out {len(MZ_e)}")
    sweep = []
    for pdeg in POWER_DEGREES:
        MP = monomial_features(torch.tensor(zt), pdeg).numpy()
        MP_e = monomial_features(torch.tensor(zt_e), pdeg).numpy()
        f = fit_balance_pair(MZ, MZn, at, MP=MP)
        mu = MZ.mean(0, keepdims=True); mp = MP.mean(0, keepdims=True)
        rb = _resid(MZ_e, MZn_e, MP_e, at_e, mu, mp, f["c"], f["q"])
        rc = _resid(MZ_e, MZn_e, MP_e, at_e, mu, mp, f["c_conserved_only"], None)
        C = (MZ_e - mu) @ f["c"]; pw = (MP_e - mp) @ f["q"]
        rec = {"power_degree": pdeg, "residual_balance": rb, "residual_conserved_only": rc,
               "ratio_vs_conserved": rc / max(rb, 1e-30),
               "rho_C_energy": cor(C, E), "rho_C_shadow": cor(C, H),
               "rho_power_thetadot": cor(pw, thd_e)}
        print(f"    pdeg {pdeg}  resid {rb:.5f}  ratio {rec['ratio_vs_conserved']:6.2f}x  "
              f"rho(C,E) {rec['rho_C_energy']:.4f}  rho(C,H~) {rec['rho_C_shadow']:.4f}  "
              f"rho(q,thd) {rec['rho_power_thetadot']:.4f}", flush=True)
        sweep.append(rec)
    b = sweep[0]
    res = {"embed_dim": embed_dim, "seed": seed, "n_traj": n_traj, "sweep": sweep,
           "control_recovers": bool(b["rho_C_shadow"] >= 0.8 and b["rho_power_thetadot"] >= 0.8
                                    and b["ratio_vs_conserved"] >= 5.0)}
    print(f"  control_recovers = {res['control_recovers']}  (registered: rho(C,H~)>=0.8, "
          f"rho(q,thd)>=0.8, ratio>=5x at power degree 1)")
    op = pathlib.Path(out); attach(res, op, inputs=[])
    op.write_text(json.dumps(res, indent=1) + "\n"); print(f"wrote {op}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--n-traj", type=int, default=N_TRAJ)
    p.add_argument("--out", default="runs/f1_positive_control.json")
    a = p.parse_args()
    main(a.seed, a.embed_dim, a.out, a.n_traj)
