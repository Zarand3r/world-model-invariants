"""F1: does an action-conditioned world model learn a latent BALANCE law?

Pre-registered in `docs/F1_PREREG.md` with the 2026-08-28 amendment. The reported object is the
power-degree sweep, degrees 1-4, against the ground-truth reference curve committed in that
amendment. P1-P3 are read at power degree 1.

  P1  |rho(C, true E)| >= 0.8                       -- the scalar is energy-like
  P2  |rho(power coefficient, true thetadot)| >= 0.8 -- the ACTION COUPLING is the physical one
  P3  balance residual at least 5x below the conserved-only fit on the same latent
  P4  balance residual at least 5x below 20 random (C, P) pairs at matched coefficient norm

P2 is the one that matters. A 34-term power basis reaches a low residual while recovering nothing
(rho 0.51 / 0.07 on ground truth), so a good residual alone proves nothing.

Physical labels are used ONLY here, after the fit is frozen, per the roadmap's rule at line 79. The
fit sees `z` and `a`.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.balance import fit_balance_pair
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args

DEG, LD, WARMUP = 4, 12, 10
ANALYSIS = slice(204, None)
POWER_DEGREES = (1, 2, 3, 4)


def _resid(MZ, MZn, MP, a, mu, mp, c, q):
    """Residual of a FROZEN (c, q) on whatever split is passed in."""
    D = (MZn - mu) - (MZ - mu)
    Rp = a.reshape(-1, 1) * (MP - mp)
    r = D @ c - (Rp @ q if q is not None else 0.0)
    return float(np.median(np.abs(r)) / max(np.std((MZ - mu) @ c), 1e-30))


def run(ckpt, data, n_random=20):
    from latent_noether.dreamer_adapter import DreamerV3Adapter
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    av = torch.as_tensor(d["actions"][ANALYSIS]).float().cuda()
    st = d["states"][ANALYSIS]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr, actions=av).detach()
    H = hs[:, WARMUP:]; A = av[:, WARMUP:, 0]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    # next latent under the model's OWN transition, driven by the action actually applied at t
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1]), a=A.reshape(-1, 1)).reshape(H.shape)
    Zn = ((nxt - hm) @ U) @ R

    # HELD-OUT SPLIT (amendment 2, 2026-08-28). Degree-4 monomials in LD=12 give 1819 coefficients
    # against ~5720 samples, and the in-sample balance residual reaches 0.00000 at degree 4 -- an
    # exact fit that is useless out of sample. Every residual below is therefore fitted on the first
    # half of the analysis trajectories and evaluated on the second.
    half = Z.shape[0] // 2
    def _prep(sl):
        return (Z[sl].reshape(-1, LD).double().cpu().numpy(),
                Zn[sl].reshape(-1, LD).double().cpu().numpy(),
                A[sl].reshape(-1).double().cpu().numpy())
    zt, zn, at = _prep(slice(0, half))
    zt_e, zn_e, at_e = _prep(slice(half, None))
    MZ = monomial_features(torch.tensor(zt), DEG).numpy()
    MZn = monomial_features(torch.tensor(zn), DEG).numpy()
    MZ_e = monomial_features(torch.tensor(zt_e), DEG).numpy()
    MZn_e = monomial_features(torch.tensor(zn_e), DEG).numpy()

    # labels, evaluation only, on the HELD-OUT half
    E_true = d["energy"][ANALYSIS][half:, WARMUP:].reshape(-1)
    thd_true = st[half:, WARMUP:, 1].reshape(-1)
    cor = lambda x, y: float(abs(np.corrcoef(x, y)[0, 1]))

    sweep = []
    for pdeg in POWER_DEGREES:
        MP = monomial_features(torch.tensor(zt), pdeg).numpy()
        MP_e = monomial_features(torch.tensor(zt_e), pdeg).numpy()
        f = fit_balance_pair(MZ, MZn, at, MP=MP)
        mu = MZ.mean(0, keepdims=True); mp = MP.mean(0, keepdims=True)
        rb = _resid(MZ_e, MZn_e, MP_e, at_e, mu, mp, f["c"], f["q"])
        rc = _resid(MZ_e, MZn_e, MP_e, at_e, mu, mp, f["c_conserved_only"], None)
        C = (MZ_e - mu) @ f["c"]; pw = (MP_e - mp) @ f["q"]
        Ccon = (MZ_e - mu) @ f["c_conserved_only"]
        rec = {"power_degree": pdeg, "n_power_terms": int(MP.shape[1]),
               "residual_balance": rb,
               "residual_balance_in_sample": f["residual_balance"],
               "residual_conserved_only": rc,
               "residual_conserved_only_in_sample": f["residual_conserved_only"],
               "ratio_vs_conserved": rc / max(rb, 1e-30),
               "rho_C_energy": cor(C, E_true),
               "rho_power_thetadot": cor(pw, thd_true),
               "rho_conserved_only_energy": cor(Ccon, E_true)}
        print(f"    pdeg {pdeg}  terms {rec['n_power_terms']:3d}  resid {rec['residual_balance']:.5f}  "
              f"ratio {rec['ratio_vs_conserved']:6.1f}x  rho(C,E) {rec['rho_C_energy']:.4f}  "
              f"rho(q,thd) {rec['rho_power_thetadot']:.4f}", flush=True)
        sweep.append(rec)

    # P4: random (C, P) pairs at matched coefficient norm, power degree 1
    MP1 = monomial_features(torch.tensor(zt), 1).numpy()
    MP1_e = monomial_features(torch.tensor(zt_e), 1).numpy()
    base = next(s for s in sweep if s["power_degree"] == 1)
    f1 = fit_balance_pair(MZ, MZn, at, MP=MP1)
    mu = MZ.mean(0, keepdims=True); mp1 = MP1.mean(0, keepdims=True)
    D = (MZn_e - mu) - (MZ_e - mu)
    Rp = at_e.reshape(-1, 1) * (MP1_e - mp1)
    rand = []
    for dr in range(n_random):
        g = np.random.default_rng(1000 + dr)
        rc = g.standard_normal(len(f1["c"])); rc *= np.linalg.norm(f1["c"]) / np.linalg.norm(rc)
        rq = g.standard_normal(len(f1["q"])); rq *= np.linalg.norm(f1["q"]) / max(np.linalg.norm(rq), 1e-30)
        r = D @ rc - Rp @ rq
        rand.append(float(np.median(np.abs(r)) / max(np.std((MZ_e - mu) @ rc), 1e-30)))
    return {"ckpt": ckpt, "sweep": sweep, "random_residuals": rand,
            "P1_pass": bool(base["rho_C_energy"] >= 0.8),
            "P2_pass": bool(base["rho_power_thetadot"] >= 0.8),
            "P3_pass": bool(base["ratio_vs_conserved"] >= 5.0),
            "P4_pass": bool(float(np.median(rand)) / max(base["residual_balance"], 1e-30) >= 5.0)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data", default="runs/pendulum_actuated.npz")
    p.add_argument("--out", default="runs/f1_balance.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        print(f"[F1] {ck}", flush=True)
        rec = run(ck, a.data)
        print(f"    P1 {rec['P1_pass']}  P2 {rec['P2_pass']}  P3 {rec['P3_pass']}  P4 {rec['P4_pass']}")
        out["models"].append(rec); op.write_text(json.dumps(out, indent=1) + "\n")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
