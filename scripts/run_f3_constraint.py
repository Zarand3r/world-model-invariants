"""F3 Stage A: can a constraint residual G(z) = 0 be extracted from a world model's latent?

Pre-registered in `docs/F3_PREREG.md`. Validated on a constraint we already have: the pendulum's rod
length is fixed, so the ink centroid's distance from the calibrated pivot is constant across every
frame of every trajectory. Measured on the training data: relative std 0.67%, with within-trajectory
and across-trajectory spread BOTH tiny -- the signature of a constraint, not an orbit label.

Why the existing search cannot find it, measured rather than argued: that radius has an invariance
ratio of ~0.97 (within-variance over total-variance), so `polynomial_invariants` ranks it as one of
the WORST conserved directions available. The objective here is different:

    minimise  E[ G(z)^2 ]   subject to ||a|| = 1,        G = a . phi(z)

i.e. the smallest eigenvector of the feature second-moment matrix, not of `T^-1 W`.

Two details that matter, both learned the hard way in this project:

  * Features are scaled by their RMS, NOT centred. Centring would delete the constant monomial, and
    a constraint needs that offset to be expressible as an affine function equal to zero.
  * Everything is evaluated on HELD-OUT trajectories, because a second-moment objective over 1819
    coefficients will happily memorise (F1's balance fit reached a residual of exactly 0.00000
    in-sample and was useless out of it).
"""
import argparse, json, pathlib
import numpy as np, torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import centroids_from_frames, load_pivot
from latent_noether.polynomial import monomial_features, polynomial_invariants
from latent_noether.provenance import attach, inputs_from_args

DEG, LD, WARMUP = 4, 12, 10
ANALYSIS = slice(204, None)
N_RANDOM = 20


def _rms_scale(P):
    s = np.sqrt((P ** 2).mean(0))
    s[s < 1e-12] = 1.0
    return s


def run(ckpt, data, seed=0):
    d = np.load(data)
    frames = d["frames"][ANALYSIS][:, WARMUP:]
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr).detach()
    H = hs[:, WARMUP:]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R

    # the known constraint, used ONLY for evaluation after the fit is frozen
    piv = load_pivot()
    cy, cx, _ = centroids_from_frames(frames)
    G_true = (np.sqrt((cy - piv) ** 2 + (cx - piv) ** 2) - 7.8871).ravel()

    n_traj = Z.shape[0]; half = n_traj // 2
    def prep(sl):
        return monomial_features(Z[sl].reshape(-1, Z.shape[-1]).double().cpu(), DEG).numpy()
    P_tr, P_te = prep(slice(0, half)), prep(slice(half, None))
    G_true_te = G_true.reshape(n_traj, -1)[half:].ravel()

    scale = _rms_scale(P_tr)
    X_tr, X_te = P_tr / scale, P_te / scale
    M = X_tr.T @ X_tr / len(X_tr)
    M = 0.5 * (M + M.T) + 1e-8 * np.trace(M) / M.shape[0] * np.eye(M.shape[0])
    w, V = np.linalg.eigh(M)
    a = V[:, 0]                                     # smallest second moment, unit norm

    sec = lambda vec: float(((X_te @ vec) ** 2).mean())
    G_te = X_te @ a
    cor = lambda x, y: float(abs(np.corrcoef(x, y)[0, 1]))

    # A2: the best ORBIT-LABEL invariant the existing search returns, in the same scaled basis
    cands = polynomial_invariants(Z.double().cpu(), degree=DEG, max_results=8)
    inv_secs = []
    for c in cands:
        v = np.asarray(c["coeffs"], float) * scale   # raw-space coeffs -> scaled basis
        n = np.linalg.norm(v)
        if n > 0:
            inv_secs.append(sec(v / n))
    best_inv = float(min(inv_secs)) if inv_secs else float("nan")

    # A3: random unit directions
    rng = np.random.default_rng(1000 + seed)
    rand = []
    for _ in range(N_RANDOM):
        v = rng.standard_normal(len(a)); rand.append(sec(v / np.linalg.norm(v)))

    return {"ckpt": ckpt, "n_train_traj": half, "n_test_traj": n_traj - half,
            "second_moment_G": sec(a),
            "second_moment_best_invariant": best_inv,
            "second_moment_random_median": float(np.median(rand)),
            "rho_G_Gtrue": cor(G_te, G_true_te),
            "ratio_vs_invariant": best_inv / max(sec(a), 1e-30),
            "ratio_vs_random": float(np.median(rand)) / max(sec(a), 1e-30),
            "A1_pass": bool(cor(G_te, G_true_te) >= 0.5),
            "A2_pass": bool(best_inv / max(sec(a), 1e-30) >= 5.0),
            "A3_pass": bool(float(np.median(rand)) / max(sec(a), 1e-30) >= 5.0)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3, 4, 5)])
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--out", default="runs/f3_constraint.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        r = run(ck, a.data)
        print(f"  {ck.split('/')[-1]:28s} E[G^2] {r['second_moment_G']:.3e}  "
              f"rho(G,Gtrue) {r['rho_G_Gtrue']:.4f}  vs-inv {r['ratio_vs_invariant']:7.1f}x  "
              f"vs-rand {r['ratio_vs_random']:8.1f}x   A1 {r['A1_pass']} A2 {r['A2_pass']} A3 {r['A3_pass']}",
              flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
