"""ARCHITECTURE DIAGNOSIS: does Dreamer under-protect the state that matters physically?

S4 established the causal bridge: the model represents physical structure, its imagination lets that
structure drift, and correcting the drift from outside improves prediction. The architectural
question that follows is WHERE that structure lives, and whether the model's own representation
gives it any prominence.

For each latent direction u (unit norm, in the extracted subspace) we measure two things:

    V(u)  = Var(u^T z)                      statistical prominence -- what PCA sees
    D_H(u) = E[ L_H(z + eps u) - L_H(z) ]   causal leverage -- how much an H-step rollout degrades
                                            when that direction is displaced

and separately how hard the S4 projection actually pushes on each direction.

THE HYPOTHESIS, from the reviewer and from N7/D37. If the directions S4 edits have SMALL V and
LARGE D, then Dreamer assigns little statistical prominence to state with disproportionately large
physical consequence. That is an architectural failure mode, and it names its own fix: protect
latent state by downstream consequence rather than by variance.

REGISTERED PREDICTIONS (before running):
  P1  rank correlation between V(u) and D_H(u) across directions is NEGATIVE or near zero.
      A strongly POSITIVE correlation kills the hypothesis: it would mean variance already tracks
      consequence and there is nothing for an architecture to fix.
  P2  the directions the S4 projection moves most are concentrated in the low-V, high-D quadrant.
  KILL: if D_H is flat across directions (max/min < 2), there is no leverage structure to protect
  and the diagnosis is void rather than negative.

eps is one fixed absolute displacement for every direction, so D is comparable across them: it
answers "how bad is it to be wrong by this much, in this direction". Scaling eps per direction
would instead answer "how bad is a typical error", which is the question PCA already answers.

CALIBRATING eps, because the first attempt at this measurement was worthless. At eps = 0.05|z| the
damage came out at -0.5% of baseline with the sign varying between directions, and the verdict logic
duly reported that variance tracks leverage. The rollout is exactly deterministic (a repeated
zero-perturbation rollout differs by 0.0), so that was not RNG noise: eps was simply too small for
the response to leave the second-order regime. Damage against eps for the leading direction:

    eps/|z|   0.05     0.10     0.25     0.50     1.00
    D/base   -0.5%    +3.1%   +18.1%   +46.4%  +128.8%

We therefore fix eps = 0.25|z|, the smallest displacement whose damage clearly exceeds 10% of
baseline while the response is still growing smoothly. Reported so the choice is visible.

THE DISSIPATIVE CONTROL, REQUIRED BY M26. D59 reported an error-geometry result on the conservative
arm and read it as physical structure; D60 killed that reading by running the same measurement on
the damped models, which showed the same thing (flow share 0.545 vs 0.564). Every claim here is of
the same kind, so point this script at the damped checkpoints and data before reading any of it as
being about conservation:

    --ckpts runs/dreamer_damped_s{0,1,2}.pt --data runs/pendulum_pixels_damped.npz

Run 2026-08-13, that control SEPARATES rather than matching: rho(V, D) is negative on 3/3
conservative models and positive on 3/3 damped ones, and rho(D, edit) likewise inverts. So the
variance/consequence mismatch is specific to models that carry a conserved quantity. Caveat kept in
the paper: damped trajectories collapse toward a fixed point, so their latent geometry differs for
reasons beyond the absence of an invariant, and the separation is not attributed to conservation
alone without a smooth-dynamics control that has no physics at all.
"""
import argparse
import json

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP, HORIZON = 4, 12, 10, 40
ALPHA = 0.4                       # the strongest projection in the S4 grid


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()), 1e-30))


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def run(ckpt, data, eps_frac=0.25):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
    m.eval()

    val = slice(204, None)
    with torch.no_grad():
        hs = m.encode(fr[val]).detach()
    H = hs[:, WARMUP:]
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD)
    Z = (H - h_mean) @ U
    R = effective_rank_basis(Z)
    Z = Z @ R
    P = U @ R                                        # (D, r) hidden <- latent
    r = Z.shape[-1]

    # V(u): statistical prominence of each extracted direction
    V = Z.reshape(-1, r).var(0).cpu().numpy()
    eps = float(eps_frac * Z.reshape(-1, r).norm(dim=-1).mean())

    ref = fr[val][:, WARMUP: WARMUP + HORIZON]
    h0 = hs[:, WARMUP].clone()

    def rollout_loss(h_start):
        with torch.no_grad():
            h, preds = h_start, []
            for _ in range(HORIZON):
                preds.append(m.readout_from_h(h))
                h = m.transition(h)
            return float(torch.nn.functional.mse_loss(torch.stack(preds, 1), ref))

    base = rollout_loss(h0)

    # Does an injected displacement decay through this transition, or grow? A contracting transition
    # would make small latent errors self-correcting and the whole diagnosis moot. Measured, because
    # this number was quoted in the paper from a docstring with no run behind it (2026-08-13 audit):
    # retention = ||h_perturbed - h_clean|| / ||initial displacement|| after HORIZON steps, averaged
    # over the extracted directions.
    def retention(frac):
        e = float(frac * Z.reshape(-1, r).norm(dim=-1).mean())
        keep = []
        for i in range(r):
            with torch.no_grad():
                h_a, h_b = h0.clone(), h0 + e * P[:, i]
                for _ in range(HORIZON):
                    h_a, h_b = m.transition(h_a), m.transition(h_b)
                keep.append(float((h_b - h_a).norm(dim=-1).mean() / e))
        return float(np.mean(keep))

    retain = {f"{f}": retention(f) for f in (0.05, 0.25, 0.5)}
    # D_H(u): causal leverage. One fixed absolute displacement per direction, both signs averaged,
    # so a direction is not scored well merely because the model happens to sit on one side of it.
    D = []
    for i in range(r):
        dirn = P[:, i]
        D.append(0.5 * ((rollout_loss(h0 + eps * dirn) - base)
                        + (rollout_loss(h0 - eps * dirn) - base)))
    D = np.array(D)

    # how hard does the S4 projection push on each direction?
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - h_mean) @ U) @ R) - Z
    fit = fit_hamiltonian_pair(Z.double().cpu(), F.double().cpu(), degree=DEGREE, n_basis=8)
    coeffs = torch.as_tensor(fit["coeffs"], dtype=Z.dtype, device=Z.device)
    with torch.no_grad():
        h, moves = h0.clone(), []
        z0 = (h - h_mean) @ P
        C0, _ = _C_and_grad(z0, coeffs)
        for _ in range(HORIZON):
            h = m.transition(h)
            z = (h - h_mean) @ P
            Cv, g = _C_and_grad(z, coeffs)
            step = ALPHA * ((Cv - C0) / g.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * g
            moves.append(step.abs().mean(0).cpu().numpy())
            h = h - (step @ torch.linalg.pinv(P))
    edit_move = np.mean(moves, 0)

    # Q1 of D52: the correction is rank-1 at every state, so the spectrum of M_C = E[gg^T/||g||^2]
    # measures how much grad C ROTATES across state space. Reported here rather than only in the
    # A2d script so that pointing this script at the damped arm returns every section-7 statistic
    # in one place (M26).
    zz = Z.reshape(-1, r).detach().requires_grad_(True)
    with torch.enable_grad():
        gv = monomial_features(zz, DEGREE) @ coeffs
        gg, = torch.autograd.grad(gv.sum(), zz)
    gn = gg.detach()
    gn = gn / gn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    ev = torch.linalg.eigvalsh(((gn.T @ gn) / gn.shape[0]).double()).flip(0)
    top3 = float(ev[:3].sum() / ev.sum())

    return {"ckpt": ckpt, "eps": eps, "baseline_loss": base, "top3_eigenmass": top3,
            "retention_by_eps_frac": retain,
            "variance": V.tolist(), "leverage": D.tolist(), "edit_move": edit_move.tolist(),
            "rho_V_D": spearman(V, D), "rho_V_edit": spearman(V, edit_move),
            "rho_D_edit": spearman(D, edit_move),
            "leverage_ratio": float(D.max() / max(D.min(), 1e-30)) if D.min() > 0 else float("inf")}


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{i}.pt" for i in (3, 4, 5)])
    a.add_argument("--data", default="runs/pendulum_pixels.npz")
    a.add_argument("--out", default="runs/dreamer_leverage.json")
    a = a.parse_args()
    print(__doc__.split("\n\n")[0])
    print("\nREGISTERED  P1: spearman(V, D) negative or near zero (a strongly positive value kills it)")
    print("            P2: the directions the edit moves most sit in the low-V, high-D quadrant")
    print("            KILL: D flat across directions (max/min < 2) -> no leverage structure\n")

    rows = []
    for ck in a.ckpts:
        r = run(ck, a.data)
        rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=2)
        V, D, E = np.array(r["variance"]), np.array(r["leverage"]), np.array(r["edit_move"])
        order = np.argsort(-V)
        print(f"  {ck.split('/')[-1]}   baseline rollout MSE {r['baseline_loss']:.5f}, eps {r['eps']:.3f}")
        print(f"    {'dir (by variance)':<20}{'V':>10}{'D_H':>12}{'edit move':>12}")
        for j in order:
            print(f"    {'dim '+str(j+1):<20}{V[j]:>10.3f}{D[j]:>12.2e}{E[j]:>12.2e}")
        print(f"    spearman(V, D) {r['rho_V_D']:+.3f}   spearman(V, edit) {r['rho_V_edit']:+.3f}"
              f"   spearman(D, edit) {r['rho_D_edit']:+.3f}", flush=True)

    med = lambda k: float(np.median([x[k] for x in rows]))
    print(f"\n--- medians over {len(rows)} seeds")
    print(f"    spearman(variance, leverage)  {med('rho_V_D'):+.3f}")
    print(f"    spearman(variance, edit move) {med('rho_V_edit'):+.3f}")
    print(f"    spearman(leverage, edit move) {med('rho_D_edit'):+.3f}")
    print(f"    leverage max/min ratio        {med('leverage_ratio'):.1f}x")
    print("\n--- VERDICT")
    if med("leverage_ratio") < 2:
        print("  VOID (kill criterion): rollout damage is flat across directions, so there is no")
        print("  leverage structure for an architecture to protect.")
    elif med("rho_V_D") < 0.2:
        print(f"  P1 SUPPORTED. Variance does not track causal leverage (spearman "
              f"{med('rho_V_D'):+.3f}).")
        print("  Dreamer gives little statistical prominence to directions whose displacement most")
        print("  damages its own rollouts. Variance-based protection would protect the wrong state.")
        if med("rho_D_edit") > 0.2:
            print(f"  P2 SUPPORTED: the edit pushes hardest where leverage is highest "
                  f"(spearman {med('rho_D_edit'):+.3f}).")
        else:
            print(f"  P2 NOT supported: the edit's push does not track leverage "
                  f"(spearman {med('rho_D_edit'):+.3f}); the correction is distributed rather than")
            print("  targeted, which argues for a learned corrector over a protected sub-state.")
    else:
        print(f"  P1 FAILED: variance DOES track leverage (spearman {med('rho_V_D'):+.3f}). There is")
        print("  no variance/consequence mismatch to fix, and the architectural motivation weakens.")
