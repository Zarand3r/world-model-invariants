"""S4: does enforcing the recovered law improve a real DreamerV3's imagination?

Pre-registered in docs/S4_PREREG.md. The edit and the scoring rule are ported unchanged from the
GRU protocol (scripts/run_projection_edit.py, D9) -- nothing is re-derived here.

    z <- z - alpha (C(z) - C0) grad C(z) / ||grad C(z)||^2

then mapped back through pinv(U R) so the component of h outside the extracted subspace is
untouched. Decoding-time only; the model is never adapted.

SCORING IS THE SLOPE OVER A FIXED ALPHA GRID, NOT THE BEST ALPHA (D9). Taking the minimum over the
grid once reported -5.9% for a damped arm whose curve rose monotonically to twice its baseline.

ARMS -- arm B is the one that can kill this. If projection helps regardless of WHICH C is enforced,
the edit is just regularising the rollout and says nothing about physics.
    A  conservative + its own recovered C      expect: error DECREASES with alpha
    B  conservative + a RANDOM matched-complexity C   expect: no improvement, or harm
    C  damped + its own recovered C            expect: no improvement, or harm

ARM B IS A DISTRIBUTION, NOT A DRAW (M24). Until 2026-08-13 this drew ONE random polynomial:
`manual_seed(0)` was called inside `run()`, so all three seeds received the *same* coefficient
vector and the control was n = 1. That is the defect D50 caught in the A2c random arm and fixed
there with 100 draws, and it survived here because the arm looked like three samples. It is now
N_RANDOM_LAWS independent draws per checkpoint, and the reported statistic is the null's median
with the recovered law's percentile inside it.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP = 4, 12, 10
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.4)      # fixed grid, identical for every arm (D9)
HORIZON = 50
N_RANDOM_LAWS = 20                       # arm B draws per checkpoint; a null with n=1 is not a null


def _C_and_grad(z, coeffs):
    """C and grad C. enable_grad is required: the rollout runs under no_grad but the projection
    needs the gradient even though nothing is being trained."""
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        vals = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(vals.sum(), zz)
    return vals.detach(), g.detach()


def run(ckpt, data, random_law=False, draw=0):
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
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - h_mean) @ U) @ R) - Z

    fit = fit_hamiltonian_pair(Z.double().cpu(), F.double().cpu(), degree=DEGREE, n_basis=8)
    coeffs = torch.as_tensor(fit["coeffs"], dtype=Z.dtype, device=Z.device)
    if random_law:
        # ARM B: a random polynomial over the SAME basis, matched in norm. Same edit, same grid,
        # wrong law -- so any improvement here is the projection regularising, not physics.
        # The seed varies with `draw`, so the arm is a distribution over laws rather than one law
        # evaluated on three models. Seeding from `draw` alone (not the ckpt) keeps the SAME twenty
        # polynomials across seeds, which pairs the comparison: any difference between checkpoints
        # is the model, not the draw.
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)
        rc = torch.randn(coeffs.shape[0], generator=g, dtype=torch.float64)
        coeffs = (rc / rc.norm() * coeffs.norm().cpu()).to(Z.dtype).to(Z.device)

    P = U @ R                                   # (D, r)
    P_pinv = torch.linalg.pinv(P)               # (r, D)
    ref = fr[val][:, WARMUP: WARMUP + HORIZON]

    errs = {}
    for alpha in ALPHAS:
        with torch.no_grad():
            h = hs[:, WARMUP].clone()
            z0, _ = (h - h_mean) @ P, None
            C0, _ = _C_and_grad(z0, coeffs)
            preds = []
            for _ in range(HORIZON):
                preds.append(m.readout_from_h(h))
                h = m.transition(h)
                if alpha > 0.0:
                    z = (h - h_mean) @ P
                    Cv, gr = _C_and_grad(z, coeffs)
                    step = alpha * ((Cv - C0) / gr.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * gr
                    h = h - (step @ P_pinv)
            errs[float(alpha)] = float(torch.nn.functional.mse_loss(torch.stack(preds, 1), ref))
    a = np.array(ALPHAS, dtype=np.float64)
    e = np.array([errs[float(x)] for x in ALPHAS])
    slope = float(np.polyfit(a, e / max(e[0], 1e-30), 1)[0])     # normalised, so arms compare
    return {"ckpt": ckpt, "random_law": random_law, "draw": draw if random_law else None,
            "pairing_residual": fit["residual"],
            "rollout_by_alpha": errs, "baseline": e[0],
            "normalised_slope": slope,
            "relative_change_at_max_alpha": float((e[-1] - e[0]) / max(e[0], 1e-30))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/dreamer_edit.json")
    p.add_argument("--max-models", type=int, default=0)
    a = p.parse_args()
    ARMS = [("A_conservative_own", f"runs/dreamer_ref_s{i}.pt", "runs/pendulum_pixels.npz", False, 0)
            for i in (3, 4, 5)]
    ARMS += [("B_conservative_random", f"runs/dreamer_ref_s{i}.pt", "runs/pendulum_pixels.npz",
              True, k) for i in (3, 4, 5) for k in range(N_RANDOM_LAWS)]
    ARMS += [("C_damped_own", f"runs/dreamer_damped_s{i}.pt", "runs/pendulum_pixels_damped.npz",
              False, 0) for i in range(3)]

    out = json.load(open(a.out)) if pathlib.Path(a.out).exists() else {}
    print(__doc__.split("\n\n")[0])
    print(f"\nfixed alpha grid {ALPHAS}; scored by SLOPE (D9), full curve reported")
    print(f"arm B is {N_RANDOM_LAWS} independent draws per checkpoint, not one law (M24)\n")
    done_before = sum(len(v) for v in out.values())
    for arm, ck, data, rnd, draw in ARMS:
        key = f"{arm}|{ck}" + (f"|draw{draw}" if rnd else "")
        if any(r["key"] == key for r in out.get(arm, [])):
            continue
        r = run(ck, data, random_law=rnd, draw=draw)
        r["key"] = key
        out.setdefault(arm, []).append(r)
        json.dump(out, open(a.out, "w"), indent=2)
        torch.cuda.empty_cache()
        curve = "  ".join(f"{k}:{v:.2e}" for k, v in r["rollout_by_alpha"].items())
        print(f"  {arm:<24}{ck.split('/')[-1]:<22} slope {r['normalised_slope']:+.3f}   "
              f"@max {r['relative_change_at_max_alpha']:+.1%}")
        print(f"      {curve}", flush=True)
        if a.max_models and sum(len(v) for v in out.values()) - done_before >= a.max_models:
            print(f"\n  [stopped after {a.max_models}; re-run to continue]")
            raise SystemExit(0)

    print(f"\n  {'arm':<24}{'n':>4}{'median slope':>14}{'median @max alpha':>20}")
    med = {}
    for arm in ("A_conservative_own", "B_conservative_random", "C_damped_own"):
        if not out.get(arm):
            continue
        s = float(np.median([r["normalised_slope"] for r in out[arm]]))
        c = float(np.median([r["relative_change_at_max_alpha"] for r in out[arm]]))
        med[arm] = (s, c)
        print(f"  {arm:<24}{len(out[arm]):>4}{s:>14.3f}{c:>19.1%}")

    # Where does each model's own law sit inside ITS OWN null? The median above pools three
    # checkpoints; the percentile is per-checkpoint and is the statistic the claim rests on.
    if out.get("A_conservative_own") and out.get("B_conservative_random"):
        print(f"\n  {'checkpoint':<22}{'own slope':>11}{'null median':>13}{'pctile':>8}"
              f"{'draws improving':>17}")
        for ra in out["A_conservative_own"]:
            null = [r["normalised_slope"] for r in out["B_conservative_random"]
                    if r["ckpt"] == ra["ckpt"]]
            imp = [r for r in out["B_conservative_random"]
                   if r["ckpt"] == ra["ckpt"] and r["relative_change_at_max_alpha"] < 0]
            if not null:
                continue
            pct = 100.0 * float(np.mean([x < ra["normalised_slope"] for x in null]))
            print(f"  {ra['ckpt'].split('/')[-1]:<22}{ra['normalised_slope']:>+11.4f}"
                  f"{float(np.median(null)):>+13.4f}{pct:>7.0f}%{len(imp):>12} of {len(null)}")
    if len(med) < 3:
        print("\n  INCOMPLETE — re-run to finish the remaining arms.")
        raise SystemExit(0)
    A, B, C = (med[k][0] for k in ("A_conservative_own", "B_conservative_random", "C_damped_own"))
    print("\n--- VERDICT (registered: A decreases with alpha; B and C do not)")
    if A < -0.02 and B > A + 0.02 and C > A + 0.02:
        print(f"  PASSED. Enforcing the recovered law improves the model's own imagination")
        print(f"  (slope {A:+.3f}) while a random law ({B:+.3f}) and the dissipative model")
        print(f"  ({C:+.3f}) do not. recover -> refuse -> INTERVENE, on a real pixel world model.")
    elif A < -0.02 and B < -0.02:
        print(f"  KILLED BY ARM B. The projection improves rollouts regardless of which law is")
        print(f"  enforced (A {A:+.3f}, random B {B:+.3f}). The edit is regularising the rollout,")
        print(f"  not enforcing physics, and is retired as evidence about the recovered law.")
    else:
        print(f"  NO EFFECT: A {A:+.3f}, B {B:+.3f}, C {C:+.3f}. Enforcing the recovered law does")
        print(f"  not measurably improve Dreamer's imagination. The invariant is present and")
        print(f"  conserved (D44) but not shown to be causally load-bearing.")
