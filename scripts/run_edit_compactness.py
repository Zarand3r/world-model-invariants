"""A2b: is the S4 correction a compact fix to a small physical state, or a distributed adjustment?

This decides which architecture to build, so it runs before either is built. The original roadmap
treated it as something to infer from the corrector's arms; that was backwards.

The test needs no training. Apply the S4 projection but restrict the correction to the top-k
directions and sweep k. If most of the benefit survives at small k the useful correction is compact,
and a protected sub-state is the right design. If the benefit grows steadily with k the correction is
distributed, and a learned corrector free to act anywhere is the right design.

    z <- z - alpha * Pi_k [ (C(z) - C0) grad C(z) / ||grad C(z)||^2 ]

Pi_k projects onto the top-k directions under one of two rankings, which may disagree:
    edit    where the unrestricted projection naturally pushes hardest
    lever   where a displacement most damages the model's own rollout

REGISTERED: report the fraction of the full-subspace benefit recovered at each k. Compact if k=3 of
12 recovers >= 70%.

ALSO MEASURED from the same rollouts, all named in review and missing from the first roadmap draft:
  - the drift of C under the edit, not just pixel error
  - when in the rollout the intervention starts to help (error against step)
"""
import argparse, json, pathlib
import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP, HORIZON, ALPHA = 4, 12, 10, 50, 0.4
KS = (1, 2, 3, 4, 6, 8, 12)


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def run(ckpt, data):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    val = slice(204, None)
    with torch.no_grad():
        hs = m.encode(fr[val]).detach()
    H = hs[:, WARMUP:]
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - h_mean) @ U
    R = effective_rank_basis(Z); Z = Z @ R
    P = U @ R; Ppinv = torch.linalg.pinv(P); r = Z.shape[-1]
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - h_mean) @ U) @ R) - Z
    fit = fit_hamiltonian_pair(Z.double().cpu(), F.double().cpu(), degree=DEGREE, n_basis=8)
    coeffs = torch.as_tensor(fit["coeffs"], dtype=Z.dtype, device=Z.device)
    ref = fr[val][:, WARMUP: WARMUP + HORIZON]

    def rollout(mask=None):
        """mask: (r,) 0/1 restricting which directions the correction may touch. None = no edit."""
        with torch.no_grad():
            h = hs[:, WARMUP].clone()
            C0, _ = _C_and_grad((h - h_mean) @ P, coeffs)
            preds, drift, push = [], [], torch.zeros(r, device=Z.device)
            for _ in range(HORIZON):
                preds.append(m.readout_from_h(h))
                h = m.transition(h)
                z = (h - h_mean) @ P
                Cv, g = _C_and_grad(z, coeffs)
                drift.append(float((Cv - C0).abs().mean()))
                if mask is not None:
                    step = ALPHA * ((Cv - C0) / g.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * g
                    step = step * mask
                    push += step.abs().mean(0)
                    h = h - (step @ Ppinv)
            pr = torch.stack(preds, 1)
            per_step = ((pr - ref) ** 2).flatten(2).mean(-1).mean(0).cpu().numpy()
            return float(torch.nn.functional.mse_loss(pr, ref)), np.array(drift), per_step, push

    base, drift_base, step_base, _ = rollout(None)
    full, drift_full, step_full, push = rollout(torch.ones(r, device=Z.device))
    full_gain = base - full
    # A2c: four rankings at matched rank. The question is whether the invariant's gradient picks a
    # better sub-state to protect than variance does, which is the architectural claim in one line.
    order = {"edit": torch.argsort(push, descending=True).cpu().numpy(),
             "variance": torch.argsort(Z.reshape(-1, r).var(0), descending=True).cpu().numpy(),
             # A SINGLE random permutation is not a control. The first one we drew shared its
             # first two entries with the edit ranking by chance (both start 9, 2), so the random
             # arm reproduced the gradC arm exactly at k=1 and k=2 and looked like a refutation.
             # The random arm is now a DISTRIBUTION over draws, scored per k in `random_draws`.
             "lever": None}
    lev = json.load(open("runs/dreamer_leverage.json"))
    match = [x for x in lev if x["ckpt"] == ckpt]
    if match:
        order["lever"] = np.argsort(-np.array(match[0]["leverage"]))

    out = {"ckpt": ckpt, "baseline": base, "full_edit": full, "full_gain_frac": full_gain / base,
           "drift_base_final": float(drift_base[-1]), "drift_full_final": float(drift_full[-1]),
           "step_base": step_base.tolist(), "step_full": step_full.tolist(), "by_rank": {}}
    for name, idx in order.items():
        if idx is None:
            continue
        row = {}
        for k in KS:
            mask = torch.zeros(r, device=Z.device)
            mask[torch.as_tensor(idx[:k].copy(), device=Z.device)] = 1.0
            err = rollout(mask)[0]
            row[k] = {"err": err, "frac_of_full": float((base - err) / full_gain) if full_gain else 0.0}
        out["by_rank"][name] = row
    rng = np.random.default_rng(1)
    out["random_draws"] = {}
    for k in (2, 3, 4):
        fr_ = []
        for _ in range(12):
            mk = torch.zeros(r, device=Z.device)
            mk[torch.as_tensor(rng.choice(r, k, replace=False), device=Z.device)] = 1.0
            fr_.append(float((base - rollout(mk)[0]) / full_gain) if full_gain else 0.0)
        out["random_draws"][k] = fr_
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{i}.pt" for i in (3, 4, 5)])
    a.add_argument("--data", default="runs/pendulum_pixels.npz")
    a.add_argument("--out", default="runs/edit_compactness.json")
    a.add_argument("--max-models", type=int, default=0)
    a = a.parse_args()
    print(__doc__.split("\n\n")[0])
    print("\nREGISTERED: compact if k=3 of 12 recovers >= 70% of the full-subspace benefit.\n")
    rows = json.load(open(a.out)) if pathlib.Path(a.out).exists() else []
    done = {x["ckpt"] for x in rows}; start = len(rows)
    for ck in a.ckpts:
        if ck in done: continue
        r = run(ck, a.data); rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=2); torch.cuda.empty_cache()
        print(f"  {ck.split('/')[-1]}  baseline {r['baseline']:.5f} -> full edit {r['full_edit']:.5f}"
              f"  ({-r['full_gain_frac']:+.1%})")
        print(f"    |C - C0| at final step: {r['drift_base_final']:.4f} without the edit -> "
              f"{r['drift_full_final']:.4f} with it")
        for name, row in r["by_rank"].items():
            frac = "  ".join(f"k={k}:{row[k]['frac_of_full']:>5.0%}" for k in KS)
            print(f"    by {name:<6} {frac}", flush=True)
        if a.max_models and len(rows) - start >= a.max_models:
            print(f"\n  [stopped after {a.max_models}; re-run to continue]"); raise SystemExit(0)

    if len(rows) < 3:
        print(f"\n  INCOMPLETE ({len(rows)}/3)."); raise SystemExit(0)
    print(f"\n  {'rank by':<10}" + "".join(f"{'k='+str(k):>8}" for k in KS))
    med = {}
    for name in rows[0]["by_rank"]:
        v = [float(np.median([r["by_rank"][name][str(k)]["frac_of_full"] for r in rows]))
             for k in KS]
        med[name] = v
        print(f"  {name:<10}" + "".join(f"{x:>8.0%}" for x in v))
    print("\n--- VERDICT")
    i3 = KS.index(3)
    grad3, var3, rnd3 = med["edit"][i3], med.get("variance", [0]*9)[i3], med.get("random", [0]*9)[i3]
    print(f"\n  AT MATCHED RANK 3:  gradC {grad3:.0%}   variance {var3:.0%}   random {rnd3:.0%}")
    if grad3 > var3 + 0.15 and grad3 > rnd3 + 0.15:
        print("  A2c SUPPORTED: the invariant's gradient picks a better sub-state to protect than")
        print("  variance does, at matched rank. That is the architectural claim, and it is testable")
        print("  without any leverage measurement.")
    else:
        print("  A2c NOT SUPPORTED at rank 3: gradient selection does not clearly beat variance.")
    best3 = max(med[n][i3] for n in med)
    if best3 >= 0.7:
        print(f"  COMPACT. k=3 of 12 directions recovers {best3:.0%} of the benefit, so the useful")
        print("  correction lives in a small sub-state. A protected sub-state (A4) is the right")
        print("  architecture and the learned corrector is over-general.")
    elif best3 >= 0.4:
        print(f"  INTERMEDIATE. k=3 recovers {best3:.0%}. Neither design is clearly indicated; report")
        print("  the curve and decide on A3 vs A4 by what the corrector arms show.")
    else:
        print(f"  DISTRIBUTED. k=3 recovers only {best3:.0%} of the benefit, so the correction is")
        print("  spread across the latent. A learned corrector free to act anywhere (A3) is the right")
        print("  architecture, and the protected sub-state (A4) should NOT run.")
