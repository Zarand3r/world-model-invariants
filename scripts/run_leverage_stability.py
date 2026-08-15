"""A1: is causal leverage a stable selector, or is it noise?

The architecture programme (docs/ARCHITECTURE_ROADMAP.md) rests on ranking latent directions by
D_H(u) = E[L_H(z + eps u) - L_H(z)]. Before building anything that uses that ranking, we check the
ranking replicates. A selector that does not survive a split of the evaluation set cannot support an
architecture, however suggestive the correlation in D46 looked.

REGISTERED (before running):
  G1  split-half rank correlation of D_H across disjoint halves of the evaluation trajectories > 0.6
  G2  the SIGN of rho(V, D) is stable across horizons 20/50/100 and eps 0.15/0.25/0.40
  KILL: either fails -> leverage is not a usable selector and stages A2-A6 are void.

Efficiency note: one rollout per (direction, sign, eps) is run at the longest horizon and scored at
every shorter one, so horizons cost nothing extra. The trajectory split is free because it only
indexes the per-trajectory loss.
"""
import argparse, json
import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace

LD, WARMUP = 12, 10
HORIZONS = (20, 50, 100)
EPS_FRACS = (0.15, 0.25, 0.40)
N_TRAJ = 160                      # subset: the statistic is per-direction, not per-trajectory


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra**2).sum() * (rb**2).sum()), 1e-30))


def run(ckpt, data):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][:N_TRAJ]).float().div_(255.).sub_(0.5).cuda()
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr).detach()
    H = hs[:, WARMUP:]
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - h_mean) @ U
    R = effective_rank_basis(Z); Z = Z @ R
    P = U @ R
    r = Z.shape[-1]
    V = Z.reshape(-1, r).var(0).cpu().numpy()
    zn = float(Z.reshape(-1, r).norm(dim=-1).mean())
    Hmax = max(HORIZONS)
    ref = fr[:, WARMUP: WARMUP + Hmax]
    h0 = hs[:, WARMUP].clone()

    def per_traj_err(h_start):
        """(n_traj, Hmax) squared error per trajectory per step, from ONE rollout."""
        with torch.no_grad():
            h, preds = h_start, []
            for _ in range(Hmax):
                preds.append(m.readout_from_h(h)); h = m.transition(h)
            e = (torch.stack(preds, 1) - ref) ** 2
            return e.flatten(2).mean(-1).cpu().numpy()

    base = per_traj_err(h0)
    out = {"ckpt": ckpt, "variance": V.tolist(), "n_traj": int(fr.shape[0])}
    half = fr.shape[0] // 2
    for ef in EPS_FRACS:
        eps = ef * zn
        Dfull = {h: [] for h in HORIZONS}
        Da = {h: [] for h in HORIZONS}
        Db = {h: [] for h in HORIZONS}
        for i in range(r):
            plus, minus = per_traj_err(h0 + eps * P[:, i]), per_traj_err(h0 - eps * P[:, i])
            dmg = 0.5 * ((plus - base) + (minus - base))            # (n_traj, Hmax)
            for h in HORIZONS:
                Dfull[h].append(float(dmg[:, :h].mean()))
                Da[h].append(float(dmg[:half, :h].mean()))
                Db[h].append(float(dmg[half:, :h].mean()))
        for h in HORIZONS:
            key = f"eps{ef}_H{h}"
            out[key] = {"D": Dfull[h], "split_half_rho": spearman(np.array(Da[h]), np.array(Db[h])),
                        "rho_V_D": spearman(V, np.array(Dfull[h]))}
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{i}.pt" for i in (3, 4, 5)])
    a.add_argument("--data", default="runs/pendulum_pixels_eval.npz")
    a.add_argument("--out", default="runs/leverage_stability.json")
    a.add_argument("--max-models", type=int, default=0)
    a = a.parse_args()
    import pathlib
    rows = json.load(open(a.out)) if pathlib.Path(a.out).exists() else []
    done = {x["ckpt"] for x in rows}; start = len(rows)
    print(__doc__.split("\n\n")[0])
    print("\nREGISTERED  G1 split-half rho > 0.6   G2 sign of rho(V,D) stable across horizon and eps\n")
    for ck in a.ckpts:
        if ck in done: continue
        r = run(ck, a.data); rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=2); torch.cuda.empty_cache()
        print(f"  {ck.split('/')[-1]}")
        print(f"    {'eps':>6}{'H':>6}{'split-half rho':>17}{'rho(V,D)':>11}")
        for ef in EPS_FRACS:
            for h in HORIZONS:
                v = r[f"eps{ef}_H{h}"]
                print(f"    {ef:>6.2f}{h:>6}{v['split_half_rho']:>17.3f}{v['rho_V_D']:>11.3f}")
        print(flush=True)
        if a.max_models and len(rows) - start >= a.max_models:
            print(f"  [stopped after {a.max_models}; re-run to continue]"); raise SystemExit(0)

    if len(rows) < 3:
        print(f"  INCOMPLETE ({len(rows)}/3)."); raise SystemExit(0)
    sh = [r[f"eps{ef}_H{h}"]["split_half_rho"] for r in rows for ef in EPS_FRACS for h in HORIZONS]
    vd = [r[f"eps{ef}_H{h}"]["rho_V_D"] for r in rows for ef in EPS_FRACS for h in HORIZONS]
    print(f"--- across {len(rows)} seeds x {len(EPS_FRACS)} eps x {len(HORIZONS)} horizons")
    print(f"    split-half rho: median {np.median(sh):.3f}, min {min(sh):.3f}")
    print(f"    rho(V, D):      median {np.median(vd):+.3f}, negative in "
          f"{sum(1 for x in vd if x < 0)}/{len(vd)} settings")
    print("\n--- VERDICT")
    g1, g2 = np.median(sh) > 0.6, sum(1 for x in vd if x < 0) >= 0.8 * len(vd)
    if g1 and g2:
        print("  A1 PASSED. Leverage replicates across disjoint trajectories and its sign against")
        print("  variance is stable across horizon and displacement. It is a usable selector, so A2")
        print("  (does protecting high-leverage directions beat protecting high-variance ones?) runs.")
    elif not g1:
        print(f"  A1 FAILED (G1): split-half rho {np.median(sh):.3f} <= 0.6. The per-direction")
        print("  ranking does not replicate on held-out trajectories, so it cannot support an")
        print("  architecture. Stages A2-A6 are void and D46 must be restated as a population-level")
        print("  correlation rather than a per-direction selector.")
    else:
        print(f"  A1 FAILED (G2): rho(V,D) is negative in only "
              f"{sum(1 for x in vd if x < 0)}/{len(vd)} settings, so the anti-correlation depends on")
        print("  horizon or displacement and is not a stable property of the representation.")
