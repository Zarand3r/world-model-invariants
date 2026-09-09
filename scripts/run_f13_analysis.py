#!/usr/bin/env python3
"""F13: hold the evaluation data fixed, sweep only the dt conditioning, and see if the argmin moves.

Preregistered in docs/F13_PREREG.md.

The whole point: the data is IDENTICAL across arms -- same frames, same states, same encoder inputs.
Only the value fed to the model's conditioning channel changes. A data confound cannot act on that.
Ground truth says the conserved quantity follows the MAP (precondition in the prereg), so a genuinely
dt-parameterised transition must move its argmin; one that merely reproduces the data cannot.

Usage:  uv run python scripts/run_f13_analysis.py --out runs/f13_conditioned.json
"""
from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach
from scripts.run_f6_analysis import ANALYSIS, DEG, LD, W
from scripts.run_f6_timestep import energy

DTS = (0.02, 0.035, 0.05, 0.08)
SEEDS = (3, 4, 5)
EVAL_SETS = ("runs/pend_dt0.05.npz", "runs/pend_dt0.02.npz")
C_GRID = np.round(np.arange(-0.05, 0.3501, 0.0125), 5)   # absolute c, spans c*(0.02)..c*(0.08)
MGL2 = 5.0


def c_star(dt):
    return 0.5 * dt * MGL2


def sweep(m, fr, th, thd, dt_cond):
    """rho_obs over the absolute-c grid, with the conditioning channel held at dt_cond."""
    B, T = fr.shape[:2]
    a = torch.full((B, T, 1), float(dt_cond), device=fr.device, dtype=fr.dtype)
    with torch.no_grad():
        hs = m.encode(fr, actions=a).detach()
    H = hs[:, W:]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    ac = a[:, W:].reshape(-1, 1)
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1]), a=ac).reshape(H.shape)
    Zn = ((nxt - hm) @ U) @ R
    X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
    MZ = monomial_features(Z.reshape(-1, LD), DEG)
    MZn = monomial_features(Zn.reshape(-1, LD), DEG)
    E = energy(th, thd)
    rows = []
    for c in C_GRID:
        y = (E + c * thd * np.sin(th)).ravel()[:len(X)]
        w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        cf = torch.as_tensor(w / (np.linalg.norm(w) + 1e-30), dtype=Z.dtype, device=Z.device)
        with torch.no_grad():
            Cv = (MZ @ cf).reshape(Z.shape[:2]); Cn = (MZn @ cf).reshape(Z.shape[:2])
        res = (Cn - Cv).cpu().numpy()
        rows.append({"c": float(c),
                     "rho": float(np.median(np.abs(res)) / abs(float(Cv.mean(-1).std().cpu())))})
    vals = [r["rho"] for r in rows]
    best = min(rows, key=lambda r: r["rho"])
    return {"argmin_c": best["c"], "contrast": max(vals) / max(min(vals), 1e-30), "grid": rows}


def dt_use(m, fr, dt_true, seed):
    """G0, following F1's validated action-use check: true conditioning vs shuffled across trajectories."""
    B, T = fr.shape[:2]
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    perm = torch.randperm(B, generator=g).to(fr.device)
    a_true = dt_true.reshape(B, T, 1)
    out = {}
    for tag, a in (("true", a_true), ("shuffled", a_true[perm]), ("zeros", torch.zeros_like(a_true))):
        with torch.no_grad():
            hs = m.encode(fr, actions=a).detach()
            H = hs[:, W:]
            pred = m.readout_from_h(m.transition(H[:, :-1].reshape(-1, H.shape[-1]),
                                                 a=a[:, W:-1].reshape(-1, 1)))
            ref = fr[:, W + 1:].reshape(-1, *fr.shape[2:])
            out[f"mse_{tag}"] = float(torch.nn.functional.mse_loss(pred, ref))
    out["ratio_true_over_shuffled"] = out["mse_true"] / max(out["mse_shuffled"], 1e-12)
    out["G0_uses_dt"] = bool(out["ratio_true_over_shuffled"] <= 0.9)
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/f13_conditioned.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {(m["ckpt"], m["eval_data"]) for m in out["models"]}

    for s in SEEDS:
        ck = f"runs/f13_mixed_s{s}_step15000.pt"
        if not pathlib.Path(ck).exists():
            print(f"  PENDING {ck}", flush=True); continue
        m = DreamerV3Adapter(device="cuda").cuda()
        m.load_state_dict(torch.load(ck, map_location="cuda")["model"]); m.eval()
        # G0 on the mixed training distribution, where dt genuinely varies across trajectories
        mixed = np.load("runs/pend_mixed_dt.npz")
        fr_m = torch.as_tensor(mixed["frames"][:128]).float().div_(255.).sub_(0.5).cuda()
        dt_m = torch.as_tensor(mixed["actions"][:128]).float().cuda()
        g0 = dt_use(m, fr_m, dt_m, s)
        print(f"  s{s} G0: true {g0['mse_true']:.6f} shuffled {g0['mse_shuffled']:.6f} "
              f"ratio {g0['ratio_true_over_shuffled']:.3f} -> uses dt {g0['G0_uses_dt']}", flush=True)
        for ev in EVAL_SETS:
            if (ck, ev) in done:
                continue
            d = np.load(ev)
            fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
            st = d["states"][ANALYSIS]
            th, thd = st[:, W:, 0], st[:, W:, 1]
            arms = {}
            for dtc in DTS:
                r = sweep(m, fr, th, thd, dtc)
                arms[str(dtc)] = r
                print(f"    s{s} eval={ev.split('/')[-1]:17s} cond dt={dtc:<6} "
                      f"argmin c={r['argmin_c']:+.4f} (c*={c_star(dtc):.4f})  "
                      f"contrast {r['contrast']:.2f}x", flush=True)
            rec = {"ckpt": ck, "seed": s, "eval_data": ev, "G0": g0, "arms": arms,
                   "spearman_cond_vs_argmin": spearman(DTS, [arms[str(x)]["argmin_c"] for x in DTS]),
                   "G1_all_contrast_ok": bool(all(arms[str(x)]["contrast"] >= 2.0 for x in DTS))}
            out["models"].append(rec); op.write_text(json.dumps(out, indent=1) + "\n")

    ms = out["models"]
    if len(ms) >= 6:
        prim = [m for m in ms if m["eval_data"] == EVAL_SETS[0]]
        g0ok = sum(m["G0"]["G0_uses_dt"] for m in prim)
        p1 = sum(m["spearman_cond_vs_argmin"] >= 0.8 for m in prim)
        slopes = []
        for m in prim:
            x = np.array(DTS); y = np.array([m["arms"][str(t)]["argmin_c"] for t in DTS])
            slopes.append(float((x @ y) / (x @ x)))
        p3 = sum(m["spearman_cond_vs_argmin"] >= 0.8 for m in ms if m["eval_data"] == EVAL_SETS[1])
        # TRACKING FRACTION -- added 2026-09-08 after both registered statistics passed on data
        # that refutes the hypothesis. P1 (Spearman) is scale-free, so an 8% move scores 1.0 if it
        # is monotone. P2's band admits a CONSTANT: argmin fixed at the eval data's 0.125 scores
        # 2.197 against perfect tracking's 2.500. Neither can see magnitude, which is the whole
        # question. This one can: 1.0 = tracks the conditioning, 0.0 = pinned at the data's value.
        def tracking_fraction(m):
            y = np.array([m["arms"][str(t)]["argmin_c"] for t in DTS])
            pred = np.array([c_star(t) for t in DTS])
            flat = np.full_like(pred, pred[list(DTS).index(0.05)])   # the eval data's own value
            denom = float(((pred - flat) ** 2).sum())
            return float(((y - flat) * (pred - flat)).sum() / denom) if denom > 0 else float("nan")
        tf = [tracking_fraction(m) for m in prim]
        summ = {"tracking_fraction": [round(v, 3) for v in tf],
                "tracking_fraction_median": round(float(np.median(tf)), 3),
                "P1_and_P2_are_insensitive_to_magnitude": True,
                "G0_seeds_using_dt": f"{g0ok}/{len(prim)}", "G0_pass": bool(g0ok >= 2),
                "G1_pass": bool(all(m["G1_all_contrast_ok"] for m in prim)),
                "spearman_per_seed": [round(m["spearman_cond_vs_argmin"], 3) for m in prim],
                "P1_seeds": f"{p1}/{len(prim)}", "P1_pass": bool(p1 >= 2),
                "slopes": [round(s_, 3) for s_ in slopes], "predicted_slope": 2.5,
                "P2_pass": bool(np.median(slopes) > 1.25 and np.median(slopes) < 5.0),
                "P3_seeds_control_eval": f"{p3}/{len(ms)-len(prim)}"}
        out["summary"] = summ
        print(f"\n  G0 {summ['G0_pass']} ({summ['G0_seeds_using_dt']})   G1 {summ['G1_pass']}")
        print(f"  Spearman(conditioning dt, argmin c) per seed: {summ['spearman_per_seed']}")
        print(f"  slopes vs predicted 2.5: {summ['slopes']}")
        print(f"  P1 {summ['P1_pass']}   P2 {summ['P2_pass']}   P3 {summ['P3_seeds_control_eval']}")
        print(f"  TRACKING FRACTION {summ['tracking_fraction']} (median {summ['tracking_fraction_median']})")
        print( "    1.0 = argmin follows the conditioning (model);  0.0 = pinned at the eval data's value")
        print( "    P1 and P2 both PASS here and both are blind to magnitude -- read this instead.")
        if not summ["G0_pass"]:
            print("  G0 FAILED -- the model ignores the conditioning. P1 is NOT readable, and this")
            print("  is an uninformative outcome, not a negative one.")
    attach(out, op, inputs=sorted({*EVAL_SETS, "runs/pend_mixed_dt.npz"}
                                 | {m["ckpt"] for m in out["models"]}))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {op}")


if __name__ == "__main__":
    main()
