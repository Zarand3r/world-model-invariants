"""E19: does the shadow Hamiltonian explain E18's supervised-probe failure?

Pre-registered in `docs/E19_PREREG.md`. Everything -- latent, PCA frame, degree-4 basis, ridge,
direction-matched repair, eps grid, warmup, analysis slice -- is inherited unchanged from
`run_e18_supervised_baseline.py`. The ONLY thing that varies is the regression target:

    T_c(theta, thetadot) = E + c * thetadot * sin(theta)

`c = 0` is exactly E18's supervised probe, so that column doubles as a regression test on the
harness. The predicted shadow coefficient is |c| = (dt/2) * mg(l/2) = 0.125; its sign is fixed
empirically by P1 rather than asserted from a BCH convention.
"""
import argparse, datetime, hashlib, json, pathlib, subprocess
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features

DEG, LD, W, H = 4, 12, 10, 100
ANALYSIS = slice(204, None)
EPS_GRID = (0.0, 0.005, 0.01, 0.02)
C_GRID = (-0.5, -0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5)
C_SHADOW_MAG = 0.125          # (dt/2) * m*g*(l/2) = 0.025 * 5


def target(theta, thetadot, c):
    """The registered target family. c = 0 recovers textbook energy exactly."""
    return energy(theta, thetadot) + c * np.asarray(thetadot) * np.sin(np.asarray(theta))


def invariance_ratio(q):
    """mean_traj[var_t] / var_total -- the `ratio_empirical` statistic used elsewhere here."""
    q = np.asarray(q, float)
    return float(np.mean(np.var(q, axis=-1)) / max(np.var(q), 1e-30))


def physics_check(data):
    """P1: on GROUND-TRUTH states, is T_c best conserved at the predicted shadow coefficient?"""
    st = np.load(data)["states"]
    th, thd = st[..., 0], st[..., 1]
    rows = [{"c": float(c), "ratio": invariance_ratio(target(th, thd, c))} for c in C_GRID]
    best = min(rows, key=lambda r: r["ratio"])
    zero = next(r for r in rows if r["c"] == 0.0)["ratio"]
    return {"grid": rows, "argmin_c": best["c"], "ratio_at_argmin": best["ratio"],
            "ratio_at_zero": zero, "improvement_x": zero / max(best["ratio"], 1e-30),
            "P1_pass": bool(abs(abs(best["c"]) - C_SHADOW_MAG) < 1e-12 and zero / max(best["ratio"], 1e-30) >= 2.0)}


def _CG(z, c):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEG) @ c
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def _secular(E, norm):
    E = np.asarray(E, float); k = np.arange(E.shape[-1], dtype=float); kc = k - k.mean()
    return ((E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)) / norm


def run(ckpt, data, c_grid):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"]; norm = float(energy(st[..., 0], st[..., 1]).mean(-1).std())
    th_a, thd_a = st[ANALYSIS][:, W:, 0], st[ANALYSIS][:, W:, 1]
    E = energy(th_a, thd_a)                       # rho_E is always measured against TEXTBOOK energy
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    Hh = hs[:, W:]; hm = Hh.reshape(-1, Hh.shape[-1]).mean(0)
    U = pca_subspace(Hh, LD); Z = (Hh - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(Hh.reshape(-1, Hh.shape[-1])).reshape(Hh.shape)
    Zn = (((nxt - hm) @ U) @ R)
    P = U @ R; Ppinv = torch.linalg.pinv(P)
    ref = fr[ANALYSIS][:, W:W + H]
    X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
    XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])

    def stats(c):
        with torch.no_grad():
            Cv = (monomial_features(Z.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
            Cn = (monomial_features(Zn.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2])
        q = Cv.cpu().numpy(); r = (Cn - Cv).cpu().numpy(); k = min(q.shape[1], E.shape[1])
        return {"rho_E": float(abs(np.corrcoef(q[:, :k].ravel(), E[:, :k].ravel())[0, 1])),
                "rho_obs": float(np.median(np.abs(r)) / abs(float(Cv.mean(-1).std().cpu())))}

    def repair(c):
        out = {}
        for eps in EPS_GRID:
            with torch.no_grad():
                h = hs[:, W].clone(); C0, _ = _CG((h - hm) @ P, c); preds = []
                for _ in range(H):
                    preds.append(m.readout_from_h(h)); h = m.transition(h)
                    if eps > 0:
                        z = (h - hm) @ P; Cv, g = _CG(z, c)
                        u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        h = h - ((eps * torch.sign(Cv - C0).unsqueeze(-1) * u) @ Ppinv)
                img = torch.stack(preds, 1)
                pm = float(torch.nn.functional.mse_loss(img, ref))
                frames = ((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()
            ds = _secular(decode_physics(frames)["energy"], norm)
            out[float(eps)] = {"D_sec_median_abs": float(np.nanmedian(np.abs(ds))), "pixel_mse": pm}
        b = out[0.0]["D_sec_median_abs"]; e = out[0.02]["D_sec_median_abs"]
        return {"by_eps": out, "effect_pct": 100 * (e - b) / b}

    sweep = []
    for cc in c_grid:
        y = np.asarray(target(th_a, thd_a, cc)).ravel()[:len(X)]
        w = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        coef = torch.as_tensor(w / (np.linalg.norm(w) + 1e-30), dtype=Z.dtype, device=Z.device)
        rec = {"c": float(cc), **stats(coef), **repair(coef)}
        print(f"    c={cc:+.4f}  rho_E={rec['rho_E']:.4f}  rho_obs={rec['rho_obs']:.5f}  "
              f"effect={rec['effect_pct']:+.1f}%", flush=True)
        sweep.append(rec)
    return {"ckpt": ckpt, "sweep": sweep}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3, 4, 5)])
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--out", default="runs/e19_shadow_sweep.json")
    p.add_argument("--physics-only", action="store_true")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}

    phys = physics_check(a.data)
    out["physics_P1"] = phys
    print("[E19] P1 physics check (ground truth, no model):")
    for r in phys["grid"]:
        mark = "  <-- argmin" if r["c"] == phys["argmin_c"] else ""
        print(f"    c={r['c']:+.4f}  invariance ratio={r['ratio']:.6e}{mark}")
    print(f"  argmin c = {phys['argmin_c']:+.4f} (predicted |c| = {C_SHADOW_MAG})")
    print(f"  improvement over c=0: {phys['improvement_x']:.2f}x  (registered bar: >= 2x)")
    print(f"  P1_pass = {phys['P1_pass']}")
    op.write_text(json.dumps(out, indent=1) + "\n")
    if a.physics_only:
        raise SystemExit(0)
    if not phys["P1_pass"]:
        print("\nP1 FAILED -- registered response is to STOP. Not running the model sweep.")
        raise SystemExit(1)

    out.setdefault("provenance", {}).setdefault("runs", []).append({
        "data": a.data, "data_sha256": hashlib.sha256(pathlib.Path(a.data).read_bytes()).hexdigest(),
        "deg": DEG, "ld": LD, "warmup": W, "horizon": H, "eps_grid": list(EPS_GRID),
        "c_grid": list(C_GRID), "c_shadow_mag": C_SHADOW_MAG,
        "analysis_slice": [ANALYSIS.start, ANALYSIS.stop],
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists(): continue
        print(f"[E19] {ck}", flush=True)
        out["models"].append(run(ck, a.data, C_GRID)); op.write_text(json.dumps(out, indent=1) + "\n")
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
