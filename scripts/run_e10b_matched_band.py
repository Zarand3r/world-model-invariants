"""E10b: at matched (low) energy-decodability, does CONSERVATION alone predict repair?

Pre-registered in `docs/E10B_PREREG.md`, with **no direction predicted**.

This script exists because E10b's original n = 2 numbers were produced ad hoc and never committed --
a claim in the paper rested on code that was not in the repo. It reimplements the registered design
exactly, and reproducing the existing seed-0 and seed-1 records is the regression test on the
reimplementation.

Registered design, quoted from the prereg:
  - 150 candidates from the existing eigenfamily; no new fitting.
  - Band = candidates within +-0.10 of the jointly-fitted C's |rho_E|.
  - From the band, 20 stratified by quantiles of log10(invariance ratio).
  - Direction-matched intervention, eps = 0.02, H = 100, analysis split, everything else frozen.
  - PRIMARY  Spearman(invariance ratio, repair).
  - Reported alongside: Spearman(|rho_E|, repair), to confirm decodability is really held fixed.
  - Controls: the recovered C itself and 10 magnitude-matched random directions.
"""
import argparse, datetime, hashlib, json, pathlib, subprocess
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics_osc2d
from latent_noether.polynomial import monomial_features, polynomial_invariants

DEGREE, LD, WARMUP, HORIZON, EPS = 4, 12, 10, 100, 0.02
ANALYSIS = slice(204, None)
N_CANDIDATES, BAND, N_STRATA, N_RANDOM = 150, 0.10, 20, 10


def _spear(a, b):
    """Spearman rho. scipy is not a dependency of this repo; this matches `_spear` in
    `run_e11_e12_controls.py` and `_sp` in `make_results_summary.py`."""
    def rk(x):
        x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
    ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-30))


def _C_and_grad(z, coeffs):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEGREE) @ coeffs
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def _secular(E, norm):
    E = np.asarray(E, float); k = np.arange(E.shape[-1], dtype=float); kc = k - k.mean()
    return ((E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)) / norm


def run(ckpt, data, n_candidates=N_CANDIDATES):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    norm = float(d["energy"].mean(-1).std())
    Etrue = d["energy"][ANALYSIS][:, WARMUP:]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    H = hs[:, WARMUP:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - hm) @ U) @ R) - Z
    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEGREE, 8)
    ref_coeffs = torch.as_tensor(np.asarray(fit["coeffs"]), dtype=Z.dtype, device=Z.device)
    P = U @ R; Ppinv = torch.linalg.pinv(P)
    ref_frames = fr[ANALYSIS][:, WARMUP:WARMUP + HORIZON]

    def rho_E(coeffs):
        with torch.no_grad():
            Cv = (monomial_features(Z.reshape(-1, LD), DEGREE) @ coeffs).reshape(Z.shape[:2])
        q = Cv.cpu().numpy(); k = min(q.shape[1], Etrue.shape[1])
        return float(abs(np.corrcoef(q[:, :k].ravel(), Etrue[:, :k].ravel())[0, 1]))

    def repair(coeffs):
        """Percent change in secular drift of decoded 2-DoF energy, eps=0 -> eps=EPS."""
        vals = {}
        for eps in (0.0, EPS):
            with torch.no_grad():
                h = hs[:, WARMUP].clone(); C0, _ = _C_and_grad((h - hm) @ P, coeffs); preds = []
                for _ in range(HORIZON):
                    preds.append(m.readout_from_h(h)); h = m.transition(h)
                    if eps > 0:
                        z = (h - hm) @ P; Cv, g = _C_and_grad(z, coeffs)
                        u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        h = h - ((eps * torch.sign(Cv - C0).unsqueeze(-1) * u) @ Ppinv)
                img = torch.stack(preds, 1)
                frames = ((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy()
            ds = _secular(decode_physics_osc2d(frames)["energy"], norm)
            vals[eps] = float(np.nanmedian(np.abs(ds)))
        return 100.0 * (vals[EPS] - vals[0.0]) / vals[0.0]

    ref_rho = rho_E(ref_coeffs)
    cands = polynomial_invariants(Z.double().cpu(), degree=DEGREE, max_results=n_candidates)
    scored = []
    for i, c in enumerate(cands):
        cc = torch.as_tensor(np.asarray(c["coeffs"]), dtype=Z.dtype, device=Z.device)
        scored.append({"rank": i, "ratio": float(c["ratio"]), "rho_E": rho_E(cc), "_c": cc})
    band = [s for s in scored if abs(s["rho_E"] - ref_rho) <= BAND]

    # 20 strata by quantiles of log10(ratio); nearest candidate per quantile, deduplicated.
    lg = np.log10(np.array([s["ratio"] for s in band]))
    picks, seen = [], set()
    for qq in np.linspace(0, 1, N_STRATA):
        j = int(np.argmin(np.abs(lg - np.quantile(lg, qq))))
        if band[j]["rank"] not in seen:
            seen.add(band[j]["rank"]); picks.append(band[j])
    picks.sort(key=lambda s: s["ratio"])

    rows = []
    for s in picks:
        r = repair(s["_c"])
        rows.append({"rank": s["rank"], "ratio": s["ratio"], "rho_E": s["rho_E"], "repair": r})
        print(f"    rank {s['rank']:3d}  ratio {s['ratio']:.3e}  rho_E {s['rho_E']:.3f}  repair {r:+.1f}%", flush=True)

    ratios = [r["ratio"] for r in rows]; reps = [r["repair"] for r in rows]
    rhos = [r["rho_E"] for r in rows]
    sp_primary = _spear(ratios, reps)
    sp_check = _spear(rhos, reps)

    controls = {"recovered_C": {"rho_E": ref_rho, "repair": repair(ref_coeffs)}, "random": []}
    for dr in range(N_RANDOM):
        g = torch.Generator(device="cpu").manual_seed(1000 + dr)
        rc = torch.randn(ref_coeffs.shape[0], generator=g, dtype=torch.float64)
        cc = (rc / rc.norm() * ref_coeffs.norm().cpu()).to(Z.dtype).to(Z.device)
        controls["random"].append({"rho_E": rho_E(cc), "repair": repair(cc)})

    return {"ckpt": ckpt, "reference_rho_E": ref_rho, "n_candidates": len(scored),
            "band_size": len(band), "band_tolerance": BAND, "rows": rows,
            "spearman_ratio_repair": sp_primary,
            "spearman_rhoE_repair": sp_check,
            "controls": controls}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data", default="runs/osc2d_central.npz")
    p.add_argument("--n-candidates", type=int, default=N_CANDIDATES,
                   help="pool size; the 2026-08-28 amendment raises this to 400 for all seeds")
    p.add_argument("--out", default="runs/e10b_matched_band.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    out.setdefault("provenance", {}).setdefault("runs", []).append({
        "data": a.data, "data_sha256": hashlib.sha256(pathlib.Path(a.data).read_bytes()).hexdigest(),
        "degree": DEGREE, "ld": LD, "warmup": WARMUP, "horizon": HORIZON, "eps": EPS,
        "n_candidates": a.n_candidates, "band": BAND, "n_strata": N_STRATA,
        "analysis_slice": [ANALYSIS.start, ANALYSIS.stop],
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            print(f"[E10b] skip {ck}"); continue
        print(f"[E10b] {ck}", flush=True)
        out["models"].append(run(ck, a.data, a.n_candidates)); op.write_text(json.dumps(out, indent=1) + "\n")
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
