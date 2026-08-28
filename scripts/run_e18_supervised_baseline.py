"""E18: does a supervised energy probe repair as well as the label-free invariant?

Pre-registered in `docs/E18_PREREG.md`. Same latent, same coordinate frame, same polynomial family,
same direction-matched repair protocol and the same magnitude-matched random null. The only
difference is how the coefficients are chosen:

    unsupervised   fit_hamiltonian_pair  -- optimises CONSERVATION, never sees energy
    supervised     ridge least squares   -- optimises ENERGY TRACKING, sees the labels

The two objectives are different, and E10b found repair tracks conservation rather than
decodability, so the registered prediction is that the supervised probe tracks energy better and
repairs worse.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features

DEG, LD, W, H = 4, 12, 10, 100
ANALYSIS = slice(204, None)
EPS_GRID = (0.0, 0.005, 0.01, 0.02)


def _CG(z, c):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEG) @ c
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def _secular(E, norm):
    E = np.asarray(E, float); k = np.arange(E.shape[-1], dtype=float); kc = k - k.mean()
    return ((E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)) / norm


def run(ckpt, data, n_random=20):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"]; norm = float(energy(st[..., 0], st[..., 1]).mean(-1).std())
    E = energy(st[ANALYSIS][:, W:, 0], st[ANALYSIS][:, W:, 1])
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    Hh = hs[:, W:]; hm = Hh.reshape(-1, Hh.shape[-1]).mean(0)
    U = pca_subspace(Hh, LD); Z = (Hh - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(Hh.reshape(-1, Hh.shape[-1])).reshape(Hh.shape)
    Zn = (((nxt - hm) @ U) @ R); F = Zn - Z
    P = U @ R; Ppinv = torch.linalg.pinv(P)
    ref = fr[ANALYSIS][:, W:W + H]

    # --- unsupervised: optimises conservation ---
    unsup = torch.as_tensor(np.asarray(cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)["coeffs"]),
                            dtype=Z.dtype, device=Z.device)
    # --- supervised: optimises energy tracking ---
    X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
    y = np.asarray(E).ravel()[:len(X)]
    w = np.linalg.lstsq(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y, rcond=None)[0]
    sup = torch.as_tensor(w / (np.linalg.norm(w) + 1e-30), dtype=Z.dtype, device=Z.device)

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

    rec = {"ckpt": ckpt,
           "unsupervised": {**stats(unsup), **repair(unsup)},
           "supervised": {**stats(sup), **repair(sup)}, "random": []}
    for dr in range(n_random):
        g = torch.Generator(device="cpu").manual_seed(1000 + dr)
        rc = torch.randn(unsup.shape[0], generator=g, dtype=torch.float64)
        c = (rc / rc.norm() * unsup.norm().cpu()).to(Z.dtype).to(Z.device)
        rec["random"].append({**stats(c), **repair(c)})
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3, 4, 5)])
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--out", default="runs/e18_supervised_baseline.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists(): continue
        print(f"[E18] {ck}", flush=True)
        out["models"].append(run(ck, a.data, a.n_random)); op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
