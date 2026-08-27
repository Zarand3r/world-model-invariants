"""E11 step 2 and E12: does the intervention fix phase, and is the subspace on-pathway?

Pre-registered in `docs/E11_E12_PREREG.md`.

E11 step 1 already established that `D_sec` is phase-invariant (time-shifting real trajectories by
1-10 steps moves it by <= 1.6e-04 against a 1.0e-03 floor), so the Samanta & Behera phase rival
cannot explain the headline metric. Step 2 asks what the intervention actually repairs.

E12 asks whether the edited direction is one the model uses on its own: if `C` is on-pathway, its
natural variation along an UNEDITED rollout should track decoded energy. A dormant subspace
(Makelov et al., ICLR 2024) would show no such relationship until edited.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP, HORIZON, EPS = 4, 12, 10, 100, 0.02
ANALYSIS = slice(204, None)
MAX_SHIFT = 12


def _spear(a, b):
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


def _phase_error(th_hat, th_true):
    """Best constant integer time-shift per trajectory; returns (residual error, lag)."""
    n = th_hat.shape[1] - MAX_SHIFT
    best = np.full(th_hat.shape[0], np.inf); lag = np.zeros(th_hat.shape[0])
    for s in range(-MAX_SHIFT // 2, MAX_SHIFT // 2 + 1):
        a = th_hat[:, MAX_SHIFT // 2: MAX_SHIFT // 2 + n]
        b = th_true[:, MAX_SHIFT // 2 + s: MAX_SHIFT // 2 + s + n]
        e = np.nanmedian(np.abs(np.angle(np.exp(1j * (a - b)))), axis=1)
        upd = e < best
        lag[upd] = s; best[upd] = e[upd]
    return best, lag


def run(ckpt, data, random_law=False, draw=0):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"]; norm = float(energy(st[..., 0], st[..., 1]).mean(-1).std())
    th_true = st[ANALYSIS][:, WARMUP:WARMUP + HORIZON, 0]

    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    H = hs[:, WARMUP:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - hm) @ U) @ R) - Z
    fit = cached_fit(Z.double().cpu(), F.double().cpu(), DEGREE, 8)
    coeffs = torch.as_tensor(np.asarray(fit["coeffs"]), dtype=Z.dtype, device=Z.device)
    if random_law:
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)
        rc = torch.randn(coeffs.shape[0], generator=g, dtype=torch.float64)
        coeffs = (rc / rc.norm() * coeffs.norm().cpu()).to(Z.dtype).to(Z.device)
    P = U @ R; Ppinv = torch.linalg.pinv(P)

    out = {"ckpt": ckpt, "random_law": random_law, "draw": draw if random_law else None}
    for eps in (0.0, EPS):
        with torch.no_grad():
            h = hs[:, WARMUP].clone(); C0, _ = _C_and_grad((h - hm) @ P, coeffs)
            preds, Cs = [], []
            for _ in range(HORIZON):
                preds.append(m.readout_from_h(h))
                Cs.append(_C_and_grad((h - hm) @ P, coeffs)[0].cpu().numpy())
                h = m.transition(h)
                if eps > 0:
                    z = (h - hm) @ P; Cv, g = _C_and_grad(z, coeffs)
                    u = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    h = h - ((eps * torch.sign(Cv - C0).unsqueeze(-1) * u) @ Ppinv)
            frames = ((torch.stack(preds, 1) + 0.5) * 255.0).clamp(0, 255).cpu().numpy()
        ph = decode_physics(frames)
        Ck = np.asarray(Cs).T                                # (traj, H)
        dsec = _secular(ph["energy"], norm)
        perr, lag = _phase_error(ph["theta"], th_true)
        rec = {"D_sec_median_abs": float(np.nanmedian(np.abs(dsec))),
               "phase_err_median": float(np.nanmedian(perr)),
               "lag_median_abs": float(np.nanmedian(np.abs(lag)))}
        if eps == 0.0:
            # E12 (registered, FAILED -- kept for the record): within-trajectory correlation.
            rs = [_spear(Ck[i], ph["energy"][i]) for i in range(Ck.shape[0])
                  if np.isfinite(ph["energy"][i]).all()]
            rec["e12_spearman_C_vs_E_median"] = float(np.median(np.abs(rs)))
            rec["e12_spearman_signed_median"] = float(np.median(rs))
            # E12b (docs/E12B_PREREG.md): does C's DRIFT predict energy's DRIFT, across trajectories?
            dC = _secular(Ck, 1.0)
            dE = _secular(ph["energy"], 1.0)
            ok = np.isfinite(dC) & np.isfinite(dE)
            rec["e12b_spearman_Dsec_C_vs_Dsec_E"] = float(_spear(dC[ok], dE[ok]))
            rec["e12b_n"] = int(ok.sum())
            rec["e12b_Dsec_C_per_traj"] = np.where(ok, dC, np.nan).tolist()
            rec["e12b_Dsec_E_per_traj"] = np.where(ok, dE, np.nan).tolist()
        out[f"eps{eps}"] = rec
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3,4,5)])
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--out", default="runs/e11_e12.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"recovered": [], "random": []}
    save = lambda: op.write_text(json.dumps(out, indent=1) + "\n")
    for ck in a.ckpts:
        if ck in {r["ckpt"] for r in out["recovered"]} or not pathlib.Path(ck).exists(): continue
        print(f"[rec] {ck}", flush=True); out["recovered"].append(run(ck, a.data)); save()
    done = {(r["ckpt"], r["draw"]) for r in out["random"]}
    for dr in range(a.n_random):
        if (a.ckpts[0], dr) in done: continue
        out["random"].append(run(a.ckpts[0], a.data, True, dr)); save()
    print(f"wrote {op}")
