"""F2: is accumulated invariant drift a usable online trust signal?

Pre-registered in `docs/F2_PREREG.md`. Every candidate signal is computed from information available
at inference only -- no ground truth, no future frames. Ground truth enters once, to score.

The ensemble baseline is deliberately advantaged: it uses three trained models where every other
signal uses one.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics, energy
from latent_noether.polynomial import monomial_features

DEG, LD, W = 4, 12, 10
ANALYSIS = slice(204, None)


def _load(ck):
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ck, map_location="cuda")["model"]); m.eval()
    return m


def _frame(m, fr):
    with torch.no_grad(): hs = m.encode(fr).detach()
    H = hs[:, W:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - hm) @ U) @ R) - Z
    return hs, hm, U @ R, Z, F


def run(ck, data, early, late, n_random=10):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    st = d["states"][ANALYSIS]
    m = _load(ck)
    hs, hm, P, Z, F = _frame(m, fr)
    c = torch.as_tensor(np.asarray(cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)["coeffs"]),
                        dtype=Z.dtype, device=Z.device)
    Zf = Z.reshape(-1, LD).double()
    cov = torch.cov(Zf.T) + 1e-6 * torch.eye(LD, device=Z.device, dtype=torch.float64)
    Lw = torch.linalg.cholesky(torch.linalg.inv(cov)); ref_w = Zf @ Lw

    def Cval(z, coef): return (monomial_features(z, DEG) @ coef)

    randc = []
    for dr in range(n_random):
        g = torch.Generator(device="cpu").manual_seed(1000 + dr)
        rc = torch.randn(c.shape[0], generator=g, dtype=torch.float64)
        randc.append((rc / rc.norm() * c.norm().cpu()).to(Z.dtype).to(Z.device))

    with torch.no_grad():
        h = hs[:, W].clone()
        z0 = (h - hm) @ P
        C0 = Cval(z0, c); Cr0 = [Cval(z0, rc) for rc in randc]
        sig = None; preds = []
        for k in range(late + 1):
            preds.append(m.readout_from_h(h))
            if k == early:
                z = (h - hm) @ P
                zn = (m.transition(h) - hm) @ P
                sig = {
                    "acc_drift": (Cval(z, c) - C0).abs().cpu().numpy(),
                    "inst_drift": (Cval(zn, c) - Cval(z, c)).abs().cpu().numpy(),
                    "latent_disp": (z - z0).norm(dim=-1).cpu().numpy(),
                    "nn_dist": torch.cdist(z.double() @ Lw, ref_w).min(-1).values.cpu().numpy(),
                    "rand_drift": np.median(np.stack(
                        [(Cval(z, rc) - c0).abs().cpu().numpy() for rc, c0 in zip(randc, Cr0)]), 0),
                }
            h = m.transition(h)
        frames = ((torch.stack(preds, 1) + 0.5) * 255.0).clamp(0, 255).cpu().numpy()

    ph = decode_physics(frames)
    E_true = energy(st[:, W:W + late + 1, 0], st[:, W:W + late + 1, 1])
    target = np.abs(ph["energy"][:, late] - E_true[:, late])
    return {"ckpt": ck, "early": early, "late": late,
            "signals": {k: np.asarray(v).tolist() for k, v in sig.items()},
            "target_energy_error": target.tolist(),
            "imagined_frames_last": None}, frames[:, late]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3, 4, 5)])
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--early", type=int, default=25)
    p.add_argument("--late", type=int, default=100)
    p.add_argument("--out", default="runs/f2_trust_signal.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {(r["ckpt"], r["early"]) for r in out["models"]}
    last_frames = {}
    for ck in a.ckpts:
        if (ck, a.early) in done or not pathlib.Path(ck).exists(): continue
        print(f"[F2] {ck}", flush=True)
        rec, lf = run(ck, a.data, a.early, a.late)
        last_frames[ck] = lf
        out["models"].append(rec); op.write_text(json.dumps(out, indent=1) + "\n")
    # ensemble disagreement: variance across seeds of the imagined frame at `late`
    if len(last_frames) >= 2:
        stack = np.stack(list(last_frames.values()))
        dis = stack.std(0).reshape(stack.shape[1], -1).mean(-1)
        out["ensemble_disagreement"] = dis.tolist()
        op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
