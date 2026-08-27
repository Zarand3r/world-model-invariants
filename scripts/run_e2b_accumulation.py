"""E2b: does invariant violation accumulate as a random walk (k^0.5) or a systematic bias (k^1)?

Pre-registered in `docs/E2B_PREREG.md`. Authorised by the roadmap's E2 Outcome B decision branch.

Measures `|C(z_k) - C(z_0)|` along an autonomous rollout and fits the exponent `beta` in

    log median_traj |C(z_k) - C(z_0)| = a + beta log k,   k = 1..99

No fitting of `C` happens here -- it is frozen exactly as in E1/E2/E3.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features

DEGREE, LD, WARMUP, DEPTH = 4, 12, 10, 100
ANALYSIS = slice(204, None)


def run(ckpt, data, random_law=False, draw=0):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
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
    P = U @ R
    with torch.no_grad():
        h = hs[:, WARMUP].clone()
        Cs = []
        for _ in range(DEPTH):
            Cs.append((monomial_features((h - hm) @ P, DEGREE) @ coeffs).cpu().numpy())
            h = m.transition(h)
    C = np.asarray(Cs).T                       # (traj, depth)
    dC = np.abs(C - C[:, :1])                  # |C(z_k) - C(z_0)|
    return {"ckpt": ckpt, "data": data, "random_law": random_law,
            "draw": draw if random_law else None,
            "dC_per_traj": dC.tolist()}        # RAW ROW


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--conservative", nargs="*", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3,4,5)])
    p.add_argument("--damped", nargs="*", default=[f"runs/dreamer_damped_s{s}_step6500.pt" for s in (0,1,2)])
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--out", default="runs/e2b_accumulation.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"conservative": [], "damped": [], "random": []}
    save = lambda: op.write_text(json.dumps(out, indent=1) + "\n")
    for ck in a.conservative:
        if ck in {r["ckpt"] for r in out["conservative"]} or not pathlib.Path(ck).exists(): continue
        out["conservative"].append(run(ck, "runs/pendulum_pixels.npz")); save()
    for ck in a.damped:
        if ck in {r["ckpt"] for r in out["damped"]} or not pathlib.Path(ck).exists(): continue
        out["damped"].append(run(ck, "runs/pendulum_pixels_damped.npz")); save()
    done = {r["draw"] for r in out["random"]}
    for dr in range(a.n_random):
        if dr in done: continue
        out["random"].append(run(a.conservative[0], "runs/pendulum_pixels.npz", True, dr)); save()
    print(f"wrote {op}")
