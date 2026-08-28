"""E12c: interchange the C subspace MID-ROLLOUT, not at t=0.

Pre-registered in `docs/E12C_PREREG.md`. The dormant-pathway test Makelov et al. (ICLR 2024) imply:
E4 edits an encoder-produced state the model was trained to represent; this edits a state the model
reached on its own, 50 steps into autonomous imagination.
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import decode_physics
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args

DEG, LD, W = 4, 12, 10
ANALYSIS = slice(204, None)
NEWTON = 25


def _CG(z, c):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEG) @ c
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


def _dial(z, target, c):
    z0 = z.clone(); z = z.clone()
    for _ in range(NEWTON):
        Cv, g = _CG(z, c)
        z = z - ((Cv - target) / g.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * g
    return z, (z - z0).norm(dim=-1)


def run(ckpt, data, depth, post, rand=False, draw=0, tangent=False, seed=0):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad(): hs = m.encode(fr[ANALYSIS]).detach()
    H = hs[:, W:]; hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad(): nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - hm) @ U) @ R) - Z
    c = torch.as_tensor(np.asarray(cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)["coeffs"]),
                        dtype=Z.dtype, device=Z.device)
    if rand:
        g = torch.Generator(device="cpu").manual_seed(1000 + draw)
        rc = torch.randn(c.shape[0], generator=g, dtype=torch.float64)
        c = (rc / rc.norm() * c.norm().cpu()).to(Z.dtype).to(Z.device)
    P = U @ R; Ppinv = torch.linalg.pinv(P)

    def roll_from(h, n):
        with torch.no_grad():
            preds = []
            for _ in range(n):
                preds.append(m.readout_from_h(h)); h = m.transition(h)
            img = torch.stack(preds, 1)
        return ((img + 0.5) * 255.0).clamp(0, 255).cpu().numpy(), h

    # roll to `depth` autonomously
    h = hs[:, W].clone()
    with torch.no_grad():
        for _ in range(depth):
            h = m.transition(h)
    z = (h - hm) @ P
    Cv, _ = _CG(z, c)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(z.shape[0], generator=gen)
    target = Cv[perm]

    base_frames, _ = roll_from(h.clone(), post)
    base_E = np.nanmedian(decode_physics(base_frames)["energy"], axis=-1)

    if tangent:
        _, gg = _CG(z, c)
        u = gg / gg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rnd = torch.randn(z.shape, generator=torch.Generator().manual_seed(seed), device="cpu").to(z.device)
        rnd = rnd - (rnd * u).sum(-1, keepdim=True) * u
        _, dn = _dial(z, target, c)
        z_ed = z + dn.unsqueeze(-1) * rnd / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        z_ed, _ = _dial(z, target, c)
    h_ed = h + (z_ed - z) @ Ppinv
    ed_frames, _ = roll_from(h_ed, post)
    ed_E = np.nanmedian(decode_physics(ed_frames)["energy"], axis=-1)

    return {"ckpt": ckpt, "depth": depth, "post": post, "random": rand, "tangent": tangent,
            "draw": draw if rand else None,
            "intended_dC": (target - Cv).cpu().numpy().tolist(),
            "realised_dE": (ed_E - base_E).tolist()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{s}_step6500.pt" for s in (3, 4, 5)])
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--depths", nargs="+", type=int, default=[0, 50])
    p.add_argument("--post", type=int, default=50)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--n-tangent", type=int, default=3)
    p.add_argument("--out", default="runs/e12c_interchange.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"rows": []}
    key = lambda r: (r["ckpt"], r["depth"], r["random"], r["tangent"], r["draw"])
    done = {key(r) for r in out["rows"]}
    for ck in a.ckpts:
        if not pathlib.Path(ck).exists(): continue
        for dep in a.depths:
            for kind in ["rec"] + [f"rand{i}" for i in range(a.n_random)] + [f"tan{i}" for i in range(a.n_tangent)]:
                rand = kind.startswith("rand"); tan = kind.startswith("tan")
                dr = int(kind[4:]) if rand else (int(kind[3:]) if tan else 0)
                if (ck, dep, rand, tan, dr if rand else None) in done: continue
                out["rows"].append(run(ck, a.data, dep, a.post, rand, dr, tan, seed=dr))
                op.write_text(json.dumps(out, indent=1) + "\n")
        print(f"  done {ck}", flush=True)
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
