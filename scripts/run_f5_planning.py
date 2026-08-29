"""F5: does the operator-privileged direction improve CONTROL RETURN?

Pre-registered in `docs/F5_PREREG.md`. Approved to run 2026-08-29.

Task: energy targeting inside the training band. Reward `-(E_t - E*)^2 / sigma_E^2`, with the reward
for scoring computed from the SIMULATOR's true state, never from the model's own decode, so the model
cannot score its own success.

Planner: CEM over the learned world model. No actor-critic is trained; CEM plans directly.

Arms differ ONLY in the direction of an equal-size latent edit applied during imagination:

    none      no edit
    conserve  drive C toward its initial value          (naive; wrong under actuation, expected to hurt)
    balance   drive C toward C0 + accumulated tau*thetadot*dt   (the F1 object, SUPPLIED not learned)
    probe     same target, using E18's supervised energy probe
    random    same target, using a magnitude-matched random direction

Every arm uses the direction-matched step `z <- z - eps * sign(C - C_target) * gradC/||gradC||`,
never the level-set projection: the projection is scale-invariant in `C`, which is precisely the
defect the 2026-08-26 audit found in the published paper's null.
"""
import argparse, json, pathlib
import gymnasium as gym
import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.pixel_readout import energy as true_energy
from latent_noether.planning_readout import energy_from_frames
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args
from scripts.make_pendulum_pixels import RES, TH_MAX, THD_MAX, downsample

DEG, LD, WARMUP, DT = 4, 12, 10, 0.05
TAU_MAX = 1.0
ARMS = ("none", "conserve", "balance", "probe", "random")


def _C_and_grad(z, c):
    with torch.enable_grad():
        zz = z.detach().requires_grad_(True)
        v = monomial_features(zz, DEG) @ c
        g, = torch.autograd.grad(v.sum(), zz)
    return v.detach(), g.detach()


class Model:
    """World model plus the frozen extraction it will be asked to enforce."""

    def __init__(self, ckpt, data):
        d = np.load(data)
        fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
        av = torch.as_tensor(d["actions"]).float().cuda()
        self.m = DreamerV3Adapter(device="cuda").cuda()
        self.m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); self.m.eval()
        with torch.no_grad():
            hs = self.m.encode(fr, actions=av).detach()
        H = hs[:, WARMUP:]; A = av[:, WARMUP:, 0]
        self.hm = H.reshape(-1, H.shape[-1]).mean(0)
        self.U = pca_subspace(H, LD)
        Z = (H - self.hm) @ self.U
        self.R = effective_rank_basis(Z); Z = Z @ self.R
        self.P = self.U @ self.R
        self.Ppinv = torch.linalg.pinv(self.P)
        with torch.no_grad():
            nxt = self.m.transition(H.reshape(-1, H.shape[-1]),
                                    a=A.reshape(-1, 1)).reshape(H.shape)
        F = (((nxt - self.hm) @ self.U) @ self.R) - Z

        # label-free C: optimises conservation, never sees energy
        self.c_free = torch.as_tensor(np.asarray(cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)["coeffs"]),
                                      dtype=Z.dtype, device=Z.device)
        # supervised probe: E18's arm, ridge fit to TRUE energy
        E = d["energy"][:, WARMUP:].reshape(-1)
        X = monomial_features(Z.reshape(-1, LD), DEG).double().cpu().numpy()
        y = np.asarray(E).ravel()[:len(X)]
        w = np.linalg.lstsq(X.T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ y, rcond=None)[0]
        self.c_probe = torch.as_tensor(w / (np.linalg.norm(w) + 1e-30), dtype=Z.dtype, device=Z.device)
        g = torch.Generator(device="cpu").manual_seed(7)
        rc = torch.randn(self.c_free.shape[0], generator=g, dtype=torch.float64)
        self.c_rand = (rc / rc.norm() * self.c_free.norm().cpu()).to(Z.dtype).to(Z.device)

        # frozen scale mapping a change in C to a change in physical energy, so a balance target
        # expressed in joules can be written as a target value of C. Fitted once, here, on training
        # data; never refitted during planning.
        with torch.no_grad():
            Cv = (monomial_features(Z.reshape(-1, LD), DEG) @ self.c_free).cpu().numpy()
        self.dEdC = float(np.polyfit(Cv, y[:len(Cv)], 1)[0])
        self.sigma_E = float(d["energy"].mean(-1).std())

    def coeffs(self, arm):
        return {"conserve": self.c_free, "balance": self.c_free,
                "probe": self.c_probe, "random": self.c_rand}.get(arm)

    def z_of(self, h):
        return (h - self.hm) @ self.P


def imagine(M, h0, A, arm, eps):
    """Roll the model forward under action sequence A (K, H), applying the arm's edit each step.

    Returns decoded energy (K, H). The balance target accumulates tau*thetadot*dt using the
    thetadot the planner decodes as it goes -- the agent knows its own action, so the source term is
    supplied rather than learned.
    """
    K, Hn = A.shape
    c = M.coeffs(arm)
    h = h0.expand(K, -1).contiguous()
    C0 = None if c is None else _C_and_grad(M.z_of(h), c)[0]
    target = None if C0 is None else C0.clone()
    frames, th_prev = [], None
    for k in range(Hn):
        img = M.m.readout_from_h(h)
        frames.append(img)
        # decode theta incrementally so the balance target can use thetadot
        if c is not None and arm == "balance":
            e = energy_from_frames(img.unsqueeze(1))
            th = e["theta"][:, 0]
            if th_prev is not None:
                thd = (th - th_prev) / DT
                target = target + (A[:, k] * thd * DT) / max(M.dEdC, 1e-9)
            th_prev = th
        h = M.m.transition(h, a=A[:, k:k + 1])
        if c is not None and eps > 0:
            z = M.z_of(h)
            Cv, gr = _C_and_grad(z, c)
            u = gr / gr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            tgt = C0 if arm != "balance" else target
            h = h - ((eps * torch.sign(Cv - tgt).unsqueeze(-1) * u) @ M.Ppinv)
    return energy_from_frames(torch.stack(frames, 1))["energy"]


def cem_action(M, h, E_star, arm, eps, rng, K, plan_H, iters, elites):
    mu = torch.zeros(plan_H, device="cuda")
    sd = torch.full((plan_H,), 0.6, device="cuda")
    for _ in range(iters):
        noise = torch.as_tensor(rng.standard_normal((K, plan_H)), dtype=torch.float32, device="cuda")
        A = (mu + sd * noise).clamp(-TAU_MAX, TAU_MAX)
        with torch.no_grad():
            E = imagine(M, h, A, arm, eps)
        # skip t=0: the backward-difference readout has no velocity there
        R = -((E[:, 1:] - E_star) ** 2).sum(1)
        R = torch.nan_to_num(R, nan=-1e9)
        idx = torch.topk(R, elites).indices
        mu, sd = A[idx].mean(0), A[idx].std(0).clamp_min(0.05)
    return float(mu[0].clamp(-TAU_MAX, TAU_MAX))


def episode(M, arm, eps, E_star, seed, steps, K, plan_H, iters, elites, random_policy=False):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    u = env.unwrapped
    rng = np.random.default_rng(seed)
    env.reset(seed=int(rng.integers(0, 2 ** 31)))
    u.state = np.array([rng.uniform(-TH_MAX, TH_MAX), rng.uniform(-THD_MAX, THD_MAX)])
    buf_f, buf_a, ret, clipped = [], [], 0.0, False
    for _ in range(WARMUP):                       # build a posterior with zero action
        buf_f.append(downsample(env.render())); buf_a.append(0.0)
        env.step(np.array([0.0], dtype=np.float32))
    for _ in range(steps):
        fr = torch.as_tensor(np.stack(buf_f[-WARMUP:])).float().div_(255.).sub_(0.5).cuda().unsqueeze(0)
        av = torch.as_tensor(np.array(buf_a[-WARMUP:], dtype=np.float32)).view(1, WARMUP, 1).cuda()
        with torch.no_grad():
            h = M.m.encode(fr, actions=av)[:, -1]
        a = (float(rng.uniform(-TAU_MAX, TAU_MAX)) if random_policy
             else cem_action(M, h, E_star, arm, eps, rng, K, plan_H, iters, elites))
        env.step(np.array([a], dtype=np.float32))
        th, thd = float(u.state[0]), float(u.state[1])
        if abs(thd) >= u.max_speed - 1e-6:
            clipped = True
        ret += -((true_energy(th, thd) - E_star) ** 2) / (M.sigma_E ** 2)
        buf_f.append(downsample(env.render())); buf_a.append(a)
    env.close()
    return {"return": ret / steps, "clipped": clipped, "E_star": E_star, "seed": seed}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/f1_act_s{s}.pt" for s in (3, 4, 5)])
    p.add_argument("--data", default="runs/pendulum_actuated.npz")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--eps", type=float, default=0.02)
    p.add_argument("--K", type=int, default=96)
    p.add_argument("--plan-H", type=int, default=10)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--elites", type=int, default=12)
    p.add_argument("--gate0", action="store_true", help="only compare arm 'none' against random actions")
    p.add_argument("--out", default="runs/f5_planning.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    # E* targets are drawn ONCE and shared by every arm and model, so arms never differ by task.
    g = np.random.default_rng(20260829)
    targets = [(float(g.uniform(-1.5, 3.5)), int(g.integers(0, 2 ** 31))) for _ in range(a.episodes)]

    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        print(f"[F5] {ck}", flush=True)
        M = Model(ck, a.data)
        rec = {"ckpt": ck, "dEdC": M.dEdC, "sigma_E": M.sigma_E, "arms": {}}
        arms = ["none", "__random_policy__"] if a.gate0 else a.arms
        for arm in arms:
            rp = arm == "__random_policy__"
            eps = 0.0 if arm in ("none", "__random_policy__") else a.eps
            rows = [episode(M, "none" if rp else arm, eps, E, sd, a.steps,
                            a.K, a.plan_H, a.iters, a.elites, random_policy=rp)
                    for E, sd in targets]
            r = [x["return"] for x in rows]
            rec["arms"][arm] = {"rows": rows, "mean_return": float(np.mean(r)),
                                "median_return": float(np.median(r)),
                                "clipped": int(sum(x["clipped"] for x in rows))}
            print(f"    {arm:18s} mean {np.mean(r):+8.3f}  median {np.median(r):+8.3f}  "
                  f"clipped {rec['arms'][arm]['clipped']}/{len(rows)}", flush=True)
        out["models"].append(rec); op.write_text(json.dumps(out, indent=1) + "\n")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
