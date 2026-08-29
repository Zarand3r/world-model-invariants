"""F1 (amendment 5): does the VALIDATED extraction's scalar obey a balance law?

No new extractor. `C` comes from `fit_hamiltonian_pair` -- the same call, unchanged, that produced
every result in the manuscript -- and is frozen before any balance quantity is computed. The only
question asked here is whether its change under the model's own transition tracks the action's power.

    predicted   dC_pred = tau * thetadot * dt / (dE/dC)
    observed    dC_obs  = C(z_{t+1}) - C(z_t)

A2's control shuffles `tau * thetadot` across transitions, destroying the pairing with the state while
leaving the marginal distribution identical.
"""
import argparse, json, pathlib
import numpy as np, torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.fit_cache import cached_fit
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.polynomial import monomial_features
from latent_noether.provenance import attach, inputs_from_args

DEG, LD, WARMUP, DT = 4, 12, 10, 0.05
ANALYSIS = slice(204, None)


def _spear(a, b):
    def rk(x):
        x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
    ra, rb = rk(a) - rk(a).mean(), rk(b) - rk(b).mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-30))


def run(ckpt, data, seed=0):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    av = torch.as_tensor(d["actions"][ANALYSIS]).float().cuda()
    st = d["states"][ANALYSIS]
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    with torch.no_grad():
        hs = m.encode(fr, actions=av).detach()
    H = hs[:, WARMUP:]; A = av[:, WARMUP:, 0]
    hm = H.reshape(-1, H.shape[-1]).mean(0)
    U = pca_subspace(H, LD); Z = (H - hm) @ U; R = effective_rank_basis(Z); Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1]), a=A.reshape(-1, 1)).reshape(H.shape)
    Zn = ((nxt - hm) @ U) @ R
    F = Zn - Z

    # THE PAPER'S OWN EXTRACTION, frozen here before anything about balance is computed
    c = torch.as_tensor(np.asarray(cached_fit(Z.double().cpu(), F.double().cpu(), DEG, 8)["coeffs"]),
                        dtype=Z.dtype, device=Z.device)
    with torch.no_grad():
        Cv = (monomial_features(Z.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2]).cpu().numpy()
        Cn = (monomial_features(Zn.reshape(-1, LD), DEG) @ c).reshape(Z.shape[:2]).cpu().numpy()

    E = d["energy"][ANALYSIS][:, WARMUP:]
    thd = st[:, WARMUP:, 1]
    tau = A.cpu().numpy()
    rho_E = float(abs(np.corrcoef(Cv.ravel(), E.ravel())[0, 1]))
    dEdC = float(np.polyfit(Cv.ravel(), E.ravel(), 1)[0])

    dC_obs = (Cn - Cv).ravel()
    power = (tau * thd * DT).ravel()
    dC_pred = power / dEdC
    rng = np.random.default_rng(seed)
    shuffled = dC_pred[rng.permutation(len(dC_pred))]

    return {"ckpt": ckpt, "rho_C_energy": rho_E, "dEdC": dEdC,
            "spearman_pred_obs": _spear(dC_pred, dC_obs),
            "spearman_shuffled_obs": _spear(shuffled, dC_obs),
            "pearson_pred_obs": float(np.corrcoef(dC_pred, dC_obs)[0, 1]),
            "scale_obs_over_pred": float(np.std(dC_obs) / max(np.std(dC_pred), 1e-30)),
            "n": int(len(dC_obs))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=[f"runs/f1_act_s{s}.pt" for s in (3, 4, 5)])
    p.add_argument("--data", default="runs/pendulum_actuated.npz")
    p.add_argument("--out", default="runs/f1_balance_measured.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        r = run(ck, a.data)
        print(f"  {ck.split('/')[-1]:16s} rho(C,E) {r['rho_C_energy']:.4f}   "
              f"Spearman(pred,obs) {r['spearman_pred_obs']:+.4f}   "
              f"shuffled {r['spearman_shuffled_obs']:+.4f}   "
              f"|obs|/|pred| {r['scale_obs_over_pred']:.2f}", flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
