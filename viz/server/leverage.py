"""Which latent directions the model gives prominence to, and which ones actually matter.

Two numbers per extracted direction u:

    V(u)   = Var(u^T z)                     statistical prominence — what PCA sees
    D_H(u) = E[ L_H(z + eps u) - L_H(z) ]   causal leverage — how much an H-step rollout degrades
                                            when that direction is displaced

plus how hard the projection edit pushes on each one. The hypothesis these test is that the
directions the edit moves have small V and large D — that the architecture assigns little
statistical prominence to state with large physical consequence.

**eps is one fixed absolute displacement for every direction**, so D is comparable across them: it
answers "how bad is it to be wrong by this much, in this direction", where scaling per direction
would instead answer "how bad is a typical error" — the question PCA already answers. The value
eps = 0.25 |z| is inherited from `run_dreamer_leverage.py`, where it was calibrated: at 0.05 |z| the
damage does not leave the second-order regime and comes out negative.

Both signs of the displacement are averaged, so a direction cannot score well merely because the
model happens to sit on one side of it.
"""
import numpy as np
import torch

from latent_noether.extraction import Bundle
from viz.server.rollout import _C_and_grad, analysis_frames, start_states

HORIZON = 40
EPS_FRAC = 0.25
ALPHA = 0.4          # the strongest projection in the published grid


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()), 1e-30))


def measure(model, b: Bundle, horizon: int = HORIZON, eps_frac: float = EPS_FRAC) -> dict:
    frames = analysis_frames(b)
    dev = frames.device
    P = torch.as_tensor(b.P, dtype=torch.float32, device=dev)
    h_mean = torch.as_tensor(b.h_mean, device=dev)
    Z = torch.as_tensor(b.Z, dtype=torch.float32, device=dev)
    r = Z.shape[-1]

    V = Z.reshape(-1, r).var(0).cpu().numpy()
    eps = float(eps_frac * Z.reshape(-1, r).norm(dim=-1).mean())

    h0 = start_states(model, b)[:, b.warmup].clone()
    ref = frames[:, b.warmup: b.warmup + horizon]

    def loss(h_start):
        with torch.no_grad():
            h, preds = h_start, []
            for _ in range(horizon):
                preds.append(model.readout_from_h(h))
                h = model.transition(h)
            return float(torch.nn.functional.mse_loss(torch.stack(preds, 1), ref))

    base = loss(h0)
    D = np.array([0.5 * ((loss(h0 + eps * P[:, i]) - base) + (loss(h0 - eps * P[:, i]) - base))
                  for i in range(r)])

    coeffs = torch.as_tensor(b.coeffs, dtype=torch.float32, device=dev)
    P_pinv = torch.as_tensor(b.P_pinv, dtype=torch.float32, device=dev)
    with torch.no_grad():
        h, moves = h0.clone(), []
        C0, _ = _C_and_grad((h - h_mean) @ P, coeffs, b.degree)
        for _ in range(horizon):
            h = model.transition(h)
            z = (h - h_mean) @ P
            Cv, g = _C_and_grad(z, coeffs, b.degree)
            step = ALPHA * ((Cv - C0) / g.pow(2).sum(-1).clamp_min(1e-12)).unsqueeze(-1) * g
            moves.append(step.abs().mean(0).cpu().numpy())
            h = h - (step @ P_pinv)
    edit_move = np.mean(moves, 0)

    return {"eps": eps, "baseline_loss": base, "horizon": horizon,
            "variance": V.tolist(), "damage": D.tolist(), "edit_move": edit_move.tolist(),
            "rho_V_D": spearman(V, D), "rho_D_edit": spearman(D, edit_move),
            "damage_ratio": float(np.max(D) / max(np.min(D), 1e-30))}
