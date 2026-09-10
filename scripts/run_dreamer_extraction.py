"""S2 — THE CRITICAL GATE: does our extraction recover anything from a real DreamerV3?

This is the N9 replication at real scale. On our toy RSSM, recovery succeeded on only **3 of 22**
runs and six explanations were eliminated (D19, D20, D21, D28a). A real DreamerV3 is more
stochastic, higher-capacity, and learns its state from pixels rather than being handed
coordinates — so the honest prior is that this fails.

Substrate: gymnasium Pendulum-v1, free evolution, 64x64 RGB, reference DreamerV3 world model
(NM512/dreamerv3-torch) behind a verified adapter (our `transition` matches the reference
`img_step` to 5.1e-08).

Pipeline, unchanged from the toy work: PCA the deterministic state -> effective-rank basis ->
one-step displacement as the flow -> joint `f = B grad C` fit -> correlate `C` with true energy.

**What the true energy is here, and it is not the textbook one.** gymnasium integrates with
semi-implicit (symplectic) Euler, which exactly conserves a SHADOW Hamiltonian H~ = H + O(dt),
not H. Textbook E oscillates ~11% with no secular drift. The model learns gymnasium's map, so the
quantity it should conserve is H~. We score against textbook E and treat that ~11% as a noise
floor rather than pretending the ground truth is exact.

REGISTERED (before running):
  PASS if |rho| between the recovered C and true energy exceeds 0.7 on >= 2 of 3 seeds.
  FAIL -> the headline becomes "the method does not reach real world-model architectures", a
  legitimate and publishable limitation. S3-S5 are void. **Do NOT tune to rescue it.**

  Measured regardless of outcome: participation ratio and the observability spectrum of the
  Dreamer latent, against the GRU's 2.13-of-6. A wildly different geometry is the first clue for
  any failure.
"""
import argparse, hashlib, json, sys
import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features

DEGREE, LD = 4, 6      # LD overridden by --ld
G, M, L = 10.0, 1.0, 1.0


def true_energy(states):
    th, thd = states[..., 0], states[..., 1]
    return 0.5 * (M * L ** 2 / 3) * thd ** 2 + M * G * (L / 2) * np.cos(th)


def run(ckpt, data, warmup=10, LD=LD, untrained=False):
    """`untrained=True` skips loading the checkpoint: a randomly-initialised DreamerV3.

    THE CONTROL THIS PIPELINE WAS MISSING. N1 measured a *random feature map* recovering energy at
    rho = 0.998 on the GRU, and `gauge.decodability` states the rule outright -- an untrained
    network is a random feature map and its score must be subtracted, never used alone. The Dreamer
    headline (|rho|_E = 0.973 pre-registered) was reported WITHOUT this arm, so it could not
    distinguish "DreamerV3 learned the physics" from "a randomly-initialised recurrent net driven
    by these frames is already a faithful state encoder, and energy is the only function constant
    along trajectories on a 1-DOF conservative system".

    The seed is taken from the checkpoint path so the three untrained arms differ from each other.
    """
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    E_all = true_energy(d["states"])
    # DETERMINISTIC seed. `hash()` on a str is salted per process (PYTHONHASHSEED), so seeding
    # from it made the untrained control unreproducible -- the first run drew three arbitrary,
    # unrecoverable initialisations. The untrained arm IS the null distribution, so its seeds have
    # to be recoverable or the null cannot be checked.
    torch.manual_seed(int(hashlib.sha256(ckpt.encode()).hexdigest()[:8], 16))
    m = DreamerV3Adapter(device="cuda").cuda()
    if not untrained:
        m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
    # The untrained arm deliberately does NOT open the file. Its seed comes from the checkpoint
    # PATH (above), and the weights are discarded, so loading them only forced reproducers to
    # possess six checkpoints whose contents cannot affect the answer. The paper's null was run
    # over runs/dreamer_ref_s{0..5}.pt; three of those models no longer exist anywhere, and with
    # this change the arm still reproduces exactly, because only the path strings ever mattered.
    m.eval()

    val = slice(204, None)
    with torch.no_grad():
        H = m.encode(fr[val])[:, warmup:].detach()          # (B, T, deter)
    E = torch.as_tensor(E_all[val][:, warmup:], dtype=H.dtype, device=H.device)

    U = pca_subspace(H, LD)
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)
    Z = (H - h_mean) @ U
    R = effective_rank_basis(Z)
    Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - h_mean) @ U) @ R) - Z

    cov = torch.cov(Z.reshape(-1, Z.shape[-1]).T)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-30)
    pr = float(1.0 / (p ** 2).sum().clamp_min(1e-30))

    Zc, Fc = Z.double().cpu(), F.double().cpu()
    fit = fit_hamiltonian_pair(Zc, Fc, degree=DEGREE, n_basis=8)
    c = torch.as_tensor(fit["coeffs"], dtype=Zc.dtype)
    C = (monomial_features(Zc.reshape(-1, Zc.shape[-1]), DEGREE) @ c).reshape(Zc.shape[:2])
    n = min(C.shape[1], E.shape[1])
    rho = abs(float(torch.corrcoef(torch.stack(
        [C[:, :n].reshape(-1), E[:, :n].double().cpu().reshape(-1)]))[0, 1]))
    return {"ckpt": ckpt, "pairing_residual": fit["residual"], "rho_energy": rho,
            "participation_ratio": pr, "latent_dim": int(Z.shape[-1]),
            "eigenvalues": ev.flip(0).tolist()[:6]}


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpts", nargs="+", default=["runs/dreamer_ref_s0.pt"])
    a.add_argument("--data", default="runs/pendulum_pixels.npz")
    a.add_argument("--ld", type=int, default=LD)
    a.add_argument("--untrained", action="store_true",
                   help="randomly-initialised control: the N1 analogue, previously missing")
    a.add_argument("--out", default="runs/dreamer_extraction.json")
    a = a.parse_args()
    print("S2 GATE: does extraction recover anything from a REAL DreamerV3?")
    print("PASS: |rho|_E > 0.7 on >=2 of 3 seeds.  FAIL: the method does not reach real")
    print("      world-model architectures -- a limitation, not something to tune away.\n")
    rows = []
    for ck in a.ckpts:
        r = run(ck, a.data, LD=a.ld, untrained=a.untrained)
        rows.append(r)
        print(f"  {ck}\n    |rho|_E {r['rho_energy']:.3f}   pairing residual "
              f"{r['pairing_residual']:.4f}   participation ratio {r['participation_ratio']:.2f}"
              f" of {r['latent_dim']}  (GRU was 2.13 of 6)", flush=True)
        json.dump(rows, open(a.out, "w"), indent=2)
    rhos = [r["rho_energy"] for r in rows]
    n_pass = sum(1 for x in rhos if x > 0.7)
    print(f"\n--- {n_pass}/{len(rhos)} seeds above 0.7   values {[round(x,3) for x in rhos]}")
    print("\n--- VERDICT")
    if len(rhos) >= 3 and n_pass >= 2:
        if a.untrained:
            print("  THIS IS THE UNTRAINED CONTROL — a 'pass' here is a FAILURE OF THE GATE, not a")
            print("  success. A randomly-initialised network clearing the same threshold means the")
            print("  threshold does not measure learning. Report trained MINUS untrained.")
        else:
            print("  PASS. Extraction reaches a real DreamerV3 world model trained on pixels.")
        print("  The substrate objection that caps the paper is answered; proceed to S3.")
    elif len(rhos) < 3:
        print(f"  PARTIAL ({len(rhos)} seed(s)); the registered criterion needs 3.")
    else:
        print("  FAIL. The method does not reach real world-model architectures. This is the")
        print("  headline limitation; S3-S5 are void. Do not tune to rescue it.")
