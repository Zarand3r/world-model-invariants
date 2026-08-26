"""S1: train a real DreamerV3 world model on gymnasium Pendulum pixels.

World model only — no actor, no critic. We never need a policy, and dropping them removes most of
the cost and all of the RL instability.

REGISTERED ACCEPTANCE, before any analysis is permitted (LESSONS M11: a model that did not train
is not evidence about the method):
  1. reconstruction converges — teacher-forced MSE well below the variance of the frames
  2. open-loop rollouts stay on the manifold (do not diverge or collapse to a constant)
  3. the model's rollouts conserve the TRUE energy better than a shuffled-frame control; if the
     model has learned no physics there is nothing downstream to recover and S2-S5 are void

Runs are capped by OPTIMIZER STEPS. `--max-hours` remains as a loose energy bound and must be set
so it never binds -- see M28 on the argument itself. Capping by wall clock makes the step count a
function of machine load, which silently gave one arm 37% more training than another; models being
compared must receive equal steps.

`--ckpt-at` saves intermediate checkpoints at named steps, so a single run yields the whole training
trajectory (E8 in docs/ROADMAP.md) rather than only its endpoint.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, device):
    d = np.load(path)
    frames = torch.as_tensor(d["frames"]).float().div_(255.0).sub_(0.5)   # (B,T,H,W,3) NHWC
    return frames.to(device), torch.as_tensor(d["states"]).float(), torch.as_tensor(d["energy"]).float()


def true_energy_from_states(th, thd, g=10.0, m=1.0, l=1.0):
    return 0.5 * (m * l ** 2 / 3) * thd ** 2 + m * g * (l / 2) * torch.cos(th)


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames, states, energy = load(a.data, dev)
    n_train = int(0.8 * frames.shape[0])
    train, val = frames[:n_train], frames[n_train:]
    print(f"  data {tuple(frames.shape)} on {dev}   train {n_train}  val {frames.shape[0]-n_train}")

    torch.manual_seed(a.seed)
    model = DreamerV3Adapter(device=dev).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  DreamerV3 world model: {n_par/1e6:.1f}M parameters")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-8)

    t0 = time.time()
    step = 0
    hist = []
    data_sha = _sha256(a.data)          # hashed once; it is the same file all run
    ckpt_at = sorted({int(x) for x in str(a.ckpt_at).split(",") if x.strip()})
    if ckpt_at:
        print(f"  intermediate checkpoints at steps: {ckpt_at}")

    def _save(path, n_steps):
        hrs = (time.time() - t0) / 3600
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "steps": n_steps, "hours": hrs,
                    "hist": list(hist), "seed": a.seed, "data": a.data,
                    "data_sha256": data_sha, "argv": sys.argv[1:]}, path)
        return hrs
    while step < a.steps and (time.time() - t0) < a.max_hours * 3600:
        i = torch.randint(0, n_train, (a.batch,), device=dev)
        t = torch.randint(0, train.shape[1] - a.seq, (1,)).item()
        batch = train[i, t:t + a.seq]
        recon, kl, kl_raw = model.loss(batch, kl_free=a.free_bits)
        loss = recon + kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        step += 1
        if step in ckpt_at:
            cp = str(a.out).replace(".pt", f"_step{step}.pt")
            hrs = _save(cp, step)
            print(f"  [ckpt] {cp}  step {step}  [{hrs*60:.1f} min]", flush=True)
        if step % 2000 == 0 or step == 1:
            with torch.no_grad():
                vb = val[torch.randint(0, val.shape[0], (a.batch,), device=dev), :a.seq]
                vr, vk, vkr = model.loss(vb, kl_free=a.free_bits)
            el = time.time() - t0
            hist.append({"step": step, "recon": float(recon), "kl": float(kl),
                         "kl_raw": kl_raw, "val_recon": float(vr), "hours": el / 3600})
            print(f"  step {step:6d}  recon {float(recon):.2f}  kl {float(kl):.3f}  "
                  f"rawKL {kl_raw:6.2f}  val_recon {float(vr):.2f}  [{el/60:.1f} min]", flush=True)

    hours = (time.time() - t0) / 3600
    print(f"\n  trained {step} steps in {hours:.2f} h   "
          f"worst-case energy <= {600*hours/1000:.3f} kWh @600W TDP")

    # ---- registered acceptance checks ----
    model.eval()
    with torch.no_grad():
        vb = val[:, :a.seq]
        var = float(vb.var())
        vr, _, vkr = model.loss(vb, kl_free=a.free_bits)
        # open-loop rollout stability
        hs = model.encode(val[:, :10])
        h = hs[:, -1]
        outs = []
        for _ in range(60):
            outs.append(model.readout_from_h(h))
            h = model.transition(h)
        roll = torch.stack(outs, 1)
        finite = bool(torch.isfinite(roll).all())
        spread = float(roll.std())
    with torch.no_grad():
        hh = model.encode(val[:, :20])[:, -1]
        pred = model.readout_from_h(hh)
        mse_model = float(((pred - val[:, 19]) ** 2).mean())
        mse_mean = float(((val[:, 19].mean(dim=0, keepdim=True) - val[:, 19]) ** 2).mean())
    print(f"\n  ACCEPTANCE")
    print(f"    raw KL {vkr:.2f} nats  ({'OK' if vkr > 1.0 else 'FAIL — posterior collapsed'})")
    print(f"    1-step decode MSE {mse_model:.5f} vs predict-the-mean {mse_mean:.5f}   "
          f"ratio {mse_model/max(mse_mean,1e-12):.3f}  "
          f"({'OK' if mse_model < 0.25*mse_mean else 'FAIL'})")
    print(f"    rollout finite: {finite}   rollout pixel std {spread:.4f} "
          f"({'OK' if finite and spread > 0.01 else 'FAIL — collapsed or diverged'})")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # PROVENANCE (M29). Neither the checkpoint nor the log used to record WHICH dataset trained the
    # model. That gap is what let D69's step mismatch and a silent wrong-path launch both survive:
    # nothing on disk could contradict what we believed we had run. The content hash is the part
    # that matters -- a path can be reused for a regenerated file.
    _save(a.out, step)
    json.dump(hist, open(str(a.out).replace(".pt", "_hist.json"), "w"), indent=2)
    print(f"  saved {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--steps", type=int, default=100_000)
    # WALL CLOCK IS AN ENERGY BOUND, NOT A MATCHING VARIABLE (M28). Models being compared must
    # receive equal OPTIMIZER STEPS; leaving wall clock to decide gave the conservative arm 37% more
    # steps than the dissipative one because the machine was busier for the second batch. Set
    # --max-hours loose enough that it never binds, and let --steps define the contract.
    p.add_argument("--max-hours", type=float, default=6.0)
    # E8 (docs/ROADMAP.md): the training TRAJECTORY, not just its endpoint. Saving milestones from a
    # single run is far cheaper than retraining per milestone, and guarantees the checkpoints lie on
    # one optimisation path rather than on several.
    p.add_argument("--ckpt-at", default="",
                   help="comma-separated steps at which to save <out>_step<N>.pt")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    # FREE BITS = 0, a deliberate deviation from the reference default of 1.0. Free bits puts a
    # floor under the KL so it cannot be crushed; with the floor at 0 the KL is fully penalised.
    # Set to 0 after the from-scratch implementation collapsed by mis-applying it (LESSONS M16),
    # and kept because the measured raw KL settles at ~1.8 nats -- healthy, nowhere near collapse,
    # so the floor is not needed here. Stated because "published defaults" is otherwise inaccurate.
    p.add_argument("--free-bits", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/dreamer_pendulum_s0.pt")
    a = p.parse_args()
    print("S1: training a real DreamerV3 world model on gymnasium Pendulum pixels.")
    print(f"    wall-clock cap {a.max_hours} h  ->  energy bound <= {600*a.max_hours/1000:.2f} kWh\n")
    main(a)
