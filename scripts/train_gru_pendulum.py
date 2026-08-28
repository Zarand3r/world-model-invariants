"""Train the deterministic conv-GRU world model (F4).

Deliberately mirrors `train_dreamer_pendulum.py`: same step-capped contract, same milestone grid,
same M29 provenance record, same registered acceptance checks minus the KL check, which is undefined
for a model with no stochastic latent.
"""
import argparse, hashlib, json, pathlib, sys, time
import numpy as np, torch
from latent_noether.gru_world_model import ConvGRUWorldModel


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    d = np.load(a.data)
    frames = torch.as_tensor(d["frames"]).float().div_(255.0).sub_(0.5)
    n_train = int(0.8 * frames.shape[0])
    train, val = frames[:n_train].to(dev), frames[n_train:].to(dev)
    model = ConvGRUWorldModel(device=dev).to(dev)
    print(f"  ConvGRU world model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    data_sha = _sha256(a.data)
    ckpt_at = sorted({int(x) for x in str(a.ckpt_at).split(",") if x.strip()})
    if ckpt_at:
        print(f"  intermediate checkpoints at steps: {ckpt_at}")
    t0 = time.time(); step = 0; hist = []

    def save(path, n_steps):
        hrs = (time.time() - t0) / 3600
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "steps": n_steps, "hours": hrs, "hist": list(hist),
                    "seed": a.seed, "data": a.data, "data_sha256": data_sha,
                    "arch": "ConvGRUWorldModel", "argv": sys.argv[1:]}, path)
        return hrs

    while step < a.steps and (time.time() - t0) < a.max_hours * 3600:
        i = torch.randint(0, train.shape[0], (a.batch,), device=dev)
        t = torch.randint(0, train.shape[1] - a.seq, (1,)).item()
        loss = model.loss(train[i, t:t + a.seq])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step(); step += 1
        if step in ckpt_at:
            p = str(a.out).replace(".pt", f"_step{step}.pt")
            print(f"  [ckpt] {p}  step {step}  [{save(p, step)*60:.1f} min]", flush=True)
        if step % 2000 == 0 or step == 1:
            with torch.no_grad():
                vb = val[torch.randint(0, val.shape[0], (a.batch,), device=dev), :a.seq]
                vl = float(model.loss(vb))
            hist.append({"step": step, "loss": float(loss), "val": vl})
            print(f"  step {step:6d}  recon {float(loss):.5f}  val {vl:.5f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    # ---- registered acceptance checks (docs/F4_PREREG.md) ----
    model.eval()
    with torch.no_grad():
        vb = val[:, :a.seq]
        h = model.encode(val[:, :20])[:, -1]
        pred = model.readout_from_h(h)
        mse_model = float(((pred - val[:, 19]) ** 2).mean())
        mse_mean = float(((val[:, 19].mean(0, keepdim=True) - val[:, 19]) ** 2).mean())
        hs = model.encode(val[:, :10])[:, -1]
        outs = []
        for _ in range(60):
            outs.append(model.readout_from_h(hs)); hs = model.transition(hs)
        roll = torch.stack(outs, 1)
        finite = bool(torch.isfinite(roll).all()); spread = float(roll.std())
    print("\n  ACCEPTANCE")
    print(f"    1-step decode MSE {mse_model:.5f} vs predict-the-mean {mse_mean:.5f}  "
          f"ratio {mse_model/max(mse_mean,1e-12):.3f}  ({'OK' if mse_model < 0.25*mse_mean else 'FAIL'})")
    # A static rollout passes a pixel-std check while being useless: F4's first run had
    # std 0.021 (above the 0.01 floor) with frame-to-frame change 0.00084, i.e. frozen.
    # The transition must actually MOVE the decoded frames.
    with torch.no_grad():
        motion = float((roll[:, 1:] - roll[:, :-1]).abs().mean())
        data_motion = float((val[:, 1:20] - val[:, :19]).abs().mean())
    print(f"    rollout finite: {finite}  pixel std {spread:.4f} "
          f"({'OK' if finite and spread > 0.01 else 'FAIL — collapsed or diverged'})")
    print(f"    rollout motion {motion:.5f} vs data {data_motion:.5f}  ratio {motion/max(data_motion,1e-12):.3f}  "
          f"({'OK' if motion > 0.2 * data_motion else 'FAIL — transition is static'})")
    save(a.out, step)
    print(f"  saved {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--ckpt-at", default="1000,3000,6500,15000,30000,60000")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="runs/gru_ref_s3.pt")
    main(p.parse_args())
