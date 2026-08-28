"""F1: does the action-conditioned model actually USE its action input?

Registered in the execution log on 2026-08-28, before any checkpoint past step 3,000 existed:

  Track the ratio across the E8 checkpoint grid. If it has not fallen meaningfully below the
  step-3000 value of 0.870 by step 60,000, the action conditioning is too weak for F1's question to
  be answerable on this dataset, and the honest move is to report that as a boundary and raise the
  torque -- NOT to proceed to the balance-law fit and interpret whatever it returns.

The check exists for the failure mode F4 taught: a model that silently ignores an input passes every
other acceptance criterion, and every downstream number computed on it would be meaningless.

Three arms, all from the same encoded start state, over an open-loop rollout:
  true      -- the actions actually applied in the data
  shuffled  -- the same actions permuted ACROSS trajectories, so the marginal distribution of
               actions is identical and only their pairing with the state is destroyed
  zeros     -- no action at all, the free-evolution path
"""
import argparse, json, pathlib
import numpy as np, torch
from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.provenance import attach, inputs_from_args

WARMUP, HORIZON, ANALYSIS = 10, 20, slice(204, None)


def run(ckpt, data, horizon=HORIZON, seed=0):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).cuda()
    av = torch.as_tensor(d["actions"][ANALYSIS]).float().cuda()
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"]); m.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(fr.shape[0], generator=g).to(fr.device)
    ref = fr[:, WARMUP:WARMUP + horizon]
    with torch.no_grad():
        h0 = m.encode(fr[:, :WARMUP], actions=av[:, :WARMUP])[:, -1]
        errs = {}
        for tag, a in (("true", av), ("shuffled", av[perm]), ("zeros", torch.zeros_like(av))):
            h, outs = h0.clone(), []
            for k in range(horizon):
                outs.append(m.readout_from_h(h))
                h = m.transition(h, a=a[:, WARMUP - 1 + k])
            errs[tag] = float(((torch.stack(outs, 1) - ref) ** 2).mean())
    return {"ckpt": ckpt, **{f"mse_{k}": v for k, v in errs.items()},
            "ratio_true_over_shuffled": errs["true"] / max(errs["shuffled"], 1e-12),
            "ratio_true_over_zeros": errs["true"] / max(errs["zeros"], 1e-12)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--data", default="runs/pendulum_actuated.npz")
    p.add_argument("--out", default="runs/f1_action_use.json")
    a = p.parse_args()
    op = pathlib.Path(a.out)
    out = json.loads(op.read_text()) if op.exists() else {"models": []}
    done = {r["ckpt"] for r in out["models"]}
    for ck in a.ckpts:
        if ck in done or not pathlib.Path(ck).exists():
            continue
        r = run(ck, a.data)
        print(f"  {ck.split('/')[-1]:26s} true {r['mse_true']:.6f}  shuf {r['mse_shuffled']:.6f}  "
              f"zeros {r['mse_zeros']:.6f}   ratio {r['ratio_true_over_shuffled']:.3f}", flush=True)
        out["models"].append(r); op.write_text(json.dumps(out, indent=1) + "\n")
    attach(out, op, inputs=inputs_from_args(a))
    op.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {op}")
