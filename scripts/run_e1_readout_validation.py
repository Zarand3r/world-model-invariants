"""E1 step 1: validate the geometric pixel readout and measure its noise floor.

`docs/E1_PREREG.md` requires this to be reported BEFORE any intervention number, on
  (a) true rendered frames, and
  (b) Dreamer reconstructions of observation-conditioned frames.
An intervention effect smaller than the floor measured here is reported as "below readout
resolution", never as an effect.

Arm (b) is skipped when no checkpoint is present, and the run log records that it was skipped.

The primary E1 statistic is the normalised secular drift

    D_sec = OLS slope of Ehat_k against k, divided by the across-trajectory std of true E,

so this script also measures D_sec on REAL frames, where the true value is known to be ~0: gymnasium
integrates semi-implicitly and conserves a shadow Hamiltonian, giving oscillation in textbook E but
no secular component. Whatever D_sec this script reports on real frames is the floor below which an
E1 effect cannot be resolved.
"""
import argparse
import json
import pathlib

import numpy as np

from latent_noether.pixel_readout import decode_physics, load_pivot, energy

ANALYSIS = slice(204, None)          # the split E1 scores on; the pivot was fitted on 0:204
WARMUP = 10                          # matches run_dreamer_edit.py
HORIZON = 50


def _err_stats(name, err):
    err = np.asarray(err)[np.isfinite(np.asarray(err))]
    return {f"{name}_median": float(np.median(np.abs(err))),
            f"{name}_p90": float(np.percentile(np.abs(err), 90)),
            f"{name}_p99": float(np.percentile(np.abs(err), 99))}


def secular_drift(E, norm):
    """OLS slope of E against step index, per trajectory, normalised by `norm`."""
    E = np.asarray(E, dtype=np.float64)
    k = np.arange(E.shape[-1], dtype=np.float64)
    kc = k - k.mean()
    slope = (E - E.mean(-1, keepdims=True)) @ kc / (kc @ kc)
    return slope / norm


def validate_real(data, window):
    d = np.load(data)
    fr = d["frames"][ANALYSIS][:, window]
    st = d["states"][ANALYSIS][:, window]
    th_true, thd_true = st[..., 0], st[..., 1]
    E_true = energy(th_true, thd_true)
    norm = float(energy(d["states"][..., 0], d["states"][..., 1]).mean(-1).std())

    got = decode_physics(fr)
    # theta is unwrapped on both sides, but the readout's branch is arbitrary per trajectory:
    # remove a per-trajectory multiple of 2*pi before comparing.
    off = np.round((got["theta"] - th_true).mean(-1, keepdims=True) / (2 * np.pi)) * 2 * np.pi
    th_hat = got["theta"] - off

    out = {"n_traj": int(fr.shape[0]), "n_steps": int(fr.shape[1]),
           "across_traj_E_std": norm, "pivot": load_pivot()}
    out.update(_err_stats("theta_err_rad", th_hat - th_true))
    out.update(_err_stats("thetadot_err", got["thetadot"] - thd_true))
    out.update(_err_stats("E_err", got["energy"] - E_true))
    out.update(_err_stats("E_err_over_std", (got["energy"] - E_true) / norm))
    out["theta_err_deg_median"] = float(np.degrees(out["theta_err_rad_median"]))

    d_hat = secular_drift(got["energy"], norm)
    d_true = secular_drift(E_true, norm)
    out["D_sec_decoded_median"] = float(np.median(d_hat))
    out["D_sec_decoded_iqr"] = [float(np.percentile(d_hat, 25)), float(np.percentile(d_hat, 75))]
    out["D_sec_true_median"] = float(np.median(d_true))
    out["D_sec_true_iqr"] = [float(np.percentile(d_true, 25)), float(np.percentile(d_true, 75))]
    out["D_sec_noise_floor_abs_median"] = float(np.median(np.abs(d_hat - d_true)))
    return out


def validate_recon(ckpt, data, window):
    """Arm (b): the same readout applied to the model's reconstruction of real frames."""
    import torch
    from latent_noether.dreamer_adapter import DreamerV3Adapter
    d = np.load(data)
    fr_u8 = d["frames"][ANALYSIS][:, window]
    st = d["states"][ANALYSIS][:, window]
    norm = float(energy(d["states"][..., 0], d["states"][..., 1]).mean(-1).std())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = DreamerV3Adapter(device=dev).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    m.eval()
    fr = torch.as_tensor(d["frames"][ANALYSIS]).float().div_(255.).sub_(0.5).to(dev)
    with torch.no_grad():
        h = m.encode(fr)[:, window]
        rec = m.readout_from_h(h.reshape(-1, h.shape[-1])).reshape(*h.shape[:2], 64, 64, 3)
        rec = ((rec + 0.5) * 255.0).clamp(0, 255).cpu().numpy()
    got = decode_physics(rec)
    th_true, thd_true = st[..., 0], st[..., 1]
    off = np.round((got["theta"] - th_true).mean(-1, keepdims=True) / (2 * np.pi)) * 2 * np.pi
    out = {"ckpt": ckpt, "n_traj": int(rec.shape[0]), "n_steps": int(rec.shape[1])}
    out.update(_err_stats("theta_err_rad", got["theta"] - off - th_true))
    out.update(_err_stats("thetadot_err", got["thetadot"] - thd_true))
    out.update(_err_stats("E_err_over_std", (got["energy"] - energy(th_true, thd_true)) / norm))
    out["theta_err_deg_median"] = float(np.degrees(out["theta_err_rad_median"]))
    out["blank_decode_fraction"] = float(np.mean(~np.isfinite(got["theta"])))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/pendulum_pixels.npz")
    p.add_argument("--ckpts", nargs="*", default=[f"runs/dreamer_ref_s{s}.pt" for s in (3, 4, 5)])
    p.add_argument("--out", default="runs/e1_readout_validation.json")
    a = p.parse_args()
    window = slice(WARMUP, WARMUP + HORIZON)

    rec = {"window": [WARMUP, WARMUP + HORIZON], "real_frames": validate_real(a.data, window),
           "reconstructions": [], "skipped": []}
    for ck in a.ckpts:
        if pathlib.Path(ck).exists():
            rec["reconstructions"].append(validate_recon(ck, a.data, window))
        else:
            rec["skipped"].append(ck)
    pathlib.Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
