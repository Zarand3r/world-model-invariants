"""Both timing instruments must report the SAME sign for the same physical shift.

The Dreamer mechanism test reported a coefficient b ~ -1 stable across a 5.5x range of horizons,
against the GRU's +1. A coefficient near MINUS one rather than near zero is the signature of a
convention flip, not of an absent effect -- so it gets settled by calibration against a known
shift, never by flipping a sign until the result looks right.

The GRU instrument is  Delta_tau = <e, f> / |f|^2  with e = model - true, f = d(true)/dt.
If model(t) = true(t + delta) then e ~ delta * f, so it reports +delta: POSITIVE MEANS AHEAD.

The Dreamer instrument uses the oscillator phase, because |f|^2 varies 26,579x on this data.
This test pins its sign to the same meaning.
"""
import numpy as np

from scripts.run_dreamer_mechanism import DT, OMEGA0, f_true, phase, sim_step

DELTA_STEPS = 3          # the known shift; model runs AHEAD of truth by this many steps


def _reference_pair():
    """A true trajectory and a copy advanced by DELTA_STEPS of the simulator's own map."""
    th, thd = np.pi - 0.9, 0.0
    T = 80
    seq = []
    for _ in range(T + DELTA_STEPS):
        seq.append((th, thd))
        th, thd = sim_step(th, thd)
    seq = np.array(seq)
    true = seq[:T][None]                       # (1, T, 2)
    ahead = seq[DELTA_STEPS:][None]            # same trajectory, DELTA_STEPS further along
    return true, ahead


def test_projection_instrument_reports_ahead_as_positive():
    true, ahead = _reference_pair()
    e = ahead - true
    f = f_true(true[..., 0], true[..., 1])
    dtau = (e * f).sum(-1) / (f ** 2).sum(-1)
    assert np.median(dtau) > 0, "GRU instrument must report a leading model as POSITIVE"
    assert abs(np.median(dtau) - DELTA_STEPS * DT) < 0.3 * DELTA_STEPS * DT


def test_phase_instrument_agrees_in_sign_and_magnitude():
    """The phase must be NEGATED to mean the same thing.

    phi = atan2(thdot/omega, th - pi) runs BACKWARDS: for th - pi = A cos(wt) and
    thdot/omega = -A sin(wt), atan2 gives -wt. So (phi_model - phi_true)/omega is a LAG, and the
    timing offset with the project's established sign is its negative.
    """
    true, ahead = _reference_pair()
    ph_t = phase(true[..., 0], true[..., 1])
    ph_m = phase(ahead[..., 0], ahead[..., 1])
    raw = (ph_m - ph_t) / OMEGA0
    assert np.median(raw) < 0, "phi runs backwards in time; the raw difference is a LAG"
    corrected = -raw
    assert abs(np.median(corrected) - DELTA_STEPS * DT) < 0.3 * DELTA_STEPS * DT, (
        f"corrected offset {np.median(corrected):.4f} should be ~{DELTA_STEPS*DT:.4f} s")


def test_dreamer_encode_is_deterministic():
    """`encode` must return the same latent for the same frames, on every call.

    The reference `obs_step` SAMPLES the categorical posterior and `observe` does not expose the
    flag, so this silently drifted: repeated identical measurement runs gave Spearman
    0.522 / 0.524 / 0.531 with R^2 wandering 0.062 -> 0.022. A result that moves between identical
    runs cannot be reported, so the adapter takes the posterior mode (see `_deterministic_posterior`).
    """
    import numpy as np
    import torch

    from latent_noether.dreamer_adapter import DreamerV3Adapter

    ck = "runs/dreamer_ref_s3.pt"
    if not __import__("pathlib").Path(ck).exists():
        import pytest
        pytest.skip("trained Dreamer checkpoint not present")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load("runs/pendulum_pixels.npz")
    fr = torch.as_tensor(d["frames"][:4, :12]).float().div_(255.).sub_(0.5).to(dev)
    m = DreamerV3Adapter(device=dev).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)["model"])
    m.eval()
    with torch.no_grad():
        a, b = m.encode(fr), m.encode(fr)
    assert torch.equal(a, b), f"encode is non-deterministic: max diff {(a-b).abs().max():.3e}"
