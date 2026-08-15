"""The equation of motion in gauge-invariant form: the frequency–action relation.

Asking "did the model encode `dω/dt = −4 sin θ`?" is asking for a **gauge-dependent**
quantity — the coefficient only means something relative to a coordinate system, which is
why a ground-truth-fitted alignment could manufacture it from an untrained network
(docs/THEORY.md §5).

By the Arnold–Liouville theorem, the coordinate-free content of an integrable system is
the foliation of phase space by invariant level sets together with the **frequency on each
leaf**. Action–angle variables are a canonical gauge fixed by the dynamics alone. So the
invariant question is:

    does the model's OWN period vary with the model's OWN conserved quantity
    the way the true system's does?

Harmonic: period constant (isochronous). Pendulum: period grows with amplitude along an
elliptic curve. Anharmonic (`1 + εr²`): period shrinks as amplitude grows. That curve is
the equation of motion, stated invariantly.

Both ingredients come from the model — period from its own rollouts, invariant from our own
search. Ground truth enters only at the final comparison, never inside the pipeline.
"""
import torch


def dominant_period(traj: torch.Tensor, dt: float, min_periods: float = 1.5) -> torch.Tensor:
    """Period of each trajectory, from the dominant peak of its power spectrum.

    traj: (n_traj, T, d). Returns (n_traj,) periods in time units; entries are NaN where
    the record is too short to resolve the peak (fewer than `min_periods` cycles), which
    must be dropped rather than trusted.
    """
    if traj.ndim != 3:
        raise ValueError(f"traj must be (n_traj, T, d), got {tuple(traj.shape)}")
    x = traj - traj.mean(dim=1, keepdim=True)
    # sum power over coordinates: a period is a property of the orbit, not of one axis
    power = (torch.fft.rfft(x, dim=1).abs() ** 2).sum(-1)          # (n_traj, F)
    power[:, 0] = 0.0                                              # ignore DC
    idx = power.argmax(dim=1)
    freqs = torch.fft.rfftfreq(traj.shape[1], d=dt)
    f = freqs[idx]
    period = torch.where(f > 0, 1.0 / f.clamp_min(1e-12), torch.full_like(f, float("nan")))
    too_short = period * min_periods > traj.shape[1] * dt
    return torch.where(too_short, torch.full_like(period, float("nan")), period)


def frequency_action_curve(traj: torch.Tensor, invariant_values: torch.Tensor,
                           dt: float) -> dict:
    """Pair each trajectory's period with its conserved-quantity value.

    invariant_values: (n_traj,) one value per trajectory (the invariant is constant along
    it, so any summary works; use the mean).
    Returns the finite pairs plus the monotone trend — the sign and strength of dT/dC is
    the qualitative signature that distinguishes the systems.
    """
    period = dominant_period(traj, dt)
    ok = torch.isfinite(period) & torch.isfinite(invariant_values)
    if ok.sum() < 4:
        raise ValueError("fewer than 4 usable trajectories; cannot fit a trend")
    p, c = period[ok], invariant_values[ok]
    rp = p.argsort().argsort().to(p.dtype)
    rc = c.argsort().argsort().to(c.dtype)
    rp, rc = rp - rp.mean(), rc - rc.mean()
    spearman = float((rp * rc).sum() / (rp.norm() * rc.norm() + 1e-12))
    return {"period": p, "invariant": c, "n_used": int(ok.sum()),
            "spearman": spearman,
            "period_spread": float((p.max() - p.min()) / p.median()),
            "median_period": float(p.median())}


def curve_agreement(model_curve: dict, true_curve: dict, n_bins: int = 6) -> dict:
    """Do the model's and the true system's frequency–action curves agree in SHAPE?

    Both curves are rank-normalized before comparison, because the invariant is only
    defined up to monotone reparameterization (identifiability, proposal §8) and the
    action variable's scale is not observable. What is invariant is the ordering and the
    relative spread — i.e. the shape of T as a function of which leaf you are on.
    """
    out = {}
    for name, curve in (("model", model_curve), ("true", true_curve)):
        c, p = curve["invariant"], curve["period"]
        # An empty bin makes `median()` return NaN, and the returned disagreement is then NaN
        # rather than an error -- a silently meaningless number. `frequency_action_curve` admits
        # as few as 4 trajectories, so the default n_bins=6 reaches this.
        if len(p) < n_bins:
            raise ValueError(f"{name} curve has {len(p)} trajectories but n_bins={n_bins}; "
                             f"empty bins would make the comparison NaN")
        order = c.argsort()
        p_sorted = p[order]
        edges = torch.linspace(0, len(p_sorted), n_bins + 1).long()
        binned = torch.stack([p_sorted[edges[i]:edges[i + 1]].median()
                              for i in range(n_bins)])
        out[name] = binned / binned.median()          # scale-free profile
    a, b = out["model"], out["true"]
    rel_err = float(((a - b).abs() / b.abs().clamp_min(1e-9)).mean())
    return {"model_profile": a.tolist(), "true_profile": b.tolist(),
            "mean_relative_error": rel_err,
            "trend_sign_matches": bool(torch.sign(a[-1] - a[0]) == torch.sign(b[-1] - b[0]))}
