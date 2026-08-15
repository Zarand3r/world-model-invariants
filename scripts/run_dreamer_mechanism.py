"""S3 TEST 1 — does the frequency-action mechanism hold on a REAL DreamerV3?

**Why this is the experiment that decides the paper.** S2 showed extraction recovers an
energy-correlated invariant from a reference DreamerV3 trained on gymnasium pixels. But the paper's
actual claim is the MECHANISM: that a world model's growing phase error is the accumulated
downstream consequence of its small, bounded invariant error,

    Delta_tau(t) = Delta_tau(0) + integral_0^t kappa(E_s) Delta_E(s) ds,   kappa = d ln(omega)/dE

Every mechanism result so far (D29, D32, D33) is GRU-only. If it does not reproduce here, we have a
recovery result on a real architecture and a mechanism that is toy-specific.

**The measurement problem, and the rule we hold ourselves to.** Dreamer emits PIXELS, not
coordinates, so the model's physical state must be read out of decoded frames. That needs a probe.
The probe is fitted on TRUE frames with known states and is used **only for measurement, never for
extraction** -- exactly the discipline stated in the JEPA roadmap. Two gates guard it, because a
bad instrument would manufacture or destroy the result:

  G1  held-out angle accuracy on TRUE frames: R^2 > 0.95 for both cos(th) and sin(th)
  G2  no distribution shift onto DECODED frames: at rollout step 0, where the model is known
      accurate, probe-on-decoded must agree with truth to < 0.15 rad
  G3  the instrument must RESOLVE the signal: running probe + differencing + energy on TRUE frames
      must give an energy error < half the model's median |dE|
  If any gate fails the test is VOID -- not negative. A failed instrument is not evidence.

**G3 exists because the first version of this test failed it silently.** Every model scored held-out
R^2 ~ 0, INCLUDING linear-in-t -- which is a statement about the measurement, not the mechanism. Two
defects, both now fixed and both previously recorded as lessons: a central difference for thdot
(biased by the simulator's semi-implicit integrator, putting the energy floor at 0.4115 against a
signal of 0.3612) and the along-flow projection <e,f>/|f|^2 (whose denominator ranges over 26,579x
here). See `velocity` and `phase`. The earlier KILL verdict was void and is not recorded as a result.

**REGISTERED PREDICTIONS (written before running):**
  P1  M3 (the integral predictor X) beats M2 (a fitted power law a + b t^p) on HELD-OUT
      trajectories. This is the pre-registration the review asked for: R^2_integral > R^2_{t^p}.
  P2  M3's coefficient b is near 1 -- the integral form has no free scale if kappa is right.
      Reported either way; b far from 1 with high R^2 means right shape, wrong magnitude.
  KILL: if M2 matches or beats M3, the mechanism does no explanatory work on this substrate and
  the paper's central claim stays GRU-only. That is a real outcome and gets reported as one.

Nothing here touches the extraction pipeline. kappa comes from the SIMULATOR (an independent
quantity), Delta_E and Delta_tau come from the MODEL's own rollout against the simulator.
"""
import argparse
import json

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter

DT = 0.05           # gymnasium Pendulum-v1
G, M, L = 10.0, 1.0, 1.0
WARMUP = 10         # encoder context before the autonomous rollout begins
HORIZON = 40        # overridden by --horizon; see the horizon sweep for why this default is weak
OMEGA0 = np.sqrt(3 * 10.0 / (2 * 1.0))   # small-oscillation frequency, for the phase coordinate


def velocity(th):
    """thdot from an angle sequence, using the simulator's OWN discrete map. Drops the first step.

    Gymnasium integrates semi-implicitly: th_{k+1} = th_k + thdot_{k+1} dt. So the BACKWARD
    difference is EXACT -- (th_k - th_{k-1})/dt IS thdot_k -- while a central difference is biased
    by (3g/2l) sin(th) dt / 2 ~ 0.375 rad/s. Measured (run_dreamer_mechanism_diag): the central
    difference put the instrument's ENERGY noise floor at 0.4115, LARGER than the model's own
    median |dE| of 0.3612. The first version of this test was measuring its own instrument.

    Returns thdot for indices 1..T-1 only. There is no honest velocity at index 0, and faking one
    by copying index 1 left a single sample wrong by ~0.75 rad/s, which alone held the floor at
    0.1665. Callers roll out one extra step and slice `[1:]` so nothing is imputed.
    """
    return np.diff(th, axis=1) / DT


def phase(th, thd):
    """Unwrapped oscillator phase about the hanging point.

    Replaces the along-flow projection <e,f>/|f|^2 used for the GRU. That projection is valid only
    where |f|^2 is roughly constant; here it ranges over 26,579x within the data (thdot ~ 0 at the
    turning points, sin th ~ 0 at the bottom), so it is dominated by its denominator -- final-time
    spread 2.93 against 0.045 for this phase measure. Same quantity, conditioned instrument.
    """
    return np.unwrap(np.arctan2(thd / OMEGA0, th - np.pi), axis=1)


def true_energy(th, thd):
    return 0.5 * (M * L ** 2 / 3) * thd ** 2 + M * G * (L / 2) * np.cos(th)


def f_true(th, thd):
    """Gymnasium's continuous vector field: (d th/dt, d thd/dt)."""
    return np.stack([thd, (3 * G / (2 * L)) * np.sin(th)], -1)


def sim_step(th, thd, n=1):
    """Gymnasium's own semi-implicit Euler map, reimplemented for cheap period measurement."""
    for _ in range(n):
        thd = thd + (3 * G / (2 * L)) * np.sin(th) * DT
        th = th + thd * DT
    return th, thd


SEPARATRIX = 5.0          # M g (l/2): at rest, upright. Above it the pendulum ROTATES.


def _period(th0, thd0, max_steps=20000):
    """Period of the librating orbit through (th0, thd0), timed on the SIMULATOR's own map.

    Sub-step precision matters here. The period is ~38 steps, so integer-step timing quantises
    omega at ~2.6% -- useless for a derivative. The thdot sign change is therefore located by
    linear interpolation between the bracketing steps.
    """
    th, thd = th0, thd0
    prev_thd, cross = thd, []
    for k in range(1, max_steps):
        th, thd = sim_step(th, thd)
        if prev_thd != 0 and np.sign(thd) != np.sign(prev_thd):
            cross.append(k - 1 + prev_thd / (prev_thd - thd))       # sub-step zero crossing
            if len(cross) == 2:
                return 2 * (cross[1] - cross[0]) * DT
        prev_thd = thd
    return np.nan


def kappa_of_E(n_probe=60, e_lo=None, e_hi=None):
    """kappa = d ln(omega)/dE for LIBRATING pendulum orbits, measured on the simulator.

    **This function previously returned a near-constant -0.0344 and that was an artefact that
    corrupted an entire analysis.** It probed amplitudes 0.25-1.6 rad, covering E in [-4.84, -0.15],
    then fitted a quadratic in E -- but the energies the model actually visits are [-1.01, +5.95].
    Every kappa in use came from EXTRAPOLATING that quadratic far outside its support, which
    flattened it to a constant. The conclusion drawn from it ("the pendulum cannot identify the
    frequency-action mechanism, so train a new world model on a new environment") was wrong and
    expensive.

    Measured properly, over the visited range, omega varies 2.38x and kappa spans -0.34 .. -0.03
    among librating orbits alone -- a 10x variation, which is ample to identify the mechanism.

    The grid is built from the ENERGIES REQUESTED by the caller, and `kappa` raises rather than
    extrapolates outside it. Refusing is the whole point: silent extrapolation is what caused the
    original error.
    """
    lo = -SEPARATRIX + 1e-3 if e_lo is None else max(e_lo, -SEPARATRIX + 1e-3)
    hi = SEPARATRIX - 0.05 if e_hi is None else min(e_hi, SEPARATRIX - 0.05)
    Es = np.linspace(lo, hi, n_probe)
    ws = []
    for E in Es:
        kin = E + SEPARATRIX                       # KE at the hanging point, th = pi
        th0, thd0 = (np.pi, float(np.sqrt(6 * kin))) if kin > 0 else (float(np.arccos(E / 5)), 0.0)
        T = _period(th0, thd0)
        ws.append(2 * np.pi / T if np.isfinite(T) and T > 0 else np.nan)
    ws = np.array(ws)
    ok = np.isfinite(ws)
    Es, lw = Es[ok], np.log(ws[ok])
    # smooth ln(omega)(E) with a local quadratic, then differentiate -- the period timing is exact
    # to sub-step precision but still discrete, and kappa is a derivative.
    k = np.gradient(np.convolve(lw, np.ones(5) / 5, mode="same"), Es)
    k[:2], k[-2:] = k[2], k[-3]                    # the convolution edge is not trustworthy

    def kappa(E):
        E = np.asarray(E, dtype=np.float64)
        if np.any(E < Es[0] - 1e-6) or np.any(E > Es[-1] + 1e-6):
            raise ValueError(
                f"kappa requested outside its measured support [{Es[0]:.3f}, {Es[-1]:.3f}]: "
                f"got [{np.min(E):.3f}, {np.max(E):.3f}]. Extrapolating here is exactly the bug "
                f"that produced a spurious constant kappa; widen the probe range instead.")
        return np.interp(E, Es, k)

    kappa.support = (float(Es[0]), float(Es[-1]))    # callers filter trajectories to this range
    return kappa, float(np.median(k))


# ---------------------------------------------------------------------------------------
# the measurement instrument
# ---------------------------------------------------------------------------------------

def fit_angle_probe(frames, states, n_fit):
    """Ridge from a frame to (cos th, sin th). MEASUREMENT ONLY -- never used for extraction."""
    X = frames[:n_fit].reshape(-1, np.prod(frames.shape[2:])).astype(np.float64) / 255.0 - 0.5
    th = states[:n_fit, :, 0].reshape(-1)
    Y = np.stack([np.cos(th), np.sin(th)], -1)
    mu = X.mean(0)
    A = X - mu
    W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ (Y - Y.mean(0)))
    return {"mu": mu, "W": W, "b": Y.mean(0)}


def apply_probe(p, frames_flat):
    return (frames_flat - p["mu"]) @ p["W"] + p["b"]


def probe_r2(p, frames, states):
    X = frames.reshape(-1, np.prod(frames.shape[2:])).astype(np.float64) / 255.0 - 0.5
    th = states[:, :, 0].reshape(-1)
    Y = np.stack([np.cos(th), np.sin(th)], -1)
    P = apply_probe(p, X)
    return [float(1 - ((Y[:, j] - P[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum())
            for j in range(2)]


# ---------------------------------------------------------------------------------------
# held-out model comparison
# ---------------------------------------------------------------------------------------

def spearman(a, b):
    """Rank correlation. Implemented here because scipy is not a dependency of this repo."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / max(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()), 1e-30))


def heldout_r2(design, y, tr, te):
    """Fit a + b*design on `tr` trajectories, score R^2 on `te`. Returns (R2, coefficients)."""
    A = np.concatenate([np.ones((design[tr].reshape(-1, design.shape[-1]).shape[0], 1)),
                        design[tr].reshape(-1, design.shape[-1])], 1)
    yt = y[tr].reshape(-1)
    beta = np.linalg.lstsq(A, yt, rcond=None)[0]
    Ae = np.concatenate([np.ones((design[te].reshape(-1, design.shape[-1]).shape[0], 1)),
                         design[te].reshape(-1, design.shape[-1])], 1)
    ye = y[te].reshape(-1)
    pred = Ae @ beta
    return float(1 - ((ye - pred) ** 2).sum() / max(((ye - ye.mean()) ** 2).sum(), 1e-30)), beta


def fit_power_law(t, y, tr, te):
    """M2: a + b*t^p with p fitted by scan. The phenomenological competitor P1 must beat."""
    best = (-np.inf, None, None)
    for p in np.linspace(0.3, 3.0, 55):
        d = (t ** p)[None, :, None].repeat(y.shape[0], 0)
        r2, beta = heldout_r2(d, y, tr, te)
        if r2 > best[0]:
            best = (r2, float(p), beta)
    return best


def run(ckpt, data, seed_tag, HORIZON=HORIZON, return_arrays=False, probe_data=None):
    """`probe_data` calibrates the measurement probe on a DIFFERENT dataset from the one analysed.

    With it, every trajectory in `data` is available for the across-trajectory statistic instead of
    the 20% left over after fitting the probe. That statistic has ONE SAMPLE PER TRAJECTORY, so n is
    the number of trajectories and nothing else -- it was the binding constraint at n = 51 (D38).
    """
    d = np.load(data)
    frames_u8, states = d["frames"], d["states"]
    if probe_data is None:
        n_val = frames_u8.shape[0] // 5
        probe = fit_angle_probe(frames_u8[:-n_val], states[:-n_val],
                                n_fit=min(80, frames_u8.shape[0] - n_val))
        fr_va, st_va = frames_u8[-n_val:], states[-n_val:]
    else:
        pd_ = np.load(probe_data)
        probe = fit_angle_probe(pd_["frames"], pd_["states"], n_fit=min(80, pd_["frames"].shape[0]))
        n_val = frames_u8.shape[0]                      # the whole disjoint set is analysable
        fr_va, st_va = frames_u8, states
    g1 = probe_r2(probe, fr_va, st_va)
    print(f"    G1 probe on TRUE held-out frames: cos R2 {g1[0]:.4f}  sin R2 {g1[1]:.4f}", flush=True)

    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
    m.eval()

    fr = torch.as_tensor(fr_va[:, :WARMUP]).float().div_(255.).sub_(0.5).cuda()
    with torch.no_grad():
        h = m.encode(fr)[:, -1]
        dec = []
        for _ in range(HORIZON + 1):        # +1: the extra step makes velocity[0] a real difference
            dec.append(m.readout_from_h(h).cpu())
            h = m.transition(h)
        dec = torch.stack(dec, 1).numpy()                       # (B, H, 64, 64, 3) in [-0.5, 0.5]

    flat = dec.reshape(-1, np.prod(dec.shape[2:])).astype(np.float64)
    P = apply_probe(probe, flat).reshape(dec.shape[0], HORIZON + 1, 2)
    th_m = np.unwrap(np.arctan2(P[..., 1], P[..., 0]), axis=1)

    # truth on the SAME clock. encode(obs[:, :W]) leaves state W-1, which predicts frame W-1.
    # Index 0 of the rollout is dropped everywhere (see `velocity`), so truth starts at W.
    th_t = st_va[:, WARMUP: WARMUP + HORIZON, 0]
    thd_t = st_va[:, WARMUP: WARMUP + HORIZON, 1]

    g2 = float(np.abs(np.angle(np.exp(1j * (th_m[:, 1] - th_t[:, 0])))).mean())
    print(f"    G2 probe on DECODED frames at t=0: mean |angle error| {g2:.4f} rad", flush=True)

    # G3 -- ADDED after the first run scored R^2 ~ 0 for EVERY model including linear-in-t. Running
    # the whole instrument (probe + differencing + energy) on TRUE frames gives its noise floor. If
    # that floor is not small against the model error the test measures itself, and is VOID.
    tf = fr_va[:, WARMUP - 1: WARMUP + HORIZON].astype(np.float64)
    Pi = apply_probe(probe, tf.reshape(-1, np.prod(tf.shape[2:])) / 255.0 - 0.5)
    Pi = Pi.reshape(-1, HORIZON + 1, 2)
    th_i = np.unwrap(np.arctan2(Pi[..., 1], Pi[..., 0]), axis=1)
    th_i = th_i - 2 * np.pi * np.round((th_i[:, 1:2] - th_t[:, :1]) / (2 * np.pi))
    floor_E = float(np.sqrt(((true_energy(th_i[:, 1:], velocity(th_i))
                              - true_energy(th_t, thd_t)) ** 2).mean()))
    print(f"    G3 instrument energy noise floor {floor_E:.4f}", flush=True)
    if min(g1) < 0.95 or g2 > 0.15:
        return {"ckpt": ckpt, "VOID": True, "g1": g1, "g2": g2, "floor_E": floor_E}

    # the probe reads angle only; align its unwrapping branch to truth, which fixes a global 2*pi
    th_m = th_m - 2 * np.pi * np.round((th_m[:, 1:2] - th_t[:, :1]) / (2 * np.pi))
    thd_m, th_m = velocity(th_m), th_m[:, 1:]

    dE = true_energy(th_m, thd_m) - true_energy(th_t, thd_t)
    # NEGATED, and the negation is not a fudge: phi = atan2(thdot/omega, th - pi) runs BACKWARDS
    # in time (for th - pi = A cos wt, atan2 returns -wt), so the raw difference is a LAG while the
    # project's established instrument <e,f>/|f|^2 reports a LEAD. Calibrated against a known
    # 3-step shift in tests/test_timing_convention.py, which pins both to "positive means ahead".
    # Before this fix the coefficient came out b ~ -1 stable across a 5.5x horizon range -- near
    # MINUS one rather than near zero, which is the signature of a flipped convention, not absence.
    dtau = -(phase(th_m, thd_m) - phase(th_t, thd_t)) / OMEGA0     # timing offset, seconds

    # RESTRICT TO LIBRATING ORBITS, and say why. 17.4% of these trajectories ROTATE (E > 5, the
    # separatrix) and 30% pass within 0.5 of it. Two things break there and neither is repairable
    # by averaging: the phase coordinate atan2(thdot/omega, th - pi) assumes oscillation about the
    # hanging point, and kappa is a different function on rotating orbits. Both regimes are
    # physical; mixing them in one scalar law is not. The cut is on the TRUE energy, fixed before
    # any Delta_tau is examined, and the retained fraction is reported.
    # The kappa table is built FIRST, and trajectories are filtered to the range where its period
    # timing actually succeeded. Building it from the data's own range instead fails at the edge:
    # period measurement breaks down within ~0.1 of the separatrix, so the table silently ends
    # short of what was asked for -- and `kappa` then (correctly) refuses.
    kfun, kbar = kappa_of_E()
    k_lo, k_hi = kfun.support
    Et = true_energy(th_t, thd_t)
    lib = (Et.max(1) <= k_hi) & (Et.min(1) >= k_lo)
    kept = float(lib.mean())
    th_m, thd_m, th_t, thd_t = th_m[lib], thd_m[lib], th_t[lib], thd_t[lib]
    dE, dtau, Et = dE[lib], dtau[lib], Et[lib]
    integrand = kfun(Et) * dE
    X = np.cumsum(integrand, axis=1) * DT
    t = np.arange(HORIZON) * DT

    # IDENTIFIABILITY GATE. If X is nearly collinear with the unweighted integral, the frequency
    # weighting cannot be distinguished from it and the experiment cannot test the mechanism --
    # only the weaker "invariant error predicts drift". Registered threshold |rho| < 0.9.
    unw = np.cumsum(dE, axis=1) * DT
    collinearity = abs(float(np.corrcoef(X[:, -1], unw[:, -1])[0, 1]))

    n = dtau.shape[0]
    tr, te = np.arange(n // 2), np.arange(n // 2, n)

    # ACROSS-TRAJECTORY test at the final time. Added after the pooled comparison came out near
    # zero for EVERY model. It is post hoc and labelled so -- but it is the better-posed
    # discriminator, for a structural reason: at a fixed final time T, `t` and `t^p` are CONSTANT
    # across trajectories and can only ever explain within-trajectory time trend. X(T) varies per
    # trajectory because each has its own accumulated bias. So "does X predict who drifts most?"
    # is a question no function of t alone can answer, which is exactly the mechanism-vs-curve-fit
    # distinction P1 was trying to draw. The pooled numbers are still reported, unchanged.
    if return_arrays:
        return (X, dtau, dE, th_m - th_t, Et) if return_arrays == "ext" else (X, dtau, dE)
    xT, yT = X[:, -1], dtau[:, -1]
    ax = np.stack([np.ones(len(tr)), xT[tr]], 1)
    bx = np.linalg.lstsq(ax, yT[tr], rcond=None)[0]
    pr = np.stack([np.ones(len(te)), xT[te]], 1) @ bx
    r2_cross = float(1 - ((yT[te] - pr) ** 2).sum() /
                     max(((yT[te] - yT[te].mean()) ** 2).sum(), 1e-30))
    r2_lin, _ = heldout_r2((t[None, :, None]).repeat(n, 0), dtau, tr, te)
    r2_pow, p_hat, _ = fit_power_law(t, dtau, tr, te)
    r2_int, beta_int = heldout_r2(X[..., None], dtau, tr, te)

    if floor_E > 0.5 * float(np.median(np.abs(dE))):
        print("    VOID (G3): instrument noise is not small against the signal.", flush=True)
        return {"ckpt": ckpt, "VOID": True, "g1": g1, "g2": g2, "floor_E": floor_E,
                "dE_median_abs": float(np.median(np.abs(dE)))}
    return {"ckpt": ckpt, "VOID": False, "g1": g1, "g2": g2, "floor_E": floor_E,
            "kappa_mean": kbar, "librating_fraction": kept, "collinearity": collinearity,
            "n_traj": int(dtau.shape[0]),
            "r2_linear": r2_lin, "r2_power": r2_pow, "power_p": p_hat,
            "r2_integral": r2_int, "b": float(beta_int[1]),
            "dE_median_abs": float(np.median(np.abs(dE))),
            "spearman_XT": spearman(X[:, -1], dtau[:, -1]),
            "r2_cross": r2_cross, "b_cross": float(bx[1]),
            "dtau_final_median": float(np.median(dtau[:, -1]))}


def mechanism_arrays_ext(ckpt, data, horizon, probe_data=None):
    """(X, dtau, dE, dtheta, E_true) — the extra series the control arms need."""
    return run(ckpt, data, ckpt, HORIZON=horizon, return_arrays="ext", probe_data=probe_data)


def mechanism_arrays(ckpt, data, horizon, probe_data=None):
    """(X, dtau, dE) for one checkpoint — the raw series behind `run`, for downstream analyses."""
    return run(ckpt, data, ckpt, HORIZON=horizon, return_arrays=True, probe_data=probe_data)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--ckpts", nargs="+", default=[f"runs/dreamer_ref_s{i}.pt" for i in range(3)])
    a.add_argument("--data", default="runs/pendulum_pixels.npz")
    a.add_argument("--horizon", type=int, default=HORIZON)
    a.add_argument("--probe-data", default=None,
                   help="calibrate the probe on THIS dataset so every trajectory in "
                        "--data is analysable (the across-trajectory statistic has "
                        "one sample per trajectory, so n is the binding constraint)")
    a.add_argument("--out", default="runs/dreamer_mechanism.json")
    a = a.parse_args()
    print("S3 TEST 1: does the frequency-action mechanism hold on a real DreamerV3?")
    print("REGISTERED  P1: held-out R2(integral) > R2(fitted t^p).   P2: coefficient b near 1.")
    print("GATES  G1 probe cos/sin R2 > 0.95 on true frames;  G2 decoded-frame agreement < 0.15 rad")
    print("VOID if a gate fails -- a failed instrument is not evidence.\n")
    rows = []
    for ck in a.ckpts:
        print(f"  {ck}", flush=True)
        r = run(ck, a.data, ck, HORIZON=a.horizon, probe_data=a.probe_data)
        rows.append(r)
        json.dump(rows, open(a.out, "w"), indent=2)
        if r["VOID"]:
            print("    VOID — gate failed.\n", flush=True)
            continue
        print(f"    kappa(E) median {r['kappa_mean']:+.4f}   median |dE| {r['dE_median_abs']:.4f}"
              f"   librating {r['librating_fraction']:.0%} (n={r['n_traj']})")
        print(f"    IDENTIFIABILITY |rho(X, int dE)| = {r['collinearity']:.4f}  "
              f"({'OK — the frequency weighting is separable' if r['collinearity'] < 0.9 else
                 'COLLINEAR — cannot test the mechanism'})")
        print(f"    pooled held-out R2:  linear t {r['r2_linear']:+.3f}   "
              f"t^p (p={r['power_p']:.2f}) {r['r2_power']:+.3f}   "
              f"INTEGRAL {r['r2_integral']:+.3f}   b = {r['b']:+.3f}")
        print(f"    ACROSS-TRAJECTORY at final time: R2 {r['r2_cross']:+.3f}  b {r['b_cross']:+.3f}"
              f"  spearman {r['spearman_XT']:+.3f}   (t and t^p score 0 here by construction)\n",
              flush=True)

    ok = [r for r in rows if not r["VOID"]]
    if not ok:
        print("--- ALL VOID. No conclusion about the mechanism on this substrate.")
        raise SystemExit(0)
    med = lambda k: float(np.median([r[k] for r in ok]))
    ri, rp = med("r2_integral"), med("r2_power")
    print(f"--- medians over {len(ok)} seeds:  R2 integral {ri:+.3f}   R2 t^p {rp:+.3f}   "
          f"b {med('b'):+.3f}")
    print("\n--- VERDICT")
    if ri < 0.2:
        print(f"  INCONCLUSIVE / UNDERPOWERED: R2_integral = {ri:+.3f}. Every model, including")
        print(f"  linear-in-t, is near zero, so the ordering {ri:+.3f} > {rp:+.3f} compares noise")
        print("  to noise and is NOT a confirmation. Neither P1 nor its negation is established.")
    elif ri > rp:
        print(f"  P1 CONFIRMED: the integral predictor beats the fitted power law "
              f"({ri:+.3f} vs {rp:+.3f}) on held-out")
        print("  trajectories. The frequency-action mechanism reproduces on a real DreamerV3, so")
        print("  the paper's central claim is about LEARNED WORLD MODELS, not about our GRU.")
        b = med("b")
        print(f"  P2: b = {b:+.3f} " + ("— near 1 as predicted." if 0.4 < b < 2.5 else
              "— NOT near 1. Right shape, wrong magnitude; report as such."))
    else:
        print(f"  P1 FAILED / KILL: a fitted t^p matches or beats the integral predictor "
              f"({rp:+.3f} vs {ri:+.3f}).")
        print("  The mechanism does no explanatory work on this substrate. It stays GRU-only.")
