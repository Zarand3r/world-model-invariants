"""Does the frequency weighting add anything BEYOND unweighted accumulation? A nested test.

D41 concluded that kappa does no work because the unweighted control scored R^2 0.289 against the
weighted predictor's 0.270, at one horizon. **That conclusion was stated too strongly and this
script is the retest.** Three problems with it:

  1. A gap of 0.019 on ~256 held-out points, median of 3 seeds, was never checked against its own
     sampling error. "No detectable difference" is not "the weighting does nothing".
  2. Comparing two STANDALONE R^2 values is a weak instrument when the predictors correlate at
     0.78-0.87 -- they share ~65% of their variance, so both will score similarly almost regardless
     of which one carries the signal. The question is nested, so the test should be:
         does X = integral kappa dE ds explain variance that integral dE ds does not?
     and symmetrically, does the unweighted one add anything over X?
  3. Controls ran only at H=80, where the mechanism's Spearman is 0.314. It reaches 0.512 at
     H=160. Testing where the signal is weakest is not a fair test of it.

METHOD. Per seed and horizon, held out by trajectory:
    R2_unw     y ~ integral dE
    R2_wgt     y ~ X
    R2_both    y ~ both
    delta_wgt  = R2_both - R2_unw    the weighting's UNIQUE contribution
    delta_unw  = R2_both - R2_wgt    the unweighted term's unique contribution
with a bootstrap over trajectories for a 95% interval on each increment.

READING IT. delta_wgt > 0 with an interval excluding 0 means the frequency weighting carries
information the unweighted integral does not -- the mechanism claim, properly tested. If both
increments straddle 0, the two are not separable at this power, and THAT is the honest result:
neither confirmation nor refutation.
"""
import json

import numpy as np

import scripts.run_dreamer_mechanism as MECH

CKPTS = [f"runs/dreamer_ref_s{i}.pt" for i in range(3)]
DATA, PROBE = "runs/pendulum_pixels_eval.npz", "runs/pendulum_pixels.npz"
HORIZONS = [80, 120, 160]
N_BOOT = 400


def r2(cols, y, tr, te):
    A = np.column_stack([np.ones(len(tr))] + [c[tr] for c in cols])
    b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    p = np.column_stack([np.ones(len(te))] + [c[te] for c in cols]) @ b
    return float(1 - ((y[te] - p) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-30))


def increments(unw, wgt, y, rng):
    n = len(y)
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    base = {"unw": r2([unw], y, tr, te), "wgt": r2([wgt], y, tr, te),
            "both": r2([unw, wgt], y, tr, te)}
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        u, w, yy = unw[idx], wgt[idx], y[idx]
        boot.append((r2([u, w], yy, tr, te) - r2([u], yy, tr, te),
                     r2([u, w], yy, tr, te) - r2([w], yy, tr, te)))
    boot = np.array(boot)
    return base, np.percentile(boot[:, 0], [2.5, 97.5]), np.percentile(boot[:, 1], [2.5, 97.5])


print(__doc__.split("\n\n")[0])
print(f"\nn = 512 disjoint eval trajectories, librating only;  {N_BOOT} bootstrap resamples\n")
print(f"  {'H':>4}{'seed':>6}{'R2_unw':>9}{'R2_wgt':>9}{'R2_both':>10}"
      f"{'d_wgt [95% CI]':>24}{'d_unw [95% CI]':>24}")
rows = []
for H in HORIZONS:
    for i, ck in enumerate(CKPTS):
        X, dtau, dE, _, Et = MECH.mechanism_arrays_ext(ck, DATA, H, probe_data=PROBE)
        y = dtau[:, -1]
        unw = (np.cumsum(dE, 1) * MECH.DT)[:, -1]
        base, ci_w, ci_u = increments(unw, X[:, -1], y, np.random.default_rng(i))
        rows.append({"horizon": H, "seed": i, **base,
                     "d_wgt": base["both"] - base["unw"], "ci_wgt": ci_w.tolist(),
                     "d_unw": base["both"] - base["wgt"], "ci_unw": ci_u.tolist()})
        print(f"  {H:>4}{i:>6}{base['unw']:>9.3f}{base['wgt']:>9.3f}{base['both']:>10.3f}"
              f"{rows[-1]['d_wgt']:>10.3f} [{ci_w[0]:+.3f},{ci_w[1]:+.3f}]"
              f"{rows[-1]['d_unw']:>10.3f} [{ci_u[0]:+.3f},{ci_u[1]:+.3f}]", flush=True)
        json.dump(rows, open("runs/dreamer_nested_kappa.json", "w"), indent=2)

print("\n--- VERDICT")
w_pos = sum(1 for r in rows if r["ci_wgt"][0] > 0)
u_pos = sum(1 for r in rows if r["ci_unw"][0] > 0)
print(f"  the WEIGHTED term adds significantly (CI excludes 0) in {w_pos}/{len(rows)} seed-horizons")
print(f"  the UNWEIGHTED term adds significantly                 in {u_pos}/{len(rows)}")
if w_pos > u_pos and w_pos >= len(rows) // 2:
    print("\n  D41 WAS WRONG. The frequency weighting carries information the unweighted integral")
    print("  does not, and more often than the reverse. The mechanism claim survives on Dreamer.")
elif u_pos > w_pos and u_pos >= len(rows) // 2:
    print("\n  D41 STANDS, and now on the right test: the unweighted integral carries information")
    print("  the weighted one does not, not the other way round. kappa is not the conversion factor.")
else:
    print("\n  NEITHER SEPARABLE. Both increments straddle zero, so the two predictors cannot be")
    print("  told apart at this power. D41's negative verdict is NOT supported -- but neither is")
    print("  the mechanism. The honest statement is that this substrate cannot decide it, and")
    print("  D41 must be softened from a refutation to an inconclusive.")
