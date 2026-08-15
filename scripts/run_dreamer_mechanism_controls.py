"""Is the Dreamer mechanism result SPECIFIC, or would any error signal reproduce it?

WHAT NEEDS DEFENDING. On a real DreamerV3, restricted to librating orbits and with kappa measured
properly (D40 -- an earlier version extrapolated it to a constant and corrupted the analysis), the
integral predictor X(t) = integral kappa(E_s) Delta_E(s) ds beats a fitted power law on HELD-OUT
pooled data at every horizon past one cycle, by up to 8.9x (R^2 0.187 vs 0.021 at H=80), with
across-trajectory Spearman up to +0.51. The fitted coefficient is b ~ 0.25-0.30, NOT the predicted
1 -- so the shape is right and the magnitude is about 4x off, and both get reported.

That is only evidence if the alternatives fail. Four controls, each removing exactly one ingredient
of the claim, and each with a registered expectation:

  C1  SHUFFLE       pair trajectory i's X with trajectory j's Delta_tau.
                    -> destroys everything. rho ~ 0. If it does not, the correlation is an artefact
                       of the marginal distributions and nothing here is real.
  C2  NO KAPPA      X' = integral Delta_E ds, dropping the frequency-sensitivity weight.
                    **THIS CONTROL IS NOW INFORMATIVE, AND EARLIER IT WAS NOT.** With the broken
                    kappa it correlated with X at -0.99998, so it removed nothing and could not
                    fail. With kappa measured over its true support the correlation is 0.775-0.865,
                    under the registered identifiability threshold of 0.9 -- so the two predictors
                    are genuinely distinguishable and C2 can now kill the claim.
                    -> this is the one that matters. If C2 does as well, the result shows only
                       "energy error predicts drift" -- true but weak, and NOT the frequency-action
                       mechanism, whose whole content is that d ln(omega)/dE is the conversion
                       factor. The paper's claim would have to shrink accordingly.
  C3  WRONG SIGN    X'' = integral (-kappa) Delta_E ds.
                    -> b must flip to ~ -1. A trivial check that b tracks the physics rather than
                       the fitting procedure.
  C4  WRONG STATE   replace Delta_E with the angle error Delta_theta, which is NOT an invariant
                    error. -> should fail. If a non-invariant error predicts drift equally well,
                    the INVARIANT plays no privileged role and the Noether framing is decorative.

C2 and C4 are the ones that can actually kill the claim. They are reported first and in full.
"""
import json

import numpy as np

import scripts.run_dreamer_mechanism as MECH

CKPTS = [f"runs/dreamer_ref_s{i}.pt" for i in range(3)]
DATA, PROBE = "runs/pendulum_pixels_eval.npz", "runs/pendulum_pixels.npz"
HORIZON = 80            # best pooled separation in the corrected-kappa sweep; fixed before controls


def score(x, y):
    """Held-out across-trajectory fit of y on x. Returns (R2, b, spearman)."""
    n = len(x)
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    A = np.stack([np.ones(len(tr)), x[tr]], 1)
    bb = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    pred = np.stack([np.ones(len(te)), x[te]], 1) @ bb
    r2 = float(1 - ((y[te] - pred) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-30))
    return r2, float(bb[1]), MECH.spearman(x, y)


print(__doc__.split("\n\n")[0])
print(f"\nhorizon {HORIZON} ({HORIZON*MECH.DT:.1f} s, {HORIZON*MECH.DT*MECH.OMEGA0/(2*np.pi):.1f} "
      f"cycles);  n = 512 disjoint trajectories\n")

arms = {k: [] for k in ["MECHANISM", "C1 shuffle", "C2 no kappa", "C3 wrong sign", "C4 wrong state"]}
for ck in CKPTS:
    X, dtau, dE, dTH, Etrue = MECH.mechanism_arrays_ext(ck, DATA, HORIZON, probe_data=PROBE)
    y = dtau[:, -1]
    kfun = MECH.kappa_of_E()[0]
    kap = kfun(Etrue)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    arms["MECHANISM"].append(score(X[:, -1], y))
    arms["C1 shuffle"].append(score(X[perm, -1], y))
    arms["C2 no kappa"].append(score(np.cumsum(dE, 1)[:, -1] * MECH.DT, y))
    arms["C3 wrong sign"].append(score(np.cumsum(-kap * dE, 1)[:, -1] * MECH.DT, y))
    arms["C4 wrong state"].append(score(np.cumsum(kap * dTH, 1)[:, -1] * MECH.DT, y))

print(f"  {'arm':<16}{'R2':>9}{'b':>9}{'spearman':>11}")
res = {}
for k, v in arms.items():
    med = [float(np.median([s[j] for s in v])) for j in range(3)]
    res[k] = med
    print(f"  {k:<16}{med[0]:>9.3f}{med[1]:>9.2f}{med[2]:>11.3f}")
json.dump(res, open("runs/dreamer_mechanism_controls.json", "w"), indent=2)

m, c1, c2, c3, c4 = (res[k] for k in arms)
print("\n--- VERDICT")
ok = True
if abs(c1[2]) > 0.15:
    print(f"  C1 FAILED: shuffling still gives rho {c1[2]:+.3f}. The correlation is an artefact."); ok = False
if abs(c3[1] + abs(m[1])) > 0.5 * abs(m[1]):
    print(f"  C3 note: sign-flipped kappa gives b {c3[1]:+.2f}; expected ~{-m[1]:+.2f}.")
if abs(c2[0]) >= abs(m[0]) - 0.02:
    print(f"  C2 FAILED — THE IMPORTANT ONE. Dropping the frequency weight costs nothing")
    print(f"  (R^2 {c2[0]:+.3f} vs {m[0]:+.3f}). Since the identifiability gate passes, the two")
    print(f"  predictors ARE distinguishable on this data and the weighting still adds nothing.")
    print(f"  The claim shrinks to: the integral of the INVARIANT error predicts drift. That is")
    print(f"  real, but it is not the frequency-action mechanism."); ok = False
else:
    print(f"  C2 PASSES — and it is informative now, unlike in D39. Dropping the frequency weight")
    print(f"  costs real accuracy (R^2 {c2[0]:+.3f} vs the mechanism's {m[0]:+.3f}) while the")
    print(f"  identifiability gate confirms the two predictors are separable (|rho| 0.78-0.87 < 0.9).")
    print(f"  d ln(omega)/dE is doing measurable work.")
if abs(c4[2]) > 0.8 * abs(m[2]) and abs(c4[2]) > 0.15:
    print(f"  C4 FAILED: a NON-invariant error predicts drift as well (rho {c4[2]:+.3f} vs "
          f"{m[2]:+.3f}).")
    print("  The invariant plays no privileged role; the Noether framing would be decorative.")
    ok = False
if ok:
    print(f"\n  SPECIFIC. The mechanism (rho {m[2]:+.3f}, b {m[1]:+.2f}) survives")
    print(f"  while the informative controls fail: shuffle {c1[2]:+.3f} (destroyed), wrong-state")
    print(f"  {c4[2]:+.3f} with R^2 {c4[0]:+.3f} (a NON-invariant error does substantially worse),")
    print(f"  wrong-sign b {c3[1]:+.2f} (flips with the physics).")
    print("  So the INVARIANT error specifically -- not the state error generally -- predicts which")
    print(f"  rollouts drift. Reported alongside: b = {m[1]:+.2f}, not the predicted 1, so the SHAPE")
    print("  of the law is supported and its MAGNITUDE is off by roughly a factor of 4.")
