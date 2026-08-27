# Pre-registration — E17, two degrees of freedom

**Written 2026-08-27, before any 2-DoF data was generated.** `docs/ROADMAP.md` Phase IV.
Claim addressed: **C6 — generality.** Approved by Richard 2026-08-27, after E10 turned out not to be
constructible on the pendulum.

## Two independent reasons this system is needed

1. **The 1-DoF objection, stated in the roadmap.** On a conservative pendulum, energy is essentially
   the only nontrivial scalar constant within a trajectory and varying across them. "We found
   energy" is close to "we found the one thing there was to find."
2. **E10 is not constructible on a pendulum.** Of 250 eigenfamily candidates, exactly 1 had
   `|rho_E|` > 0.3. The decodability-matched null — the control that most directly tests the paper's
   probe-vs-dynamics thesis — requires variation in decodability at matched conservation, and a
   1-DoF system does not supply it. A 4-dimensional state should.

## The system

A single particle in a 2D anharmonic potential, rendered as one disk at position `(q1, q2)`.
State is 4-dimensional. One object, no occlusion, and velocity must still be inferred from the
sequence — the same epistemic situation as the pendulum.

**Two arms, identical renderer and pipeline, differing only in how many quantities are conserved.**

> **Design corrected 2026-08-27, before any data was generated.** At `w1 = w2` the central potential
> expands exactly: `1/4 a r^4 = 1/4 a(q1^4 + q2^4) + 1/2 a q1^2 q2^2`. So **central is simply the
> case `b = a`**, and the two arms become one system with **one parameter** changed — a strictly
> cleaner matched pair than two separately-specified potentials.

    V = 1/2 (q1^2 + q2^2) + 1/4 a (q1^4 + q2^4) + 1/2 b q1^2 q2^2

| arm | parameter | conserved |
|---|---|---|
| **central** | `b = a` | energy **and** angular momentum |
| **non-central** | `b != a` | energy only |

This is a matched pair in the same sense as conservative/damped: identical in every respect except
one physical fact. It converts "does it generalise?" into a **prediction the method can be scored
against** — the eigenfamily should yield one well-conserved energy-like direction in the non-central
arm and a two-dimensional conserved subspace in the central arm.

**Integrator: semi-implicit (symplectic) Euler**, deliberately matching gymnasium's pendulum
convention, so that `p_k = (q_k - q_{k-1})/dt` holds exactly and the geometric readout's
backward-difference convention transfers unchanged. Using a different integrator would silently
break the readout that took an iteration to get right on the pendulum.

## CHAOS GATE — runs before any pixels are rendered

Quartic coupling can be chaotic, and a chaotic system would fail this experiment for reasons that
have nothing to do with the hypothesis.

**Gate statistic:** maximal Lyapunov exponent `lambda_max`, by two-trajectory renormalisation, over
the intended initial-condition distribution.

- **PASS** if `lambda_max * T_run < 1` for the intended rollout length, i.e. neighbouring
  trajectories separate by less than a factor of `e` over the horizon used, on **at least 95%** of
  sampled initial conditions.
- **FAIL** otherwise. On failure, reduce `a` and `b` and re-test, **recording every parameter set
  tried and its result**, until the gate passes. The parameters are then frozen before any data is
  generated.

Reported alongside: the fraction of trajectories whose Poincare section (`q2 = 0`, `p2 > 0`) forms a
closed curve rather than a scattered set, as a qualitative cross-check.

## GATE RESULT AND FROZEN PARAMETERS — 2026-08-27

The gate ran before any pixels were rendered, and it changed the design twice.

**First failure: weak coupling leaves extra invariants.** At `w1 = 1.0, w2 = 1.3, a = b = 0.20` the
per-mode energies `E1`, `E2` are conserved almost as well as the total (0.052 vs 0.033), so the
"non-central" arm would secretly have had **three** approximate invariants. KAM tori survive a 10%
perturbation. Strengthening the coupling to `a = b = 0.50` broke them but **failed the chaos gate**
(frac 0.832).

**Second finding: symmetry-breaking and chaos trade off directly.** Holding `a = 0.30` and raising
`b`, the gate fails from `b = 0.60` upward. The resolution is *low overall anharmonicity with a large
`b/a` ratio*. Full search recorded in `runs/e17_chaos_gate_search.json` (16 parameter sets).

**FROZEN**, in `runs/e17_chaos_gate_frozen.json`:

    w1 = w2 = 1.0     a = 0.05     dt = 0.05
    non-central: b = 0.40      central: b = 0.05

| arm | horizon 100 | horizon 190 | horizon 200 | E invariance | L invariance |
|---|---|---|---|---|---|
| non-central | 1.000 PASS | 0.977 PASS | 0.980 PASS | 0.0290 | **0.1006** (broken) |
| central | 1.000 PASS | 1.000 PASS | 1.000 PASS | 0.0283 | **0.0000** (exact) |

Angular momentum is exactly conserved in the central arm and 3.5x less conserved than energy in the
non-central arm, while total energy is equally well conserved in both. The gate passes on both arms
at every horizon the analysis uses.

## Method transfer — frozen first, adapted only on record

`docs/ROADMAP.md` is explicit and is followed: **first attempt the already-fixed extraction
hyperparameters** — LD = 12, degree 4, `n_basis` = 8, WARMUP = 10 — with no tuning.

If recovery fails at those settings, the failure is **recorded as a result**, and any adapted
setting is treated as a **separate experiment** with its own entry. LD = 12 was chosen for a
2-dimensional physical state; a 4-dimensional state may need more, and that is precisely the kind of
finding worth reporting rather than tuning away.

## Registered predictions

1. **Recovery.** The non-central arm yields one scalar with `|rho_E|` comparable to the pendulum's
   (> 0.8) and an invariance ratio within an order of magnitude of it.
2. **Two invariants.** The central arm yields a **two**-dimensional well-conserved subspace, with
   the second direction correlating with angular momentum at `|rho_L|` > 0.8.
3. **Repair.** Direction-matched projection reduces decoded physical energy drift, with 0 of 20
   magnitude-matched random directions beating it.
4. **E10 becomes constructible.** The eigenfamily contains >= 8 candidates within +-0.10 of the
   recovered `C`'s `|rho_E|`, so the decodability-matched null can finally be run.

**Falsifiers.** Prediction 1 failing at frozen hyperparameters is a real negative result about
generality and will be reported as one. Prediction 2 failing while 1 holds would say the method finds
*an* invariant but not the full conserved structure — also a result. Prediction 3 failing while 1
holds would confine the repair claim to 1-DoF.

## Cost and staging

Chaos gate: minutes, pure simulation. Renderer and dataset: hours. Training 3 conservative seeds per
arm at the frozen step-capped contract (60,000 steps, E8 milestone grid): ~85 min each. Analysis
reuses every existing script unchanged.

Staged so the gate can stop the work before any expensive step.
