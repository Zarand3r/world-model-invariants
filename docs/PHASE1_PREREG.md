# Phase-1 pre-registration — Jacobian-closure vs probe kill-test

**Status: CLOSED — registered runs executed; verdict KILL (see "Registered outcome" below).**
Margins below were committed before the runs and are preserved verbatim as registered.
Registered margins must be committed before full-scale results are computed. A `--smoke`
plumbing run (tiny sizes, seed 0, model-quality gate FAILED, 2026-07-21) was executed to
verify the pipeline only; its numbers are void for inference and were not used to set the
margins below (which come from `RESEARCH_NOTES.md`, written before any code existed).

## Hypothesis (Wedge 1)

The subspace the frozen model's transition Jacobian preserves (`fit_closed_subspace`,
`latent_noether/closure.py`) is a better mechanistic object than the subspace a linear
probe reads (`latent_noether/probe.py`), measured by model-error fidelity
(`latent_noether/fidelity.py`) and intervention consistency
(`latent_noether/interventions.py`).

## Registered setup

- Environment: harmonic oscillator, ω = 1.0, dt = 0.05, unit mass; position-only
  observations (velocity must live in the hidden state).
- Model: `GRUWorldModel(hidden_dim=64)`, predictive MSE only, `Config()` defaults in
  `scripts/run_phase1.py`.
- Seeds: 0–4; report per-seed values and the median. No seed selection.
- Latent dimension d = 2 (the system's true dimension; d-sweep is a pre-declared
  refinement, not a fishing license).

## Validity gate (before any wedge conclusion)

- `model_quality_ok`: open-loop rollout MSE over the 50-step horizon must beat the
  constant predictor by ≥ 10×. If it fails, fix world-model training — the run is void
  for the wedge either way.
- `probe_r2` ≥ 0.9 (if a probe can't even decode (q, p), the comparison is degenerate).

## Decision rule (medians over seeds 0–4)

- **PURSUE** if `fidelity(closed) − fidelity(probe) ≥ +0.10` **and**
  `intervention(closed) − intervention(probe) ≥ +0.10` **and** `fidelity(closed) > 0`.
- **KILL** if the closed subspace beats the probe on **neither** metric by ≥ +0.10 —
  Wedge 1 is dead at ~2 weeks of cost, as planned.
- **AMBIGUOUS** (exactly one margin met, or fidelity(closed) ≤ 0 with margins met): one
  pre-declared refinement round only — extraction layer choice, d ∈ {2, 3, 4}, Jacobian
  sample count — then a single re-run of the full protocol and a final verdict. No
  further iteration on the registered systems.
- `fits_environment_better = true` for the closed subspace at full scale is a red flag
  that the extraction identified the environment, not the model; report it prominently
  regardless of verdict.

## Protocol amendment 1 (2026-07-21, before the registered multi-seed runs)

**Disclosure:** a full-scale seed-0 run under the original protocol was executed and its
wedge metrics were seen before this amendment. Under the original protocol both arms
scored catastrophically (closed: fidelity −101; probe: −1.96, `fits_environment_better`),
and a spectral-candidate audit showed the failure was construction, not measurement: (a)
pure Jacobian closure is degenerate — readout-irrelevant and fast-decaying subspaces are
also exactly invariant (every candidate preserved ≤ 20% of the model's own readout
signal); (b) decoding laws through `readout(μ + Uz)` destroys the signal for any subspace.
Amendments, identical for both arms:

1. The closed subspace is found by `fit_sufficient_closed_subspace` — closure residual
   + λ·readout-insufficiency (λ = 1), per proposal §5.1 constraint 1 + §5.2. Both terms
   are model-internal; no simulator quantities enter subspace discovery.
2. Laws decode to observation space through a per-arm linear decoder g(z) fit to the
   **model's own predictions** (never simulator states), same capacity in both arms.
3. Margins, seeds, gates, and the decision rule are unchanged. Seed 0 remains in the
   registered set; since the amendment was designed after seeing seed-0's *broken-protocol*
   numbers (not its amended numbers), the median over seeds 0–4 under the amended protocol
   is reported with this history disclosed.
4. Pre-declared escalation (to use only if BOTH arms score fidelity ≤ 0 at full scale):
   replace linear g and linear latent dynamics with equal-capacity MLPs in both arms —
   one escalation, then verdict.

The damped-oscillator control (`--env damped`, ζ = 0.1) runs the identical protocol;
its `invariance_ratio` must be materially larger than the harmonic one (refusal), and
trajectories span ≥ 2 periods so refusal is tested against the correct mathematical fact.

## Outcome (registered, 2026-07-21)

**Verdict: KILL.** Both validity gates passed (all 5 harmonic models beat the constant
predictor by ≥ 10×; probe R² ≥ 0.9). Linear law class: both arms fidelity ≤ 0 (medians
−9.18 closed / −1.50 probe) → escalation per amendment 1.4. Escalated (MLP, final):
median fidelity −2.06 closed vs −1.63 probe (margin not met); median intervention R²
0.61 closed vs 0.99 probe (margin not met). Wedge 1, as operationalized here, is dead
on this system. Per-seed: the closed arm wins outright on seeds 0 and 3 (fidelity
+0.10 / +0.08 vs probe −0.86 / −1.63) and fails catastrophically on seeds 1–2 —
the closure objective's optimum is unstable across training seeds, while the probe arm
is uniformly stable. Secondary results that stand: the damped control refused an exact
invariant (median invariance ratio 0.062 vs 0.0036 harmonic); the model-error-fidelity
metric and environment-fit flag behaved as designed. See `runs/registered_verdict_phase1mlp.json`.

## Post-hoc correction 1 (2026-07-21, after the verdict) — disclosed, verdict unchanged

A code review after the verdict found a **one-step misalignment in the fidelity
reference**: `rollout(h0)[j]` predicts observation index `warmup + j`, but `sim_roll`
was sliced from `warmup − 1`. Every registered run therefore compared the model's
prediction of obs *t* against true obs *t−1*, inflating MSE(model, sim) — the
denominator of model-error fidelity — by ~2× (0.00664 vs 0.00313 on seed 0). It does
not touch MSE(law, model), since the law and model rollouts share the correct
alignment.

Re-analysis was run **from the existing checkpoints** (no retraining, so the correction
is isolated from training variance), together with an observability-Krylov init fix
(`span{w, J̄ᵀw, …}`); the old-init results are retained as `corrlinOldK_*` and are
**bit-identical** to the new ones, confirming the init is not the deciding factor.

| law class | fidelity closed | fidelity probe | interv. closed | interv. probe | verdict |
|---|---|---|---|---|---|
| linear (registered) | −20.633 | −3.951 | 0.675 | 0.971 | KILL |
| MLP (escalated, final) | −3.308 | −2.955 | 0.613 | 0.989 | **KILL** |

Both margins remain unmet and both gates still pass, so **the KILL verdict stands**;
only the magnitudes change (the original run reported −2.06 vs −1.63). Damped-control
invariance ratios are unaffected (0.0036 harmonic vs 0.0616 damped). Artifacts:
`runs/registered_verdict_corrlinear.json`, `runs/registered_verdict_corrmlp.json`.

## Known deviations and limitations (disclosed)

- **Damped-control trajectory length violated this document's own requirement — now
  fixed, and the corrected control is STRONGER.** The registered damped runs spanned
  6.0 time units (< 1 period); the ≥ 2-period requirement exists because smooth
  spiral-phase invariants genuinely exist on shorter segments. Re-run at
  `--n-steps 260` (13.0 units, ~2.1 periods), all 5 seeds, quality gate passing on all
  of them (previously seed 2 failed):

  | | span | median invariance ratio | separation from harmonic (0.0036) |
  |---|---|---|---|
  | original (violating) | < 1 period | 0.0616 | 17× |
  | corrected | ~2.1 periods | **0.2472** | **69×** |

  The violation had made the refusal look *weaker* than it is — exactly as the
  mathematics predicts, since the sub-period search could partially succeed on a
  spiral-phase invariant. Artifacts: `runs/damplong_damped_seed*.json`.
- **No extraction/evaluation split.** Subspaces, latent dynamics, decoders and
  invariants are fit on the same validation trajectories used to score fidelity and
  interventions. This is symmetric across both arms — the comparison and the verdict
  hold — but absolute fidelity numbers are optimistic.
- **Damped seed 2 failed the model-quality gate** yet its invariance ratio is included
  in the control median (excluding it moves the median 0.062 → 0.060; conclusion
  unchanged).
- **Not implemented from the proposal:** Markov-closure metric (§5.1), OOD fidelity
  (§7.4), Stage G symbolic regression / SINDy (§5.7). CLAUDE.md's test table names a
  SINDy-recovery test that does not exist. These are out of Phase-1 scope, not silently
  dropped.
  *(Superseded 2026-07-28: `latent_noether/symbolic.py` and `tests/test_symbolic.py` now
  implement Stage G with four known-answer recovery tests. Recorded here rather than
  edited away, because this is a registered document.)*

## After the gate

PURSUE → pendulum (nonlinear law), then damped oscillator (the method must **refuse**
an exact invariant), per `RESEARCH_NOTES.md`. Invariant discovery is **not** part of
this gate and no conservation claim may be made from Phase 1.
