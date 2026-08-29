# Reviewer handoff — world-model invariants

Everything a reviewer, collaborator or future maintainer needs: what is claimed, what the evidence
is, what failed, how to reproduce it, and what to do next. Every number here is checked mechanically
by `scripts/verify_paper_numbers.py` (**63/63 passing**) against the immutable records in `runs/`.

---

## 1. The claim, in one paragraph

A DreamerV3 world model trained only on pendulum video learns a conserved scalar. That scalar is
**not** textbook energy — it is the **shadow Hamiltonian of the simulator's integrator**,
`H~ = H + (dt/2) mg(l/2) thetadot sin(theta)`. The coefficient is a property of the *discretisation*,
and the model tracks it across a 4x range of simulator timestep with a parameter-free slope of
**2.484 ± 0.058** against a predicted **2.500**. Consequently a probe fitted to *true* energy —
reaching correlation `0.9999` — identifies a direction the model's own transition does **not**
preserve, and enforcing it during imagination makes the model's physics worse. **World models learn
their simulator's numerical scheme, not the physics that scheme approximates.**

---

## 2. Results

### 2.1 The headline (F6): the model tracks the simulator's timestep

Four timesteps x three seeds, sweep expressed relatively as `r = c/(2.5 dt)` so the prediction is a
single value of `r` everywhere.

| dt | predicted `c*` | argmin `r` per seed | `rho_obs` r=0 / r=1 / r=-1 | separation |
|---|---|---|---|---|
| 0.02 | 0.0500 | +0.75, +0.75, +1.00 | 0.01714 / 0.01657 / 0.01966 | 1.03x |
| 0.035 | 0.0875 | +1.00 x3 | 0.02311 / 0.01033 / 0.04473 | 2.24x |
| 0.05 | 0.1250 | +1.00 x3 | 0.04487 / 0.00785 / 0.09264 | 5.72x |
| 0.08 | 0.2000 | +1.00 x3 | 0.11884 / 0.00882 / 0.23781 | **13.5x** |

- Argmin **exactly** at the predicted `r = 1` on **10 of 12** models; wrong-sign control beaten
  **12/12**.
- Origin-forced slope **2.484 ± 0.058** vs predicted **2.500** (0.6% agreement, nothing fitted).
- Two unfitted scalings make the mechanism visible: `rho_obs` at textbook energy grows as `dt^1.41`
  while `rho_obs` at the shadow stays at the model's own floor (`dt^-0.49`, flat at 0.008–0.017).
- **Registered failure, reported as such:** the preregistration required slope *and* intercept; the
  two-parameter intercept is `-0.0072 ± 0.0057`, excluding zero, driven by a real ~12% low bias at
  `dt = 0.02` where the whole separation is only 1.03x. A post-hoc fine grid there puts the argmin at
  `0.875` on 3/3 seeds, so it is a genuine bias, not grid coarseness.

Records: `runs/f6_models.json`, `runs/f6_physics.json`. Prereg: `docs/F6_PREREG.md`.

### 2.2 The mechanism (E19)

Sweeping the regression target over `T_c = E + c*thetadot*sin(theta)` — a family containing both
signs and wrong magnitudes — `rho_obs` is minimised at exactly the integrator-predicted `c* = 0.125`
on 3/3 seeds, **12x** better than the wrong-sign control. On ground truth with no model, `T_{c*}` is
**591x** better conserved than textbook energy. Residual gap to the label-free scalar: **1.10x** — the
shadow accounts for essentially the whole effect.

### 2.3 The dissociation (E18 + five axes)

A supervised probe fitted to true energy reaches `|rho|_E = 0.9999` yet is **6.7x** less conserved by
the transition and **never repairs** a rollout (increases drift on 2 of 3 seeds, reduces on none),
where the label-free scalar repairs 3/3 at a median **-42.2%**.

Decodability survives while dynamical structure fails on **five** independent axes: untrained models,
out-of-distribution states, a second architecture, an untrained transition, and **actuation** — where
models that demonstrably use their actions (0.740–0.761 true vs shuffled) carry an energy-like scalar
whose motion the action's power explains **0.31%** of.

### 2.4 Causal deployment

Enforcing the recovered scalar cuts drift in decoded physical energy by **55–76%** on unseen
trajectories, with **0 of 60** magnitude-matched random directions beating it; setting it steers the
imagined world's true energy (rank correlation **0.80–0.91**); the effect survives an interchange
**50 steps** into autonomous imagination.

---

## 3. What failed — four preregistered negatives, each with a measured cause

These are load-bearing, not embarrassments. Each bounds the claim with a number.

| id | question | outcome | measured cause |
|---|---|---|---|
| **F2** | is invariant drift an online trust signal? | **NO** | Spearman `-0.02/+0.09/-0.24` vs `+0.27/+0.29/+0.31` for plain latent displacement; a random control beats it on 2/3 |
| **F5** | does the correction improve control return? | **NO** | largest arm effect `0.0014` against a return SD of `0.575` — **0.24%**. The correction shifts the plan-ranking signal by **0.3%**, because it acts on *accumulated* drift and a re-planning controller never accumulates any. Survives a 3x horizon; planning quality itself degrades with horizon |
| **E10b** | does conservation, not decodability, drive repair? | **NOT ESTABLISHED** | Spearman `+0.19/+0.57/+0.19`, CI excludes zero on 1 of 3. The previously reported `+0.71` came from an uncommitted script and does not reproduce |
| **F3** | can constraints `G(z)=0` be extracted? | **NO** | recovered `G` correlates with the true rod-length residual at `0.004–0.032`. Minimising `E[G^2]` ranks the true constraint **1526th of 1819**, and the constraint is barely encoded (held-out `rho` 0.19). **A quantity that never varies carries no information**, so it lives in decoder weights |

**F3's cause generalises and is why no released-checkpoint experiment is reported:** a walker's limb
lengths do not vary either.

---

## 4. Reproducibility

- **63/63** mechanical number checks (`scripts/verify_paper_numbers.py`) recompute every headline
  figure from `runs/*.json` and grep the sources for it. Guards include standing checks against
  specific past errors returning.
- **56 tests** pass, including a regression test pinning the raw-vs-standardised coefficient
  convention, verified to fail when the convention is broken.
- `docs/RESULTS.md` regenerates **byte-identical**; **17 claims, all at n >= 3**.
- **28 preregistrations**, written before results, with falsifiers. Amendments are dated and state
  the direction they move the expected result.
- **Four headline records re-derived from scratch**: E9, E4 and E12c reproduce to every reported
  digit; E10b did not — and it was the only one produced by an uncommitted script. That failure drove
  `latent_noether/provenance.py`, which now stamps argv, git state and input sha256 into records.
- Artifacts: **W&B** `richardbao419-substrate/world-model-invariants` (26 training runs + a results
  run); **Hugging Face** `Zarand3r/world-model-invariants` (56 files, 1,968 MB, **private** for
  double-blind anonymity), with `docs/ARTIFACT_MANIFEST.md` giving each file's sha256 and the claim it
  backs.

### Known defects, disclosed

- The **published** `paper/` version describes its random-constraint null as norm-matched; the
  projection is scale-invariant in `C`, so random draws took steps **29x larger**. Corrected in
  `paper1.2` with a magnitude-matched null, which *strengthens* the result (2/3 → 3/3 seeds).
- `paper1.2` main body measures **12 pages at ICLR geometry against a 9-page limit**.

---

## 5. Extensions, ranked

**F7 — does it track the *scheme*, not just the timestep?** *(strongest, cheapest)*
Train on explicit Euler and semi-implicit Euler at the **same** `dt`. Explicit Euler is not symplectic
and has **no** conserved shadow, so the prediction is sharp and the conditions cleanly separated. This
upgrades "learns the timestep" to "learns the integrator". ~3 GPU-hours; infrastructure exists.

**F8 — higher-order integrators.** RK4's leading shadow correction is `O(dt^4)`, so the predicted
`c*` is ~0. If the model tracks that too, the claim covers integrator *order*, not just step size.
A clean null-prediction test.

**F9 — does inherited discretisation predict transfer failure?** *(first plausible practical payoff)*
Train at `dt = 0.05`, evaluate on `dt = 0.02` dynamics, and test whether prediction error is
predicted by the mismatch in `c*`. F2 and F5 both failed for the same effect-size reason; this one
targets a regime where the effect is large by construction. **This is the most promising route to a
downstream consequence**, which the paper currently lacks.

**F10 — mixed-timestep training.** Train on data pooling two timesteps. Does the model learn one
shadow, an average, or neither? Probes whether the learned scheme is a single global property of the
representation.

**F11 — real video.** With no integrator, what does the model learn? Highest impact, hardest, and the
direct sim-to-real question.

**Explicitly not recommended:** more toy systems, more seeds, another intervention variant, or a sixth
dissociation axis. Effect sizes are small, interventions do not transfer, and the marginal claim is
already well supported. The binding constraint is presentation, not evidence.

---

## 6. Where to look

| what | where |
|---|---|
| claim architecture | `paper1.2/CLAIMS.md` |
| every experiment, dated, with negatives | `docs/EXECUTION_LOG.md` |
| plan + outcome per item | `docs/ROADMAP.md` (execution-status table) |
| generated summary | `docs/RESULTS.md` |
| preregistrations | `docs/*_PREREG.md` |
| number verification | `scripts/verify_paper_numbers.py` |
| published artifacts | `docs/ARTIFACTS.md`, `docs/ARTIFACT_MANIFEST.md` |
