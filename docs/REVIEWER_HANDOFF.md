# Review handoff — *World models learn their simulator's integrator, not the physics it approximates*

For a co-author, an internal reviewer, or a future maintainer. What the paper claims, what the
evidence is, what failed, what to distrust, and how to check any of it yourself.

**Status:** ICLR 2025 submission format, **9-page main text** (16 with appendix and references),
double-blind anonymised. **66/66** mechanical number checks and **56** tests pass. `origin/main`
remains pinned at the last published state (`20fa8b4`, 2026-08-24); all work is local on
`roadmap/stage1` behind a `pre-push` hook.

---

## 1. The claim

A DreamerV3 world model trained only on pendulum video learns a conserved scalar. That scalar is
**not textbook energy** — it is the **shadow Hamiltonian of the simulator's integrator**,
`H~ = H + (dt/2) mg(l/2) thetadot sin(theta)`. The correction coefficient is a property of the
*discretisation*, so varying the simulator's timestep must move it, and it does: across a 4x range
the recovered coefficient tracks the integrator's prediction with slope **2.484 ± 0.058** against a
parameter-free **2.500**.

The interpretability consequence: a probe fitted to *true* energy reaches `|rho| = 0.9999` yet is
**6.7x** less preserved by the model's own transition than a label-free scalar, and enforcing it
during imagination makes the model's physics **worse**. **The perfect probe is fitted to a quantity
the generating process does not conserve**, and no probe hygiene detects that, because nothing is
wrong with the probe.

---

## 2. Evidence

### 2.1 The headline (F6) — the model tracks the simulator's timestep

| dt | predicted `c*` | argmin `r` per seed | `rho_obs` r=0 / r=1 | separation |
|---|---|---|---|---|
| 0.02 | 0.0500 | +0.75, +0.75, +1.00 | 0.01714 / 0.01657 | 1.03x |
| 0.035 | 0.0875 | +1.00 x3 | 0.02311 / 0.01033 | 2.24x |
| 0.05 | 0.1250 | +1.00 x3 | 0.04487 / 0.00785 | 5.72x |
| 0.08 | 0.2000 | +1.00 x3 | 0.11884 / 0.00882 | **13.5x** |

Argmin exactly on prediction for **10 of 12** models; wrong-sign control beaten **12/12**. Two
unfitted scalings make the mechanism visible: `rho_obs` at textbook energy grows as `dt^1.41` while
`rho_obs` at the shadow stays at the model's floor (`0.008`--`0.017`, flat).

**Registered failure, reported as such.** The preregistration demanded slope *and* intercept; the
two-parameter intercept is `-0.0072 ± 0.0057`, excluding zero, from a real ~12% low bias at the
smallest timestep where the whole separation is 1.03x. A post-hoc fine grid there gives argmin
`0.875` on 3/3, so it is genuine, not grid coarseness.

### 2.2 The mechanism (E19)

Sweeping `T_c = E + c*thetadot*sin(theta)` over a family containing both signs and wrong magnitudes,
`rho_obs` is minimised at exactly `c* = 0.125` on 3/3 seeds, **12x** better than the wrong-sign
control. On ground truth with no model, `T_{c*}` is **591x** better conserved than textbook energy.
Residual gap to the label-free scalar: **1.10x**.

### 2.3 Dissociation, five axes

Untrained models (74x), out-of-distribution states (55--267x), a second architecture (767x), an
untrained transition (~10,000x), and **actuation** — models that demonstrably use their actions
(0.740--0.761 true vs shuffled) carry an energy-like scalar whose motion the action's power explains
**0.31%** of.

### 2.4 Causal deployment

Drift in decoded physical energy cut **55--76%** on unseen trajectories (**0 of 60** matched random,
**0 of 15** tangent); steering at rank correlation **0.802--0.914** (**0 of 75** controls); survives
an interchange **50 steps** into autonomous imagination (0.762--0.849 vs 0.803--0.924 at depth 0).

---

## 3. Four preregistered negatives, each with a measured cause

Load-bearing, not embarrassments — each bounds the claim with a number.

| id | question | outcome | cause |
|---|---|---|---|
| **F5** | does it improve control return? | **no** | largest arm effect `0.0014` vs return SD `0.575` = **0.24%**. The correction shifts the plan-ranking signal by **0.3%**, because it acts on *accumulated* drift and a re-planning controller never accumulates any. Survives a 3x horizon; planning quality itself degrades with horizon |
| **F2** | is drift an online trust signal? | **no** | Spearman `-0.02/+0.09/-0.24` vs `+0.27/+0.29/+0.31` for plain latent displacement |
| **F1** | does it learn the balance law under action? | **no** | power explains **0.31%** of the variance in `dC`; observed change 4.5--5.4x larger than predicted |
| **F3** | can constraints `G(z)=0` be extracted? | **no** | `E[G^2]` ranks the true constraint **1526th of 1819**, and the constraint is barely encoded (held-out `rho` 0.19). **A quantity that never varies carries no information** — it lives in decoder weights. This is why no released-checkpoint experiment is reported: limb lengths do not vary either |
| **E10b** | does conservation *alone* predict repair? | **not established** | CI excludes zero on 1 of 3 seeds; the previously reported `+0.71` came from an uncommitted script and does not reproduce |

---

## 4. What to distrust, and what we did about it

Read this section first if you are reviewing adversarially.

- **The published `paper/` version has a defect.** Its random-constraint null is described as
  norm-matched, but the projection is scale-invariant in `C`, so random draws took steps **29x
  larger**. Corrected in `paper1.2` with a magnitude-matched null — which *strengthens* the result
  (2/3 to 3/3 seeds). Disclosed in `paper/README.md`.
- **E10b's numbers changed.** The original came from a script that was never committed. On
  reimplementation seed 1 reproduced and seed 0 did not; no update rule tested reconstructs the old
  values. This drove `latent_noether/provenance.py`, which now stamps argv, git state and input
  sha256 into every record.
- **Four headline records were re-derived from scratch.** E9, E4 and E12c reproduce to every reported
  digit. E10b did not — and it was the only one produced by an uncommitted script.
- **Several errors were mine and are logged.** An overclaim ("increases drift on 3 of 3" when one seed
  reduced it), a flattering-seed quote (14x was seed 3 of 14/10/12), an arithmetic slip (768 vs 767),
  a count understating our own result (9 of 12 when it was 10), and a page-length metric that was
  wrong for several passes. Each is dated in `docs/EXECUTION_LOG.md` with how it was found.
- **The guards themselves have failed in both directions**, and the fixes are recorded: one matched
  an ASCII rendering and missed a stale `6.3\times` for many iterations; its replacement matched bare
  digits and flagged two legitimate uses. The rule now is **match the claim, not its rendering**.

---

## 5. How to check any of it

```bash
uv run python scripts/verify_paper_numbers.py   # 66 checks: recompute from runs/, grep the sources
uv run python -m pytest tests/ -q               # 56 tests, incl. a pinned coefficient-convention test
uv run python scripts/make_results_summary.py   # regenerates docs/RESULTS.md byte-identically
./scripts/make_arxiv_archive.sh paper1.2        # archive; extract anywhere and it compiles standalone
```

**17 claims, all at n >= 3.** 28 preregistrations with falsifiers written before results; amendments
are dated and state which direction they move the expected outcome. Artifacts: W&B
`richardbao419-substrate/world-model-invariants`; Hugging Face `Zarand3r/world-model-invariants`
(**private**, for double-blind), with `docs/ARTIFACT_MANIFEST.md` giving each file's sha256 and the
claim it backs.

---

## 6. Extensions, ranked

1. **F7 — does it track the *scheme*, not just the timestep?** Train on explicit vs semi-implicit
   Euler at the same `dt`. Explicit Euler is not symplectic and has **no** conserved shadow, so the
   prediction is sharp. Upgrades "learns the timestep" to "learns the integrator". ~3 GPU-hours,
   infrastructure exists. *Strongest and cheapest.*
2. **F9 — does inherited discretisation predict transfer failure?** Train at one `dt`, evaluate on
   another, test whether error tracks the mismatch in `c*`. F2 and F5 both failed for the same
   effect-size reason; this targets a regime where the effect is large by construction. *Best route
   to the downstream consequence the paper lacks.*
3. **F8 — higher-order integrators.** RK4's shadow correction is `O(dt^4)`, so predicted `c*` ~ 0.
   A clean null-prediction test extending the claim to integrator *order*.
4. **F10 — mixed-timestep training.** One shadow, an average, or neither?
5. **F11 — real video.** No integrator exists; what is learned then? Highest impact, hardest, and the
   direct sim-to-real question.

**Not recommended:** more toy systems, more seeds, another intervention variant, or a sixth
dissociation axis. Effect sizes are small, interventions do not transfer, and the marginal claim is
already well supported.

---

## 7. Honest assessment

The contribution is a **positive, general, parameter-free result** (F6) with a **mechanism** (E19)
explaining a **methodological correction** (probing over-reports), bounded by **four measured
negatives**. The evidence is unusually well controlled for this area.

The weaknesses are real and stated in the paper: two smooth Hamiltonian toy systems, 13.5M
parameters, no released checkpoint, and **no downstream benefit** — the two practical uses we tested
both fail, for one measured reason.

Estimated main-track probability: **ICLR 55--65%**, **NeurIPS 40--50%**, **ICML 35--45%**. F6 is what
moved these; before it the paper was diagnostic-only and I would have said 40--50 / 25--30 / 20--25.
The largest remaining risk is presentation rather than evidence — and a reviewer who wants scale will
not find it here.

---

## 8. Where to look

| what | where |
|---|---|
| claim architecture | `paper1.2/CLAIMS.md` |
| every experiment, dated, negatives included | `docs/EXECUTION_LOG.md` |
| plan + outcome per item | `docs/ROADMAP.md` (execution-status table) |
| generated summary | `docs/RESULTS.md` |
| preregistrations | `docs/*_PREREG.md` |
| number verification | `scripts/verify_paper_numbers.py` |
| published artifacts | `docs/ARTIFACTS.md`, `docs/ARTIFACT_MANIFEST.md` |
| build + figure provenance | `paper1.2/README.md` |
