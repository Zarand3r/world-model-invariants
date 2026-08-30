# Review handoff — *World models learn their simulator's integrator, not the physics it approximates*

> **The title above is not currently supported.** Read §0 before anything else. The paper's headline
> claim is **unresolved**, four experiments built to test it were **withdrawn as invalid**, and one
> recommendation I made on 2026-08-30 was **retracted the same day**. The negatives, the
> dissociation, the architecture gap and the causal interventions are unaffected, and the
> interventions are now measured immune rather than argued immune.

For a co-author, an internal reviewer, or a future maintainer. What the paper claims, what the
evidence is, what failed, what to distrust, and how to check any of it yourself.

**Status:** ICLR 2025 submission format, **9-page main text** (16 with appendix and references),
double-blind anonymised. **70/70** mechanical number checks (all now anchored to the claim they
support, with a mutation harness proving 26/26 catch their claim being falsified) and **58** tests
pass. `origin/main`
remains pinned at the last published state (`20fa8b4`, 2026-08-24); all work is local on
`roadmap/stage1` behind a `pre-push` hook.

---

## 0. What changed on 2026-08-29/30, and what it costs

**The short version: the paper's headline claim is untested, not refuted, and I spent a week testing
it on an axis where the difference does not exist in the data.**

### 0.1 Four experiments withdrawn as invalid design

F7, F7b, F9 and F10 compared **semi-implicit Euler** against **velocity Verlet**. Eliminating
velocity, both reduce to the *identical* three-term position recurrence

    th_{t+1} = 2 th_t - th_{t-1} + a(th_t) dt^2

verified to `8.9e-16` synthetically and `~1e-15` on both datasets. They differ only in the first step
and in **which finite difference is called the velocity**: the semi-implicit dataset's recorded
`thetadot` equals the backward difference exactly, the Verlet dataset's equals the central difference
exactly (both to `0.00000`).

The models see **pixels, which show position**. Velocity is never rendered. So the entire contrast
lived in a bookkeeping label the model cannot observe, and those four experiments could not have
worked. `tests/test_observable_difference.py` now pins this, and pins that `dt` — unlike the scheme —
*is* observable.

### 0.2 A recommendation made and retracted the same day

On the strength of F10 I recommended **withdrawing F6 and E19**. That was wrong: F10's control
assumed the two schemes are distinguishable in the observations. F6 and E19 revert to **unresolved**.

### 0.3 What F6 can and cannot claim today

Models trained at timestep `dt` recover a coefficient matching `c*(dt)`. **Whether that reflects the
model or the data it was measured on is untested**, and is **not testable by cross-evaluation with
these assets**: every model is only in distribution on its own timestep. F11 quantified the wall — a
model's one-step error rises from `~0.007` rad on its own data to `~0.6` rad on another timestep's,
nine times the separation the test needs to resolve. This is a **limitation to state**, not a
falsification.

What *is* properly separated is F4b: it varies the **model** with the data held fixed (conv-GRU
against RSSM on identical trajectories, `767x`). That supports the **level** claim — this model
conserves something, that one does not — and says nothing about the coefficient.

### 0.4 A gate that worked, for contrast

F11 registered its readability gates *before* running. Without them the `dt = 0.08` models read as
**0.844 / 0.852 / 0.849, 3 of 3, `p < 1e-100`** — and it is pure artefact: at `0.6` rad the
prediction is near neither candidate, and the mirrored cells sit at chance (`0.512 / 0.502 / 0.511`),
which a real effect would not do. Four experiments failed this week because I had not registered the
gate that mattered; this one did not.

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

### 2.1 The headline (F6) — the recovered coefficient matches the simulator's timestep

**Read §0.3 first.** The measurement below is reproducible and the numbers stand. What is *not*
established is that it reflects the **model** rather than the **data it was measured on**.

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

### 2.2 The mechanism (E19) — same attribution caveat as §2.1

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
| **E10b** | does conservation *alone* predict repair? | **not established** | CI excludes zero on 1 of 3 seeds; the figure reported before 2026-08-28 came from an uncommitted script and does not reproduce |

---

## 4. What to distrust, and what we did about it

Read this section first if you are reviewing adversarially.

- **I spent a week testing the headline on an axis that does not exist in the data**, and four
  preregistrations, six training runs and five measurement designs sat on that gap before I checked
  it. The check costs three lines and no GPU (§0.1). Every gate I had written asked whether the
  *instrument* could see a difference; none asked whether the difference was *there*.
- **I recommended withdrawing F6 and E19, then retracted it the same day** (§0.2). Treat my
  confidence statements in this document accordingly — see §7.
- **Four checking scripts were weaker or wronger than the claims they enforced**, all found this
  week: `run_f6_cross.py` printed "argmin follows the data" while its own rows read `0/3`;
  `run_f7_gate0.py` recorded `G2_pass: true` off an `inf` from a comparison that never ran (F7's
  Gate 0 had therefore **not** passed as registered); F7's `P3` checked two of three registered
  criteria; F7b's `P2` checked a two-arm gap where the prereg registered a three-arm ordering. The
  first version of the provenance audit reported `0/153` off a schema misread, and the first version
  of the prereg audit reported ten false positives.
- **M29 provenance had silently degraded across every script written that week.**
  `inputs_from_args` only sees paths reaching the argparse Namespace, and those scripts hardcoded
  their checkpoints and datasets, so artefacts recorded `inputs: {}` while looking stamped. Repaired
  without recomputing (rows verified unchanged); the limitation is now documented in
  `provenance.py` and swept by `scripts/audit_provenance.py`. **70 pre-M29 artefacts are deliberately
  left unstamped** — asserting a provenance nobody can verify would be worse than an honest gap.
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
- **Several errors were mine and are logged.** <!-- superseded: an overclaim ("increases drift on 3 of 3" when one seed reduced it), a flattering-seed quote (14x was seed 3 of 14/10/12), an arithmetic slip (768 vs 767), a count understating our own result (9 of 12 when it was 10), and a page-length metric wrong for several passes -->
  Five of them, each dated in `docs/EXECUTION_LOG.md` with how it was found: an overclaim, a
  flattering-seed quote, an arithmetic slip, a count that understated our own result, and a
  page-length metric that was wrong for several passes.
- **The guards themselves have failed in both directions**, and the fixes are recorded: one matched
  an ASCII rendering and missed a stale `6.3\times` for many iterations; its replacement matched bare
  digits and flagged two legitimate uses. The rule now is **match the claim, not its rendering**.

---

## 5. How to check any of it

```bash
uv run python scripts/verify_paper_numbers.py   # 70 checks, ALL anchored to their claim
uv run python scripts/mutate_guards.py          # breaks each claim; 26/26 guards must catch it
uv run python scripts/audit_preregs.py          # registered predictions with no recorded verdict
uv run python scripts/audit_provenance.py       # artefacts with no record of their inputs
uv run python -m pytest tests/ -q               # 58 tests, incl. the observable-difference guard
uv run python scripts/make_results_summary.py   # regenerates docs/RESULTS.md byte-identically
./scripts/make_arxiv_archive.sh paper1.2        # archive; extract anywhere and it compiles standalone
```

**33 preregistrations** with falsifiers written before results; amendments are dated and state which
direction they move the expected outcome. (This previously read "17 claims, all at n >= 3" and "28
preregistrations". The count was stale --- five were added on 2026-08-29/30 --- and the "17 claims"
figure could not be substantiated against `CLAIMS.md`, which sets out **three**, so it is replaced
here by what is checkable rather than restated.)

**Verified 2026-08-30, having been asserted here unchecked until then:** the 50 artifact hashes in
`docs/ARTIFACT_MANIFEST.md` all match their local files (50/50, `scripts/verify_artifact_manifest.py`);
`docs/RESULTS.md` regenerates byte-identically; and the arXiv archive extracts to a clean directory
and compiles standalone with `tectonic` to a **16-page** PDF, matching the page count claimed above. Artifacts: W&B
`richardbao419-substrate/world-model-invariants`; Hugging Face `Zarand3r/world-model-invariants`
(**private**, for double-blind), with `docs/ARTIFACT_MANIFEST.md` giving each file's sha256 and the
claim it backs.

---

## 6. Extensions, ranked

**The previous version of this list led with "F7 — does it track the *scheme*, not just the
timestep?", called it the strongest and cheapest, and estimated three GPU-hours. It was run, and it
was invalid by construction (§0.1).** Kept visible here rather than quietly deleted, because the
error was in the reasoning that produced the recommendation, not in the execution.

1. **Mixed-timestep or `dt`-conditioned training.** *Now the only route to the paper's headline.*
   Attributing the recovered coefficient to the model rather than the data needs one model that is
   in distribution on more than one timestep; F11 showed cross-evaluation cannot do it (§0.3). Ask
   whether such a model recovers one shadow, an average, or neither. Needs new training compute.
2. **Real video.** No integrator exists, so what is learned then? Highest impact, hardest, and the
   direct sim-to-real question. Unaffected by anything in §0.
3. **Higher-order integrators (RK4).** Predicted `c* ~ 0` from an `O(dt^4)` correction. **Check
   first** whether RK4 differs from the current schemes *in the position sequence* — the test in
   `tests/test_observable_difference.py` — since that is exactly what sank F7.
4. **Does inherited discretisation predict transfer failure?** F11 already shows the transfer
   penalty is enormous (one-step error `0.007 -> 0.6` rad across a 4x timestep change). Attributing
   that to `c*` mismatch rather than to generic distribution shift runs into the same confound, so
   this needs a design that separates them before it is worth running.

**Not recommended:** more toy systems, more seeds, another intervention variant, or a sixth
dissociation axis.

---

## 7. Honest assessment

**The estimates that stood here are withdrawn.** They read *ICLR 55--65%, NeurIPS 40--50%, ICML
35--45%*, and were driven by F6 — the claim now known to be unattributed. Replacing them with new
numbers would repeat the mistake: over two days I called F7 "the cleanest result in the project",
recommended withdrawing F6 and E19, and retracted that recommendation, all on the same evidence base.
The calibration failure is the finding; a fresh point estimate would not be worth more than the last.

What can be said without a forecast:

**What the evidence supports today.** A methodological correction with a mechanism — a probe fitted
to true energy reaches `|rho| = 0.9999` yet is **6.7x** less preserved by the model's own transition
than a label-free scalar, and enforcing it makes the model's physics worse. A five-axis dissociation.
A `767x` architecture gap that *is* properly model-versus-data separated. Four preregistered
negatives, each with a measured cause. Causal interventions that are immune to the confound by
construction **and were measured immune**: repair survives 3/3 with `C` fitted on a different
dataset entirely.

**What it does not support.** That the model learns its simulator's integrator, or its timestep. The
measurement reproduces; the attribution to the model is untested and needs §6.1.

**The honest framing of the paper as it stands** is a probing-methodology result with an unusually
well-controlled negative section — not the positive general result the current title claims. Whether
that is worth a main-track submission as-is, or whether to run §6.1 first, is a judgement call for
Richard, and it is the open decision this document exists to inform.

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
