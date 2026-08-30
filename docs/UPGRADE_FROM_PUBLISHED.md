# Adopting these changes into a clone of the published repository

**For anyone working from `github.com/Zarand3r/world-model-invariants` at or near the arXiv state
(`20fa8b4`, 2026-08-24).** What changed, why, and which commits to take.

Read §1 first even if you take nothing else. **One published result is wrong**, and the fix
strengthens it.

---

## 1. Tier 1 — the correction. This is not a cherry-pick.

**Checked, not assumed:** every file carrying these fixes is *new* since `20fa8b4`, and
`scripts/run_dreamer_edit.py` — the script that produced the published intervention result — **has
never been modified**. So there is no patch series to apply. What follows is the defect stated
precisely enough to fix in place, and the replacement pipeline if you would rather adopt it whole.

### The E1 null was not matched — the published result is affected

Introduced by `77d64ec` *(2026-08-26)*, as a **new** script rather than a fix to the old one.

**This defect is in the published paper.** The E1 edit is a Newton step to the level set, so its size
scales with how badly the constraint is violated. Random draws took steps **29x larger** than the
recovered `C`. So *"0 of 20 random constraints improve anything"* could equally have meant *"random
constraints perturb 29x harder"*.

Norm-matching the coefficients cannot fix this: the Newton step is invariant under `C -> lambda C`,
so coefficient norm has **no effect on edit size at all**. The paper describes the null as matched in
norm; that description is accurate and irrelevant.

**The fix, precisely.** `scripts/run_dreamer_edit.py` line 6 documents the published edit:

    z <- z - alpha (C(z) - C0) grad C(z) / ||grad C(z)||^2        # magnitude depends on the violation

Its step magnitude is `alpha |C - C0| / ||grad C||`, which varies with how badly each candidate
constraint is violated — and is unchanged by rescaling `C`. Replace it with a **fixed step along the
normal**, so arms differ only in direction:

    u    = grad C / ||grad C||                                    # unit normal
    step = eps * sign(C - C0) * u                                 # magnitude is eps for every arm
    z    = z - step

and add an **equal-norm tangent control**: a random direction with its normal component removed
(`r <- r - (r.u) u`, renormalised to `eps`), which cannot change `C` to first order and so separates
"moving along this direction" from "moving at all".

**The result survives and gets stronger.** At matched magnitude the recovered direction improves
drift up to `-51%` while random directions *worsen* it `+17%` and tangent steps `+8%`, with 0/20
random directions beating recovered at any `eps`. Specificity goes from **2/3 seeds to 3/3**.

The same commit fixes a genuine crash in `_secular`: `np.arange(n, float)` reads the float as *stop*.

### `f32a6cf`, `55a7a5f` — the tangent control was unseeded *(2026-08-28)*

It drew from a bare `torch.randn_like`, so unlike the random-law null it was **not reproducible**:
re-running gave different numbers, and the recorded `draw` index labelled the run without controlling
it. Now seeded and reset per `eps`, which is what makes the `eps` comparison a comparison of
magnitude rather than of magnitude confounded with direction. `55a7a5f` applies the same fix to E17.

### `6528fee` — `balance.py` conditioning *(2026-08-28)*

A real bug: no feature standardisation and an absolute rather than relative ridge, giving
`cond(T) = 9.9e38`. Only affects you if you use the balance-law extractor.

### `cad6bc1` — the conv-GRU comparison was confounded by its own design *(2026-08-27)*

F4's first run compared architectures under conditions that differed in more than architecture. The
confound is itself reported as a finding. Only affects you if you run the second-architecture arm.

---

## 2. Tier 2 — new results that strengthen the published claims

| commit | date | what it adds |
|---|---|---|
| `c5d35ba` | 08-27 | **E18** — a probe fitted to *true* energy reaches `\|rho\| = 0.9999` yet is `6.7x` less preserved by the model's own transition, and **never repairs**. The control the published paper lacks, and the sharpest single addition. |
| `4421b75`, `75bac91`, `a22f8e8` | 08-27 | **E17** — recovery and repair both transfer to **two degrees of freedom**; survives disjoint evaluation at `-62.6%` on 512 unseen trajectories. The published paper is pendulum-only. |
| `d02bbdc` | 08-27 | **E12c** — the effect survives an interchange **50 steps** into autonomous imagination, closing the dormant-pathway objection. |
| `15163e7` | 08-27 | **E14b** — out of distribution, energy stays decodable (free probe `0.999`) while conservation degrades `155--267x`. |
| `79ce130` | 08-27 | **F4b** — a conv-GRU on **identical data** is `767x` worse; seven times more training moves the median barely. Architecture, not training amount. |
| `7e6ebc3` | 08-27 | **F2 (negative)** — invariant drift is *not* a usable online trust signal; beaten 3/3 by plain latent displacement. |
| `a8fe3b4` | 08-28 | **F5 (negative)** — no control-return benefit; the correction is `0.3%` of the planner's ranking signal. |
| `e2d949d` | 08-28 | **F3 (negative)** — constraints `G(z)=0` are not extractable; a quantity that never varies carries no information. |
| `c9d2cf5` | 08-28 | **F1** — under actuation the model represents the quantity and does not implement the relation it obeys. |

---

## 3. Tier 3 — infrastructure

- `0da2cda`, `2529455` — **provenance stamping** (argv, cwd, git HEAD, dirty flag, python, input sha256)
  written into every run record. `2529455` fixes a silent degradation: `inputs_from_args` only sees
  paths that reach the argparse namespace, so scripts hardcoding their inputs recorded `inputs: {}`
  while *looking* stamped.
- `scripts/verify_paper_numbers.py` — recomputes every number from `runs/` and checks it appears
  **next to the claim it supports**, not merely somewhere in the corpus.
- `scripts/mutate_guards.py` — breaks each guarded claim and asserts the guard notices. A guard that
  never fails is not a guard.
- `scripts/audit_preregs.py`, `audit_provenance.py`, `verify_artifact_manifest.py` — sweep for
  registered predictions with no verdict, artefacts with no input record, and manifest hashes that
  do not match their files.

---

## 4. Not included, deliberately

Commits from `00eb797` (2026-08-28) onward are the **F6--F11 extension line**, built around the claim
that the model learns its simulator's integrator. **That claim is unsupported** — the attribution to
the model rather than to the data it was measured on is untested, and F7/F7b/F9/F10 were withdrawn as
invalid by construction (the two integrators compared reduce to the same position recurrence and
differ only in a velocity labelling the pixels never show). Do not adopt those results.

`docs/EXECUTION_LOG.md` and `docs/REVIEWER_HANDOFF.md` §0 record all of it.

---

## 5. How to adopt

**There is no patch series, because there is nothing to patch.** Every file involved
(`latent_noether/fit_cache.py`, `pixel_readout.py`, `provenance.py`, `balance.py`,
`gru_world_model.py`, and `scripts/run_e1_direction_matched_null.py`) is new since `20fa8b4`. I
generated a `git am` series first and tested it against a clean worktree at the published state; it
failed on the first patch, which is how this section got rewritten.

Two honest options:

**A. Fix in place.** Apply the step-size change and the tangent control from §1 to your own
`scripts/run_dreamer_edit.py`. It is a few lines, it needs nothing else from this repository, and it
is enough to correct the published specificity claim. **This is the minimum.**

**B. Adopt the replacement pipeline.** Take these files wholesale — they have no dependency on the
extension line:

    latent_noether/fit_cache.py          latent_noether/pixel_readout.py
    latent_noether/provenance.py         scripts/run_e1_direction_matched_null.py

then run it as `run_e1_direction_matched_null.py --ckpt <ckpt> --data <npz> --horizon 100`.

**Distribution.** This repository is private, so for anyone outside it to adopt either option the
material has to reach the public repository. The `pre-push` hook blocks that deliberately, so it must
be an explicit act:

    git push --no-verify origin <branch>

The minimum honest step is option A, because the published paper's specificity comparison is
unmatched without it.
