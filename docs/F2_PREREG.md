# Pre-registration — F2, invariant drift as an online trust signal

**Written 2026-08-28, before any F2 quantity was computed.** `docs/ROADMAP.md` Phase V.

## Why this is the experiment that matters for significance

Everything established so far is diagnostic: the model has a conserved quantity, it violates it, and
correcting it helps. None of it tells a practitioner anything they can act on. F2 asks the usable
question:

> **At rollout time, with no access to ground truth, does accumulated invariant drift predict that
> the model's imagination is about to become unreliable — better than the signals already available?**

If yes, the work supplies a **physical trust horizon**: stop trusting autonomous imagination when
learned physical consistency crosses a calibrated threshold. That is a concrete deliverable and it is
what would move this from an interpretability finding to a usable one.

## Design

At an early point in an autonomous rollout (`k = 25`), compute each candidate signal from
**information available at inference only** — no ground truth, no future frames. Then measure the
model's actual physical error much later (`k = 100`), and ask which early signal predicts it.

**Target (ground truth, used only for scoring):** absolute error of decoded physical energy against
the true trajectory's energy, at `k = 100`.

**Candidate signals, all computable online:**

| signal | rationale |
|---|---|
| **accumulated invariant drift** `|C(z_k) - C(z_0)|` | ours |
| **instantaneous drift** `|C(T(z_k)) - C(z_k)|` | the local defect, no accumulation |
| latent displacement `||z_k - z_0||` | does anything that moves predict failure? |
| whitened NN distance to observation-conditioned latents | off-manifold detection, the standard alternative |
| **ensemble disagreement** across the 3 independently trained seeds | the standard practitioner's answer |

## Primary metric

**Spearman correlation between each early signal at `k = 25` and the decoded physical energy error at
`k = 100`, across trajectories.**

- **Registered prediction:** accumulated invariant drift attains a **higher** absolute Spearman than
  every baseline, on all three models.
- **Falsifier:** any baseline matches or beats it on a majority of models. Then invariant drift is not
  the best available online signal, the practical claim is unsupported, and the paper should not make
  it. Ensemble disagreement in particular is cheap and widely used; if it wins, that is the honest
  finding.

Reported alongside, not decisive: the same at `k = 50 -> 150`, to check the result is not specific to
one lead time; and the AUROC of each signal for the binary event *"final energy error in the worst
quartile"*, since a practitioner needs a decision rule rather than a correlation.

## Controls

**Random-constraint drift**, 10 draws: accumulated `|C_rand(z_k) - C_rand(z_0)|`. Registered
expectation near zero. This separates "invariant drift predicts failure" from "any latent quantity
drifting predicts failure", and is the same control that has carried every other claim here.

## Scope and limits

Pendulum seeds 3/4/5 at step 6,500, analysis split, `C` and the coordinate frame frozen.

The ensemble baseline is favoured by this design: it uses **three trained models** while every other
signal uses one. If invariant drift wins anyway, it wins against a baseline with more information.

A correlation is not a deployed trust monitor. F2 establishes whether the signal carries the
information, not that a threshold on it is well calibrated across systems, which is untested.
