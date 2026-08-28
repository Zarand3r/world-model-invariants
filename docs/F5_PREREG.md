# Pre-registration — F5, does the operator-privileged direction improve CONTROL RETURN?

**Written 2026-08-28, before any F5 quantity exists. NOT YET APPROVED TO RUN.**

## Why this is the experiment that matters

Every result in the paper is diagnostic. It shows that probing over-reports, that a statistic on the
transition operator discriminates, and that the direction it selects is causally deployed inside the
model's imagination. It never shows that anyone is **better off** using it. `limits.tex` says so
outright, and the one practical use tested -- F2, invariant drift as a trust signal -- **failed its
registered falsifier**.

This is the paper's most reviewer-visible gap, and closing it is worth more than any further
diagnostic.

## The obstacle, stated first because it shapes the design

The recovered `C` is a **conserved** quantity, defined for **free** evolution. Under actuation energy
is no longer conserved, so naively enforcing `C = const` during action-conditioned planning would
**cancel the action's own legitimate effect on energy** -- it would fight the task rather than help
it. F1 was meant to supply the actuated object (a balance law) and its extraction is blocked.

**F5 does not need F1 to succeed.** At plan time the agent *knows its own action*, so the source term
does not have to be learned. The decoded readout supplies `thetadot`, the plan supplies `tau`, and
Gate 0 of F1 already established -- on ground truth, with no model -- that the discrete balance
relation closes to **1.7%** using the shadow energy and midpoint power. So the correction can be made
**balance-aware from known quantities**:

    enforce   C(z_{t+1}) - C(z_t)  =  tau_t * thetadot_t * dt      (instead of  = 0)

This is the F1 object supplied rather than discovered, and it is legitimate precisely because a
planner is not label-free about its own actions.

## Task, chosen to be in-distribution and NOT to flatter the method

**Energy targeting.** Reward `r_t = -( E_t - E* )^2 / sigma_E^2`, with `E_t` the decoded physical
energy and `E*` drawn inside the **training band** (per-trajectory mean energy spans `[-2.58, 5.07]`
on `runs/pendulum_actuated.npz`). Episodes start from the training initial-condition distribution.

Two properties make this a hard test rather than a favourable one:

1. The task **requires changing** the quantity the naive correction conserves. A conservation
   correction should therefore *hurt*, and arm (b) below exists to measure exactly that.
2. It stays inside the distribution the world model was trained on, so a failure cannot be blamed on
   distribution shift -- which swing-up would have invited, since the model has never seen the large
   torques swing-up needs.

## Planner

CEM over the learned world model: sample `K` action sequences of horizon `H`, imagine each with
`transition(h, a)`, decode frames with the validated `decode_physics` readout, score cumulative
reward, refit, take the best first action, execute in the real environment, repeat. **No actor-critic
is trained** -- the world model already exists, and CEM plans directly in it.

## Gate 0 -- can this model support planning at all?

**Run before any arm is compared.** Plain imagination (arm a) must beat a random-action policy on
true environment return by a margin excluding zero over 20 episodes.

> **If G0 fails, STOP.** A world model that cannot plan cannot show that a correction helps planning,
> and every arm below would be uninterpretable noise. Report as a negative about the model's planning
> capacity and make no claim about the correction.

## Arms, all sharing the same planner, seeds, episodes and reward

| arm | correction applied during imagination |
|---|---|
| **a** | none |
| **b** | conservation, `Delta C = 0` -- the naive transfer, expected to HURT |
| **c** | **balance-aware**, `Delta C = tau * thetadot * dt`, on the label-free `C` |
| **d** | balance-aware, on the **supervised energy probe** from E18 |
| **e** | balance-aware, on a **magnitude-matched random** direction |

## Registered predictions

**P1 (the headline).** Arm **c** attains higher true return than arm **a**, on at least 2 of 3 model
seeds, with the per-episode difference excluding zero.

**P2 (specificity).** Arm **c** beats arm **e**. Without this, any gain is "editing the latent at all
helps", not "editing *this* direction helps".

**P3 (the E18 contrast, and the reason this is worth running).** Arm **c** beats arm **d**. E18
showed the perfect probe is worse *inside imagination*; P3 asks whether that dissociation survives
contact with a task. **This is the prediction most likely to fail**, and it is the one that would
most change what the paper can claim.

**P4 (the obstacle, measured).** Arm **b** is *worse* than arm **c**, confirming that a conserved
quantity is the wrong object under actuation and that the balance-aware form is what carries the
effect.

## Falsifiers, stated plainly

- **G0 fails** -> stop; no claim about the correction.
- **P1 fails** -> the correction does not help control. **This is a real possibility and it would be
  reported as the headline negative**, in the same terms as F2. The paper would then say that the
  privileged direction is causally deployed inside imagination and *still* confers no control
  benefit, which is a sharper and more useful statement than the current silence.
- **P2 fails** -> any gain is non-specific; report as such.
- **P3 fails** -> the probe-versus-operator dissociation does **not** transfer to task performance.
  The paper's diagnostic claims stand; its practical significance is bounded, and we say so.

## Scope and honesty constraints, fixed now

- 3 model seeds; the model seed is the independent unit, as everywhere else in this project.
- The reward uses decoded physical energy, whose readout error is already characterised
  (2.09% of across-trajectory spread on the actuated data). True return is computed from the
  **simulator's** state, never from the model's own decode, so the model cannot score its own success.
- Planner hyperparameters (`K`, `H`, elites, iterations) are frozen from a single pilot on **seed 3
  only**, before any arm comparison, and reused unchanged for all arms and seeds. No per-arm tuning.
- Physical labels are used for the reward and for evaluation only; the extraction of `C` is unchanged
  and frozen beforehand.
- Records carry the `latent_noether.provenance` stamp.

## Cost

No training. Three existing checkpoints, CEM rollouts only. Estimated 2-4 GPU-hours total.

## Status

**AWAITING APPROVAL.** Registered so the design, its falsifiers and its most likely failure are on
record before any number exists, and so the decision to run it can be made against a concrete
protocol rather than an idea.
