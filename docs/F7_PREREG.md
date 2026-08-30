# F7 --- Does the recovered coefficient identify the *scheme*, or only the *timestep*?

**Registered 2026-08-29, before running anything.**

## Why this exists

The paper is titled *World models learn their simulator's integrator*, and the word "integrator"
appears 17 times. What F6 actually varied was the **timestep**, at one fixed scheme (semi-implicit
Euler). The functional form `theta_dot sin(theta)` and the coefficient `(dt/2) mg(l/2)` were both
derived from that scheme with no free parameters, and the model's optimum matched --- but no
alternative scheme was ever trained, so "the model would recover a *different* quantity under a
different integrator" is **asserted, not measured**.

A reviewer asks this first. Before spending training compute on F7 proper, this gate asks the
cheaper question that decides whether F7 is even well-posed:

> At a **fixed** timestep, do different integrators predict **different** `c*`?

If they do not, then `c*` is a fingerprint of `dt` alone, F7 has nothing to discriminate, and the
paper's language must narrow from *integrator* to *timestep*. This gate can therefore falsify the
paper's own title, which is why it runs first.

## Design

Pure ground truth --- no world model, no training, deterministic given the seed. At each
`dt` in {0.02, 0.035, 0.05, 0.08}, integrate the same pendulum from the same initial conditions
under three schemes:

| scheme | symplectic | leading local error | shadow prediction for the `theta_dot sin(theta)` term |
|---|---|---|---|
| **SI** semi-implicit Euler (what the paper's data uses) | yes | `O(dt^2)` | `c* = (dt/2) mg(l/2)` |
| **VV** velocity Verlet | yes | `O(dt^3)` | no `O(dt)` term, so `c* ~ 0` |
| **EE** explicit Euler | **no** | `O(dt^2)` | none --- energy grows secularly, no nearby conserved quantity |

Sweep the same family `C_c = E + c * theta_dot sin(theta)` used everywhere else in the paper and
minimise the same `invariance_ratio` F6's physics arm uses, so the numbers are directly comparable.

## Registered predictions

- **G0 (positive control).** SI's argmin sits at `r = c/c* = 1` at all four timesteps. This
  re-derives F6's physics arm; if it fails, the harness is wrong and nothing else here is readable.
- **G1 (scheme separation).** Velocity Verlet's argmin satisfies `|c_VV| <= 0.25 * c_SI` at every
  timestep --- i.e. at the *same* `dt`, the two symplectic schemes want visibly different
  coefficients.
- **G2 (non-symplectic has no answer).** At `dt = 0.05`, explicit Euler's *best* ratio over the whole
  grid is at least `5x` worse than semi-implicit Euler's best. No choice of `c` makes it conserve.

## Gate

- **G1 and G2 both pass** -> `c*` is scheme-discriminating at fixed `dt`. F7 proper (train a model on
  a second scheme at the same `dt`, ask which `c*` it recovers) is well-posed and decisive, and the
  title's claim is supportable in principle.
- **G1 fails** -> `c*` identifies the timestep only. F7 is pointless, and the paper is **narrowed**:
  *integrator* becomes *discretisation timestep* throughout, including the title. Recorded as a
  negative regardless of how much rewriting it costs.

## What this gate does *not* establish

Even a clean pass shows only that the *measurement* can tell schemes apart on ground truth. It says
nothing about whether a **model trained on pixels** tracks the scheme it was trained under. That is
F7 proper and needs training. This gate decides only whether that experiment is worth running.

---

# F7 proper --- train on a second integrator at the *same* timestep

**Registered 2026-08-29, after Gate 0 passed and before generating any data or training anything.**

Gate 0 passed: at `dt = 0.05`, semi-implicit Euler wants `c = 0.125` and velocity Verlet wants
`c = 0.000`, with explicit Euler diverging. So the instrument discriminates schemes on ground truth.
This is the model-side test.

## Design

Everything is held identical to F6's `dt = 0.05` arm except the integrator: same renderer, same
initial-condition ranges, same 256 x 120 trajectories, same clip rejection, same seeds (3, 4, 5),
same 6,500 training steps, same checkpoint, same analysis. Only the state update changes, from
gymnasium's semi-implicit Euler to velocity Verlet.

**The timestep is held fixed at 0.05.** That is the whole point: "the model merely learned `dt`"
cannot explain any difference, because `dt` does not differ.

## Power (computed on ground truth before registering, from `runs/f7_gate0.json`)

At `dt = 0.05` the two schemes give near mirror-image sweeps:

| data from | ratio at `r = 0` | ratio at `r = 1` | separation |
|---|---|---|---|
| semi-implicit Euler | 3.36e-02 | 5.33e-05 | **631x** favouring `r = 1` |
| velocity Verlet | 8.63e-05 | 3.32e-02 | **384x** favouring `r = 0` |

So the ground-truth contrast is two-to-three orders of magnitude in *both* directions. Whether the
**model** resolves it is the open question --- F6's semi-implicit models showed a 5.7x separation at
this timestep, well short of the 631x available, so the model tracks a fraction of the signal, not
all of it.

## Registered predictions

- **P1 (primary).** Verlet-trained models put their argmin closer to `0` than to `1`:
  `|r_argmin| <= 0.5` on at least **2 of 3** seeds.
- **P2 (contrast).** Median argmin `r` across the three Verlet models is at least **0.5 below** the
  median across F6's three semi-implicit models at the same timestep (which was `1.0`).
- **P3 (model-quality control).** The Verlet models pass the same acceptance checks F6's did ---
  1-step decode MSE ratio below 0.05, finite rollout, rollout pixel std in F6's observed range. A
  model that failed to train is uninformative and must not be read as a scheme effect.

## Falsifier --- and what it costs

If the Verlet-trained models put argmin at `r ~ 1` (that is, `c ~ 0.125`) on 2 of 3 seeds, then the
recovered coefficient tracks the **timestep regardless of the scheme**. The word *integrator* would
then be unearned, and the paper narrows to *discretisation timestep* throughout --- **including the
title**. That is recorded as the outcome whatever it costs the paper.

This is the experiment most likely to kill the paper's headline framing, which is why it runs.

## Interpretation if P1 passes

Verlet's shadow in this family is plain energy, so a Verlet-trained model recovering `r ~ 0` means it
conserves **textbook energy** --- the very quantity the semi-implicit models were shown *not* to
conserve. The claim becomes: *the model conserves what its simulator conserves*, and the paper's
central dissociation is a property of the **scheme**, not a fixed fact about world models.
