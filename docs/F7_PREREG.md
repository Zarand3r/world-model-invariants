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
