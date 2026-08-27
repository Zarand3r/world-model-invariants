# Pre-registration — E12b, does invariant drift predict physical energy drift?

**Written 2026-08-27, before any E12b quantity was computed**, and after validating the statistic on
known-answer signals — the standing rule adopted after E3, E2b and E12 each registered a threshold
on a statistic that turned out to lack the range to detect anything.

Replaces nothing: `E11_E12_PREREG`'s E12 test failed and is reported as failed. This is a **new**
test of the same underlying question, not a substitution.

## Statistic validation, done FIRST

Does `Spearman(D_sec(A), D_sec(B))` across trajectories detect coupling between two slowly-drifting
quantities? On synthetic series with known answers, n = 52, T = 100:

| construction | measured | truth |
|---|---|---|
| `B = 2.5A` + small independent noise | **+0.987** | near +1 |
| `B = 2.5A` + large independent drift | **+0.417** | moderate |
| `B` independent of `A` | **-0.025** | 0 |
| `B` coupled, plus strong oscillation | **+0.980** | still high |

The statistic has the range to detect coupling and correctly returns ~0 for independence.

**A prediction of mine that the validation refuted, recorded because it changes the interpretation.**
I expected the endpoint-difference version used in E12's diagnostics to be badly degraded by
oscillation. It is not: on the same coupled-plus-oscillation series it returns **+0.982**. So the
weak endpoint result already measured on real data (+0.356, -0.124, +0.167) is **not obviously an
artefact of a weak statistic**, and may reflect something real. E12b is therefore a genuine test with
a live possibility of a negative outcome, not a rescue of a foregone conclusion.

## The question

The mechanism implied by E1/E2/E2b is: the transition nudges `C` off its level set, and that
accumulated deviation *is* the physical energy error. If so, across trajectories, **how much `C`
drifts should predict how much decoded physical energy drifts.**

## Primary metric

Across the 52 analysis trajectories, on **unedited** rollouts, H = 100:

    rho = Spearman( D_sec(C(z_k)) , D_sec(E_decoded_k) )

- **Registered prediction:** `rho > 0.5`, CI excluding 0, on all three conservative seeds.
- **Falsifier:** `rho` is near 0 or inconsistent in sign across seeds. Then invariant drift and
  physical energy drift are not coupled trajectory-by-trajectory, the mechanism implied by
  E1/E2/E2b is wrong or incomplete, and the intervention's success needs a different account.

The intervention result does **not** depend on this. E1/E4 establish by direct manipulation that
acting on `C` changes physical energy. E12b asks whether the *natural, unforced* relationship holds
in the same direction, which is what the dormant-pathway objection turns on.

## Registered secondary — gain normalisation, fixed now

E4 showed the `C` -> energy gain varies about threefold **across seeds**. If it also varies **across
trajectories**, `dC` and `dE` would correlate weakly even under perfect coupling.

E4a supplies a per-trajectory gain estimate directly: `gain_i = realised_dE_i / intended_dC_i`.
Registered secondary: repeat the primary with `D_sec(C)` multiplied by `gain_i`. Reported alongside
the primary regardless of outcome, and **not** substituted for it.

## Controls

- **Random `C`** (20 draws): registered expectation `rho` near 0.
- **Damped models** (3 seeds): reported descriptively, no registered expectation.

## Splits

Analysis split `204:`, `WARMUP = 10`, H = 100, `C` and the coordinate frame frozen as in E1/E2/E3.
Conservative seeds 3/4/5 and damped seeds 0/1/2 at step 6,500. No new fitting.
