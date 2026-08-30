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

---

# F7b --- the sign flip: a third scheme, equally rough, predicting `-c*`

**Registered 2026-08-30, after F7 landed and before generating any data or training anything.**

## The objection this answers

F7 showed Verlet-trained models recover `c = 0` where semi-implicit models recover `c = 0.125`. The
obvious reviewer objection: **the Verlet dataset is smoother.** Its textbook-energy oscillation is
0.006 against semi-implicit's 0.186, so perhaps the model simply latches onto whatever varies least,
and "tracks the integrator" is a story told over a data-difficulty artefact.

That objection cannot be answered by any scheme whose data is smoother. It needs a scheme that is
**exactly as rough** and predicts a **different** coefficient.

## The scheme

Gymnasium's semi-implicit Euler updates velocity first, then position from the *new* velocity.
Reversing the two lines --- position first, then velocity from the *new* position --- is still
symplectic, still first order, and its shadow Hamiltonian carries the same `O(dt)` term with the
**opposite sign**. Verified on ground truth at `dt = 0.05` before registering:

| scheme | argmin `c` | best ratio | separation from `c = 0` |
|---|---|---|---|
| semi-implicit (velocity first) | `+0.1250` | 5.329e-05 | **630.9x** |
| **reversed (position first)** | **`-0.1250`** | 5.333e-05 | **630.1x** |
| velocity Verlet | `0.0000` | 8.635e-05 | 1.0x |

The forward and reversed schemes are **equally difficult by every measure available** --- same best
ratio to three digits, same separation to one part in a thousand. The only difference is the sign,
and it is caused by swapping two lines of simulator code.

## Registered predictions

Three seeds (3, 4, 5) on reversed semi-implicit Euler at `dt = 0.05`; everything else identical to
F6's `dt = 0.05` arm and F7's Verlet arm.

- **P1 (primary).** Reversed-scheme models put their argmin at `r = -1`: `|r_argmin + 1| <= 0.5` on
  at least **2 of 3** seeds.
- **P2 (ordering).** The three arms' median argmin `r` are strictly ordered
  `reversed < Verlet < semi-implicit`, with `reversed <= -0.5` and `semi-implicit >= +0.5`.
- **P3 (model quality).** All **three** registered acceptance criteria, checked and reported
  individually: decode ratio below 0.05, finite rollout, and pixel std inside F6's observed
  `0.0694--0.0699`. (F7's script checked only the first two; that is fixed and this arm uses the
  corrected check.)

## Falsifier

If the reversed-scheme models land at `r ~ +1` --- the same place as the forward scheme --- then the
recovered sign is a property of the **model or the training**, not of the simulator. F7's Verlet
result would then have to be reinterpreted as a data-smoothness effect, and the paper's "learns the
integrator" claim would weaken back to "learns something that co-varies with the integrator".

This is the experiment that can most cheaply destroy F7's interpretation, which is why it runs next.

## What a pass would establish

That the recovered coefficient tracks a **sign flip produced by reordering two lines of the
simulator**, on data that is by every available measure exactly as hard. At that point three schemes
at one timestep predict `+0.125`, `0.000`, `-0.125` and the model reproduces all three, with `dt`
held fixed throughout.

## F7b amendment 1 --- correcting the "equally rough" claim, before any result

**Written 2026-08-30, after generating the dataset and before analysing any model.**

I registered above that the forward and reversed schemes are "exactly as difficult by every measure
available". Generating the dataset shows that is **too strong**, and the record should say so before
results exist rather than after.

| measure | forward (semi-implicit) | reversed | Verlet |
|---|---|---|---|
| ground-truth invariance ratio at its own optimum | 5.329e-05 | 5.333e-05 | 8.635e-05 |
| ground-truth separation from `c = 0` | 630.9x | 630.1x | 1.0x |
| **dataset textbook-`E` relative oscillation** | **0.186** | **0.115** | **0.006** |

On the invariance-ratio measure the experiment actually uses, the two are equal to within 0.1%. On
relative oscillation they are **not**: the reversed dataset is about **1.6x smoother**. My claim of
equality across "every measure available" was wrong.

**Does this damage the design? No, and it is worth being precise about why.** A data-difficulty
confound can explain a change in *magnitude* --- an easier dataset letting the model sit closer to
textbook energy. It cannot explain a change in **sign**. Smoothness has no direction. The registered
prediction is that the coefficient moves to `-0.125`, on the far side of `c = 0` from the forward
scheme, and no amount of "this data is easier" produces that.

So the correct statement of the control is the weaker, sufficient one: the reversed dataset is
**1.6x** from the forward scheme on the roughness measure where Verlet is **31x** away, and it
predicts an outcome that difficulty cannot produce at all. P1 and its falsifier are unchanged.

## F7 amendment 2 --- cross-evaluation control: does the argmin follow the model or the data?

**Registered 2026-08-30, before running it. No training required --- existing checkpoints only.**

The measurement evaluates `rho_obs` under **the model's own learned transition**
(`m.transition`), not under the simulator, so the argmin is a property of the model. But the
**evaluation dataset** still does real work: it supplies the frames that are encoded, the PCA
subspace, and the regression that defines the candidate direction in latent space. A sceptic can
therefore ask whether F7's contrast lives in the readout construction rather than in the model.

The control separates them by crossing checkpoint against evaluation data:

| checkpoint trained on | evaluated on | if argmin follows the **model** | if it follows the **data** |
|---|---|---|---|
| semi-implicit | Verlet data | `r = +1` | `r = 0` |
| Verlet | semi-implicit data | `r = 0` | `r = +1` |

- **P1 (registered).** The argmin follows the **checkpoint** in both off-diagonal cells: the
  semi-implicit models stay at `r = +1` on Verlet data, and the Verlet models stay at `r = 0` on
  semi-implicit data, each on at least **2 of 3** seeds.
- **Falsifier.** If either off-diagonal cell flips to match its evaluation data, the sweep is reading
  the trajectories used to build the readout rather than the model's dynamics, and **F7's
  interpretation collapses** --- the result would be a property of the analysis, not of the model.

Off-diagonal cells are mildly out-of-distribution for the encoder (same initial conditions and
similar images, different integration), so a degradation in absolute `rho_obs` is expected and is not
itself a failure. Only the **argmin location** is registered.

## Amendment 3 --- the same control applied to F6, to size the damage

**Registered 2026-08-30, before running. No training --- existing F6 checkpoints only.**

Amendment 2 falsified F7. F6 uses the same measurement and every F6 model was only ever evaluated on
data from its own timestep, so the confound was invisible there by construction. This control asks
directly whether F6's model arm is affected, because that fact --- not my guess about it --- is what
the decision about re-auditing the paper should rest on.

Cross the `dt = 0.02` and `dt = 0.08` checkpoints (4x apart, F6's widest separation) against both
datasets. The sweep is in relative units `r = c / c*(dt_eval)`, so the two hypotheses predict
different, separable locations:

| checkpoint | eval data | if argmin follows the **model** | if it follows the **data** |
|---|---|---|---|
| `dt = 0.02` | `dt = 0.08` | `r = 0.05/0.20 = 0.25` | `r = 1.0` |
| `dt = 0.08` | `dt = 0.02` | `r = 0.20/0.05 = 4.0` | `r = 1.0` |

`r = 0.25` is on F6's grid. `r = 4.0` is **off** it (the grid stops at 3.0), so in that cell
"follows the model" is registered as **argmin at the grid edge, `r >= 2.0`**, and "follows the data"
as `r = 1.0`. Stated now so the reading is not chosen after seeing the numbers.

- **P1 (registered).** If F6's model arm is sound, the off-diagonal argmins follow the **checkpoint**
  on at least 2 of 3 seeds per cell.
- **Falsifier.** If they follow the evaluation data --- `r = 1.0` in both off-diagonal cells --- then
  F6's model arm is confounded in exactly the way F7's was, and F6's headline claim that *a model
  trained only on pixels recovers a coefficient tracking `(dt/2) mg(l/2)`* is re-deriving its own
  physics arm rather than measuring the model.

I expect the falsifier to fire, given amendment 2. Registering the expectation so that confirming it
counts for nothing extra and refuting it counts fully.

## Amendment 4 --- does E1's repair survive fitting `C` on the wrong integrator?

**Registered 2026-08-30, before running. No training --- existing checkpoints only.**

Yesterday I recorded that E1 and E12c are "immune by construction" to the F7 confound, because they
roll the model forward freely and the data supplies only the initial condition. I also flagged, and
did not test, the one hole in that claim: **the direction `C` is still identified using the falsified
one-step statistic on real frames.** After three checking artefacts this week that were weaker than
the claims they existed to test, leaving my own caveat untested is not acceptable.

E1 already supports the needed separation via `--eval-data` (built for E9): `C`, `h_mean`, `U` and
`R` are frozen from `--data`, and the rollout starts from `--eval-data`'s latents.

Two arms, same semi-implicit-trained checkpoint, same rollout initial conditions:

| arm | `C` fitted on | rollout from |
|---|---|---|
| **M (matched)** | semi-implicit `dt = 0.05` | semi-implicit `dt = 0.05` |
| **X (mismatched)** | **velocity Verlet `dt = 0.05`** | semi-implicit `dt = 0.05` |

Only the data used to *identify* `C` differs. This is a fair test rather than a distribution-shift
artefact for the reason F7's control was clean and F6's was not: at a fixed timestep the two schemes
differ at `O(dt^2)`, and a crossed model scores `rho_obs` 0.00765 against a matched model's 0.00766.

- **P1 (registered).** In arm X the recovered direction still repairs: median `|D_sec|` at its best
  `eps` is below its value at `eps = 0` by at least **half** the relative improvement arm M shows, on
  at least **2 of 3** seeds.
- **Falsifier.** If repair vanishes in arm X, or is indistinguishable from the tangent control there,
  then `C`'s identification carries the effect, the intervention results inherit the F7 confound, and
  **my "immune by construction" claim was wrong** and must be retracted alongside F7.

**E1's hard gate applies unchanged**: improvement in `pixel_mse` without improvement in decoded
physical energy is not repair, and is to be read as a failure rather than a partial success.

I am not stating an expected direction. Last time I registered an expectation ("I expect the
falsifier to fire") I was wrong, and in a way I had not imagined.
