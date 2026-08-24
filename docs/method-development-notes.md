> Method-development notes for the paper *Correcting a learned physical invariant improves
> world-model rollouts*. These preliminary experiments on smaller recurrent models motivated the
> untrained, dissipative and intervention controls used in the paper. They are unpublished and are
> **not** evidence for the DreamerV3 results; the paper says so and does not depend on them.

# Earlier recurrent-model study

The extraction method was developed on a smaller recurrent model before we applied it to DreamerV3.
This study is unpublished and does not support the DreamerV3 claim directly. We include it to
document which findings motivated the controls in the main paper and which findings failed to
transfer.

## Setup and extraction

We trained a single-layer gated recurrent unit with 48-64 hidden units, about 13k parameters, to
predict the next observation in low-dimensional mechanical systems. The model received position but
not velocity, so its recurrent state had to infer velocity from the sequence. We studied two
conservative systems, an anharmonic central-force system and a pair of coupled anharmonic
oscillators, and matched dissipative variants. The extraction procedure never used physical labels.

Let `z` denote a low-dimensional representation of the hidden state and `f` the latent flow induced
by the trained transition. We jointly searched for a scalar `C` and an antisymmetric field `B`
satisfying `f ~ B C`. Because `B` is antisymmetric, ` C^ B  C = 0`, so a
good fit implies that `C` changes little along the flow. We score the fit with the normalized
residual

```
R = {\|f - B C\|^2}{\|f\|^2}.
```

## Results

A randomly initialized recurrent network already contained a latent scalar with
`|| ~ 0.998` correlation with true energy. A high-dimensional random representation of a
low-dimensional trajectory can preserve enough state information for a flexible readout to
reconstruct energy. This result motivated the untrained control in the DreamerV3 study.

With the joint criterion, the recovered scalar reached `|_E| >= 0.972` on conservative models,
sometimes approaching `0.999`. The residual separated conservative and dissipative models in a
representative comparison (`p = 5.4x10^{-6}`, `d = 3.10`). During free rollout, correcting the
hidden state to keep `C` near its initial value reduced trajectory error by up to 68% and
true-energy drift by 69%; the same intervention increased error on dissipative models.

About 77% of rollout error lay along the true physical flow, while only 14% changed the conserved
quantity. The model therefore tended to stay near the correct orbit while moving along it at the
wrong rate. For a nonlinear oscillator, frequency depends on the conserved quantity, so a small
energy error can create a persistent frequency error that accumulates into phase error. We tested
the predictor

```
X(t) = _0^t (E_s)\, E(s)\,ds,  (E) = {d}{dE},
```

against generic functions of time on held-out trajectories.

predictor | central force | coupled oscillators
|--|--|--|
`a + bt` | `0.124` | `-0.028`
`a + bt^p` (best `p`) | `0.131` | `-0.024`
`a + b\, E\,ds` | `{0.658}` | `{0.358}`
 plus an oscillatory correction | `0.703` | --
 plus an angular-momentum term | `0.657` | --
fitted coefficient `b` | `0.91` | `0.67`

The fitted coefficient has roughly the scale predicted by the mechanism, and `` comes from the
simulator rather than from a fit. Adding angular momentum does not improve the central-force
predictor, and the coupled system has no rotational symmetry, so a scalar-energy account is
sufficient for these errors. A one-time intervention on the invariant also changed later phase drift
with the predicted sign in 94% of cases, with held-out `R^2 = 0.887`.

## Negative results

**The invariant gradient is not the physical energy gradient.** Matching perturbations so
that each direction produces the same instantaneous change in true energy does not make their
downstream effects equal. Perturbing along ` C` changes later phase drift the most per unit
energy change, a random direction less, and a direction tangent to `C` the least, on every seed. The
size of the gap depends on how the coefficient is estimated, so we report only the ordering.
Separately, ` C` changes true energy only `1.08` times more per unit displacement than a random
direction. Recovering the right scalar does not imply that the surrounding latent geometry matches
the physical state space.

**The learned model breaks a physical symmetry.** For a rotationally symmetric central-force
system, radial frequency must be even in angular momentum: `_r(E,L) = _r(E,-L)`. At fixed
energy, the simulator obeys this constraint. `|L|` alone predicts radial frequency with
`R^2 = 0.89`, while signed `L` adds no explanatory power (`R^2 = 0.00`). The learned model violates
the symmetry: signed `L` predicts its radial frequency with `R^2 = 0.30`. Recovering an energy-like
scalar therefore does not imply that the model has recovered the full governing law.

**Structural explanations that did not hold.** A low-dimensional part of the hidden state
carried most of the information used to decode future observations, which suggested a quotient
interpretation. The corresponding divergence statistic, however, separated conservative from
dissipative systems at only `{AUC} = 0.68`, so we treat the observation as descriptive rather
than explanatory. A subspace selected by transition-closure criteria also failed to outperform PCA at
matched dimension. Finally, extraction depends on the representation: whitening can destroy recovery,
and restricting the analysis to only the highest-variance directions can remove physical information.

## Comparison with DreamerV3

finding | earlier model | DreamerV3 (this paper)
|--|--|--|
energy-like invariant recoverable | `|_E| >= 0.972` | replicates, `~ 0.97`
untrained networks score high | up to `0.998` | replicates, up to `0.908`
conservative and dissipative models separate | `p = 5.4x10^{-6}` | replicates, no seed overlap
enforcing the invariant improves rollouts | up to 68% | same direction, 2.9-3.5%
residual indicates recovery quality | holds | does not hold
frequency weighting improves the fit | `R^2 = 0.658` vs `0.131` | unresolved at `n = 512`
` C` is the physical energy direction | false | remains unsupported
a correct scalar implies correct geometry | false | remains unsupported

The intervention transfers in direction but not in magnitude, and the percentages are not directly
comparable. The earlier study measures trajectory error in a two-dimensional state space, while this
paper measures pixel error against video. Much of each video frame is static and can remain correct
even when the latent dynamics drift, so the pixel metric naturally compresses the apparent effect
size. Any remaining difference could come from the architecture, system, or extraction dimension.

The flow-generation residual also fails to transfer as a recovery metric. In the earlier recurrent
model it separated the relevant cases; in DreamerV3 it sits near the middle of the random-polynomial
distribution, while conservation provides the stronger signal.
