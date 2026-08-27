# Pre-registration — E11 (phase rival) and E12 (subspace-illusion check)

**Written 2026-08-27, before any E11 or E12 quantity was computed.** `docs/ROADMAP.md` Phase III.

---

# E11 — is the effect really about phase?

## The rival

Samanta & Behera (arXiv:2608.07189) decompose latent reduced-order-model rollout error and find
**95-98% of it is pure phase error**, correctable offline with one parameter per latent coordinate.
If the intervention is really fixing timing, the physics story is wrong.

## Step 1 — a structural claim that must be checked, not assumed

The primary metric `D_sec` is the **slope of decoded energy over time**. Energy does not depend on
phase, so a rollout with correct energy and wrong phase should show `D_sec` ~ 0. If that holds,
`D_sec` is **phase-invariant by construction** and the rival cannot explain the headline result.

**Registered check:** time-shift real rendered trajectories by `s` in {1, 2, 5, 10} steps and
recompute `D_sec`. Registered prediction: unchanged to within the readout floor (1.0e-03).
**Falsifier:** `D_sec` moves materially with a pure time shift, in which case it is *not*
phase-invariant, this reasoning is void, and E11 must be run as a full competing-correction
comparison.

## Step 2 — what the intervention actually repairs

For eps = 0 and eps = 0.02, on the same rollouts, report:

- **energy drift** (`D_sec`)
- **phase error**: median absolute decoded `theta` error after removing the single best constant
  time-shift per trajectory, so it measures shape mismatch rather than lag
- **lag**: the best-fit shift itself

**Registered prediction:** the intervention reduces energy drift substantially more, in relative
terms, than it reduces phase error or lag. **Falsifier:** phase error falls by as much as or more
than energy drift, in which case the intervention is a timing correction wearing a physics costume.

---

# E12 — is the edited subspace one the model actually uses?

## The rival

Makelov, Lange & Nanda (ICLR 2024, arXiv:2311.17030) show a subspace activation edit can produce the
expected output change through a **dormant pathway the model does not normally use**. Level-set
projection is exactly a subspace intervention, so this is the sharpest available objection to E1 and
E4, and the roadmap lists it.

## Registered test — on-pathway evidence from unmodified rollouts

If the `C` direction is a pathway the model genuinely uses, then on **unedited** rollouts the natural
variation in `C` should track the natural variation in decoded physical energy. A dormant subspace
would carry no such relationship until it is edited.

**PRIMARY STATISTIC:** Spearman between `C(z_k)` and decoded energy `E_k` along **unedited**
rollouts, pooled within trajectory then medianed across trajectories.

- **Registered prediction:** strongly positive in absolute value, `|rho| > 0.5`, on all three seeds.
- **Falsifier:** `|rho|` is small on unedited rollouts while the edit still produces large energy
  changes. That is the dormant-pathway signature, and E1/E4 would need reinterpretation.

## Controls

- **Random `C`** (20 draws): registered expectation `|rho|` near 0 on unedited rollouts.
- **Equal-rank random subspace**: already covered by the direction-matched null (0/60), and cited
  rather than re-run.

## Known limitation

A correlation on unedited rollouts shows the direction is *informative* about energy in the model's
own dynamics. It does not by itself prove the forward pass *reads* it. Combined with E4's transfer
correlation (setting `C` moves energy) the two together are much stronger than either alone, but
neither is a proof of mechanism and no such claim will be made.
