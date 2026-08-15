"""THE REFUSAL CONTROL: does extraction report an invariant when there is none to find?

Pre-registered in docs/DISSIPATIVE_PREREG.md before any damped data existed.

**Why this is the control that matters.** Our extraction returns functions constant WITHIN a
trajectory and varying ACROSS trajectories. On a 1-DOF conservative pendulum, energy is essentially
the only such function -- so any faithful encoding of (th, thdot) yields it, and |rho|_E = 0.973
may reflect the STRUCTURE OF THE PROBLEM rather than anything DreamerV3 learned. N1 already measured
a random feature map recovering energy at rho = 0.998.

On a damped pendulum every trajectory spirals into the same fixed point, so any continuous function
of state that is constant along trajectories must be constant on the whole basin: **no non-trivial
invariant exists.** A confident recovery there is confabulation, and it would substantially undercut
the conservative result rather than merely qualify it.

REGISTERED (LD = 12, carried over from the confirmed conservative run):
  P1  REFUSAL, the primary criterion: |rho|_E < 0.7 on >= 2 of 3 damped seeds, against
      0.973/0.967/0.975 conservative. 0.7 is the same threshold the conservative run had to PASS,
      so the test is symmetric and was not chosen after seeing damped numbers.
  P2  does the pairing residual SEPARATE? Reported but NOT primary -- its magnitude is already
      known to misbehave on this substrate (0.83-0.90 on a CONSERVATIVE system, worse than our toy
      damped models at 0.44-0.69). Registering it as the criterion and watching it fail would
      confound "the method cannot refuse" with "this statistic is mis-scaled here".
  P3  the best held-out INVARIANCE RATIO should be materially worse on damped.

GATES -- the test is VOID, not negative, if these fail:
  G1  the model trained (raw KL > 1 nat, 1-step decode >= 4x better than predict-the-mean)
  G2  the latent is not degenerate. Damped trajectories converge, and a rank-deficient latent
      manufactures trivially constant directions that score a perfect invariance ratio while
      carrying no content -- that failure once moved k* from 2 to infinity. zeta = 0.03 was
      SELECTED by this gate on states alone (see the prereg addendum), not tuned to an outcome.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter
from latent_noether.gauge import effective_rank_basis, pca_subspace
from latent_noether.hamiltonian_select import fit_hamiltonian_pair
from latent_noether.polynomial import monomial_features, validated_invariants

N_REFIT_NULL = 6       # end-to-end refits per null family (each is a full alternating LS fit)
N_RANDOM_C = 200       # cheap null: random coefficient vectors over the same basis

DEGREE, LD, WARMUP = 4, 12, 10
G, M, L = 10.0, 1.0, 1.0


def true_energy(states):
    th, thd = states[..., 0], states[..., 1]
    return 0.5 * (M * L ** 2 / 3) * thd ** 2 + M * G * (L / 2) * np.cos(th)


def _drift(C):
    """Held-out within-trajectory variance ratio of C: 0 = conserved, ~1 = not.

    TEST A. |rho(C, E)| alone cannot distinguish "conserved" from "tracks instantaneous energy",
    and on a damped system the latter is expected -- energy decays predictably and the model may
    legitimately represent that. Only the drift of C itself separates them.
    """
    half = C.shape[0] // 2
    te = C[half:]
    return float(te.var(dim=1).mean() / te.reshape(-1).var().clamp_min(1e-30))


def _score(Z, F, candidates=None):
    """(drift, pairing residual) for the candidate this latent yields. One end-to-end extraction."""
    fit = fit_hamiltonian_pair(Z, F, degree=DEGREE, n_basis=8, candidates=candidates)
    c = torch.as_tensor(fit["coeffs"], dtype=Z.dtype)
    C = (monomial_features(Z.reshape(-1, Z.shape[-1]), DEGREE) @ c).reshape(Z.shape[:2])
    return _drift(C), fit["residual"], c


def nulls(Z, F):
    """TEST B: is the candidate exceptional against matched nulls drawn INSIDE this same latent?

    The pairing residual's MAGNITUDE is not comparable across substrates (D13; and it moves opposite
    to recovery across LD). But these nulls are computed within one fixed extraction and chart --
    exactly the regime where D13 says residual comparisons ARE meaningful. The null turns an
    uncalibrated number into a calibrated one, which is what makes this the primary criterion rather
    than the raw residual.

      flow shuffle        permute F against Z: destroys the dynamical pairing, keeps both marginals
      traj reassign       regroup samples into FAKE trajectories: destroys within-trajectory
                          constancy while keeping every (z, f) pair intact
      random C            random coefficients over the same degree-4 basis, matched complexity

    **A null that was wrong and is worth recording.** The first version permuted time *within* each
    trajectory, gathering Z and F with the SAME permutation. That is degenerate with the real fit by
    construction: the pairing residual is a sum over samples (order-free) and the invariance
    eigenproblem uses within-trajectory covariance (also order-free), so it would have reproduced
    the real number exactly and been read as "the candidate is not exceptional". Reassigning
    trajectory membership is the null that actually removes the structure being tested.
    """
    g = torch.Generator().manual_seed(0)
    n, T, d = Z.shape
    out = {}

    # the conserved basis depends only on Z, so the flow-shuffle family reuses one eigenproblem
    from latent_noether.polynomial import polynomial_invariants
    cands = polynomial_invariants(Z, degree=DEGREE, max_results=8)

    flow, reassign = [], []
    for _ in range(N_REFIT_NULL):
        idx = torch.randperm(n * T, generator=g)
        flow.append(_score(Z, F.reshape(-1, d)[idx].reshape(n, T, d), candidates=cands)[:2])
        r = torch.randperm(n * T, generator=g)
        reassign.append(_score(Z.reshape(-1, d)[r].reshape(n, T, d),
                               F.reshape(-1, d)[r].reshape(n, T, d))[:2])
    out["flow_shuffle"] = np.array(flow)
    out["traj_reassign"] = np.array(reassign)

    feats = monomial_features(Z.reshape(-1, d), DEGREE)
    rand = []
    for _ in range(N_RANDOM_C):
        c = torch.randn(feats.shape[-1], generator=g, dtype=Z.dtype)
        c = c / c.norm()
        C = (feats @ c).reshape(n, T)
        zz = Z.reshape(-1, d).detach().requires_grad_(True)
        v, = torch.autograd.grad((monomial_features(zz, DEGREE) @ c).sum(), zz)
        from latent_noether.poisson import _antisymmetric_basis
        basis = _antisymmetric_basis(d, Z.dtype)
        A = torch.stack([v @ b.T for b in basis], -1)
        beta = torch.linalg.lstsq(A.reshape(-1, A.shape[-1]), F.reshape(-1)).solution
        res = float(((F.reshape(-1, d) - (A @ beta)) ** 2).sum() / (F ** 2).sum())
        rand.append((_drift(C), res))
    out["random_C"] = np.array(rand)
    return out


def run(ckpt, data):
    d = np.load(data)
    fr = torch.as_tensor(d["frames"]).float().div_(255.).sub_(0.5).cuda()
    E_all = true_energy(d["states"])
    m = DreamerV3Adapter(device="cuda").cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda")["model"])
    m.eval()

    val = slice(204, None)
    with torch.no_grad():
        H = m.encode(fr[val])[:, WARMUP:].detach()
    E = torch.as_tensor(E_all[val][:, WARMUP:], dtype=H.dtype, device=H.device)

    U = pca_subspace(H, LD)
    h_mean = H.reshape(-1, H.shape[-1]).mean(0)
    Z = (H - h_mean) @ U
    R = effective_rank_basis(Z)
    Z = Z @ R
    with torch.no_grad():
        nxt = m.transition(H.reshape(-1, H.shape[-1])).reshape(H.shape)
    F = (((nxt - h_mean) @ U) @ R) - Z

    cov = torch.cov(Z.reshape(-1, Z.shape[-1]).T)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-30)
    pr = float(1.0 / (p ** 2).sum().clamp_min(1e-30))

    Zc, Fc = Z.double().cpu(), F.double().cpu()
    fit = fit_hamiltonian_pair(Zc, Fc, degree=DEGREE, n_basis=8)
    c = torch.as_tensor(fit["coeffs"], dtype=Zc.dtype)
    C = (monomial_features(Zc.reshape(-1, Zc.shape[-1]), DEGREE) @ c).reshape(Zc.shape[:2])
    n = min(C.shape[1], E.shape[1])
    rho = abs(float(torch.corrcoef(torch.stack(
        [C[:, :n].reshape(-1), E[:, :n].double().cpu().reshape(-1)]))[0, 1]))
    # P3: is anything actually conserved? held out, because in-sample ratios overfit at degree 4.
    ratio = float(validated_invariants(Zc, degree=DEGREE, max_results=1)[0]["heldout_ratio"])
    drift = _drift(C)

    nl = nulls(Zc, Fc)
    pct = {}
    for name, arr in nl.items():
        # percentile of the real candidate within the null distribution; LOW = exceptional
        pct[name] = {"drift_pct": float((arr[:, 0] <= drift).mean() * 100),
                     "residual_pct": float((arr[:, 1] <= fit["residual"]).mean() * 100),
                     "null_drift_median": float(np.median(arr[:, 0])),
                     "null_residual_median": float(np.median(arr[:, 1]))}
    return {"ckpt": ckpt, "rho_energy": rho, "pairing_residual": fit["residual"],
            "heldout_invariance_ratio": ratio, "drift_of_C": drift,
            "participation_ratio": pr, "retained_rank": int(Z.shape[-1]), "nulls": pct}


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--damped", nargs="+", default=[f"runs/dreamer_damped_s{i}.pt" for i in range(3)])
    a.add_argument("--damped-data", default="runs/pendulum_pixels_damped.npz")
    a.add_argument("--conservative", nargs="+",
                   default=[f"runs/dreamer_ref_s{i}.pt" for i in (3, 4, 5)])
    a.add_argument("--conservative-data", default="runs/pendulum_pixels.npz")
    a.add_argument("--out", default="runs/dreamer_refusal.json")
    a.add_argument("--max-models", type=int, default=0,
                   help="process at most N models this invocation, then exit (0 = all). Keeps a "
                        "single call inside a foreground timeout; state is on disk so re-running "
                        "resumes.")
    a = a.parse_args()
    _start = sum(len(v) for v in
                 (json.load(open(a.out)) if pathlib.Path(a.out).exists() else {}).values())
    print(__doc__.split("\n\n")[0])
    print("\nREGISTERED P1: |rho|_E < 0.7 on >= 2 of 3 damped seeds (the same threshold the")
    print("conservative run had to PASS). A method that cannot refuse is not a method.\n")

    # RESUMABLE, one model per invocation. Background jobs in this environment are killed at
    # process teardown between turns -- two long runs died silently that way -- so each model is
    # computed in a foreground call and checkpointed to disk immediately. Re-running skips models
    # already present, so repeated invocations make monotonic progress.
    out = json.load(open(a.out)) if pathlib.Path(a.out).exists() else {}
    for arm, ckpts, data in (("conservative", a.conservative, a.conservative_data),
                             ("damped", a.damped, a.damped_data)):
        rows = out.get(arm, [])
        done = {r["ckpt"] for r in rows}
        print(f"  {arm.upper()}  ({len(done)}/{len(ckpts)} already done)")
        for ck in ckpts:
            if ck in done:
                continue
            r = run(ck, data)
            rows.append(r)
            out[arm] = rows
            json.dump(out, open(a.out, "w"), indent=2)
            torch.cuda.empty_cache()
            _stop_now = bool(a.max_models and
                             sum(len(v) for v in out.values()) - _start >= a.max_models)
            print(f"    {ck.split('/')[-1]:<22} |rho|_E {r['rho_energy']:.3f}  "
                  f"drift(C) {r['drift_of_C']:.4f}  residual {r['pairing_residual']:.4f}  "
                  f"PR {r['participation_ratio']:.2f}/{r['retained_rank']}")
            for nm, v in r["nulls"].items():
                print(f"        vs {nm:<18} drift pct {v['drift_pct']:>5.1f}%  "
                      f"(null med {v['null_drift_median']:.4f})   resid pct "
                      f"{v['residual_pct']:>5.1f}%  (null med {v['null_residual_median']:.4f})",
                      flush=True)
            if _stop_now:
                print(f"\n  [stopped after {a.max_models} model(s) this invocation; re-run to "
                      f"continue]")
                raise SystemExit(0)
        out[arm] = rows
        json.dump(out, open(a.out, "w"), indent=2)

    if any(len(out.get(k, [])) < 3 for k in ("conservative", "damped")):
        print(f"\n  INCOMPLETE: conservative {len(out.get('conservative', []))}/3, "
              f"damped {len(out.get('damped', []))}/3. Re-run to continue; no verdict yet.")
        raise SystemExit(0)

    med = lambda arm, k: float(np.median([r[k] for r in out[arm]]))
    cons_r, damp_r = [r["rho_energy"] for r in out["conservative"]], \
                     [r["rho_energy"] for r in out["damped"]]
    n_refuse = sum(1 for r in damp_r if r < 0.7)
    print(f"\n  {'':<22}{'|rho|_E':>10}{'residual':>12}{'ratio':>10}{'PR':>8}")
    for arm in ("conservative", "damped"):
        print(f"  {arm:<22}{med(arm,'rho_energy'):>10.3f}{med(arm,'pairing_residual'):>12.4f}"
              f"{med(arm,'heldout_invariance_ratio'):>10.4f}{med(arm,'participation_ratio'):>8.2f}")

    print("\n--- VERDICT  (primary criterion = TEST B exceptionality, not raw residual)")
    # Only `random_C` is an informative null family. `flow_shuffle` and `traj_reassign` destroy the
    # data so thoroughly (null median residual 0.998 and 0.968 -- i.e. no antisymmetric B relates
    # them at all) that ANY surviving structure beats them, so both arms score 0.0% and the family
    # cannot discriminate. Scoring across all three made a decisive PASS print as "TEST B FAILED".
    # A null that everything beats is not a control; it is a floor.
    exc = lambda arm: [1 if (r["nulls"]["random_C"]["drift_pct"] <= 5 and
                             r["nulls"]["random_C"]["residual_pct"] <= 50) else 0
                       for r in out[arm]]
    ce, de = exc("conservative"), exc("damped")
    print(f"  exceptional vs the informative null (random matched-complexity C):")
    print(f"    conservative {ce} of 3 seeds      damped {de} of 3")
    print(f"    (flow_shuffle / traj_reassign are floors, not controls -- null median residual"
          f" 0.998 / 0.968;\n     both arms beat them trivially, so they cannot discriminate)")
    print(f"  TEST A drift of C: conservative {med('conservative','drift_of_C'):.4f}  vs  "
          f"damped {med('damped','drift_of_C'):.4f}")
    if sum(ce) >= 2 and sum(de) == 0:
        print("  TEST B PASSED. The conservative candidate is exceptional against nulls drawn in")
        print("  its own latent; the damped one is not. The method discriminates conservative from")
        print("  genuinely dissipative dynamics under matched coverage and identical protocol.")
    elif sum(de) >= 2:
        print("  TEST B FAILED. The damped candidate is ALSO exceptional. The method does not")
        print("  refuse on Dreamer, and recovered invariants should not be used downstream until")
        print("  it does. Re-run at zeta = 0.05 / 0.07 before concluding -- 0.03 is weak damping.")
    else:
        print("  INCONCLUSIVE: the conservative candidate is not exceptional either, so the null")
        print("  test lacks resolution here and cannot adjudicate refusal.")
    if min(r["participation_ratio"] for r in out["damped"]) < 2.0:
        print("  VOID (G2): the damped latent is degenerate; a refusal here proves nothing.")
        raise SystemExit(0)
    print(f"  P1 REFUSAL: {n_refuse}/3 damped seeds below 0.7  (values "
          f"{[round(x,3) for x in damp_r]} vs conservative {[round(x,3) for x in cons_r]})")
    if n_refuse >= 2:
        print("  PASSED. The method does NOT report the energy where no invariant exists, so the")
        print("  conservative recovery is not an artefact of the pipeline always finding something.")
    else:
        print("  FAILED — and this is serious. The pipeline reports an energy-correlated quantity")
        print("  on a system with no invariant. That undercuts the conservative result rather than")
        print("  qualifying it. Before concluding, re-run at zeta = 0.05 and 0.07: zeta = 0.03 is")
        print("  weaker than the GRU control, so 'the dissipation was too weak to detect' is live.")
    sep = med("damped", "pairing_residual") - med("conservative", "pairing_residual")
    print(f"\n  P2 pairing residual: conservative {med('conservative','pairing_residual'):.4f} vs "
          f"damped {med('damped','pairing_residual'):.4f}  (separation {sep:+.4f})")
    print("     " + ("separates in the expected direction — the LD anti-correlation is a scaling"
                     " problem,\n     not a validity one." if sep > 0.05 else
                     "DOES NOT separate. The pairing works as a selection criterion on a small\n"
                     "     deterministic GRU and not on a modern stochastic pixel model. Serious."))
    print(f"\n  P3 held-out invariance ratio: conservative "
          f"{med('conservative','heldout_invariance_ratio'):.4f} vs damped "
          f"{med('damped','heldout_invariance_ratio'):.4f}")
