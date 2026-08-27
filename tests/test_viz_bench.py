"""The bench must keep agreeing with the paper.

Two kinds of test here. The first kind is arithmetic and runs anywhere: the claim that re-mixing an
invariant is `G @ a` rather than a refit is the whole reason the sliders are interactive, so it is
pinned against a direct polynomial evaluation. The second kind needs a GPU and the released
checkpoints, and asserts that what the browser displays still equals what the committed run logs
say — the gate every phase of this bench was built against.
"""
import json
import pathlib

import numpy as np
import pytest
import torch

from latent_noether.extraction import Bundle, drift_of_C, rho_energy
from latent_noether.polynomial import monomial_features

ROOT = pathlib.Path(__file__).resolve().parent.parent
CKPT = ROOT / "runs" / "dreamer_ref_s3.pt"
DATA = ROOT / "runs" / "pendulum_pixels.npz"
needs_gpu = pytest.mark.skipif(
    not (CKPT.exists() and DATA.exists() and torch.cuda.is_available()),
    reason="needs the released checkpoints (scripts/fetch_assets.py) and a GPU")


def _toy_bundle(seed: int = 0, n: int = 6, T: int = 9, r: int = 3, k: int = 4, degree: int = 2):
    g = np.random.default_rng(seed)
    Z = g.standard_normal((n, T, r))
    basis = g.standard_normal((k, monomial_features(torch.zeros(1, r), degree).shape[-1]))
    Phi = monomial_features(torch.as_tensor(Z.reshape(-1, r)), degree).numpy()
    return Bundle(
        model_key="toy", ckpt="toy", ld=r, degree=degree, warmup=0, split_start=0,
        h_mean=np.zeros(r), P=np.eye(r), P_pinv=np.eye(r), Z=Z, flow=g.standard_normal((n, T, r)),
        G=Phi @ basis.T, basis_coeffs=basis, weights=g.standard_normal(k),
        energy=g.standard_normal((n, T)), eigenvalues=np.ones(r), participation_ratio=float(r),
        residual=0.0, rank_and_test_residual=0.0)


def test_remixing_equals_a_direct_polynomial_evaluation():
    """C(a) = G @ a must equal evaluating the collapsed coefficients from scratch.

    This identity is what makes the law sliders instant. If it ever drifts, the bench would show
    scores for one invariant while its rollouts enforced another.
    """
    b = _toy_bundle()
    for w in (b.weights, np.array([1.0, 0, 0, 0]), np.array([0.3, -0.7, 0.1, 0.5])):
        direct = (monomial_features(torch.as_tensor(b.Z.reshape(-1, b.Z.shape[-1])), b.degree)
                  .numpy() @ (w @ b.basis_coeffs)).reshape(b.Z.shape[:2])
        assert np.allclose(b.C(w), direct, atol=1e-10)


def test_drift_uses_the_held_out_half_and_unbiased_variance():
    """Copied from `run_dreamer_refusal._drift`; the two conventions move it by ~6%."""
    b = _toy_bundle()
    C = b.C()
    te = C[C.shape[0] // 2:]
    want = float(np.mean(np.var(te, axis=1, ddof=1))) / float(np.var(te.ravel(), ddof=1))
    assert drift_of_C(b) == pytest.approx(want, rel=1e-12)
    # and it must NOT equal the all-trajectories, biased-variance version
    naive = float(np.mean(np.var(C, axis=1))) / float(np.var(C))
    assert abs(drift_of_C(b) - naive) > 1e-12


def test_rho_energy_is_symmetric_in_sign():
    """The paper reports |rho|, so flipping the sign of C must not change the score."""
    b = _toy_bundle()
    assert rho_energy(b, b.weights) == pytest.approx(rho_energy(b, -b.weights), rel=1e-12)


def test_bundle_survives_the_disk_cache():
    from viz.server import bundles
    b = _toy_bundle()
    tmp = bundles.CACHE / "__test_roundtrip"
    np.savez_compressed(tmp.with_suffix(".npz"),
                        **{n: getattr(b, n) for n in bundles._ARRAYS})
    z = np.load(tmp.with_suffix(".npz"))
    for n in bundles._ARRAYS:
        assert np.allclose(z[n], getattr(b, n)), n
    tmp.with_suffix(".npz").unlink()


@needs_gpu
def test_bench_reproduces_the_published_extraction():
    """|rho|_E, the pairing residual and the participation ratio, against the committed log."""
    from viz.server import bundles
    want = {r["ckpt"].split("/")[-1]: r
            for r in json.loads((ROOT / "runs" / "dreamer_extraction_prereg_ld12.json").read_text())
            }["dreamer_ref_s3.pt"]
    b = bundles.build("dreamer_ref_s3", 12, 4)
    assert rho_energy(b) == pytest.approx(want["rho_energy"], abs=1e-6)
    assert b.residual == pytest.approx(want["pairing_residual"], abs=1e-6)
    assert b.participation_ratio == pytest.approx(want["participation_ratio"], rel=1e-6)


@needs_gpu
def test_bench_reproduces_the_published_dose_response():
    """The alpha grid the browser draws must be the one `run_dreamer_edit.py` published."""
    from viz.server import bundles, registry, rollout
    want = {r["ckpt"].split("/")[-1]: r
            for r in json.loads((ROOT / "runs" / "dreamer_edit.json").read_text())
            ["A_conservative_own"]}["dreamer_ref_s3.pt"]
    alphas = (0.0, 0.05, 0.1, 0.2, 0.4)
    b = bundles.build("dreamer_ref_s3", 12, 4)
    m = registry.get("dreamer_ref_s3")
    with registry.GPU:
        r = rollout.imagine(m, b, None, 50, alphas, keep_frames=False)
    got = np.asarray(r["mse_by_alpha"], dtype=np.float64)
    for a, v in zip(alphas, got):
        assert v == pytest.approx(want["rollout_by_alpha"][str(a)], rel=1e-4), f"alpha {a}"
    change = float((got[-1] - got[0]) / got[0])
    assert change == pytest.approx(want["relative_change_at_max_alpha"], rel=1e-3)


def test_rollout_rejects_an_out_of_range_trajectory_before_touching_the_gpu():
    """The bug this guards is not a 500; it is an unrecoverable process.

    An out-of-range index reached a CUDA kernel and fired a device-side assert, which poisons the
    whole CUDA context: every later request failed while the page kept returning 200, so the bench
    looked alive and was dead. `max` on a number input is advisory, so two keystrokes reached it.
    """
    from viz.server import rollout
    b = _toy_bundle()
    n = b.Z.shape[0]
    with pytest.raises(ValueError, match="out of range"):
        rollout.check_request(b, [n], 5, (0.0,))
    with pytest.raises(ValueError, match="out of range"):
        rollout.check_request(b, [-1], 5, (0.0,))
    assert rollout.check_request(b, None, 5, (0.0,)) == list(range(n))


def test_rollout_rejects_a_horizon_past_the_available_frames():
    """The UI offered 200 against 110 real frames; the mismatch surfaced as a tensor-size error."""
    from viz.server import rollout
    b = _toy_bundle()
    h = rollout.max_horizon(b)
    assert h == b.Z.shape[1]
    with pytest.raises(ValueError, match="horizon must be"):
        rollout.check_request(b, [0], h + 1, (0.0,))
    with pytest.raises(ValueError, match="horizon must be"):
        rollout.check_request(b, [0], 0, (0.0,))


def test_rollout_rejects_unusable_alphas():
    from viz.server import rollout
    b = _toy_bundle()
    with pytest.raises(ValueError, match="finite"):
        rollout.check_request(b, [0], 5, (float("nan"),))
    with pytest.raises(ValueError, match="at most"):
        rollout.check_request(b, [0], 5, (99.0,))
    rollout.check_request(b, [0], 5, (-0.4,))          # anti-correction control, allowed


def test_law_rejects_weights_that_do_not_describe_an_invariant():
    from viz.server import law
    b = _toy_bundle()
    k = b.basis_coeffs.shape[0]
    with pytest.raises(ValueError, match="expected"):
        law.check_weights(b, np.ones(k + 1))
    with pytest.raises(ValueError, match="finite"):
        law.check_weights(b, np.full(k, np.nan))
    with pytest.raises(ValueError, match="all zero"):
        law.check_weights(b, np.zeros(k))


def test_scores_do_not_depend_on_the_scale_or_sign_of_C():
    """Why `scores` no longer normalises: there was nothing for the normalisation to do."""
    from viz.server import law
    b = _toy_bundle()
    base = law.scores(b, b.weights)
    for s in (3.7, 0.02, -1.0):
        got = law.scores(b, b.weights * s)
        for k in ("rho_energy", "drift_of_C", "pairing_residual"):
            assert got[k] == pytest.approx(base[k], rel=1e-10), k


def test_gradients_are_cached_on_the_bundle_not_on_its_id():
    """id(bundle) is recycled aggressively — 3000 short-lived Bundles reused 7 ids — so an
    id-keyed cache serves one bundle's gradients for another once the first is collected."""
    from viz.server import law
    a, b = _toy_bundle(seed=1), _toy_bundle(seed=2)
    ga, gb = law.basis_grads(a), law.basis_grads(b)
    assert not np.allclose(ga, gb)
    assert a.basis_grads is ga and b.basis_grads is gb
    assert law.basis_grads(a) is ga            # cached, and still a's
