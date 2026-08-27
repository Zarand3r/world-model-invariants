"""The fit cache must be exact and must be keyed on content, not on paths.

A path-keyed cache would silently serve a stale invariant after a dataset regeneration or a
retrain -- the failure mode M29's provenance recording exists to prevent. These tests pin both
properties.
"""
import numpy as np
import torch

from latent_noether.fit_cache import _key, cached_fit
from latent_noether.hamiltonian_select import fit_hamiltonian_pair


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(12, 40, 6)), rng.normal(size=(12, 40, 6)) * 0.01


def test_cached_fit_matches_uncached(tmp_path):
    Z, F = _toy()
    got = cached_fit(Z, F, 3, 4, cache_dir=tmp_path)
    ref = fit_hamiltonian_pair(torch.tensor(Z), torch.tensor(F), degree=3, n_basis=4)
    assert np.allclose(np.asarray(got["coeffs"]), np.asarray(ref["coeffs"]), atol=1e-10)
    assert abs(got["residual"] - ref["residual"]) < 1e-12


def test_hit_is_identical_to_miss(tmp_path):
    Z, F = _toy()
    a = cached_fit(Z, F, 3, 4, cache_dir=tmp_path)
    b = cached_fit(Z, F, 3, 4, cache_dir=tmp_path)
    assert np.array_equal(np.asarray(a["coeffs"]), np.asarray(b["coeffs"]))


def test_key_changes_with_data_params_and_flow():
    Z, F = _toy()
    base = _key(Z, F, 3, 4)
    Z2 = Z.copy(); Z2[0, 0, 0] += 1e-9
    F2 = F.copy(); F2[0, 0, 0] += 1e-9
    assert _key(Z2, F, 3, 4) != base, "a perturbed Z must miss the cache"
    assert _key(Z, F2, 3, 4) != base, "a perturbed F must miss the cache"
    assert _key(Z, F, 4, 4) != base, "a changed degree must miss the cache"
    assert _key(Z, F, 3, 8) != base, "a changed n_basis must miss the cache"
