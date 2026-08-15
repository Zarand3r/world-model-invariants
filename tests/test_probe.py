"""The probe baseline must decode a linearly embedded state and its subspace must
span the embedding's row space (that is exactly what 'static decodability' gives you)."""
import numpy as np
import torch

from latent_noether.closure import max_principal_angle
from latent_noether.probe import fit_probe, probe_subspace

D, k, N = 24, 2, 4096


def _embedded_state(seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((N, k))
    M = rng.standard_normal((D, k))
    H = z @ M.T + noise * rng.standard_normal((N, D))
    return H, z, M


def test_probe_decodes_embedded_state():
    H, z, _ = _embedded_state()
    W, b, r2 = fit_probe(H, z)
    assert W.shape == (k, D)
    assert r2 > 0.98


def test_probe_subspace_matches_embedding():
    H, z, M = _embedded_state()
    W, _, _ = fit_probe(H, z)
    U = probe_subspace(W)
    assert U.shape == (D, k)
    M_basis, _ = torch.linalg.qr(torch.as_tensor(M, dtype=torch.get_default_dtype()))
    assert max_principal_angle(U, M_basis).item() < 0.1
