"""Linear-probe baseline: the statically decodable subspace that Wedge 1 must beat."""
import numpy as np
import torch


def fit_probe(H: np.ndarray, targets: np.ndarray, ridge: float = 0.05):
    """Ridge regression targets ≈ (H − mean) Wᵀ + b. H: (N, D), targets: (N, k).
    ridge is relative to the mean feature variance — without that scaling, directions
    with near-zero variance dominate the decoder's rows and the probe subspace is
    noise. Returns (W (k, D), b (k,), mean R² across target dims)."""
    Hc = H - H.mean(axis=0)
    tc = targets - targets.mean(axis=0)
    gram = Hc.T @ Hc
    gram += ridge * np.trace(gram) / H.shape[1] * np.eye(H.shape[1])
    W = np.linalg.solve(gram, Hc.T @ tc).T
    b = targets.mean(axis=0)
    pred = Hc @ W.T
    ss_res = ((tc - pred) ** 2).sum(axis=0)
    ss_tot = (tc ** 2).sum(axis=0)
    r2 = float((1.0 - ss_res / ss_tot).mean())
    return W, b, r2


def probe_subspace(W: np.ndarray) -> torch.Tensor:
    """Orthonormal basis (D, k) of the probe's row space — 'what a probe reads'."""
    U, _ = torch.linalg.qr(torch.as_tensor(W.T, dtype=torch.get_default_dtype()))
    return U
