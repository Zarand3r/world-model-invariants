"""Content-addressed cache for `fit_hamiltonian_pair`.

The invariant fit is a generalized eigenproblem over 1819 degree-4 monomials in 12 dimensions. It
costs ~21 s. Every null-arm sweep recomputes it once per random draw even though the fit depends
only on `(Z, F)` and the fit parameters -- the random coefficients are drawn *afterwards*. Measured
on seed 3: 21.2 s fit against 0.2 s for the rollout it feeds, so a 20-draw arm spent ~95% of its
time re-deriving one identical answer.

**Keyed on content, not on paths.** The key is a hash of the actual `Z` and `F` arrays plus the fit
parameters, so a regenerated dataset, a retrained checkpoint, or a changed extraction dimension all
miss the cache automatically. A path-keyed cache would silently serve a stale fit after a re-run,
which is exactly the class of error M29's provenance recording exists to prevent.
"""
import hashlib
import pathlib

import numpy as np

from latent_noether.hamiltonian_select import fit_hamiltonian_pair

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "runs" / "fit_cache"


def _key(Z, F, degree, n_basis):
    h = hashlib.sha256()
    for arr in (np.ascontiguousarray(np.asarray(Z, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(F, dtype=np.float64))):
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    h.update(f"deg{degree}|nb{n_basis}|v1".encode())
    return h.hexdigest()[:32]


def cached_fit(Z, F, degree, n_basis, cache_dir=CACHE_DIR):
    """Same contract as `fit_hamiltonian_pair(Z, F, degree=, n_basis=)`, memoised on disk."""
    Zc = Z.detach().cpu().numpy() if hasattr(Z, "detach") else np.asarray(Z)
    Fc = F.detach().cpu().numpy() if hasattr(F, "detach") else np.asarray(F)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{_key(Zc, Fc, degree, n_basis)}.npz"
    if p.exists():
        d = np.load(p)
        return {"coeffs": d["coeffs"], "residual": float(d["residual"])}
    import torch
    Zt = Z if hasattr(Z, "detach") else torch.as_tensor(Zc)
    Ft = F if hasattr(F, "detach") else torch.as_tensor(Fc)
    fit = fit_hamiltonian_pair(Zt, Ft, degree=degree, n_basis=n_basis)
    np.savez(p, coeffs=np.asarray(fit["coeffs"]), residual=float(fit["residual"]))
    return fit
