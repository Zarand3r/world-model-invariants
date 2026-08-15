"""Jacobian-closed subspace discovery (Wedge 1).

Find an orthonormal basis U minimizing the normalized closure residual
E_t ‖(I − UUᵀ) J_t U‖²_F / E_t ‖J_t U‖²_F — the subspace the model's own transition
approximately preserves. The invariant-subspace objective itself is standard
Koopman/model-reduction math; the contribution is applying it post hoc to a frozen
world model's transition and testing it against the probe subspace.
"""
import torch


def closure_residual(U: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """U: (D, d) orthonormal; J: (B, D, D). Scale-free residual in [0, 1]."""
    JU = J @ U
    leak = JU - U @ (U.T @ JU)
    return (leak ** 2).sum(dim=(-2, -1)).mean() / (JU ** 2).sum(dim=(-2, -1)).mean()


def _spectral_candidates(J_mean: torch.Tensor, d: int) -> list[torch.Tensor]:
    """Candidate subspaces from the mean Jacobian's eigenstructure: group eigenvectors
    into conjugate-closed real units (dim 1 per real eigenvalue, dim 2 per complex
    pair), sort by |λ|, and take every contiguous window totalling dimension d.
    An exactly invariant subspace of every J_t is invariant for the mean, so it
    appears among these candidates — this is what saves the non-convex fit from
    local minima that pure random restarts fall into."""
    evals, evecs = torch.linalg.eig(J_mean)
    order = torch.argsort(evals.abs(), descending=True)
    units, used = [], set()
    for idx in order.tolist():
        if idx in used:
            continue
        used.add(idx)
        lam, v = evals[idx], evecs[:, idx]
        if abs(lam.imag.item()) < 1e-9:
            units.append(v.real.unsqueeze(1))
        else:
            units.append(torch.stack([v.real, v.imag], dim=1))
            used.add(torch.argmin((evals - lam.conj()).abs()).item())
    candidates = []
    for i in range(len(units)):
        cols, dim = [], 0
        for j in range(i, len(units)):
            cols.append(units[j])
            dim += units[j].shape[1]
            if dim >= d:
                break
        if dim == d:
            Q, _ = torch.linalg.qr(torch.cat(cols, dim=1))
            candidates.append(Q)
    return candidates


def fit_closed_subspace(J: torch.Tensor, d: int, n_iters: int = 500, lr: float = 2e-2,
                        n_random_inits: int = 4, seed: int = 0) -> tuple[torch.Tensor, list[float]]:
    """Minimize closure_residual over the Stiefel manifold via QR reparameterization.
    Non-convex: initialize from spectral candidates of the mean Jacobian plus random
    restarts, polish each with Adam, keep the best. Returns (U, winning loss history)."""
    D = J.shape[-1]
    g = torch.Generator().manual_seed(seed)
    inits = _spectral_candidates(J.mean(dim=0), d)
    inits += [torch.linalg.qr(torch.randn(D, d, generator=g))[0] for _ in range(n_random_inits)]
    best_U, best_hist = None, None
    for U0 in inits:
        W = U0.clone().requires_grad_(True)
        opt = torch.optim.Adam([W], lr=lr)
        hist = []
        for _ in range(n_iters):
            U, _ = torch.linalg.qr(W)
            loss = closure_residual(U, J)
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(loss.item())
        if best_hist is None or hist[-1] < best_hist[-1]:
            with torch.no_grad():
                best_U, _ = torch.linalg.qr(W)
            best_hist = hist
    return best_U.detach(), best_hist


def readout_insufficiency(U: torch.Tensor, H: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Fraction of the model's own readout signal lost by projecting states onto span(U).
    H: (N, D) recorded hidden states; W: (D,) or (k, D) readout weights. Model-internal —
    no simulator quantities enter. 0 = the subspace carries everything the readout reads."""
    Wm = W.unsqueeze(0) if W.ndim == 1 else W
    Hc = H - H.mean(dim=0)
    signal = Hc @ Wm.T
    kept = (Hc @ U) @ (U.T @ Wm.T)
    return ((signal - kept) ** 2).sum(-1).mean() / (signal ** 2).sum(-1).mean()


def fit_sufficient_closed_subspace(J: torch.Tensor, H: torch.Tensor, W: torch.Tensor,
                                   d: int, lam: float = 1.0, n_iters: int = 500,
                                   lr: float = 2e-2, n_random_inits: int = 4,
                                   seed: int = 0) -> tuple[torch.Tensor, list[float]]:
    """Stage A + Stage B jointly (proposal §5.1 constraint 1 + §5.2): minimize
    closure_residual + lam · readout_insufficiency. Pure closure alone is degenerate —
    fast-decaying or readout-irrelevant subspaces are exactly invariant too.

    Seeded with the OBSERVABILITY Krylov space span{w, J̄ᵀw, (J̄ᵀ)²w, …}: what the readout
    reads about the future is wᵀT(h) ≈ (J̄ᵀw)ᵀh, so J̄ᵀ (not J̄) propagates readout
    relevance backwards. W: (D,) or (k, D)."""
    D = J.shape[-1]
    g = torch.Generator().manual_seed(seed)
    J_mean = J.mean(dim=0)
    Wm = W.unsqueeze(0) if W.ndim == 1 else W
    krylov, v = [], Wm
    while len(krylov) < d:
        krylov.extend(list(v))
        v = v @ J_mean  # rows of (W J̄) are (J̄ᵀ wᵢ)ᵀ — the observability directions
    inits = [torch.linalg.qr(torch.stack(krylov[:d], dim=1))[0]]
    inits += _spectral_candidates(J_mean, d)
    inits += [torch.linalg.qr(torch.randn(D, d, generator=g))[0] for _ in range(n_random_inits)]
    best_U, best_hist = None, None
    for U0 in inits:
        P = U0.clone().requires_grad_(True)   # NOT named W: would shadow the readout weights
        opt = torch.optim.Adam([P], lr=lr)
        hist = []
        for _ in range(n_iters):
            U, _ = torch.linalg.qr(P)
            loss = closure_residual(U, J) + lam * readout_insufficiency(U, H, W)
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(loss.item())
        if best_hist is None or hist[-1] < best_hist[-1]:
            with torch.no_grad():
                best_U, _ = torch.linalg.qr(P)
            best_hist = hist
    return best_U.detach(), best_hist


def max_principal_angle(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Largest principal angle (radians) between equal-rank subspaces spanned by U, V."""
    s = torch.linalg.svdvals(U.T @ V).clamp(-1.0, 1.0)
    return torch.acos(s.min())
