"""Properties the library must hold that no single module's tests cover.

Both cases here are defects that reached the repository: a silently-NaN comparison, and two copies
of one function that disagreed on dtype and crashed every float64 caller (D39).
"""
import torch, pytest
from latent_noether.frequency_action import curve_agreement
from latent_noether.poisson import _antisymmetric_basis
from latent_noether.hamiltonian_select import _antisym_basis

def test_curve_agreement_refuses_more_bins_than_trajectories():
    """Empty bins make `median()` return NaN and the disagreement metric silently meaningless."""
    c = {"invariant": torch.linspace(0, 1, 5), "period": torch.linspace(1, 2, 5)}
    with pytest.raises(ValueError, match="bins"):
        curve_agreement(c, c, n_bins=6)

def test_antisymmetric_basis_has_one_implementation():
    """Two copies disagreed on dtype and crashed every float64 caller (D39)."""
    assert _antisym_basis is _antisymmetric_basis
    for dt in (torch.float32, torch.float64):
        B = _antisymmetric_basis(4, dt)
        assert B.dtype == dt and B.shape == (6, 4, 4)
        torch.testing.assert_close(B, -B.transpose(-1, -2))
