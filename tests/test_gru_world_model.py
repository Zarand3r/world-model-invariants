"""The GRU world model must satisfy the same interface contract as the DreamerV3 adapter.

The contract that matters is the timing convention: state `k` has consumed `obs[:k]`, so
`readout_from_h(h[k])` is a one-step-ahead PREDICTION, not an autoencoding. Getting this wrong
would make every downstream number measure reconstruction rather than dynamics, and it is invisible
from the numbers alone -- which is why the DreamerV3 adapter has the same test.
"""
import torch

from latent_noether.gru_world_model import ConvGRUWorldModel


def _model(dev="cpu"):
    torch.manual_seed(0)
    return ConvGRUWorldModel(device=dev).to(dev).eval()


def test_interface_shapes():
    m = _model()
    obs = torch.randn(2, 5, 64, 64, 3) * 0.1
    h = m.encode(obs)
    assert h.shape == (2, 5, m.deter)
    assert m.transition(h[:, -1]).shape == (2, m.deter)
    assert m.readout_from_h(h[:, -1]).shape == (2, 64, 64, 3)


def test_state_k_has_not_yet_seen_obs_k():
    """h[0] must be the zero prior: it has consumed nothing, so it cannot depend on obs[0]."""
    m = _model()
    a = torch.randn(1, 3, 64, 64, 3) * 0.1
    b = a.clone(); b[:, 0] += 5.0                      # change obs[0] only
    ha, hb = m.encode(a), m.encode(b)
    assert torch.equal(ha[:, 0], hb[:, 0]), "h[0] depends on obs[0]; the timing convention is wrong"
    assert not torch.allclose(ha[:, 1], hb[:, 1]), "h[1] ignores obs[0]; the recurrence is broken"


def test_transition_is_deterministic_and_pure():
    m = _model()
    torch.manual_seed(3)
    h = torch.randn(4, m.deter)
    assert torch.equal(m.transition(h), m.transition(h))
