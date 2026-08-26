"""The adapter's `transition` must agree with the reference RSSM's own `img_step`.

docs/REPRODUCE.md calls this out as the check to run before trusting any result, and records the
measured agreement as ~5.1e-08 -- but until now it lived only in a docstring in
`scripts/run_dreamer_extraction.py`, with nothing executing it. Two adapter bugs (a reversed
`get_feat` concatenation, and feeding soft probabilities to networks trained on one-hot samples)
both presented as *model* failure and were caught only by this comparison, so it is worth a test.

The correspondence being asserted. The reference `img_step(prev_state, prev_action)` reads
`prev_state["stoch"]`, feeds `cat(stoch, action)` through `_img_in_layers`, and steps the GRU on
`prev_state["deter"]`. Our `transition(h)` closes that map on `deter` alone by *deriving* the stoch
from `h` via `prior_stoch`. So for the two to be comparable the reference must be handed exactly the
state our adapter implies:

    s_t = {"deter": h, "stoch": prior_stoch(h)}   =>   img_step(s_t, 0)["deter"] == transition(h)

This runs on a randomly-initialised model: the identity is architectural, not learned, so it does
not need a trained checkpoint and is not skipped in a fresh clone.
"""
import numpy as np
import pytest
import torch

from latent_noether.dreamer_adapter import DreamerV3Adapter

TOL = 1e-6      # REPRODUCE.md records 5.1e-08; this leaves room for fp32 nondeterminism


def _adapter(dev):
    torch.manual_seed(0)
    m = DreamerV3Adapter(device=dev).to(dev)
    m.eval()
    return m


@pytest.mark.parametrize("batch", [1, 8])
def test_transition_matches_reference_img_step(batch):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _adapter(dev)
    torch.manual_seed(1234)
    h = torch.randn(batch, m.rssm._deter, device=dev)

    with torch.no_grad():
        ours = m.transition(h)

        stoch = m.prior_stoch(h)                       # the stoch our map implies for this h
        prev = {
            "deter": h,
            "stoch": stoch.reshape(batch, m.stoch, m.discrete),
        }
        action = torch.zeros(batch, 1, device=dev)
        theirs = m.rssm.img_step(prev, action, sample=False)["deter"]

    diff = (ours - theirs).abs().max().item()
    assert diff < TOL, (
        f"adapter.transition disagrees with reference img_step by {diff:.3e} "
        f"(tolerance {TOL:.0e}); REPRODUCE.md records 5.1e-08"
    )


def test_transition_is_deterministic():
    """Two calls on the same `h` must return the same next state.

    `img_step` samples by default; anything stochastic here would make every downstream measurement
    move between identical runs, which is the failure `test_dreamer_encode_is_deterministic`
    documents for the posterior side.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _adapter(dev)
    torch.manual_seed(7)
    h = torch.randn(4, m.rssm._deter, device=dev)
    with torch.no_grad():
        a, b = m.transition(h), m.transition(h)
    assert torch.equal(a, b), f"transition is non-deterministic: max diff {(a - b).abs().max():.3e}"
