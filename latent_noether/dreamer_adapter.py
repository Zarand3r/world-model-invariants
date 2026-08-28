"""Adapter exposing NM512/dreamerv3-torch's RSSM through the interface our extraction expects.

**Why this file exists.** An earlier attempt reimplemented DreamerV3 from scratch
(`latent_noether/dreamer.py`, ~180 lines). It failed three training runs with total posterior
collapse — posterior entropy 3.465 against a uniform 3.466, posterior output differing between
very different frames by 1e-4. The causes were both details that mature implementations already
get right:

  1. **free bits applied to the scalar mean.** `clamp(kl.mean(), min=1.0)` with a raw KL of 0.0375
     returns a CONSTANT with zero gradient, so the KL term was inert for 20,707 steps — including
     `kl_dyn`, the term that trains the prior. Our `transition()` runs entirely on the prior.
     The reference implementation clips **per element** and uses separate `dyn_scale=0.5` /
     `rep_scale=0.1` rather than a convex combination.
  2. **no unimix.** DreamerV3 mixes 1% uniform into every categorical to stop degenerate
     distributions. Absent from the reimplementation.

Re-deriving a solved problem was the wrong call: our pipeline needs only three methods, which is
an adapter, not a reimplementation. This file is that adapter — ~30 lines of interface over the
reference RSSM, encoder and decoder, at their published defaults.

**The interface, and why `deter` alone is a valid autonomous state.** The reference `img_step`
computes `stoch_t` from `deter_t` (via `_img_out_layers` -> stat layer), then
`deter_{t+1} = GRU(img_in(cat(stoch_t, a_t)), deter_t)`. So the stochastic state is a *function*
of the deterministic one, and the map on `deter` closes:

    T(h) = GRU( img_in( cat( prior_probs(h), 0 ) ), h )

`prior_probs` uses the categorical **probabilities** (with unimix), not `.mode()`. The reference
`get_stoch` calls `.mode()`, which is an argmax and therefore non-differentiable — and we need
Jacobians. Stated explicitly because it is a deliberate deviation from Dreamer's own imagination
rollout, which samples.
"""
import contextlib
import functools
import sys
import pathlib

import torch
from torch import nn

_EXT = pathlib.Path(__file__).resolve().parent.parent / "external" / "dreamerv3-torch"
if str(_EXT) not in sys.path:
    sys.path.insert(0, str(_EXT))

import networks  # noqa: E402  (reference implementation)


class DreamerV3Adapter(nn.Module):
    """Reference DreamerV3 world model, exposing encode / transition / readout_from_h."""

    def __init__(self, image_size: int = 64, deter: int = 512, stoch: int = 32,
                 discrete: int = 32, hidden: int = 512, cnn_depth: int = 32,
                 device: str = "cuda"):
        super().__init__()
        shapes = {"image": (image_size, image_size, 3)}
        self.encoder = networks.MultiEncoder(
            shapes, mlp_keys="$^", cnn_keys="image", act="SiLU", norm=True,
            cnn_depth=cnn_depth, kernel_size=4, minres=4, mlp_layers=5,
            mlp_units=1024, symlog_inputs=True)
        self.rssm = networks.RSSM(
            stoch=stoch, deter=deter, hidden=hidden, rec_depth=1, discrete=discrete,
            act="SiLU", norm=True, mean_act="none", std_act="sigmoid2", min_std=0.1,
            unimix_ratio=0.01, initial="learned", num_actions=1,
            embed=self.encoder.outdim, device=device)
        feat = deter + stoch * discrete
        self.decoder = networks.MultiDecoder(
            feat, shapes, mlp_keys="$^", cnn_keys="image", act="SiLU", norm=True,
            cnn_depth=cnn_depth, kernel_size=4, minres=4, mlp_layers=5, mlp_units=1024,
            cnn_sigmoid=False, image_dist="mse", vector_dist="symlog_mse", outscale=1.0)
        self.deter, self.stoch, self.discrete = deter, stoch, discrete
        self._device = device

    # ---- the three methods the extraction pipeline needs -------------------------------

    def prior_stoch(self, h, straight_through: bool = True):
        """Prior stochastic state at `h`: ONE-HOT forward, differentiable backward.

        Two bugs lived here and both cost a failed acceptance run (2026-08-10):

        1. **Probabilities instead of one-hot.** The decoder and `_img_in_layers` are trained on
           SAMPLED one-hot stochastic states. Feeding soft probabilities is a distribution shift
           the network never saw, and it degraded 1-step decoding by ~700x (0.00002 -> 0.01373).
           Straight-through gives the exact one-hot value forward while keeping the gradient, so
           we get the trained regime AND the Jacobians the extraction needs.
        2. See `readout_from_h` for the concatenation-order bug.
        """
        x = self.rssm._img_out_layers(h)
        stats = self.rssm._suff_stats_layer("ims", x)
        probs = self.rssm.get_dist(stats).mean                    # unimix already applied
        idx = probs.argmax(-1)
        oh = torch.nn.functional.one_hot(idx, self.discrete).to(probs.dtype)
        z = (oh + probs - probs.detach()) if straight_through else oh
        return z.reshape(h.shape[0], self.stoch * self.discrete)

    def transition(self, h, a=None):
        """One autonomous step. With `a=None` this is free evolution, exactly as before.

        ACTION CONVENTION, read off `RSSM.img_step(prev_state, prev_action)`: the action passed here
        is the one applied *at* the current state `h`, producing the next state. That is the same
        indexing the F1 data uses (`actions[t]` is the torque applied at state `t`), so no shift is
        needed here -- unlike `encode`, which does need one. Getting this backwards would make F1
        measure the model's response to the wrong action, so the two conventions are stated
        explicitly rather than inferred at each call site.
        """
        z = self.prior_stoch(h)
        if a is None:
            a = torch.zeros(h.shape[0], 1, dtype=h.dtype, device=h.device)   # free evolution
        else:
            a = a.reshape(h.shape[0], -1).to(dtype=h.dtype, device=h.device)
        x = self.rssm._img_in_layers(torch.cat([z, a], -1))
        _, deter = self.rssm._cell(x, [h])
        return deter[0]

    def readout_from_h(self, h):
        """Decode an image from `h` alone, using the prior."""
        # ORDER MATTERS: the reference `get_feat` returns cat([stoch, deter]) — stoch FIRST.
        # An earlier version concatenated cat([deter, stoch]) and produced garbage decodes.
        # The reference MultiDecoder also expects (B, T, feat) and permutes a 5-D output, so a
        # singleton time axis is added and stripped.
        feat = torch.cat([self.prior_stoch(h), h], -1).unsqueeze(1)
        return self.decoder(feat)["image"].mode().squeeze(1)

    @contextlib.contextmanager
    def _deterministic_posterior(self):
        """Make `rssm.observe` take the posterior MODE instead of sampling it.

        The reference `obs_step` samples the categorical posterior and `observe` does not expose the
        flag, so every call to `encode` returned a different `h`. Measured cost: repeated identical
        measurement runs gave Spearman 0.522 / 0.524 / 0.531 and R^2 drifting 0.062 -> 0.022. Results
        that move between identical runs cannot be reported.

        Taking the mode is the CONSISTENT choice, not merely a convenient one: `prior_stoch` already
        uses an argmax (see its docstring — the pipeline needs determinism and Jacobians), and the
        reference's own `get_stoch` returns `dist.mode()`. Sampling here was the odd one out.
        """
        orig = self.rssm.obs_step
        self.rssm.obs_step = functools.partial(orig, sample=False)
        try:
            yield
        finally:
            self.rssm.obs_step = orig

    @staticmethod
    def _rssm_actions(actions, B, T, device, dtype):
        """Data convention -> RSSM convention.

        `RSSM.observe` scans `obs_step(prev_state, prev_act, embed[t], is_first[t])`, so its
        `action[:, t]` is the action that led INTO state `t`. The F1 data stores `actions[t]` as the
        torque applied AT state `t`. The two differ by one step, so shift and zero-pad the first.
        `obs_step` already zeroes `prev_action` wherever `is_first`, so the pad is consistent.
        """
        if actions is None:
            return torch.zeros(B, T, 1, device=device, dtype=dtype)
        a = actions.reshape(B, T, -1).to(device=device, dtype=dtype)
        return torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)

    def encode(self, obs, actions=None, deterministic: bool = True):
        """Teacher-forced pass. obs: (B, T, H, W, 3) in [-0.5, 0.5] -> deter states (B, T, deter).

        INDEXING, verified empirically 2026-08-11 (an earlier docstring said obs[:k+1] and was
        wrong): state `k` has consumed `obs[:k]`, so `readout_from_h(h[k])` predicts `obs[k]` —
        a genuine one-step-ahead prediction. Measured on the trained model:
        readout(h[19]) vs obs[19] = 1.8e-05, vs obs[18] = 3.7e-03, vs obs[17] = 5.7e-03.
        """
        B, T = obs.shape[:2]
        embed = self.encoder({"image": obs})
        action = self._rssm_actions(actions, B, T, obs.device, obs.dtype)
        is_first = torch.zeros(B, T, device=obs.device, dtype=obs.dtype)
        is_first[:, 0] = 1.0
        ctx = self._deterministic_posterior() if deterministic else contextlib.nullcontext()
        with ctx:
            post, _ = self.rssm.observe(embed, action, is_first)
        return post["deter"]

    # ---- training ---------------------------------------------------------------------

    def loss(self, obs, actions=None, kl_free: float = 1.0, dyn_scale: float = 0.5, rep_scale: float = 0.1):
        """Reference DreamerV3 world-model loss at published defaults."""
        B, T = obs.shape[:2]
        embed = self.encoder({"image": obs})
        action = self._rssm_actions(actions, B, T, obs.device, obs.dtype)
        is_first = torch.zeros(B, T, device=obs.device, dtype=obs.dtype)
        is_first[:, 0] = 1.0
        post, prior = self.rssm.observe(embed, action, is_first)
        kl_loss, kl_value, _, _ = self.rssm.kl_loss(post, prior, kl_free, dyn_scale, rep_scale)
        feat = self.rssm.get_feat(post)
        recon = -self.decoder(feat)["image"].log_prob(obs).mean()
        return recon, kl_loss.mean(), float(kl_value.mean())
