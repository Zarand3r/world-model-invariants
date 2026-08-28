"""A deterministic conv-GRU world model, exposing the same interface as the DreamerV3 adapter.

F4 in `docs/ROADMAP.md`, preregistered in `docs/F4_PREREG.md`. Every result in this project comes
from one architecture; this is the contrast that says whether the recovered invariant is a property
of learned world models or of DreamerV3's particular latent design.

**Deliberately minimal.** Conv encoder -> GRU -> conv decoder, trained on pure reconstruction. No
stochastic latent, no KL, no unimix, no free bits — every mechanism that makes an RSSM an RSSM is
removed. What remains is the least that still counts as a pixel world model.

The recurrent state is 512, matching DreamerV3's `deter`, so the extraction operates on a latent of
the same dimensionality and `LD = 12` means the same thing in both. Parameter count is not matched;
matching it would require changing the architecture, which is the thing under test.

Interface is exactly `encode` / `transition` / `readout_from_h`, so every analysis script runs
unchanged.
"""
import torch
from torch import nn


class ConvGRUWorldModel(nn.Module):
    def __init__(self, image_size: int = 64, deter: int = 512, embed: int = 256,
                 depth: int = 32, device: str = "cuda"):
        super().__init__()
        self.deter = deter
        self.device = device
        c = depth
        self.encoder = nn.Sequential(                      # 64 -> 4
            nn.Conv2d(3, c, 4, 2, 1), nn.SiLU(),           # 32
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.SiLU(),       # 16
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.SiLU(),   # 8
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.SiLU(),   # 4
            nn.Flatten(), nn.Linear(c * 8 * 16, embed),
        )
        self.cell = nn.GRUCell(embed, deter)
        # The autonomous transition needs an input when no observation is available. A learned
        # constant is the deterministic analogue of DreamerV3's prior: the model must carry the
        # dynamics in the recurrent state rather than read them from the frame.
        self.prior_input = nn.Parameter(torch.zeros(embed))
        self.dec_in = nn.Linear(deter, c * 8 * 16)
        self.decoder = nn.Sequential(                      # 4 -> 64
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(c, 3, 4, 2, 1),
        )
        self._depth = c

    # ---- the three methods the extraction requires -------------------------
    def encode(self, obs, deterministic: bool = True):
        """(B, T, H, W, 3) in [-0.5, 0.5] -> (B, T, deter).

        Indexing matches the DreamerV3 adapter: state `k` has consumed `obs[:k]`, so
        `readout_from_h(h[k])` predicts `obs[k]` — a genuine one-step-ahead prediction, not an
        autoencoding. Verified by `tests/test_gru_world_model.py`.
        """
        B, T = obs.shape[:2]
        x = obs.permute(0, 1, 4, 2, 3).reshape(B * T, 3, obs.shape[2], obs.shape[3])
        e = self.encoder(x).reshape(B, T, -1)
        h = torch.zeros(B, self.deter, device=obs.device, dtype=obs.dtype)
        outs = []
        for t in range(T):
            outs.append(h)                    # state BEFORE consuming obs[t]
            h = self.cell(e[:, t], h)
        return torch.stack(outs, 1)

    def transition(self, h):
        """One autonomous step: a pure, differentiable function of `h` alone."""
        return self.cell(self.prior_input.expand(h.shape[0], -1).to(h.dtype), h)

    def readout_from_h(self, h):
        x = self.dec_in(h).reshape(-1, self._depth * 8, 4, 4)
        return self.decoder(x).permute(0, 2, 3, 1)

    # ---- training ----------------------------------------------------------
    def loss(self, batch, open_loop: int = 8, w_open: float = 1.0):
        """Teacher-forced reconstruction PLUS an open-loop rollout term.

        **The open-loop term is not optional, and F4's first run proved it.** Trained on
        teacher-forced reconstruction alone, `prior_input` appears in no loss, so `transition()` --
        the map the entire extraction analyses -- is never trained. Measured on that model: the
        autonomous step was 2.5x larger than a true step, the 50-step rollout changed by 0.00084 per
        frame (essentially frozen), and the recovered scalar's one-step conservation defect was
        ~10,000x worse than DreamerV3's while its energy correlation was *better* (0.97).

        DreamerV3 trains its prior through the `kl_dyn` term. This is the deterministic analogue:
        encode a prefix, then roll autonomously and require the decoded frames to match. Without it
        the architecture comparison measures "a model whose transition was never trained" rather
        than a second world-model family.
        """
        h = self.encode(batch)
        pred = self.readout_from_h(h.reshape(-1, self.deter)).reshape(batch.shape)
        rec = ((pred - batch) ** 2).mean()
        if open_loop <= 0:
            return rec
        T = batch.shape[1]
        start = max(1, T - open_loop)
        hh = h[:, start - 1]
        outs = []
        for _ in range(T - start):
            hh = self.transition(hh)
            outs.append(self.readout_from_h(hh))
        roll = torch.stack(outs, 1)
        return rec + w_open * ((roll - batch[:, start:]) ** 2).mean()
