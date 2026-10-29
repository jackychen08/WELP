"""2D committor network.

Architecture matches Grant's sc_committor_rates/2D/TC.py:CommittorNet
(Linear(2, 50) -> Tanh -> Linear(50, 2)), so both methods compare models
of equivalent capacity. We add a second output head for variance estimation
(softplus) matching the (q_mean, q_var) API the training loop expects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyXYNet(nn.Module):
    """Linear(2, H) -> Tanh -> Linear(H, 2). Outputs (q_mean, q_var)."""

    def __init__(self, n_hidden: int = 50):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 2),
        )

    def forward(self, xy: torch.Tensor):
        out = self.block(xy)
        q_mean = torch.sigmoid(out[..., 0])
        q_var = F.softplus(out[..., 1])
        return q_mean, q_var

    def forward_mean(self, xy: torch.Tensor) -> torch.Tensor:
        q_mean, _ = self.forward(xy)
        return q_mean


def build_committor_net(cfg):
    return ToyXYNet(n_hidden=cfg.n_hidden)
