"""Residual distance network for this WELP example."""
import os
import torch
from torch import nn

class GrantPredictNetResidual(nn.Module):
    """Grant's pairwise-distance MLP + residual hidden trunk.

    Input  : (B, n_atoms*3) flattened ADP positions (WESTPA stores parent_pos
             flattened) OR (B, n_atoms, 3) atomic positions.
    Feature: 45 pairwise distances on 10 selected heavy atoms (no Hs).
    Trunk  : Linear(45 -> n_hidden) -> n_residual_layers x residual update
             ``h = h + LeakyReLU(W h + b)`` -> Linear(n_hidden -> 1) -> sigmoid.

    Exposes ``compute_features`` and ``forward_from_features`` so the BKE
    branch can take ``∂q/∂features`` (mean across the 45 input dims).
    """
    SEL = (0, 4, 5, 6, 8, 10, 14, 15, 16, 18)
    INPUT_DIM = 45
    N_ATOMS = 22

    def __init__(self, n_hidden=100, n_residual_layers=3, leak=0.01):
        super().__init__()
        self.input_dim = self.INPUT_DIM
        self.in_proj = nn.Linear(self.INPUT_DIM, n_hidden)
        self.activ = nn.LeakyReLU(negative_slope=leak)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(n_hidden, n_hidden) for _ in range(n_residual_layers)]
        )
        self.out_proj = nn.Linear(n_hidden, 1)
        self._sel = torch.tensor(self.SEL, dtype=torch.long)

    def _reshape_positions(self, positions):
        """Accept (B, N, 3) or (B, N*3) and return (B, N, 3)."""
        if positions.dim() == 2:
            return positions.view(positions.shape[0], self.N_ATOMS, 3)
        return positions

    def _pairwise_distances(self, positions):
        positions = self._reshape_positions(positions)
        sel = positions[:, self._sel.to(positions.device)]
        D = sel.unsqueeze(2) - sel.unsqueeze(1)
        D2 = D.square().sum(-1)
        ii, jj = torch.tril_indices(10, 10, offset=-1, device=positions.device)
        return D2[:, ii, jj].sqrt()                                  # (B, 45)

    def compute_features(self, positions):
        positions = positions.to(next(self.parameters()).dtype)
        return self._pairwise_distances(positions)

    def forward_logit_from_features(self, features):
        """Pre-sigmoid logit z = log(q/(1-q)); q = sigmoid(z). Exposed for the
        stable log-sigmoid (Grant) loss so no prediction clamp is needed."""
        h = self.activ(self.in_proj(features))
        for layer in self.hidden_layers:
            h = h + self.activ(layer(h))
        return self.out_proj(h).squeeze(-1)

    def forward_from_features(self, features):
        out = self.forward_logit_from_features(features)
        q_mean = torch.sigmoid(out)
        q_var = torch.zeros_like(q_mean)
        return q_mean, q_var

    def forward_logit(self, positions):
        """Pre-sigmoid logit z from raw positions (grant_loss path)."""
        return self.forward_logit_from_features(self.compute_features(positions))

    def forward(self, positions):
        return self.forward_from_features(self.compute_features(positions))

    def forward_mean(self, positions):
        return self.forward(positions)[0]

def build_committor_net(cfg, system=None):
    if cfg.network_type != 'grant_residual':
        raise ValueError("This example includes only grant_residual")
    return GrantPredictNetResidual(n_hidden=cfg.grant_n_hidden, n_residual_layers=cfg.grant_n_residual_layers, leak=cfg.grant_leak)
