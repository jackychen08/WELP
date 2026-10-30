"""Paper Chignolin residual distance network."""
import os
import torch
from torch import nn

def _default_chignolin_feature_stats():
    """Path to common_files/chignolin_feature_stats.pt, resolved from the shared root."""
    root = (os.environ.get("COMMITTOR_SHARED_ROOT")
            or os.environ.get("WEST_SIM_ROOT")
            or os.getcwd())
    return os.path.join(root, "common_files", "chignolin_feature_stats.pt")

class ChignolinPredictNetResidual(nn.Module):
    """Chignolin committor net: Grant's chignolin featurization on our residual trunk.

    Features = C(166, 2) = 13695 PLAIN pairwise distances over the 166 peptide
    atoms (Grant, JCTC 2024 sec 9.4: "pairwise distances between all atoms in the
    miniprotein (d = 13695)"). Unlike AIB9PredictNetResidual, chignolin uses NO
    dihedral features -- so this is that class with the sin/cos dihedral block
    removed; the residual trunk, the fixed feature standardization, the logit
    stash (required for the logsigmoid loss), and the BKE contract are identical.
    """
    N_ATOMS = 166
    INPUT_DIM = 13695                     # = C(166, 2), plain distances only

    def __init__(self, n_hidden=100, n_residual_layers=3, leak=0.01):
        super().__init__()
        self.input_dim = self.INPUT_DIM
        self.in_proj = nn.Linear(self.INPUT_DIM, n_hidden)
        self.activ = nn.LeakyReLU(negative_slope=leak)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(n_hidden, n_hidden) for _ in range(n_residual_layers)]
        )
        self.out_proj = nn.Linear(n_hidden, 1)
        ii, jj = torch.tril_indices(self.N_ATOMS, self.N_ATOMS, offset=-1)
        self.register_buffer("_tri_i", ii)
        self.register_buffer("_tri_j", jj)

        # Fixed feature standardization from the FROZEN boundary ensemble (both
        # basins), precomputed by make_chignolin_feature_stats.py. Same rationale
        # as AIB9: without it the huge distance L1 saturates the sigmoid at init.
        stats_path = (os.environ.get("CHIGNOLIN_FEATURE_STATS")
                      or _default_chignolin_feature_stats())
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Chignolin feature stats not found at {stats_path}. Training WILL "
                f"diverge without standardization -- regenerate with "
                f"make_chignolin_feature_stats.py."
            )
        _st = torch.load(stats_path, map_location="cpu")
        if _st["mu"].numel() != self.INPUT_DIM:
            raise ValueError(
                f"{stats_path}: feature stats have dim {_st['mu'].numel()} but "
                f"INPUT_DIM is {self.INPUT_DIM} -- stale stats file, regenerate it."
            )
        self.register_buffer("_mu", _st["mu"].float())
        self.register_buffer("_sd", _st["sd"].float())
        self._last_logit = None

    def _reshape_positions(self, positions):
        if positions.dim() == 2:
            return positions.view(positions.shape[0], self.N_ATOMS, 3)
        return positions

    def _distances(self, pos):
        """(B, N, 3) -> (B, 13695) plain pairwise distance d_ij (nm), lower triangular."""
        D2 = (pos.unsqueeze(2) - pos.unsqueeze(1)).square().sum(-1)
        return D2[:, self._tri_i, self._tri_j].clamp_min(1e-8).sqrt()

    def _raw_features(self, pos):
        """(B, 13695): plain pairwise distances only (no dihedrals for chignolin)."""
        return self._distances(pos)

    def compute_features(self, positions):
        """Standardized features: (raw - mu) / sd with FIXED mu/sd."""
        positions = positions.to(next(self.parameters()).dtype)
        pos = self._reshape_positions(positions)
        return (self._raw_features(pos) - self._mu) / self._sd

    def logit_from_features(self, features):
        h = self.activ(self.in_proj(features))
        for layer in self.hidden_layers:
            h = h + self.activ(layer(h))
        return self.out_proj(h).squeeze(-1)

    def forward_from_features(self, features):
        out = self.logit_from_features(features)
        self._last_logit = out           # REQUIRED for train.py's logsigmoid loss
        q_mean = torch.sigmoid(out)
        q_var = torch.zeros_like(q_mean)
        return q_mean, q_var

    def forward(self, positions):
        return self.forward_from_features(self.compute_features(positions))

    def forward_mean(self, positions):
        return self.forward(positions)[0]

def build_committor_net(cfg, system=None):
    if cfg.network_type != "chignolin_residual":
        raise ValueError("Only chignolin_residual is included")
    return ChignolinPredictNetResidual(n_hidden=cfg.grant_n_hidden, n_residual_layers=cfg.grant_n_residual_layers, leak=cfg.grant_leak)
