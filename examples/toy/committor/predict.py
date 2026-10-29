"""Tiny predict helper (committor q from (x, y) points)."""
import os

import numpy as np
import torch

from committor.config import ToyCommittorConfig
from committor.network import build_committor_net


def predict_committor(xy: np.ndarray, sim_root: str = None) -> np.ndarray:
    cfg = ToyCommittorConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_committor_net(cfg).to(device)
    if os.path.exists(cfg.checkpoint):
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["net"])
    net.eval()
    with torch.no_grad():
        t = torch.tensor(xy.astype(np.float32), device=device)
        q = net.forward_mean(t).cpu().numpy()
    return q
