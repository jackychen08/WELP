"""Committor config for the toy two-channel WESTPA run.

Stripped-down version of adp_test_mab_ss_velreinit_v3/committor/config.py:
no phi/psi basins, no PDB/top, no graphormer. Basin membership is a
Euclidean distance check in (x, y) space against cached WELL_A / WELL_B.
"""
import os
from dataclasses import dataclass
import numpy as np

@dataclass
class ToyCommittorConfig:
    alpha: float = 72.0
    basin_radius: float = 0.25
    warmup_iters: int = 3
    train_interval: int = 3
    max_iters_ahead: int = 3
    n_hidden: int = 50
    n_layers: int = 3
    dropout: float = 0.0
    network_type: str = 'toy_xy_mlp'
    lr: float = 0.001
    batch_size: int = 256
    n_epochs: int = 80
    var_loss_weight: float = 0.01
    replay_buffer_size: int = 200000
    per_alpha: float = 1.0
    per_beta_start: float = 0.4
    bke_loss_weight: float = 1
    bke_warmup_iters: int = 5
    bke_exclude_boundary: bool = False
    seed: int = 0
    ckpt_warmup_steps: int = 20
    val_interval: int = 0
    val_n_swarms: int = 64
    val_max_steps: int = 4000
    val_seeds_rel: str = 'common_files/val_seeds.npz'
    coordinator_poll_seconds: float = 15.0
    wandb_project: str = 'committor-toy-two-channel'
    wandb_run_name: str = 'welp_toy'
    wandb_entity: str = ''
    data_dir_rel: str = 'output_files/committor_data'
    test_set_rel: str = 'common_files/true_committor.npz'

    def __post_init__(self):
        self.root = os.environ.get('WEST_SIM_ROOT', os.getcwd())
        self.shared_root = os.environ.get('COMMITTOR_SHARED_ROOT', self.root)
        self.seed = int(os.environ.get('SEED', self.seed))
        self.data_dir = os.path.join(self.shared_root, self.data_dir_rel)
        self.common_files = os.path.join(self.shared_root, 'common_files')
        self.checkpoint = os.path.join(self.data_dir, 'committor_net.pt')
        self.cache_file = os.path.join(self.data_dir, 'training_cache.npz')
        self.training_state_file = os.path.join(self.data_dir, 'training_state.json')
        self.test_set_path = os.path.join(self.shared_root, self.test_set_rel)
        self.val_seeds_path = os.path.join(self.shared_root, self.val_seeds_rel)
        self.west_h5 = os.path.join(self.root, 'output_files', 'west.h5')
        self.traj_dir = os.path.join(self.root, 'output_files', 'traj_segs')
        try:
            from common_files.potential import WELL_A, WELL_B
            self.well_a_xy = tuple(WELL_A)
            self.well_b_xy = tuple(WELL_B)
        except Exception:
            self.well_a_xy = (-1.048, 0.0)
            self.well_b_xy = (+1.048, 0.0)
        sim_roots = os.environ.get('COMMITTOR_SIM_ROOTS', '')
        if sim_roots:
            self.sim_sources = [{'label': os.path.basename(r.rstrip('/')), 'root': r, 'west_h5': os.path.join(r, 'output_files', 'west.h5'), 'traj_dir': os.path.join(r, 'output_files', 'traj_segs')} for r in sim_roots.split(':') if r]
        else:
            self.sim_sources = [{'label': os.path.basename(self.root.rstrip('/')) or 'main', 'root': self.root, 'west_h5': self.west_h5, 'traj_dir': self.traj_dir}]
        if os.environ.get('WELP_SMOKE') == '1':
            self.n_epochs = 2
            self.batch_size = 16
            self.coordinator_poll_seconds = 0.2

    def in_basin_a(self, xy) -> np.ndarray:
        xy = np.asarray(xy)
        return ((xy - np.asarray(self.well_a_xy)) ** 2).sum(-1) < self.basin_radius ** 2

    def in_basin_b(self, xy) -> np.ndarray:
        xy = np.asarray(xy)
        return ((xy - np.asarray(self.well_b_xy)) ** 2).sum(-1) < self.basin_radius ** 2
