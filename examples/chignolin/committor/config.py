import os
import numpy as np
from dataclasses import dataclass
from committor.lazzeri_cv import FOLDED_Q_MIN, UNFOLDED_Q_MAX

@dataclass
class CommittorConfig:
    folded_q_min: float = FOLDED_Q_MIN
    unfolded_q_max: float = UNFOLDED_Q_MAX
    basin_a_phi: float = -1.047197
    basin_a_psi: float = -0.698132
    basin_b_phi: float = 1.047197
    basin_b_psi: float = 0.698132
    basin_radius: float = 0.174533
    warmup_iters: int = 1
    train_interval: int = 1
    network_type: str = 'chignolin_residual'
    n_layers: int = 4
    embed_dim: int = 64
    ffn_embed_dim: int = 128
    attention_heads: int = 4
    num_kernel: int = 32
    n_hidden: int = 512
    dropout: float = 0.001
    grant_n_hidden: int = 100
    grant_n_residual_layers: int = 3
    grant_leak: float = 0.01
    lr: float = 0.0001
    batch_size: int = 1024
    n_epochs: int = 120
    per_alpha: float = 1.0
    per_beta_start: float = 0.4
    var_loss_weight: float = 0.0
    n_bootstrap_steps: int = 0
    live_targets: bool = False
    replay_buffer_size: int = 65000
    semigroup_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.5
    bke_loss_weight: float = 0.05
    bke_warmup_iters: int = 3
    bke_anneal_start_iter: int = 500
    bke_anneal_end_iter: int = 1000
    bke_exclude_boundary: bool = True
    boundary_min_per_batch: int = 1
    grant_loss: bool = False
    coordinator_poll_seconds: float = 5.0
    max_drift_iters: int = 5
    drift_check_seconds: float = 5.0
    drift_warn_seconds: float = 600.0
    max_iters_ahead: int = 1
    wandb_project: str = 'committor-westpa'
    wandb_entity: str = ''
    wandb_run_name: str = 'welp_chignolin'
    n_committor_bins: int = 10
    n_rmsd_bins: int = 10
    n_surface_bins: int = 36
    seed: int = 0
    ckpt_warmup_steps: int = 20
    val_interval: int = 0
    val_n_seeds: int = 32
    val_n_swarms: int = 8
    val_max_steps: int = 8000
    val_seeds_rel: str = 'common_files/val_seeds.pt'
    data_dir_rel: str = 'output_files/committor_data'
    test_set_rel: str = 'common_files/aib9_committor_test_set.pt'
    test_set_eval_enabled: bool = False
    rate_ref_rel: str = 'common_files/lazzeri/matched_rate_ref'
    rate_eval_enabled: bool = False

    def __post_init__(self):
        self.root = os.environ.get('WEST_SIM_ROOT', os.getcwd())
        self.shared_root = os.environ.get('COMMITTOR_SHARED_ROOT', self.root)
        self.seed = int(os.environ.get('SEED', self.seed))
        self.data_dir = os.path.join(self.shared_root, self.data_dir_rel)
        self.common_files = os.path.join(self.shared_root, 'common_files', 'lazzeri', 'generated')
        self.pdb_path = os.path.join(self.common_files, 'peptide.pdb')
        self.top_path = os.path.join(self.common_files, 'system.xml')
        self.checkpoint = os.path.join(self.data_dir, 'committor_net.pt')
        self.cache_file = os.path.join(self.data_dir, 'training_cache.npz')
        self.test_set_path = os.path.join(self.shared_root, self.test_set_rel)
        self.rate_ref_dir = os.path.join(self.shared_root, self.rate_ref_rel)
        self.val_seeds_path = os.path.join(self.shared_root, self.val_seeds_rel)
        self.west_h5 = os.path.join(self.root, 'output_files', 'west.h5')
        self.traj_dir = os.path.join(self.root, 'output_files', 'traj_segs')
        sim_roots_env = os.environ.get('COMMITTOR_SIM_ROOTS', '')
        if sim_roots_env:
            self.sim_sources = [{'label': os.path.basename(r.rstrip('/')), 'root': r, 'west_h5': os.path.join(r, 'output_files', 'west.h5'), 'traj_dir': os.path.join(r, 'output_files', 'traj_segs')} for r in sim_roots_env.split(':') if r]
        else:
            self.sim_sources = [{'label': os.path.basename(self.root.rstrip('/')) or 'main', 'root': self.root, 'west_h5': self.west_h5, 'traj_dir': self.traj_dir}]
        if os.environ.get('WELP_SMOKE') == '1':
            self.n_epochs = 2
            self.batch_size = 16
            self.coordinator_poll_seconds = 0.2

    def _basin_dist(self, phi, psi, phi0, psi0):
        """Periodic EUCLIDEAN distance from (phi, psi) to a basin center, in radians.

        This is a DISC, not a per-angle box. It reproduces Grant's
        utils.dihedral_difference (sc_committor_rates/AIB9/utils.py:131-138):
            sqrt(sum((180 - ||a-b| - 180|)^2))
        exactly, just in radians instead of degrees.

        NB: our ADP code used a per-angle BOX -- `(dphi < r) & (dpsi < r)` -- which
        is a DIFFERENT region (a square of half-width r, area 4r^2, vs a disc of
        area pi*r^2: ~27% larger). Nobody noticed on ADP, and a reviewer has already
        flagged the basin-geometry mismatch in the paper. Here it matters
        concretely: our headline AIB9 metric is the MFPT compared against Grant's
        4.7 ns, and basin volume feeds straight into the rate. So we match him.
        """
        dphi = np.pi - np.abs(np.abs(phi - phi0) - np.pi)
        dpsi = np.pi - np.abs(np.abs(psi - psi0) - np.pi)
        return np.sqrt(dphi ** 2 + dpsi ** 2)

    def in_basin_a(self, x, y=None):
        """Folded A: archived Q >= .99 (inclusive); y is unused."""
        return np.asarray(x) >= self.folded_q_min

    def in_basin_b(self, x, y=None):
        """Unfolded B: archived Q <= .01 (inclusive); y is unused."""
        return np.asarray(x) <= self.unfolded_q_max
