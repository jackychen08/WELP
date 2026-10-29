"""WESTPA plugin: committor q inference for the toy run.

Mirrors adp_test_mab_ss_velreinit_v3/committor/driver.py but:
  - reads seg.npy instead of seg.dcd / seg.xml
  - inputs are (11, 2) (x, y) arrays, not (11, 22, 3) atomic positions
  - no system.pdb, no mdtraj
Training happens in a separate coordinator process; this driver only
re-reads the checkpoint and overwrites pcoord column 0 (q).

Training lockstep: pre_we blocks until the coordinator's
training_state.json shows last_trained[this_sim] >= n_iter - max_iters_ahead
so the sim cannot race ahead of the NN. Gating is skipped until the
coordinator flips warmup_complete.
"""
import json
import logging
import os
import time

import numpy as np
import torch
import westpa

from committor.config import ToyCommittorConfig
from committor.network import build_committor_net

log = logging.getLogger(__name__)


class CommittorDriver:

    def __init__(self, sim_manager, plugin_config):
        westpa.rc.pstatus("Initializing ToyCommittorDriver plugin (inference-only)")
        if not sim_manager.work_manager.is_master:
            return

        self.sim_manager = sim_manager
        self.cfg = ToyCommittorConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(plugin_config.get("batch_size", 4096))
        self.net = build_committor_net(self.cfg).to(self.device)
        self._ckpt_mtime = None
        self._maybe_reload_checkpoint()
        self.net.eval()

        # Identify this sim for the lockstep gate: matches the label the
        # coordinator uses (basename of $WEST_SIM_ROOT).
        self.sim_label = os.path.basename(self.cfg.root.rstrip("/")) or "main"
        self.gate_poll_seconds = float(plugin_config.get("gate_poll_seconds", 1.0))
        westpa.rc.pstatus(
            f"ToyCommittorDriver ready on {self.device} (sim root {self.cfg.root}, "
            f"label={self.sim_label}, max_iters_ahead={self.cfg.max_iters_ahead})"
        )

        priority = int(plugin_config.get("priority", 0))
        sim_manager.register_callback(sim_manager.pre_we, self.pre_we, priority)

    def _read_training_state(self):
        path = self.cfg.training_state_file
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def _wait_for_training(self, n_iter):
        """Block until last_trained[self.sim_label] >= n_iter - max_iters_ahead.

        No-op until the coordinator has flipped warmup_complete. If the state
        file is missing we assume the coordinator isn't running and let the
        iteration proceed (avoids hard-deadlocking a standalone sim).
        """
        logged_wait = False
        while True:
            state = self._read_training_state()
            if state is None:
                return  # coordinator not writing state — don't gate
            if not state.get("warmup_complete", False):
                return  # warmup: let sims fill the buffer freely
            last = state.get("last_trained", {}).get(self.sim_label, 0)
            gap = n_iter - last
            if gap <= self.cfg.max_iters_ahead:
                if logged_wait:
                    westpa.rc.pstatus(
                        f"ToyCommittorDriver: gate opened at iter={n_iter} "
                        f"(last_trained[{self.sim_label}]={last})"
                    )
                return
            if not logged_wait:
                westpa.rc.pstatus(
                    f"ToyCommittorDriver: waiting at iter={n_iter} — "
                    f"last_trained[{self.sim_label}]={last} "
                    f"(gap={gap} > max_iters_ahead={self.cfg.max_iters_ahead})"
                )
                logged_wait = True
            time.sleep(self.gate_poll_seconds)

    def _maybe_reload_checkpoint(self):
        path = self.cfg.checkpoint
        if not os.path.exists(path):
            return
        mtime = os.path.getmtime(path)
        if mtime == self._ckpt_mtime:
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        self.net.eval()
        self._ckpt_mtime = mtime
        westpa.rc.pstatus(f"ToyCommittorDriver reloaded checkpoint (mtime={mtime})")

    def pre_we(self):
        n_iter = self.sim_manager.n_iter
        self._wait_for_training(n_iter)
        self._maybe_reload_checkpoint()
        segments = list(self.sim_manager.segments.values())
        log.info("ToyCommittorDriver: computing q for %d segments (iter %d)",
                 len(segments), n_iter)

        all_positions = []
        seg_frame_counts = []

        for segment in segments:
            seg_dir = os.path.join(
                self.cfg.traj_dir, f"{n_iter:06d}", f"{segment.seg_id:06d}",
            )
            try:
                traj = np.load(os.path.join(seg_dir, "seg.npy")).astype(np.float32)  # (11, 2)
                all_positions.append(traj)
                seg_frame_counts.append(len(traj))
            except Exception as exc:
                log.warning("ToyCommittorDriver: failed to load seg %d: %s",
                            segment.seg_id, exc)
                seg_frame_counts.append(0)

        if all_positions:
            all_pos = np.concatenate(all_positions, axis=0)
            all_q = np.empty(len(all_pos), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(all_pos), self.batch_size):
                    end = min(start + self.batch_size, len(all_pos))
                    pos_t = torch.tensor(all_pos[start:end], device=self.device)
                    all_q[start:end] = self.net.forward_mean(pos_t).cpu().numpy()
            offset = 0
            for i, segment in enumerate(segments):
                n_frames = seg_frame_counts[i]
                if n_frames > 0:
                    segment.pcoord[:n_frames, 0] = all_q[offset:offset + n_frames]
                    offset += n_frames

        # Re-bin with corrected q values.
        we_driver = self.sim_manager.we_driver
        nbins = we_driver.bin_mapper.nbins
        we_driver.initial_binning = we_driver.bin_mapper.construct_bins()
        we_driver.final_binning = we_driver.bin_mapper.construct_bins()
        we_driver.flux_matrix = np.zeros((nbins, nbins), dtype=np.float64)
        we_driver.transition_matrix = np.zeros((nbins, nbins), dtype=np.uint64)
        we_driver.assign(segments)

        # Flush istate assignments created by recycling.
        istate_futures = self.sim_manager.get_istate_futures()
        for future in istate_futures:
            _bs, initial_state = future.get_result()
            initial_state.istate_status = (
                westpa.core.states.InitialState.ISTATE_STATUS_PREPARED
            )
            with self.sim_manager.data_manager.expiring_flushing_lock():
                self.sim_manager.data_manager.update_initial_states(
                    [initial_state], n_iter=self.sim_manager.n_iter + 1
                )
            we_driver.avail_initial_states[initial_state.state_id] = initial_state
