"""WESTPA plugin: committor q prediction (inference only).

In the two-sim steady-state setup, each sim's CommittorDriver is
inference-only. Training is owned by an external coordinator process
(``coordinator/train_coordinator.py``) that reads BOTH sims' west.h5
files and writes a single shared checkpoint. This driver:

1. ``pre_we`` — loads each segment's trajectory, batch-evaluates the
   committor network (reloading the checkpoint first if the coordinator
   has written a newer one), and overwrites column 0 of the 2D pcoord
   (q, rmsd_B). It then re-assigns all segments to bins so that WE
   splitting/merging uses the real q values.

Configuration in west.cfg::

    west:
      plugins:
        - plugin: committor.driver.CommittorDriver
"""
import json
import logging
import os
import time

import h5py
import numpy as np
import torch
import westpa

from committor.config import CommittorConfig
from committor.system import ADPSystem
from committor.network import build_committor_net

log = logging.getLogger(__name__)


class CommittorDriver:

    def __init__(self, sim_manager, plugin_config):
        westpa.rc.pstatus("Initializing CommittorDriver plugin (inference-only)")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not master process, skipping CommittorDriver")
            return

        self.sim_manager = sim_manager
        self.cfg = CommittorConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(plugin_config.get("batch_size", 512))

        # Build network once (architecture selected by cfg.network_type)
        system = ADPSystem(self.cfg.pdb_path)
        self.net = build_committor_net(self.cfg, system).to(self.device)

        # Load checkpoint if available (coordinator writes shared checkpoint)
        self._ckpt_mtime = None
        self._maybe_reload_checkpoint()

        self.net.eval()
        westpa.rc.pstatus(f"CommittorDriver ready on {self.device} (sim root {self.cfg.root})")

        # Cross-sim drift cap: peer sims = every sim_source other than mine.
        # Used in pre_we to stall the leading sim until the trailing sim
        # catches up, bounding the asymmetry of training data the
        # coordinator pools. No-op when only one sim is configured.
        my_root = os.path.realpath(self.cfg.root)
        self._peer_h5_paths = [
            src["west_h5"] for src in self.cfg.sim_sources
            if os.path.realpath(src["root"]) != my_root
        ]
        if self._peer_h5_paths:
            westpa.rc.pstatus(
                f"CommittorDriver drift cap: max_drift_iters="
                f"{self.cfg.max_drift_iters} peers={self._peer_h5_paths}"
            )

        priority = int(plugin_config.get("priority", 0))
        sim_manager.register_callback(
            sim_manager.pre_we, self.pre_we, priority
        )

    def _peer_min_iter(self):
        """Smallest latest-iteration index across all peer sims, or None.

        Returns None if no peers are configured. Returns 0 if a peer's h5
        does not exist yet (peer hasn't started) — this implicitly stalls
        the leading sim at the start, which is what we want.
        """
        if not self._peer_h5_paths:
            return None
        latest = []
        for path in self._peer_h5_paths:
            if not os.path.exists(path):
                latest.append(0)
                continue
            try:
                with h5py.File(path, "r", locking=False) as f:
                    if "iterations" not in f:
                        latest.append(0)
                        continue
                    highest = 0
                    for k in f["iterations"].keys():
                        try:
                            highest = max(highest, int(k.split("_")[-1]))
                        except ValueError:
                            pass
                    latest.append(highest)
            except Exception as exc:
                # Transient h5 lock / mid-write race: don't stall on it,
                # treat the peer as fully caught up for this cycle.
                log.debug("CommittorDriver: peer %s read failed (%s)", path, exc)
                latest.append(self.sim_manager.n_iter)
        return min(latest)

    def _wait_for_peer(self, my_iter):
        """Block until peer sims are within max_drift_iters of my_iter."""
        if not self._peer_h5_paths:
            return
        cap = int(self.cfg.max_drift_iters)
        sleep_s = float(self.cfg.drift_check_seconds)
        warn_after = float(self.cfg.drift_warn_seconds)

        t0 = time.time()
        warned = False
        while True:
            peer = self._peer_min_iter()
            if peer is None:
                return
            drift = my_iter - peer
            if drift <= cap:
                if warned:
                    westpa.rc.pstatus(
                        f"CommittorDriver: drift back within cap "
                        f"(my={my_iter} peer={peer} drift={drift})"
                    )
                return
            elapsed = time.time() - t0
            if elapsed > warn_after and not warned:
                westpa.rc.pstatus(
                    f"CommittorDriver: stalled {elapsed:.0f}s waiting on peer "
                    f"(my={my_iter} peer={peer} drift={drift} cap={cap})"
                )
                warned = True
            time.sleep(sleep_s)

    def _wait_for_coordinator(self, my_iter):
        """Block until the coordinator has trained through >= my_iter - ahead.

        Stops this sim from racing ahead of the coordinator's training (e.g.
        while the coordinator job is still queued, or training lags
        propagation). Reads coordinator_progress.json (written by
        train_coordinator after each successful train). The cap is floored at
        warmup_iters so the sim can always reach warmup_iters first and give the
        coordinator enough data for its first training pass -> no deadlock.
        """
        ahead = max(int(self.cfg.max_iters_ahead), int(self.cfg.warmup_iters))
        if ahead >= 10 ** 8:
            return  # lockstep disabled
        my_label = os.path.basename(self.cfg.root.rstrip("/"))
        prog_path = os.path.join(self.cfg.data_dir, "coordinator_progress.json")
        sleep_s = float(self.cfg.drift_check_seconds)
        warn_after = float(self.cfg.drift_warn_seconds)
        t0 = time.time()
        warned = False
        while True:
            last_trained = 0
            try:
                with open(prog_path) as f:
                    last_trained = int(json.load(f).get("last_trained", {})
                                       .get(my_label, 0))
            except Exception:
                last_trained = 0
            if my_iter - last_trained <= ahead:
                if warned:
                    westpa.rc.pstatus(
                        f"CommittorDriver: coordinator caught up "
                        f"(my={my_iter} trained={last_trained})"
                    )
                return
            elapsed = time.time() - t0
            if elapsed > warn_after and not warned:
                westpa.rc.pstatus(
                    f"CommittorDriver: stalled {elapsed:.0f}s waiting for "
                    f"coordinator training (my={my_iter} trained={last_trained} "
                    f"ahead_cap={ahead}) -- is the coordinator job running?"
                )
                warned = True
            time.sleep(sleep_s)

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
        westpa.rc.pstatus(f"CommittorDriver reloaded checkpoint (mtime={mtime})")

    def pre_we(self):
        self._maybe_reload_checkpoint()

        n_iter = self.sim_manager.n_iter
        segments = list(self.sim_manager.segments.values())
        log.info("CommittorDriver: computing q for %d segments (iter %d)",
                 len(segments), n_iter)

        # Collect positions from all segments. With BatchedADPPropagator the
        # propagator stashes positions on each segment in-memory under
        # seg.data["positions"] (shape: (pcoord_len, n_atoms, 3) float32),
        # so we skip the parent.xml + seg.dcd mdtraj round-trip.
        all_positions = []  # will be (N_segments * pcoord_len, n_atoms*3)
        seg_frame_counts = []

        for segment in segments:
            xyz = segment.data.get("positions")
            if xyz is None:
                log.warning("CommittorDriver: seg %d missing in-memory positions",
                            segment.seg_id)
                seg_frame_counts.append(0)
                continue
            xyz = np.asarray(xyz, dtype=np.float32)
            flat = xyz.reshape(len(xyz), -1)
            all_positions.append(flat)
            seg_frame_counts.append(len(xyz))

        # Batch NN inference
        if all_positions:
            all_pos = np.concatenate(all_positions, axis=0)
            all_q = np.empty(len(all_pos), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(all_pos), self.batch_size):
                    end = min(start + self.batch_size, len(all_pos))
                    pos_t = torch.tensor(all_pos[start:end], device=self.device)
                    all_q[start:end] = self.net.forward_mean(pos_t).cpu().numpy()

            # Overwrite q column in each segment's pcoord
            offset = 0
            for i, segment in enumerate(segments):
                n_frames = seg_frame_counts[i]
                if n_frames > 0:
                    segment.pcoord[:n_frames, 0] = all_q[offset:offset + n_frames]
                    offset += n_frames

        # Re-assign segments to bins with corrected q values
        we_driver = self.sim_manager.we_driver
        nbins = we_driver.bin_mapper.nbins
        we_driver.initial_binning = we_driver.bin_mapper.construct_bins()
        we_driver.final_binning = we_driver.bin_mapper.construct_bins()
        we_driver.flux_matrix = np.zeros((nbins, nbins), dtype=np.float64)
        we_driver.transition_matrix = np.zeros((nbins, nbins), dtype=np.uint64)
        we_driver.assign(segments)

        # Handle istates for segments now in target bins
        istate_futures = self.sim_manager.get_istate_futures()
        for future in istate_futures:
            _basis_state, initial_state = future.get_result()
            initial_state.istate_status = (
                westpa.core.states.InitialState.ISTATE_STATUS_PREPARED
            )
            with self.sim_manager.data_manager.expiring_flushing_lock():
                self.sim_manager.data_manager.update_initial_states(
                    [initial_state], n_iter=self.sim_manager.n_iter + 1
                )
            we_driver.avail_initial_states[initial_state.state_id] = initial_state

        # Bound cross-sim drift. After this iter's WE bookkeeping is done, if
        # we're more than max_drift_iters ahead of the slowest peer sim, sleep
        # here so the peer can catch up before we propagate the next iter.
        # No-op for single-sim runs.
        self._wait_for_peer(n_iter)

        # Bound sim->coordinator drift: don't propagate the next iter until the
        # coordinator has trained through >= n_iter - max_iters_ahead, so the
        # committor the sim bins on never lags the run by more than that (and so
        # a sim can't blast past the budget while the coordinator is still
        # queued).
        self._wait_for_coordinator(n_iter)
