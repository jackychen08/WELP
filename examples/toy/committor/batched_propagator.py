"""In-process batched propagator for the toy 2D ZJZ run.

Replaces ExecutablePropagator + runseg.sh + run_seg_toy.py. All walkers
in a given iteration are integrated as one batched tensor op; per-walker
Python cold-start, torch import, CUDA init, and most file I/O are gone.

Wired up via sim_*/west.cfg:
    propagation:
      propagator: committor.batched_propagator.BatchedToyPropagator
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import torch

import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.segment import Segment

SHARED_ROOT = os.environ.get(
    "COMMITTOR_SHARED_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
# Repo root = parent of the REAL common_files dir (resolve the per-run
# `common_files` symlink). Depth-independent, so it works whether the run dir
# is 1 level deep (original layout) or 2 (nested per-config sweep layout).
TOY_ROOT = os.path.dirname(os.path.realpath(os.path.join(SHARED_ROOT, "common_files")))
for p in (SHARED_ROOT, TOY_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from common_files.potential import (  # noqa: E402
    ALPHA_DEFAULT, GAMMA_DEFAULT, KT_DEFAULT,
    WELL_A, WELL_B, force, in_basin_a, in_basin_b,
)


# ---------------------------------------------------------------------------
# Batched integrator
# ---------------------------------------------------------------------------

def _batched_em_step(xy: torch.Tensor, alpha: float, beta: float,
                     gamma: float, dt: float, noise: torch.Tensor) -> torch.Tensor:
    """One EM step for (N, 2) walkers. `noise` is pre-sampled (N, 2) gaussian."""
    grad_U = -force(xy, alpha)
    sigma = (2.0 * dt / (beta * gamma)) ** 0.5
    return xy - grad_U * dt / gamma + sigma * noise


def _run_batched_segments(parents: torch.Tensor, alpha: float, beta: float,
                          gamma: float, dt: float, n_steps: int,
                          n_frames: int, generator: torch.Generator) -> torch.Tensor:
    """(N, 2) parents -> (N, n_frames, 2) trajectories (parent = frame 0)."""
    N = parents.shape[0]
    stride = n_steps // (n_frames - 1)

    noise = torch.randn(
        (n_steps, N, 2), generator=generator,
        device=parents.device, dtype=parents.dtype,
    )

    snapshots = [parents.clone()]
    xy = parents.clone()
    for step in range(n_steps):
        xy = _batched_em_step(xy, alpha, beta, gamma, dt, noise[step])
        if (step + 1) % stride == 0:
            snapshots.append(xy.clone())
    return torch.stack(snapshots[:n_frames], dim=0).permute(1, 0, 2).contiguous()


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

class BatchedToyPropagator(WESTPropagator):

    def __init__(self, rc=None):
        super().__init__(rc)

        self.alpha = float(os.environ.get("TOY_ALPHA", ALPHA_DEFAULT))
        self.beta = 1.0 / KT_DEFAULT
        self.gamma = GAMMA_DEFAULT
        self.dt = float(os.environ.get("TOY_DT", 1e-3))
        self.n_steps = int(os.environ.get("TOY_N_STEPS", 25))
        self.n_frames = 6

        target = os.environ.get("TARGET_BASIN", "").upper()
        if target == "B":
            self.target_well = np.asarray(WELL_B, dtype=np.float64)
            self.target_in_basin = in_basin_b
        elif target == "A":
            self.target_well = np.asarray(WELL_A, dtype=np.float64)
            self.target_in_basin = in_basin_a
        else:
            raise ValueError("TARGET_BASIN env var must be 'A' or 'B'")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sim_root = os.environ["WEST_SIM_ROOT"]
        self.traj_dir = os.path.join(self.sim_root, "output_files", "traj_segs")
        self.bstate_dir = os.path.join(self.sim_root, "bstates")

        self._global_seed = int(os.environ.get("TOY_GLOBAL_SEED", 20260424))

        westpa.rc.pstatus(
            f"BatchedToyPropagator ready: device={self.device} "
            f"target={target} dt={self.dt} n_steps={self.n_steps}"
        )

    # ---- helpers ------------------------------------------------------------

    def _bstate_xy_path(self, basis_state) -> str:
        return os.path.join(self.bstate_dir, basis_state.auxref)

    def _load_xy_for_segment(self, segment) -> np.ndarray:
        """Resolve the starting xy for this segment (continues or new-traj)."""
        if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
            path = os.path.join(
                self.traj_dir,
                f"{segment.n_iter - 1:06d}",
                f"{segment.parent_id:06d}",
                "seg_final.npy",
            )
            return np.load(path).astype(np.float64).reshape(2)

        # NEWTRAJ: parent_id = -(istate_id + 1)
        istate_id = -(segment.parent_id + 1)
        istate = self.initial_states[istate_id]
        basis_state = self.basis_states[istate.basis_state_id]
        return np.load(self._bstate_xy_path(basis_state)).astype(np.float64).reshape(2)

    def _pcoord_from_traj(self, traj: np.ndarray) -> np.ndarray:
        """(N, F, 2) trajectory -> (N, F, 3) pcoord with columns
        (q_placeholder=0, rmsd_to_target, in_target_latched)."""
        rmsd = np.linalg.norm(traj - self.target_well, axis=-1).astype(np.float32)
        hit = self.target_in_basin(traj).astype(np.float32)
        in_target = np.maximum.accumulate(hit, axis=-1)
        q_placeholder = np.zeros_like(rmsd)
        return np.stack([q_placeholder, rmsd, in_target], axis=-1)

    # ---- WESTPA API ---------------------------------------------------------

    def get_pcoord(self, state):
        """Populate `state.pcoord` for a basis / initial state.

        For basis states, `state.auxref` names a file under $WEST_SIM_ROOT/bstates.
        For initial states, the sim manager attaches the basis_state object
        (see WESimManager.prepare_iteration).
        """
        if hasattr(state, "auxref") and state.auxref:
            xy_path = os.path.join(self.bstate_dir, state.auxref)
        elif getattr(state, "basis_state", None) is not None:
            xy_path = self._bstate_xy_path(state.basis_state)
        elif getattr(state, "basis_auxref", None):
            xy_path = os.path.join(self.bstate_dir, state.basis_auxref)
        else:
            raise ValueError(
                f"Cannot resolve xy for state {state!r}: no auxref / basis_state"
            )

        xy = np.load(xy_path).astype(np.float64).reshape(2)
        rmsd = float(np.linalg.norm(xy - self.target_well))
        in_target = 1.0 if bool(self.target_in_basin(xy)) else 0.0
        state.pcoord = np.array([0.0, rmsd, in_target], dtype=np.float32)

    def gen_istate(self, basis_state, initial_state):
        """gen_istates=false in west.cfg, so this is only a safety fallback.

        Copy the basis-state pcoord forward — the istate represents the same
        point as the basis state.
        """
        initial_state.pcoord = np.array(basis_state.pcoord, dtype=np.float32)
        return initial_state

    def propagate(self, segments: List[Segment]):
        if not segments:
            return segments

        n_iter = segments[0].n_iter
        N = len(segments)

        parents_np = np.empty((N, 2), dtype=np.float64)
        for i, seg in enumerate(segments):
            parents_np[i] = self._load_xy_for_segment(seg)

        # Seed depends on n_iter AND the seg_ids in this block. Two reasons:
        # (a) if WESTPA calls propagate() more than once per iter (e.g. block_size
        #     small, or multi-worker), each call gets unique noise;
        # (b) even a single-walker call gets noise that differs from other
        #     single-walker calls in the same iteration.
        gen = torch.Generator(device=self.device)
        seg_key = 0
        for seg in segments:
            seg_key = (seg_key * 1000003 + int(seg.seg_id)) & 0xFFFFFFFFFFFFFFFF
        seed = (self._global_seed ^ (n_iter * 2654435761) ^ seg_key) & 0x7FFFFFFFFFFFFFFF
        gen.manual_seed(seed)

        parents = torch.from_numpy(parents_np).to(self.device)
        traj = _run_batched_segments(
            parents, self.alpha, self.beta, self.gamma, self.dt,
            self.n_steps, self.n_frames, gen,
        )
        traj_np = traj.detach().cpu().numpy().astype(np.float32)

        # Reject any walker whose trajectory contains inf/NaN (force blowup past
        # the exp wall). Replace with a frozen copy of the parent so downstream
        # binning/training never see a non-finite pcoord.
        bad = ~np.isfinite(traj_np).all(axis=(1, 2))
        if bad.any():
            n_bad = int(bad.sum())
            westpa.rc.pstatus(
                f"BatchedToyPropagator: rejecting {n_bad}/{len(segments)} "
                f"non-finite trajectories at iter {n_iter}; freezing at parent."
            )
            for i in np.flatnonzero(bad):
                parent_xy = parents_np[i].astype(np.float32)
                traj_np[i, :] = parent_xy

        pcoord = self._pcoord_from_traj(traj_np)

        for i, seg in enumerate(segments):
            seg_dir = os.path.join(
                self.traj_dir, f"{n_iter:06d}", f"{seg.seg_id:06d}",
            )
            os.makedirs(seg_dir, exist_ok=True)
            np.save(os.path.join(seg_dir, "seg.npy"), traj_np[i])
            np.save(os.path.join(seg_dir, "seg_final.npy"), traj_np[i, -1])

            seg.pcoord[:] = pcoord[i]
            seg.data["xy"] = traj_np[i]
            seg.status = Segment.SEG_STATUS_COMPLETE

        return segments
