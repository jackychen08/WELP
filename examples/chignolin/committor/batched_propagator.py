"""Existing WE propagator for the pinned CHARMM27 / native-contact Q system.

340 K, friction .3/ps, timestep .5 fs; retain 2 ps segments and 11 frames.
Q uses the archived periodic calculation on raw coordinates. The network sees
whole, centered protein coordinates. Full solvent states remain in seg.xml.
The existing thermal redraw, endpoint observation and WE recycling are retained.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import mdtraj
import openmm as omm
import openmm.unit as unit

import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.segment import Segment

SHARED_ROOT = os.environ.get(
    "COMMITTOR_SHARED_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if SHARED_ROOT not in sys.path:
    sys.path.insert(0, SHARED_ROOT)

from committor.config import CommittorConfig  # noqa: E402
from committor.lazzeri_cv import ArchiveQ, protein_tree, whole_protein  # noqa: E402


class BatchedChignolinPropagator(WESTPropagator):
    """Sequential in-process explicit-solvent OpenMM propagator (one CUDA Context)."""

    def __init__(self, rc=None):
        super().__init__(rc)

        self.cfg = CommittorConfig()
        self.sim_root = os.environ["WEST_SIM_ROOT"]
        self.traj_dir = os.path.join(self.sim_root, "output_files", "traj_segs")
        self.bstate_dir = os.path.join(self.sim_root, "bstates")
        self.istate_dir = os.path.join(self.sim_root, "output_files", "istates")

        target = os.environ.get("TARGET_BASIN", "").upper()
        if target not in ("A", "B"):
            raise ValueError(
                "BatchedChignolinPropagator: TARGET_BASIN env var must be 'A' or 'B'"
            )
        self.target_basin = target

        # Pinned protein topology, periodic Q evaluator and atom mapping.
        self.peptide_indices = np.load(
            os.path.join(self.cfg.common_files, "peptide_indices.npy")
        )
        self.pep_top = mdtraj.load(self.cfg.pdb_path).topology     # peptide.pdb (166)
        self.archive_q = ArchiveQ()
        self.protein_tree = protein_tree(np.load(os.path.join(self.cfg.common_files, "protein_bonds.npy")))
        self.n_peptide = self.pep_top.n_atoms

        # System: serialized OpenMM System (PME, periodic box). Explicit solvent.
        with open(self.cfg.top_path) as fh:                        # system.xml
            system = omm.XmlSerializer.deserialize(fh.read())
        self.box = system.getDefaultPeriodicBoxVectors()

        # Physical parameters transferred from the pinned SOC system.
        self.temperature = 340 * unit.kelvin
        self.friction = 0.3 / unit.picosecond
        self.timestep = 0.5 * unit.femtosecond
        self.integrator = omm.LangevinMiddleIntegrator(
            self.temperature, self.friction, self.timestep,
        )
        self.integrator.setConstraintTolerance(1e-6)
        self._global_seed = int(os.environ.get("ADP_GLOBAL_SEED", 20260425)) & 0x7FFFFFFF
        self.integrator.setRandomNumberSeed(self._global_seed)

        platform_name = os.environ.get("OPENMM_PLATFORM", "CPU")
        platform = omm.Platform.getPlatformByName(platform_name)
        properties = {}
        if platform_name in ("CUDA", "OpenCL"):
            properties["Precision"] = "mixed"
        elif platform_name == "CPU":
            properties["Threads"] = os.environ.get("OPENMM_CPU_THREADS", "1")
        self.context = omm.Context(system, self.integrator, platform, properties)

        # Preserve the default 2 ps segment and .2 ps stored-frame cadence.
        self.pcoord_len = int(os.environ.get("ADP_PCOORD_LEN", 11))
        self.total_steps = int(os.environ.get("ADP_TOTAL_STEPS", 4000))
        n_intervals = self.pcoord_len - 1
        if self.total_steps % n_intervals != 0:
            raise ValueError(
                f"total_steps ({self.total_steps}) must be a multiple of "
                f"pcoord_len-1 ({n_intervals})"
            )
        self.stride_steps = self.total_steps // n_intervals

        westpa.rc.pstatus(
            f"BatchedChignolinPropagator ready: target={target} "
            f"n_peptide={self.n_peptide} pcoord_len={self.pcoord_len} "
            f"total_steps={self.total_steps} stride_steps={self.stride_steps}"
        )

    # ---- helpers ------------------------------------------------------------

    def _bstate_xml_path(self, basis_state) -> str:
        return os.path.join(self.bstate_dir, basis_state.auxref)

    def _parent_xml_path(self, segment) -> str:
        if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
            return os.path.join(
                self.traj_dir,
                f"{segment.n_iter - 1:06d}",
                f"{segment.parent_id:06d}",
                "seg.xml",
            )
        istate_id = -(segment.parent_id + 1)
        istate = self.initial_states[istate_id]
        basis_state = self.basis_states[istate.basis_state_id]
        return self._bstate_xml_path(basis_state)

    def _load_state(self, xml_path: str):
        with open(xml_path, "r") as f:
            return omm.XmlSerializer.deserialize(f.read())

    def _peptide_slice(self, full_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
        """(n_atoms,3) full -> (166,3) peptide, COM-centered (translation-invariant
        for the distance features / CV; keeps float32 precision as the COM drifts)."""
        pep = full_xyz[self.peptide_indices]
        return whole_protein(pep, box, self.protein_tree)

    def _compute_aux(self, q_scores: np.ndarray):
        """Convert exact structural Q to (learned-q placeholder, Q, target flag).

        phi=Q and psi=0 retain the existing trainer's scalar-basin interface.
        """
        s = np.atleast_1d(q_scores).astype(np.float32)
        if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
            raise ValueError("Expected finite archived structural Q in [0,1]")
        if self.target_basin == "B":
            in_target = self.cfg.in_basin_b(s)
        else:
            in_target = self.cfg.in_basin_a(s)
        pcoord = np.zeros((len(s), 3), dtype=np.float32)
        pcoord[:, 1] = s
        pcoord[:, 2] = np.asarray(in_target, dtype=np.float32)
        psi = np.zeros_like(s)
        return pcoord, s, psi

    def _snapshot(self, state):
        full = np.asarray(state.getPositions().value_in_unit(unit.nanometer))
        box = np.asarray(state.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
        # Preserve the archived float32 calculation on the original coordinates.
        # The NN's whole/centered representation must never set basin membership.
        _, _, q = self.archive_q.evaluate(full[self.peptide_indices], box)
        return self._peptide_slice(full, box), float(q[0])

    def _integrate_one(self, parent_state, vseed=None):
        """Set parent state, redraw velocities (340 K), integrate, snapshot the
        peptide. Returns (whole peptide frames, exact Q scores, full final state)."""
        self.context.setState(parent_state)
        if vseed is None:
            self.context.setVelocitiesToTemperature(self.temperature)
        else:
            self.context.setVelocitiesToTemperature(self.temperature, int(vseed))

        snaps, scores = [], []
        st0 = self.context.getState(getPositions=True)
        xyz, q = self._snapshot(st0)
        snaps.append(xyz)
        scores.append(q)
        for _ in range(self.pcoord_len - 1):
            self.integrator.step(self.stride_steps)
            st = self.context.getState(getPositions=True)
            xyz, q = self._snapshot(st)
            snaps.append(xyz)
            scores.append(q)

        final_state = self.context.getState(
            getPositions=True, getVelocities=True,
            getEnergy=False, enforcePeriodicBox=False,
        )
        return np.stack(snaps, axis=0), np.asarray(scores, dtype=np.float32), final_state

    # ---- WESTPA API ---------------------------------------------------------

    def get_pcoord(self, state):
        """Initial pcoord for a basis/initial state (full solvated bstate State)."""
        if hasattr(state, "auxref") and state.auxref:
            xml_path = os.path.join(self.bstate_dir, state.auxref)
        elif getattr(state, "basis_state", None) is not None:
            xml_path = self._bstate_xml_path(state.basis_state)
        elif getattr(state, "basis_auxref", None):
            xml_path = os.path.join(self.bstate_dir, state.basis_auxref)
        else:
            raise ValueError(f"Cannot resolve xml for state {state!r}")

        omm_state = self._load_state(xml_path)
        _pep, q = self._snapshot(omm_state)
        pcoord, _phi, _psi = self._compute_aux(np.asarray([q]))
        state.pcoord = pcoord[0]

    def gen_istate(self, basis_state, initial_state):
        initial_state.pcoord = np.array(basis_state.pcoord, dtype=np.float32)
        return initial_state

    def propagate(self, segments: List[Segment]):
        if not segments:
            return segments

        n_iter = segments[0].n_iter
        parent_state_cache: dict[str, "omm.State"] = {}

        for seg in segments:
            xml_path = self._parent_xml_path(seg)
            parent_state = parent_state_cache.get(xml_path)
            if parent_state is None:
                parent_state = self._load_state(xml_path)
                parent_state_cache[xml_path] = parent_state
            vseed = (self._global_seed ^ (int(n_iter) * 2654435761)
                     ^ ((int(seg.seg_id) + 1) * 40503)) & 0x7FFFFFFF
            pep_traj, q_scores, final_state = self._integrate_one(parent_state, vseed)

            if not np.isfinite(pep_traj).all():
                westpa.rc.pstatus(
                    f"BatchedChignolinPropagator: non-finite traj at iter {n_iter} "
                    f"seg {seg.seg_id}; freezing at parent."
                )
                pep_traj = np.broadcast_to(pep_traj[0], pep_traj.shape).copy()
                q_scores = np.full_like(q_scores, q_scores[0])

            pcoord, phi, psi = self._compute_aux(q_scores)

            seg_dir = os.path.join(
                self.traj_dir, f"{n_iter:06d}", f"{seg.seg_id:06d}",
            )
            os.makedirs(seg_dir, exist_ok=True)
            with open(os.path.join(seg_dir, "seg.xml"), "w") as f:
                f.write(omm.XmlSerializer.serialize(final_state))
            np.save(os.path.join(seg_dir, "seg_traj.npy"), pep_traj)  # peptide-only

            seg.pcoord[:] = pcoord
            seg.data["positions"] = pep_traj        # (pcoord_len, 166, 3)
            seg.data["final_pos"] = pep_traj[-1]    # (166, 3) -> west.h5
            seg.data["phi"] = phi                   # exact structural Q (train.py compat)
            seg.data["psi"] = psi                   # = 0
            seg.status = Segment.SEG_STATUS_COMPLETE

        return segments
