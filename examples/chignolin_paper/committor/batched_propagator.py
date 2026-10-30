"""In-process single-GPU propagator for the chignolin steady-state runs.

Clone of BatchedAIB9Propagator (sc_committor_rates_run/AD/constructor.py:sample_swarm
lineage), adapted for EXPLICIT-SOLVENT chignolin:
  * loads a serialized OpenMM System (common_files/system.xml, PME + periodic box)
    instead of an AMBER prmtop -- AIB9 was vacuum.
  * 340 K, gamma = 1/ps, dt = 2 fs BAOAB (LangevinMiddleIntegrator); tau = 2 ps
    (1000 steps) matching Grant's chignolin t_stride (JCTC 2024 sec 9.4).
  * CV = SUM of 4 beta-hairpin H-bond distances (committor.chignolin_cv), computed on
    the PEPTIDE slice with RAW (non-periodic) distances -- the peptide is one bonded
    chain that OpenMM keeps whole, and the unfolded basin (sum > 3 nm) exceeds box/2,
    so minimum-image would corrupt it.
  * stores PEPTIDE-ONLY positions (166 atoms, COM-centered for float32 precision) in
    seg.data / final_pos so the net sees 166 atoms and west.h5 stays ~40x smaller; the
    FULL solvated State stays in seg.xml for exact resume.
  * pcoord = (q placeholder=0, hbond_sum, in_target). Emits phi = hbond_sum, psi = 0
    aux datasets so west.cfg and train.py (which read phi/psi and call
    cfg.in_basin_a(phi, psi)) stay byte-identical -- the maximal-reuse trick.

Wired up via sim_*/west.cfg::

    propagation:
      propagator: committor.batched_propagator.BatchedChignolinPropagator
      gen_istates: false
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
from committor.chignolin_cv import resolve_pair_indices, hbond_sum  # noqa: E402


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

        # Peptide topology + H-bond CV pairs + peptide-atom indices in the full system.
        self.peptide_indices = np.load(
            os.path.join(self.cfg.common_files, "peptide_indices.npy")
        )
        self.pep_top = mdtraj.load(self.cfg.pdb_path).topology     # peptide.pdb (166)
        self.pair_idx = resolve_pair_indices(self.pep_top)
        self.n_peptide = self.pep_top.n_atoms

        # System: serialized OpenMM System (PME, periodic box). Explicit solvent.
        with open(self.cfg.top_path) as fh:                        # system.xml
            system = omm.XmlSerializer.deserialize(fh.read())
        self.box = system.getDefaultPeriodicBoxVectors()

        # Grant's chignolin dynamics: 340 K, gamma=1/ps, dt=2 fs BAOAB (sec 9.4).
        self.temperature = 340 * unit.kelvin
        self.friction = 1.0 / unit.picosecond
        self.timestep = 2 * unit.femtosecond
        self.integrator = omm.LangevinMiddleIntegrator(
            self.temperature, self.friction, self.timestep,
        )
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

        # tau = 1000 steps x 2 fs = 2 ps (Grant's t_stride). pcoord_len=11 -> snapshot
        # every 100 steps (parent + 10 frames).
        self.pcoord_len = int(os.environ.get("ADP_PCOORD_LEN", 11))
        self.total_steps = int(os.environ.get("ADP_TOTAL_STEPS", 1000))
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

    def _peptide_slice(self, full_xyz: np.ndarray) -> np.ndarray:
        """(n_atoms,3) full -> (166,3) peptide, COM-centered (translation-invariant
        for the distance features / CV; keeps float32 precision as the COM drifts)."""
        pep = full_xyz[self.peptide_indices]
        return (pep - pep.mean(axis=0)).astype(np.float32)

    def _compute_aux(self, pep_traj: np.ndarray):
        """pep_traj: (F, 166, 3) PEPTIDE-ONLY -> (pcoord (F,3), phi (F,), psi (F,)).

        pcoord columns:
          0: q placeholder (overwritten by CommittorDriver.pre_we)
          1: hbond_sum (the state CV, nm)
          2: in_target indicator (0/1) -- inside the RECYCLING target basin
        phi = hbond_sum, psi = 0 so train.py's phi/psi aux + cfg.in_basin_a(phi,psi)
        keep working unchanged.
        """
        s = np.atleast_1d(hbond_sum(pep_traj, self.pair_idx)).astype(np.float32)  # (F,)
        if self.target_basin == "B":
            in_target = self.cfg.in_basin_b(s)
        else:
            in_target = self.cfg.in_basin_a(s)
        pcoord = np.zeros((len(pep_traj), 3), dtype=np.float32)
        pcoord[:, 1] = s
        pcoord[:, 2] = np.asarray(in_target, dtype=np.float32)
        psi = np.zeros_like(s)
        return pcoord, s, psi

    def _integrate_one(self, parent_state, vseed=None):
        """Set parent state, redraw velocities (340 K), integrate, snapshot the
        peptide. Returns (pep_traj (F,166,3) float32, final_state full for resume)."""
        self.context.setState(parent_state)
        if vseed is None:
            self.context.setVelocitiesToTemperature(self.temperature)
        else:
            self.context.setVelocitiesToTemperature(self.temperature, int(vseed))

        snaps = []
        st0 = self.context.getState(getPositions=True)
        snaps.append(self._peptide_slice(
            np.asarray(st0.getPositions().value_in_unit(unit.nanometer))))
        for _ in range(self.pcoord_len - 1):
            self.integrator.step(self.stride_steps)
            st = self.context.getState(getPositions=True)
            snaps.append(self._peptide_slice(
                np.asarray(st.getPositions().value_in_unit(unit.nanometer))))

        final_state = self.context.getState(
            getPositions=True, getVelocities=True,
            getEnergy=False, enforcePeriodicBox=False,
        )
        return np.stack(snaps, axis=0), final_state

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
        full = np.asarray(omm_state.getPositions().value_in_unit(unit.nanometer))
        pep = self._peptide_slice(full)[None, ...]        # (1,166,3)
        pcoord, _phi, _psi = self._compute_aux(pep)
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
            pep_traj, final_state = self._integrate_one(parent_state, vseed)

            if not np.isfinite(pep_traj).all():
                westpa.rc.pstatus(
                    f"BatchedChignolinPropagator: non-finite traj at iter {n_iter} "
                    f"seg {seg.seg_id}; freezing at parent."
                )
                pep_traj = np.broadcast_to(pep_traj[0], pep_traj.shape).copy()

            pcoord, phi, psi = self._compute_aux(pep_traj)

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
            seg.data["phi"] = phi                   # = hbond_sum (train.py compat)
            seg.data["psi"] = psi                   # = 0
            seg.status = Segment.SEG_STATUS_COMPLETE

        return segments
