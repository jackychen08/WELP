"""In-process single-GPU propagator for the ADP steady-state runs.

Replaces ExecutablePropagator + runseg.sh + adp_prod.py + get_distance.py.
One OpenMM CUDA Context lives for the entire master process; segments in
each WE iteration are integrated sequentially through it (Grant-style:
sc_committor_rates_run/AD_Underdamped/constructor.py:sample_swarm), so per
walker we pay only setPositions / setVelocities / step rather than a fresh
python+CUDA cold start.

Wired up via sim_*/west.cfg::

    propagation:
      propagator: committor.batched_propagator.BatchedADPPropagator
      gen_istates: false

Outputs per segment (in-memory; dumped to disk only at iteration boundary):
  * seg.xml          — final OpenMM State, used as parent for the next iter
  * seg.data["positions"] — (pcoord_len, n_atoms, 3) float32 trajectory
  * seg.data["phi"], seg.data["psi"] — (pcoord_len,) dihedrals
  * seg.pcoord       — (pcoord_len, 3): (q placeholder=0, rmsd_target, in_target)

Downstream the in-process CommittorDriver reads ``seg.data["positions"]``
directly from the in-memory segments — no DCD round-trip.
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

# Ensure shared root is importable so the propagator can reach
# committor.config when WESTPA loads it from the per-sim cwd.
SHARED_ROOT = os.environ.get(
    "COMMITTOR_SHARED_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if SHARED_ROOT not in sys.path:
    sys.path.insert(0, SHARED_ROOT)

from committor.config import CommittorConfig  # noqa: E402




class BatchedADPPropagator(WESTPropagator):
    """Sequential in-process OpenMM propagator (one CUDA Context per master)."""

    # ADP heavy-atom dihedral indices (matches common_files/get_distance.py)
    PHI_INDICES = [[4, 6, 8, 14]]
    PSI_INDICES = [[6, 8, 14, 16]]

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
                "BatchedADPPropagator: TARGET_BASIN env var must be 'A' or 'B'"
            )
        self.target_basin = target
        ref_pdb = "C7ax_ref.pdb" if target == "B" else "C7eq_ref.pdb"
        self.ref_target = mdtraj.load(
            os.path.join(self.cfg.common_files, ref_pdb)
        )

        # Topology + system
        with open(self.cfg.top_path) as handle:
            system = omm.XmlSerializer.deserialize(handle.read())
        self.mdtraj_topology = mdtraj.load(self.cfg.pdb_path).topology
        self.heavy = self.mdtraj_topology.select("not element H")

        # Integrator: matches common_files/adp_prod.py exactly
        self.temperature = 310 * unit.kelvin
        self.friction = 0.3 / unit.picosecond
        self.timestep = 1 * unit.femtosecond
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

        # 11 = pcoord_len in west.cfg. 100 fs total = 100 steps, snapshot every 10
        # to give parent + 10 frames.
        self.pcoord_len = int(os.environ.get("ADP_PCOORD_LEN", 11))
        self.total_steps = int(os.environ.get("ADP_TOTAL_STEPS", 100))
        n_intervals = self.pcoord_len - 1
        if self.total_steps % n_intervals != 0:
            raise ValueError(
                f"total_steps ({self.total_steps}) must be a multiple of "
                f"pcoord_len-1 ({n_intervals})"
            )
        self.stride_steps = self.total_steps // n_intervals

        westpa.rc.pstatus(
            f"BatchedADPPropagator ready: target={target} "
            f"pcoord_len={self.pcoord_len} total_steps={self.total_steps} "
            f"stride_steps={self.stride_steps}"
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
        # NEWTRAJ: parent_id encodes the istate id
        istate_id = -(segment.parent_id + 1)
        istate = self.initial_states[istate_id]
        basis_state = self.basis_states[istate.basis_state_id]
        return self._bstate_xml_path(basis_state)

    def _load_state(self, xml_path: str):
        with open(xml_path, "r") as f:
            return omm.XmlSerializer.deserialize(f.read())

    def _compute_aux(self, traj_xyz: np.ndarray):
        """traj_xyz: (F, n_atoms, 3) -> (pcoord (F,3), phi (F,), psi (F,)).

        pcoord columns match common_files/get_distance.py:
          0: q placeholder (overwritten by CommittorDriver.pre_we)
          1: rmsd to recycling target ref (nm)
          2: in_target indicator (0/1) via phi/psi
        """
        traj = mdtraj.Trajectory(traj_xyz, self.mdtraj_topology)
        phi = mdtraj.compute_dihedrals(traj, self.PHI_INDICES).reshape(-1)
        psi = mdtraj.compute_dihedrals(traj, self.PSI_INDICES).reshape(-1)
        rmsd = mdtraj.rmsd(traj, self.ref_target, atom_indices=self.heavy)

        if self.target_basin == "B":
            in_target = self.cfg.in_basin_b(phi, psi)
        else:
            in_target = self.cfg.in_basin_a(phi, psi)

        pcoord = np.zeros((len(traj_xyz), 3), dtype=np.float32)
        pcoord[:, 1] = rmsd
        pcoord[:, 2] = in_target.astype(np.float32)
        return pcoord, phi.astype(np.float32), psi.astype(np.float32)

    def _integrate_one(self, parent_state, vseed=None):
        """Set parent state, redraw velocities, integrate, snapshot positions.

        Returns (traj_xyz (F, n_atoms, 3) float32, final_state).
        """
        self.context.setState(parent_state)
        # Match adp_prod.py: redraw velocities at 310 K to decorrelate split
        # children whose parent state is identical. Seed the redraw (derived
        # from ADP_GLOBAL_SEED + iter + seg) so replicates with the same SEED
        # reproduce the same split-children noise.
        if vseed is None:
            self.context.setVelocitiesToTemperature(self.temperature)
        else:
            self.context.setVelocitiesToTemperature(self.temperature, int(vseed))

        snaps = []
        # Parent frame
        st0 = self.context.getState(getPositions=True)
        snaps.append(
            np.asarray(
                st0.getPositions().value_in_unit(unit.nanometer),
                dtype=np.float32,
            )
        )
        for _ in range(self.pcoord_len - 1):
            self.integrator.step(self.stride_steps)
            st = self.context.getState(getPositions=True)
            snaps.append(
                np.asarray(
                    st.getPositions().value_in_unit(unit.nanometer),
                    dtype=np.float32,
                )
            )

        final_state = self.context.getState(
            getPositions=True, getVelocities=True,
            getEnergy=False, enforcePeriodicBox=False,
        )
        return np.stack(snaps, axis=0), final_state

    # ---- WESTPA API ---------------------------------------------------------

    def get_pcoord(self, state):
        """Compute the initial pcoord for a basis or initial state."""
        if hasattr(state, "auxref") and state.auxref:
            xml_path = os.path.join(self.bstate_dir, state.auxref)
        elif getattr(state, "basis_state", None) is not None:
            xml_path = self._bstate_xml_path(state.basis_state)
        elif getattr(state, "basis_auxref", None):
            xml_path = os.path.join(self.bstate_dir, state.basis_auxref)
        else:
            raise ValueError(f"Cannot resolve xml for state {state!r}")

        omm_state = self._load_state(xml_path)
        pos = np.asarray(
            omm_state.getPositions().value_in_unit(unit.nanometer),
            dtype=np.float32,
        )[None, ...]
        pcoord, _phi, _psi = self._compute_aux(pos)
        state.pcoord = pcoord[0]

    def gen_istate(self, basis_state, initial_state):
        # gen_istates=false in west.cfg; provide a safe fallback.
        initial_state.pcoord = np.array(basis_state.pcoord, dtype=np.float32)
        return initial_state

    def propagate(self, segments: List[Segment]):
        if not segments:
            return segments

        n_iter = segments[0].n_iter

        # Cache deserialized parent States by xml path: with the
        # merge-everyone-then-split WE policy (force_split_merge), every
        # walker in a bin shares the same parent_id and therefore the same
        # xml -- without this cache we'd re-deserialize the same OpenMM state
        # bin_target_counts times per bin per iter. context.setState()
        # deep-copies into the context, and setVelocitiesToTemperature()
        # overwrites velocities anyway, so reusing the cached State is safe.
        parent_state_cache: dict[str, "omm.State"] = {}

        for seg in segments:
            xml_path = self._parent_xml_path(seg)
            parent_state = parent_state_cache.get(xml_path)
            if parent_state is None:
                parent_state = self._load_state(xml_path)
                parent_state_cache[xml_path] = parent_state
            # Reproducible per-(iter, seg) velocity-reinit seed (seg-keyed so
            # cached-parent walkers still get distinct, reproducible noise).
            vseed = (self._global_seed ^ (int(n_iter) * 2654435761)
                     ^ ((int(seg.seg_id) + 1) * 40503)) & 0x7FFFFFFF
            traj_xyz, final_state = self._integrate_one(parent_state, vseed)

            # Sanity: bail loudly if integrator blew up (shouldn't on vacuum
            # ADP at 1 fs, but easier to debug than silent NaNs).
            if not np.isfinite(traj_xyz).all():
                westpa.rc.pstatus(
                    f"BatchedADPPropagator: non-finite traj at iter {n_iter} "
                    f"seg {seg.seg_id}; freezing at parent."
                )
                traj_xyz = np.broadcast_to(
                    traj_xyz[0], traj_xyz.shape,
                ).copy()

            pcoord, phi, psi = self._compute_aux(traj_xyz)

            # Persist seg.xml (resume) + seg_traj.npy (full pcoord_len traj
            # for the coordinator's diagnostic re-evaluation; train.py only
            # needs seg.xml endpoints so it doesn't read this file).
            seg_dir = os.path.join(
                self.traj_dir, f"{n_iter:06d}", f"{seg.seg_id:06d}",
            )
            os.makedirs(seg_dir, exist_ok=True)
            with open(os.path.join(seg_dir, "seg.xml"), "w") as f:
                f.write(omm.XmlSerializer.serialize(final_state))
            np.save(os.path.join(seg_dir, "seg_traj.npy"), traj_xyz)

            seg.pcoord[:] = pcoord
            seg.data["positions"] = traj_xyz       # (pcoord_len, n_atoms, 3)
            seg.data["final_pos"] = traj_xyz[-1]   # (n_atoms, 3) -> west.h5
            seg.data["phi"] = phi
            seg.data["psi"] = psi
            seg.status = Segment.SEG_STATUS_COMPLETE

        return segments
