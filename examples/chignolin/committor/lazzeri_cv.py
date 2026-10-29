"""Pinned Lazzeri contact Q and the separate whole-protein feature coordinates.

Q is computed on original coordinates with the original periodic box, before
centering or unwrapping coordinates for the existing distance network.
"""
from pathlib import Path
import hashlib
import os
import runpy
import threading

import mdtraj as md
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "common_files/lazzeri/source"
FOLDED_Q_MIN = 0.99
UNFOLDED_Q_MAX = 0.01
PINNED = {
    "inA.py": "dc998ff21350d359b11e1ca70bb8ab75b1e8b6286ba705ddc2c18a11a4ffd0aa",
    "inB.py": "2ba234bbf3a99c0664713b441001f45a85922461429389cd4d1a2f7cb561f059",
    "signature.npy": "d88eae6bb2f9dcce60c9706a5a8b93cfa8ae903896ad9a2cedc69e060af7ad8b",
    "r0.npy": "2a3b8275262ef86da878d181714ddacca3b221cc2808ab26ead152e96021f64b",
}
_LOAD_LOCK = threading.Lock()


class ArchiveQ:
    def __init__(self, source=SOURCE):
        source = Path(source).resolve()
        for name, expected in PINNED.items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Pinned Q input changed: {name}")
        # Original scripts load relative paths. Execute once at startup, restore
        # cwd even on failure, and retain their functions and original arrays.
        with _LOAD_LOCK:
            previous = Path.cwd()
            try:
                os.chdir(source)
                self._a = runpy.run_path("inA.py")["inA"]
                self._b = runpy.run_path("inB.py")["inB"]
            finally:
                os.chdir(previous)
        self.topology = md.load_topology(str(source / "mol.pdb"))
        values = self._a.__globals__
        if (self.topology.n_atoms != 166 or values["signature"].shape != (74, 2)
                or values["r0"].shape != (74,)):
            raise ValueError("Expected the archived 166-atom, 74-contact definition")

    def evaluate(self, positions_nm, boxes_nm):
        xyz = np.asarray(positions_nm, dtype=np.float32)
        if xyz.ndim == 2:
            xyz = xyz[None]
        if xyz.shape[1:] != (166, 3) or not np.isfinite(xyz).all():
            raise ValueError("Q requires finite (frames,166,3) original protein coordinates")
        box = np.asarray(boxes_nm, dtype=np.float32)
        if box.shape == (3, 3):
            box = np.broadcast_to(box, (len(xyz), 3, 3)).copy()
        if (box.shape != (len(xyz), 3, 3) or not np.isfinite(box).all()
                or np.any(np.linalg.det(box) <= 0)):
            raise ValueError("Periodic Q requires one valid box per frame")
        trajectory = md.Trajectory(xyz, self.topology)
        trajectory.unitcell_vectors = box
        values = self._a.__globals__
        with np.errstate(over="ignore"):
            q = values["_best_hummer_q"](trajectory, values["signature"], values["r0"])
            in_a, in_b = self._a(trajectory), self._b(trajectory)
        if not np.isfinite(q).all():
            raise ValueError("Archived Q produced a nonfinite score")
        return np.asarray(in_a), np.asarray(in_b), q


def protein_tree(bonds, n_atoms=166):
    neighbors = [[] for _ in range(n_atoms)]
    for a, b in np.asarray(bonds, dtype=int):
        neighbors[a].append(b)
        neighbors[b].append(a)
    seen, queue, tree = {0}, [0], []
    for a in queue:
        for b in neighbors[a]:
            if b not in seen:
                tree.append((a, b))
                seen.add(b)
                queue.append(b)
    if len(seen) != n_atoms:
        raise ValueError("Protein bonds do not connect all 166 atoms")
    return tree


def whole_protein(positions_nm, box_nm, tree):
    """Unwrap the fixed-box bonded protein, then center for the unchanged NN."""
    raw = np.asarray(positions_nm, dtype=np.float64)
    box = np.asarray(box_nm, dtype=np.float64)
    xyz = raw.copy()
    inverse = np.linalg.inv(box)
    for a, b in tree:
        delta = (raw[b] - raw[a]) @ inverse
        delta -= np.rint(delta)
        xyz[b] = xyz[a] + delta @ box
    return (xyz - xyz.mean(axis=0)).astype(np.float32)
