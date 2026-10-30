"""ADP system data for Graphormer3D.

Provides per-atom features (one-hot element type) and pairwise edge types
(bonded vs non-bonded) derived from the molecular topology.
"""
import mdtraj
import torch


class ADPSystem:
    """Alanine dipeptide system: atom features and bond connectivity.

    Attributes:
        features:    (1, n_atoms, 4) one-hot element type [H, C, N, O]
        edge_types:  (1, n_atoms, n_atoms) long tensor, 1 if bonded else 0
        n_atoms:     22 for ADP
        topology:    mdtraj.Topology
    """

    def __init__(self, pdb_path):
        traj = mdtraj.load(pdb_path)
        top = traj.topology
        n_atoms = top.n_atoms  # 22

        # One-hot atom features: H=0, C=1, N=2, O=3
        elem_to_idx = {"H": 0, "C": 1, "N": 2, "O": 3}
        features = torch.zeros(1, n_atoms, len(elem_to_idx))
        for atom in top.atoms:
            features[0, atom.index, elem_to_idx[atom.element.symbol]] = 1.0

        # Bond connectivity: 0 = no bond, 1 = bonded
        edge_types = torch.zeros(1, n_atoms, n_atoms, dtype=torch.long)
        for bond in top.bonds:
            i, j = bond[0].index, bond[1].index
            edge_types[0, i, j] = 1
            edge_types[0, j, i] = 1

        self.features = features          # (1, 22, 4)
        self.edge_types = edge_types      # (1, 22, 22)
        self.n_atoms = n_atoms
        self.topology = top
