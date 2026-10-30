"""Chignolin state collective variable.

Grant (Mitchell & Rotskoff, JCTC 2024, sec 5.2/9.4) defines the folded/unfolded
state by the **SUM** of four beta-hairpin backbone H-bond distances of CLN025
(YYDPETGTWY):

    Tyr1 N  -- Tyr10 O
    Tyr1 O  -- Tyr10 H
    Asp3 N  -- Thr8  O
    Asp3 O  -- Thr8  N

    folded  A : sum < 1.75 nm
    unfolded B : sum > 3.0 nm

IMPORTANT -- distances are RAW (non-periodic) Euclidean distances computed on the
whole peptide. The peptide is a single bonded chain that OpenMM keeps whole, so
minimum-image must NEVER be applied: the unfolded basin (sum > 3 nm) exceeds
box/2 (~2.75 nm for a 5.5 nm box), and imaging would fold that signal back under
box/2 and destroy the basin-B definition. Always slice the peptide out of
coordinates in which it is whole, then use these raw distances.

This is the single source of truth for the CV, imported by the system builder,
the boundary-ensemble generator, the WESTPA propagator, and the greedy arm.
"""
import numpy as np

# (residue-number [1-indexed], atom-name) pairs, in Grant's order.
HBOND_PAIRS = (
    ((1, "N"), (10, "O")),
    ((1, "O"), (10, "H")),
    ((3, "N"), (8, "O")),
    ((3, "O"), (8, "N")),
)

# Basin cutoffs on the summed CV (nm). Kept here as the canonical numbers;
# committor.config re-exports them so the WESTPA/committor code has one place to
# read them from.
FOLDED_SUM_MAX = 1.75      # sum < this  -> folded   (basin A)
UNFOLDED_SUM_MIN = 3.0     # sum > this  -> unfolded (basin B)

# Indices into HBOND_PAIRS of the two Asp3--Thr8 backbone H-bonds. These are the
# INNER pair of the beta-hairpin; plotted against each other they give a
# chignolin analogue of a Ramachandran plot for the folded state (see
# committor.chignolin_plots). The Tyr1--Tyr10 pair (indices 0,1) is the outer
# pair, which frays first and is a poorer folded-state discriminator.
ASP3_THR8_PAIRS = (2, 3)
ASP3_THR8_LABELS = (
    "d(Asp3 N — Thr8 O)  [nm]",
    "d(Asp3 O — Thr8 N)  [nm]",
)


def resolve_pair_indices(mdtraj_topology):
    """Resolve the 4 H-bond pairs to atom indices in a PEPTIDE mdtraj topology.

    Returns an int64 array of shape (4, 2). Raises KeyError with a clear message
    if any (residue, atom) is missing -- e.g. a wrong terminal-oxygen name.
    """
    def find(resnum_1indexed, name):
        res = mdtraj_topology.residue(resnum_1indexed - 1)
        for a in res.atoms:
            if a.name == name:
                return a.index
        have = sorted(a.name for a in res.atoms)
        raise KeyError(
            f"atom {name!r} not in residue {resnum_1indexed} ({res.name}); "
            f"atoms present: {have}"
        )

    return np.array(
        [[find(*p0), find(*p1)] for (p0, p1) in HBOND_PAIRS], dtype=np.int64
    )


def hbond_distances(peptide_xyz, pair_idx, which=None):
    """Per-pair RAW (non-periodic) H-bond distances, in nm.

    The primitive behind both the CV (`hbond_sum`) and the diagnostic H-bond
    plane (`asp3_thr8_distances`), so the two can never drift apart.

    peptide_xyz : (F, N_pep, 3) or (N_pep, 3), nm, peptide WHOLE.
    pair_idx    : (4, 2) from resolve_pair_indices.
    which       : optional sequence of pair indices to select (default: all 4).
    Returns     : (F, n_sel) [or (n_sel,) if a single frame was passed].
    """
    xyz = np.asarray(peptide_xyz, dtype=np.float64)
    single = xyz.ndim == 2
    if single:
        xyz = xyz[None]
    pi = pair_idx if which is None else np.asarray(pair_idx)[list(which)]
    a = xyz[:, pi[:, 0], :]                 # (F, n_sel, 3)
    b = xyz[:, pi[:, 1], :]                 # (F, n_sel, 3)
    d = np.linalg.norm(a - b, axis=-1)      # (F, n_sel)
    return d[0] if single else d


def hbond_sum(peptide_xyz, pair_idx):
    """Summed CV, in nm.

    peptide_xyz : (F, N_pep, 3) or (N_pep, 3), nm, peptide WHOLE.
    pair_idx    : (4, 2) from resolve_pair_indices.
    Returns     : (F,) [or scalar float if a single frame was passed] -- the SUM
                  of the 4 raw (non-periodic) Euclidean pair distances.
    """
    d = hbond_distances(peptide_xyz, pair_idx)
    if d.ndim == 1:                          # single frame -> (4,)
        return float(d.sum())
    return d.sum(axis=-1)                    # (F,)


def asp3_thr8_distances(peptide_xyz, pair_idx):
    """The two Asp3--Thr8 backbone H-bond distances, in nm.

    Returns (F, 2) [or (2,) for a single frame]: columns are
    [d(Asp3 N -- Thr8 O), d(Asp3 O -- Thr8 N)], matching ASP3_THR8_LABELS.
    """
    return hbond_distances(peptide_xyz, pair_idx, which=ASP3_THR8_PAIRS)
