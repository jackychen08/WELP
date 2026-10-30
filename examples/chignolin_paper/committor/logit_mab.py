"""MABBinMapper variant that adapts on the committor logit log(q/(1-q)) instead of raw q.

The committor pcoord (column 0) ranges [0,1]. Binning uniformly in the logit
z = log(q/(1-q)) is *monotone* in q, so MAB's leading-edge / bottleneck semantics
are preserved exactly (unlike the q(1-q) fold in qfold_mab.py, which is 2-to-1
and therefore direction-blind). Because dq = q(1-q) dz, uniform-in-z bins are
WIDE in q near q=0.5 and NARROW near the basins: resolution is spent on the
exponentially small-committor tails, where walkers must be split repeatedly to
climb out of a basin. This is the classic rare-event / equal-weight-ratio
binning argument, and it is the opposite allocation from qfold.

q is clamped to [EPS, 1-EPS] before the logit so it can never reach exactly 0 or
1: the pcoord is float32 and sigmoid saturates to exactly 0.0/1.0 for |z| >~ 17,
which would give -inf/+inf and poison the bin assignment. EPS = 1e-6 matches the
clamp already applied by the training loss (committor/train.py:log_loss), so q is
never trusted outside that range anyway. Resulting logit range: +-log((1-EPS)/EPS)
~= +-13.8. Non-finite q (NaN) is mapped to the midpoint 0.5 -> z = 0.

Only the local copy of the coordinates used for *binning* is transformed; the
recorded pcoord (west.h5 column 0) stays the raw committor q, so every downstream
consumer (train.py, we_driver, the cross-sim peek) sees q unchanged. Recycling is
unaffected: it lives on a separate pcoord dim (in_target).

Plugged into west.cfg as the MAB sub-mapper:
    type: committor.logit_mab.LogitMABBinMapper
with the same nbins / direction / skip / bottleneck kwargs as MABBinMapper.
"""
import numpy as np
from westpa.core.binning.mab import MABBinMapper, map_mab

# Matches committor/train.py:log_loss clamp. Logit range ~ +-13.8.
EPS = 1e-6


def map_logit_mab(coords, mask, output, *args, **kwargs):
    """Map column 0 (q -> log(q/(1-q))) on a private copy, then defer to map_mab.

    ``coords`` arrives from the WE driver with the pcoord dims followed by the
    appended weight / isfinal columns; column 0 is the committor pcoord dim, so
    transforming only that column leaves the rest of MAB's machinery intact.
    """
    coords = np.array(coords, copy=True)
    q = np.asarray(coords[:, 0], dtype=np.float64)
    q = np.where(np.isfinite(q), q, 0.5)        # NaN/inf -> midpoint (z = 0)
    q = np.clip(q, EPS, 1.0 - EPS)              # never exactly 0 or 1
    coords[:, 0] = np.log(q / (1.0 - q))
    return map_mab(coords, mask, output, *args, **kwargs)


class LogitMABBinMapper(MABBinMapper):
    """MAB that adapts on log(q/(1-q)); see module docstring. Drop-in for MABBinMapper."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MABBinMapper.__init__ set self.func = map_mab; swap in the logit variant.
        self.func = map_logit_mab
