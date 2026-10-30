"""Custom WEDriver that preserves NEWTRAJ status during merging.

Stock westpa.core.WEDriver._merge_walkers picks the merge "gparent" by
weight-proportional sampling. After recycling, the new NEWTRAJ segment
(parent_id < 0, weight ~ recycled_weight) is placed in its istate's bin and
typically has weight orders of magnitude below the bin's normal-weight
walkers. Weight-proportional gparent selection then almost always picks a
non-NEWTRAJ as gparent, the glom inherits a positive parent_id, the istate
is freed back to avail_initial_states (with iter_used reset to None), and
the recycling event becomes a no-op for trajectory generation: the walker
that should have started at a bstate at iter N+1 instead continues from the
gparent's last frame.

For this codebase that breaks the steady-state setup. sim_BA walkers reach
the A basin and get marked endpoint_type=3, but their B-bstate replacements
never propagate -- so the B-basin walker population starves out, the NN
trains on a degenerate distribution, and committor predictions collapse
toward 0. (Confirmed empirically: sim_BA's
``parent_id < 0`` next-iter count is ~5% of its endpoint_type=3 count vs.
~99% for sim_AB, where the start-bin's high baseline weight prevents merges
in the recycled walker's destination bin.)

The override below forces the highest-weight NEWTRAJ in any merge group to
be the gparent. This preserves the negative parent_id, keeps the istate
marked as used, and routes the next iter's propagation through the bstate
as the recycling logic intends. WE statistics (weight conservation, bin
populations) are unchanged.
"""
from __future__ import annotations

import logging

import numpy as np

from westpa.core.binning.mab_driver import MABDriver
from westpa.core.segment import Segment

log = logging.getLogger(__name__)


class CommittorWEDriver(MABDriver):
    """MABDriver subclass that prefers NEWTRAJ segments as merge gparent.

    Inherits MABDriver's MAB-aware ``assign`` (which packs weight + initial/final
    flag into the bin_mapper input so MAB can adapt). Only override is the
    ``_merge_walkers`` gparent selection rule.
    """

    def _merge_walkers(self, segments, cumul_weight, bin):
        """Merge segments, preferring any NEWTRAJ (parent_id < 0) as gparent.

        Falls back to the stock weight-proportional selection when no
        NEWTRAJ is present in ``segments``.
        """
        newtraj = [s for s in segments if s.parent_id is not None and s.parent_id < 0]
        if not newtraj:
            return super()._merge_walkers(segments, cumul_weight, bin)

        # If multiple NEWTRAJs are merging together, pick the highest-weight
        # one. (Ties broken by parent_id to keep this deterministic; both are
        # non-positive for NEWTRAJ so subtracting works.)
        gparent_seg = max(newtraj, key=lambda s: (s.weight, -s.parent_id))

        if cumul_weight is None:
            cumul_weight = np.add.accumulate([s.weight for s in segments])

        glom = Segment(
            n_iter=segments[0].n_iter,
            weight=cumul_weight[len(segments) - 1],
            status=Segment.SEG_STATUS_PREPARED,
            pcoord=self.system.new_pcoord_array(),
        )
        glom.parent_id = gparent_seg.parent_id
        glom.pcoord[0, :] = gparent_seg.pcoord[0, :]
        # Segment.initial_state_id is a property computed from parent_id, so
        # setting glom.parent_id < 0 already gives the right initial_state_id.

        glom.wtg_parent_ids = set()
        for segment in segments:
            glom.wtg_parent_ids |= segment.wtg_parent_ids

        for segment in segments:
            if segment is gparent_seg:
                # gparent's NEWTRAJ status carries to glom; nothing to do.
                continue
            if segment.parent_id >= 0:
                self._parent_map[segment.parent_id].endpoint_type = (
                    Segment.SEG_ENDPOINT_MERGED
                )
            else:
                # Other NEWTRAJ being merged away: free its istate.
                if segment.initial_state_id in {
                    seg.initial_state_id for seg in bin
                }:
                    log.debug('initial state in use by other walker; not removing')
                else:
                    initial_state = self.used_initial_states.pop(
                        segment.initial_state_id
                    )
                    log.debug(
                        'freeing initial state %r for future use (merged)',
                        initial_state,
                    )
                    self.avail_initial_states[initial_state.state_id] = initial_state
                    initial_state.iter_used = None

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                'merging (%d) %r into 1 (NEWTRAJ-preserving): %r',
                len(segments), segments, glom,
            )

        return glom, gparent_seg
