"""Train the committor using WESTPA parent->child Markov transitions.

Self-consistent criterion (Chapman-Kolmogorov / Markov property):
    q(parent_pos) = mean( q(child_pos) )
    q = 0 in basin A,  q = 1 in basin B

Loss (symmetrized log-space, Rotskoff arXiv:2408.05879):
    L_mean = (log q - log q*)^2 + (log(1-q) - log(1-q*))^2
    L_var  = (var_pred - var_target)^2

Features: PER (prioritized replay), distributional (mean + variance).

Data (two-sim steady-state setup):
    - Parent-child segment mappings pooled from ALL ``cfg.sim_sources``
      (sim_AB + sim_BA for adp_test_mab_ss)
    - 3D positions loaded from each source's seg.xml files via mdtraj
    - Basin membership from phi/psi aux datasets (pcoord is (q, rmsd_B))

Recycling events do not produce parent-child training pairs: recycled
children have ``parent_id < 0`` in WESTPA's seg_index, so the filter
``0 <= pid`` excludes them from the 1-step pairs. The n-step bootstrap's
forward lineage tracer also terminates at recycling boundaries because
recycled walkers are keyed under pid=-1 in the forward map, which is
unreachable from any positive anchor sid.
"""
import math
import os
import shutil
import time
import numpy as np
import h5py
import mdtraj
import torch
import torch.nn.functional as F
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from committor.config import CommittorConfig
from committor.system import ADPSystem
from committor.network import build_committor_net


# ---------------------------------------------------------------------------
# BKE weight scheduling (cosine warm-up in WESTPA-iter units)
# ---------------------------------------------------------------------------

def bke_weight_for_iter(cfg, iteration):
    """Return the effective BKE weight for the given WESTPA iteration.

    Cosine warm-up from 0 at `bke_anneal_start_iter` to
    `bke_loss_weight` at `bke_anneal_end_iter`. Outside the ramp window:
      - iteration < start: 0
      - iteration >= end : full target weight
    If anneal endpoints are unset (-1), falls back to a hard zero/full
    gate at `bke_warmup_iters`.
    """
    target = float(cfg.bke_loss_weight)
    if target <= 0.0:
        return 0.0
    s = int(cfg.bke_anneal_start_iter)
    e = int(cfg.bke_anneal_end_iter)
    if s >= 0 and e > s:
        if iteration < s:
            return 0.0
        if iteration >= e:
            return target
        progress = (iteration - s) / float(e - s)
        return target * 0.5 * (1.0 - math.cos(math.pi * progress))
    if iteration < cfg.bke_warmup_iters:
        return 0.0
    return target


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _list_iter_keys(h5):
    """Return sorted (iter_int) list of available iterations in h5['iterations']."""
    out = []
    for k in h5["iterations"].keys():
        try:
            out.append(int(k.split("_")[-1]))
        except ValueError:
            pass
    out.sort()
    return out


def _iter_has_auxdata(h5, iter_key):
    """Return True iff this iteration is FULLY written end-to-end.

    Two gates are needed because WESTPA allocates pcoord/auxdata with the
    full per-iter shape at iter creation, filled with HDF5 default 0:

      (1) Every segment row must be SEG_STATUS_COMPLETE (=2). Until then,
          that row's pcoord / phi / psi is the fill value 0 -- reading it
          paints (phi=0, psi=0, q=0) points into downstream plots and
          pollutes the training cache.

      (2) Iter N+1 must already exist. CommittorDriver.pre_we (which
          overwrites pcoord[:, :, 0] with NN-predicted q) fires once per
          iter, on the master, AFTER every segment finishes and BEFORE
          WE creates iter N+1. Iter N+1 existing is the cheapest proof
          that pre_we has fired for iter N -- without this gate, even
          all-status-2 rows can still carry the q=0 placeholder that
          get_distance.py writes.
    """
    grp = h5[iter_key]
    if "auxdata" not in grp:
        return False
    if "phi" not in grp["auxdata"] or "psi" not in grp["auxdata"]:
        return False
    try:
        n_seg = grp["seg_index"].shape[0]
        n_pc = grp["pcoord"].shape[0]
        n_phi = grp["auxdata"]["phi"].shape[0]
        n_psi = grp["auxdata"]["psi"].shape[0]
    except (KeyError, AttributeError):
        return False
    if not (n_seg > 0 and n_seg == n_pc == n_phi == n_psi):
        return False
    try:
        status = grp["seg_index"]["status"]
    except (KeyError, ValueError):
        return False
    if not (status == 2).all():
        return False
    it_num = int(iter_key.rsplit("_", 1)[-1])
    if f"iter_{it_num + 1:08d}" not in h5["iterations"]:
        return False
    return True


def _read_final_pos(f, iter_key, top, traj_dir, it_num):
    """Return final-frame positions (n_segs, n_atoms, 3) for an iteration.

    Fast path: read the ``final_pos`` aux dataset BatchedADPPropagator writes
    into west.h5 — one bulk h5 read per iter, no per-segment file opens.

    Fallback: legacy ExecutablePropagator runs only have seg.xml on disk;
    mdtraj.load each one. ~50–100x slower but keeps old runs working.
    """
    grp = f[iter_key]
    if "auxdata" in grp and "final_pos" in grp["auxdata"]:
        try:
            return grp["auxdata"]["final_pos"][...]
        except Exception:
            pass

    n_segs = int(grp["seg_index"].shape[0])
    out = np.full(
        (n_segs, len(top.select("all")), 3),
        np.nan, dtype=np.float32,
    )
    for sid in range(n_segs):
        seg_xml = os.path.join(
            traj_dir, f"{it_num:06d}", f"{sid:06d}", "seg.xml"
        )
        if not os.path.exists(seg_xml):
            continue
        try:
            out[sid] = mdtraj.load(seg_xml, top=top).xyz[0]
        except Exception:
            continue
    return out


def collect_training_data(cfg):
    """Pool parent->child mappings from EVERY source in cfg.sim_sources.

    Caches to cfg.cache_file with a per-source last-iter dict so subsequent
    calls only read new iterations from each sim. Returned lists are a
    single combined buffer over all sources (shuffled by natural reading
    order, FIFO-trimmed to cfg.replay_buffer_size).

    Returns:
        parent_pos:     list of (n_atoms*3,) float32 arrays
        child_groups:   list of lists of (n_atoms*3,) float32 arrays
        child_pcoords:  list of (n_children, 4) float32 arrays (q, rmsd_B, phi, psi).
        source_list:    list of str sim labels, one per group (parallel to parent_pos)
    """
    parent_pos_list = []
    child_groups = []
    child_pcoord_list = []
    source_list = []
    last_iter_by_source = {src["label"]: 1 for src in cfg.sim_sources}

    if os.path.exists(cfg.cache_file):
        cache = np.load(cfg.cache_file, allow_pickle=True)
        parent_pos_list = list(cache["parent_pos"])
        child_groups = list(cache["child_groups"])
        child_pcoord_list = list(cache["child_pcoords"])
        if "source_list" in cache.files:
            source_list = list(cache["source_list"])
        else:
            source_list = ["unknown"] * len(parent_pos_list)
        if "last_iter_by_source" in cache.files:
            cached = cache["last_iter_by_source"].item()
            for label, it in cached.items():
                if label in last_iter_by_source:
                    last_iter_by_source[label] = int(it)
        elif "last_iter" in cache.files:
            label0 = cfg.sim_sources[0]["label"]
            last_iter_by_source[label0] = int(cache["last_iter"])
        print(f"  [collect] loaded cache: {len(parent_pos_list)} groups, "
              f"last_iter_by_source={last_iter_by_source}")

    top = mdtraj.load(cfg.pdb_path).topology
    new_groups = 0

    for src in cfg.sim_sources:
        label = src["label"]
        h5_path = src["west_h5"]
        traj_dir = src["traj_dir"]
        if not os.path.exists(h5_path):
            print(f"  [collect] {label}: west.h5 not found at {h5_path}, skipping")
            continue

        try:
            with h5py.File(h5_path, "r", locking=False) as f:
                iter_ints = _list_iter_keys(f)
                if not iter_ints:
                    continue
                max_iter = iter_ints[-1]
                start_iter = max(2, last_iter_by_source[label] + 1)
                src_new = 0
                for it in range(start_iter, max_iter + 1):
                    curr = f"iterations/iter_{it:08d}"
                    prev = f"iterations/iter_{it - 1:08d}"
                    if curr not in f or prev not in f:
                        # Stop advancing here; revisit on next call so we
                        # don't skip iter `it` forever. (HWM policy)
                        break
                    if not _iter_has_auxdata(f, curr):
                        # Partial write; stop and retry from `it` next call.
                        break

                    seg_idx = f[curr]["seg_index"][...]
                    curr_pc = f[curr]["pcoord"][...][:, -1, :]
                    prev_pc = f[prev]["pcoord"][...][:, -1, :]
                    curr_phi = f[curr]["auxdata"]["phi"][...][:, -1]
                    curr_psi = f[curr]["auxdata"]["psi"][...][:, -1]

                    # final_pos aux dataset: (n_segs, n_atoms, 3) float32 written
                    # by BatchedADPPropagator. Bulk-read once per iter; falls
                    # back to per-segment mdtraj.load(seg.xml) for legacy runs.
                    curr_final_pos = _read_final_pos(
                        f, curr, top, traj_dir, it,
                    )
                    prev_final_pos = _read_final_pos(
                        f, prev, top, traj_dir, it - 1,
                    )

                    # parent_id < 0 excludes recycled children (and new-traj children)
                    groups = {}
                    for sid in range(len(seg_idx)):
                        pid = int(seg_idx[sid]["parent_id"])
                        if 0 <= pid < len(prev_pc):
                            groups.setdefault(pid, []).append(sid)

                    for pid, child_ids in groups.items():
                        if prev_final_pos is None or pid >= len(prev_final_pos):
                            continue
                        parent_flat = prev_final_pos[pid].flatten().astype(np.float32)

                        c_positions = []
                        c_pcoords = []
                        for cid in child_ids:
                            if curr_final_pos is None or cid >= len(curr_final_pos):
                                continue
                            c_positions.append(
                                curr_final_pos[cid].flatten().astype(np.float32)
                            )
                            c_pcoords.append(np.array([
                                curr_pc[cid, 0], curr_pc[cid, 1],
                                float(curr_phi[cid]), float(curr_psi[cid]),
                            ], dtype=np.float32))

                        if c_positions:
                            parent_pos_list.append(parent_flat)
                            child_groups.append(c_positions)
                            child_pcoord_list.append(
                                np.array(c_pcoords, dtype=np.float32)
                            )
                            source_list.append(label)
                            src_new += 1

                    last_iter_by_source[label] = it

                print(f"  [collect] {label}: {src_new} new groups "
                      f"(iters {start_iter}..{last_iter_by_source[label]})")
                new_groups += src_new
        except Exception as e:
            print(f"  [collect] {label}: error reading h5: {e}")

    print(f"  [collect] total {new_groups} new groups; "
          f"buffer size now {len(parent_pos_list)}")

    if len(parent_pos_list) > cfg.replay_buffer_size:
        excess = len(parent_pos_list) - cfg.replay_buffer_size
        parent_pos_list = parent_pos_list[excess:]
        child_groups = child_groups[excess:]
        child_pcoord_list = child_pcoord_list[excess:]
        source_list = source_list[excess:]
        print(f"  [collect] FIFO trim: dropped {excess}, "
              f"keeping {len(parent_pos_list)}")

    if parent_pos_list:
        np.savez(
            cfg.cache_file,
            last_iter_by_source=np.array(last_iter_by_source, dtype=object),
            parent_pos=np.array(parent_pos_list, dtype=object),
            child_groups=np.array(child_groups, dtype=object),
            child_pcoords=np.array(child_pcoord_list, dtype=object),
            source_list=np.array(source_list),
        )

    return parent_pos_list, child_groups, child_pcoord_list, source_list


def collect_nstep_data(cfg, n_steps):
    """Build n-step ancestor->descendant groups by tracing lineages in
    each source's west.h5, then pooling across all cfg.sim_sources.

    Recycling: lineages that cross a recycling event (child parent_id < 0)
    terminate naturally in the forward tracer — recycled walkers are
    keyed under pid=-1 in fwd[it], unreachable from a positive anchor.

    Cached separately from 1-step data in nstep_cache.npz with a per-source
    last-ancestor-iter dict.
    """
    nstep_cache = os.path.join(cfg.data_dir, "nstep_cache.npz")

    anc_pos_list = []
    desc_groups = []
    desc_pcoord_list = []
    source_list = []
    last_anc_by_source = {src["label"]: 1 for src in cfg.sim_sources}

    if os.path.exists(nstep_cache):
        cache = np.load(nstep_cache, allow_pickle=True)
        anc_pos_list = list(cache["ancestor_pos"])
        desc_groups = list(cache["desc_groups"])
        desc_pcoord_list = list(cache["desc_pcoords"])
        if "source_list" in cache.files:
            source_list = list(cache["source_list"])
        else:
            source_list = ["unknown"] * len(anc_pos_list)
        if "last_anc_by_source" in cache.files:
            cached = cache["last_anc_by_source"].item()
            for label, it in cached.items():
                if label in last_anc_by_source:
                    last_anc_by_source[label] = int(it)
        elif "last_anc_iter" in cache.files:
            label0 = cfg.sim_sources[0]["label"]
            last_anc_by_source[label0] = int(cache["last_anc_iter"])
        print(f"  [nstep] loaded cache: {len(anc_pos_list)} groups, "
              f"last_anc_by_source={last_anc_by_source}")

    top = mdtraj.load(cfg.pdb_path).topology
    new_groups = 0

    for src in cfg.sim_sources:
        label = src["label"]
        h5_path = src["west_h5"]
        traj_dir = src["traj_dir"]
        if not os.path.exists(h5_path):
            continue

        try:
            with h5py.File(h5_path, "r", locking=False) as f:
                iter_ints = _list_iter_keys(f)
                if not iter_ints:
                    continue
                up_to_iter = iter_ints[-1]
                start_anc = max(2, last_anc_by_source[label] + 1)
                max_anc = up_to_iter - n_steps
                if start_anc > max_anc:
                    continue

                seg_indices = {}
                pcoord_maps = {}
                phi_maps = {}
                psi_maps = {}
                final_pos_maps = {}
                for it in range(start_anc, max_anc + n_steps + 1):
                    key = f"iterations/iter_{it:08d}"
                    if key not in f or not _iter_has_auxdata(f, key):
                        continue
                    seg_indices[it] = f[key]["seg_index"][...]
                    pcoord_maps[it] = f[key]["pcoord"][...][:, -1, :]
                    phi_maps[it] = f[key]["auxdata"]["phi"][...][:, -1]
                    psi_maps[it] = f[key]["auxdata"]["psi"][...][:, -1]
                    final_pos_maps[it] = _read_final_pos(
                        f, key, top, traj_dir, it,
                    )

                fwd = {}
                for it in range(start_anc + 1, max_anc + n_steps + 1):
                    if it not in seg_indices:
                        continue
                    fwd[it] = {}
                    for sid in range(len(seg_indices[it])):
                        pid = int(seg_indices[it][sid]["parent_id"])
                        fwd[it].setdefault(pid, []).append(sid)

                src_new = 0
                # Contiguous high-water mark: only advance the cursor as far
                # as the iter we can fully process. A missing or partially
                # written iter stops advancement so we retry on the next
                # coordinator cycle rather than skipping it permanently.
                hwm = last_anc_by_source[label]
                for anc_iter in range(start_anc, max_anc + 1):
                    if anc_iter not in seg_indices:
                        break
                    desc_iter = anc_iter + n_steps
                    if desc_iter not in pcoord_maps:
                        break
                    if not all(it in fwd for it in range(anc_iter + 1, desc_iter + 1)):
                        break

                    n_segs = len(seg_indices[anc_iter])
                    for anc_sid in range(n_segs):
                        frontier = {anc_sid}
                        broken = False
                        for it in range(anc_iter + 1, desc_iter + 1):
                            next_frontier = set()
                            for sid in frontier:
                                next_frontier.update(fwd[it].get(sid, []))
                            if not next_frontier:
                                broken = True
                                break
                            frontier = next_frontier

                        if broken or not frontier:
                            continue

                        anc_arr = final_pos_maps.get(anc_iter)
                        if anc_arr is None or anc_sid >= len(anc_arr):
                            continue
                        anc_pos = anc_arr[anc_sid]
                        if not np.isfinite(anc_pos).all():
                            continue
                        anc_flat = anc_pos.flatten().astype(np.float32)

                        desc_arr = final_pos_maps.get(desc_iter)
                        d_positions = []
                        d_pcoords = []
                        for dsid in frontier:
                            if desc_arr is None or dsid >= len(desc_arr):
                                continue
                            d_pos = desc_arr[dsid]
                            if not np.isfinite(d_pos).all():
                                continue
                            d_positions.append(
                                d_pos.flatten().astype(np.float32)
                            )
                            d_pcoords.append(np.array([
                                pcoord_maps[desc_iter][dsid, 0],
                                pcoord_maps[desc_iter][dsid, 1],
                                float(phi_maps[desc_iter][dsid]),
                                float(psi_maps[desc_iter][dsid]),
                            ], dtype=np.float32))

                        if d_positions:
                            anc_pos_list.append(anc_flat)
                            desc_groups.append(d_positions)
                            desc_pcoord_list.append(
                                np.array(d_pcoords, dtype=np.float32)
                            )
                            source_list.append(label)
                            src_new += 1
                    # Whole anc_iter processed successfully; advance HWM.
                    hwm = anc_iter

                last_anc_by_source[label] = hwm
                print(f"  [nstep] {label}: {src_new} new groups "
                      f"(anc iters {start_anc}..{hwm})")
                new_groups += src_new
        except Exception as e:
            print(f"  [nstep] {label}: error reading h5: {e}")

    print(f"  [nstep] total {new_groups} new groups; "
          f"buffer size now {len(anc_pos_list)}")

    if len(anc_pos_list) > cfg.replay_buffer_size:
        excess = len(anc_pos_list) - cfg.replay_buffer_size
        anc_pos_list = anc_pos_list[excess:]
        desc_groups = desc_groups[excess:]
        desc_pcoord_list = desc_pcoord_list[excess:]
        source_list = source_list[excess:]
        print(f"  [nstep] FIFO trim: dropped {excess}, "
              f"keeping {len(anc_pos_list)}")

    if anc_pos_list:
        np.savez(
            nstep_cache,
            last_anc_by_source=np.array(last_anc_by_source, dtype=object),
            ancestor_pos=np.array(anc_pos_list, dtype=object),
            desc_groups=np.array(desc_groups, dtype=object),
            desc_pcoords=np.array(desc_pcoord_list, dtype=object),
            source_list=np.array(source_list),
        )

    return anc_pos_list, desc_groups, desc_pcoord_list, source_list


# ---------------------------------------------------------------------------
# Self-consistent targets
# ---------------------------------------------------------------------------

def compute_targets(net, child_groups, child_pcoords, indices, cfg, device):
    """Compute target mean and variance of q for a minibatch of parents.

    For each parent in `indices`:
      - target_mean = mean(q(children))
      - target_var  = var(q(children))
    where:
      - q = 0 if child in basin A
      - q = 1 if child in basin B
      - q = net.forward_mean(child_pos) otherwise (detached, no gradient)

    Args:
        indices: 1D array of parent group indices to compute targets for.
    Returns:
        targets_mean: (len(indices),) tensor of target committor means.
        targets_var:  (len(indices),) tensor of target committor variances.
    """
    # Collect non-basin children for batch prediction
    nonbasin_positions = []
    nonbasin_lookup = {}  # (local_idx, child_idx) -> index in nonbasin_positions

    for local_i, global_i in enumerate(indices):
        pcs = child_pcoords[global_i]
        for j in range(len(pcs)):
            # pcoord columns are (q, rmsd_B, phi, psi)
            phi, psi = float(pcs[j, 2]), float(pcs[j, 3])
            if not cfg.in_basin_a(phi, psi) and not cfg.in_basin_b(phi, psi):
                nonbasin_lookup[(local_i, j)] = len(nonbasin_positions)
                nonbasin_positions.append(child_groups[global_i][j])

    # Batch-predict non-basin children (chunked to limit GPU memory).
    # Clamp the network output to [eps, 1-eps] before it can land in the
    # target mean. In float32, sigmoid(out < ~-87) underflows to *exact* 0.0
    # (and similarly to 1.0 above ~+87); without this clamp, an interior
    # parent whose children all underflow gets a target_mean that is
    # bit-exactly 0.0/1.0 and is then misclassified as a q*=0/q*=1 boundary
    # parent by the strict equality checks in log_loss_per_sample and the
    # stratified sampler. That misclassification cascades into a self-
    # reinforcing q-collapse (see the iter ~4400 collapse on
    # ss_recyclefix_grantarch_bdmin1_bke0p2anneal100to500).
    _TARGET_FLOOR_EPS = 1e-6
    nonbasin_q = np.empty(len(nonbasin_positions), dtype=np.float32)
    if nonbasin_positions:
        all_pos = np.array(nonbasin_positions, dtype=np.float32)
        chunk_size = cfg.batch_size  # reuse training batch_size for inference chunks
        with torch.no_grad():
            for start in range(0, len(all_pos), chunk_size):
                end = min(start + chunk_size, len(all_pos))
                pos_t = torch.tensor(all_pos[start:end], device=device)
                nonbasin_q[start:end] = (
                    net.forward_mean(pos_t).cpu().numpy()
                    .clip(_TARGET_FLOOR_EPS, 1.0 - _TARGET_FLOOR_EPS)
                )

    # Assemble per-parent targets (mean and variance)
    targets_mean = np.zeros(len(indices), dtype=np.float32)
    targets_var = np.zeros(len(indices), dtype=np.float32)
    for local_i, global_i in enumerate(indices):
        pcs = child_pcoords[global_i]
        q_vals = []
        for j in range(len(pcs)):
            phi, psi = float(pcs[j, 2]), float(pcs[j, 3])
            if cfg.in_basin_a(phi, psi):
                q_vals.append(0.0)
            elif cfg.in_basin_b(phi, psi):
                q_vals.append(1.0)
            else:
                q_vals.append(float(nonbasin_q[nonbasin_lookup[(local_i, j)]]))
        targets_mean[local_i] = np.mean(q_vals)
        targets_var[local_i] = np.var(q_vals)

    return (torch.tensor(targets_mean, dtype=torch.float32, device=device),
            torch.tensor(targets_var, dtype=torch.float32, device=device))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def log_loss(q_pred, q_target):
    """Symmetrized log-space loss (Rotskoff arXiv:2408.05879) with boundary terms.

    Interior (0 < q* < 1):
        L = (log q - log q*)^2 + (log(1-q) - log(1-q*))^2

    Boundary terms (limits of the above as q*->0 or q*->1):
        q* = 0 (basin A):  L_A = log(1-q)^2   (pushes q toward 0)
        q* = 1 (basin B):  L_B = log(q)^2      (pushes q toward 1)
    """
    eps = 1e-6
    q_pred = q_pred.clamp(eps, 1 - eps)

    loss = torch.tensor(0.0, device=q_pred.device)
    n_terms = 0

    # Interior samples: standard symmetrized log-space loss
    interior = (q_target > 0.0) & (q_target < 1.0)
    if interior.sum() > 0:
        qp, qt = q_pred[interior], q_target[interior].clamp(eps, 1 - eps)
        loss = loss + (torch.log(qp) - torch.log(qt)).square().mean() \
                     + (torch.log(1 - qp) - torch.log(1 - qt)).square().mean()
        n_terms += 1

    # Boundary: basin A targets (q*=0) — push q toward 0
    basin_a = q_target == 0.0
    if basin_a.sum() > 0:
        loss = loss + torch.log(1 - q_pred[basin_a]).square().mean()
        n_terms += 1

    # Boundary: basin B targets (q*=1) — push q toward 1
    basin_b = q_target == 1.0
    if basin_b.sum() > 0:
        loss = loss + torch.log(q_pred[basin_b]).square().mean()
        n_terms += 1

    if n_terms == 0:
        return q_pred.sum() * 0.0

    return loss


def log_loss_per_sample(q_pred, q_target, logit=None, drop_boundary=False):
    """Per-sample symmetrized log-space loss for PER weighting.

    Returns a (batch,) tensor of per-sample losses (not reduced).

    LOGIT-SPACE PATH (AIB9; `logit` is not None)
    --------------------------------------------
    When the network hands us its pre-sigmoid output z we compute

        log q      = logsigmoid( z)
        log(1 - q) = logsigmoid(-z)

    instead of log(clamp(sigmoid(z))). This is NOT a numerical nicety -- it is required
    for the loss to have a gradient at all once the net saturates:

      * In float32, sigmoid(z) == 1.0 EXACTLY for z >~ 16.6. Then dq/dz = q(1-q) = 0
        exactly, so the gradient dies at the SIGMOID, before the loss sees anything.
      * torch.clamp(q, eps, 1-eps) has EXACTLY zero gradient outside its range, so the
        clamp is an absorbing barrier on top of that.
      Together they make saturation PERMANENT. Measured on AIB9: one AdamW step at lr=1e-3
      moved the logit from -2.6 to -125.6; from step 2 on, grad-norm was 0.00e+00 and the
      weights never moved again (still at their init values after 8,760 steps).

    logsigmoid keeps a strong RESTORING gradient at the rails:
        d/dz logsigmoid(-z) = -sigmoid(z) -> -1  as z -> +inf
    so a saturated net is pulled back. Verified: the same net recovers from logit -28.9 to
    +4.9 within 8 steps and converges to the target.

    q-SPACE PATH (`logit` is None) -- unchanged legacy behaviour for the ADP nets, whose
    logits stay in [-7.6, +6.8] and never approach the rails. Left bit-identical so the
    published ADP results are untouched.
    """
    eps = 1e-6
    q_target_c = q_target.clamp(eps, 1 - eps)

    if logit is not None:
        log_q = F.logsigmoid(logit)          # = log(q),     exact, gradient = 1 - q
        log_1mq = F.logsigmoid(-logit)       # = log(1 - q), exact, gradient = -q
        losses = torch.zeros_like(log_q)
    else:
        q_pred = q_pred.clamp(eps, 1 - eps)
        log_q = torch.log(q_pred)
        log_1mq = torch.log(1 - q_pred)
        losses = torch.zeros_like(q_pred)

    interior = (q_target > 0.0) & (q_target < 1.0)
    if interior.sum() > 0:
        qt = q_target_c[interior]
        losses[interior] = (log_q[interior] - torch.log(qt)).square() \
                         + (log_1mq[interior] - torch.log(1 - qt)).square()

    # Grant-style (drop_boundary=True, from cfg.grant_loss): exact-0/1 targets are
    # DROPPED entirely -- matching Grant's `where(q* != 0/1)` filter -- so their
    # per-sample loss stays 0. The q=0/1 boundary conditions still enter via the
    # indicator override in compute_targets; boundary parents are additionally
    # excluded from sampling and the stratified injector is off (see train()).
    if not drop_boundary:
        basin_a = q_target == 0.0                 # q* = 0 -> push q down
        if basin_a.sum() > 0:
            losses[basin_a] = log_1mq[basin_a].square()

        basin_b = q_target == 1.0                 # q* = 1 -> push q up
        if basin_b.sum() > 0:
            losses[basin_b] = log_q[basin_b].square()

    return losses


def var_loss_per_sample(var_pred, var_target):
    """Per-sample MSE loss on variance prediction.

    Returns a (batch,) tensor of per-sample losses (not reduced).
    """
    return (var_pred - var_target).square()


# ---------------------------------------------------------------------------
# Committor surface
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def log_diagnostics(*args, **kwargs):
    return {}


# --- wandb media throttle -------------------------------------------------
# Diagnostic-image uploads (ramachandran / scatter / loss-curve / transition
# plots) dominate wandb media storage, especially for the greedy-like cadence
# that trains every iteration. Upload them only every _WANDB_IMAGE_EVERY-th
# training call; scalar metrics (pearson r / MAE / loss) still log every call so
# the comparison curves stay dense. Override per-run via WANDB_IMAGE_LOG_EVERY.
_WANDB_IMAGE_EVERY = max(1, int(os.environ.get("WANDB_IMAGE_LOG_EVERY", "20")))


def _log_images_now(iteration):
    return (int(iteration) % _WANDB_IMAGE_EVERY) == 0


# ---------------------------------------------------------------------------
# Test-set evaluation (ground-truth empirical committors)
# ---------------------------------------------------------------------------

_RATE_REF = {}          # lazily-loaded frozen boundary ensemble + equilibrium fluxes


def estimate_rate(net, cfg, device):
    """Rate / MFPT from the CURRENT committor, using GRANT'S OWN estimator.

        k_AB = Phi_A * <q>_{dA}        k_BA = Phi_B * <1 - q>_{dB}

    (the Hill / TPT relation; constructor.compute_rate_estimate in sc_committor_rates).
    Phi_X is the equilibrium flux OUT of basin X, measured once by sample_basins, and
    the average is over that basin's boundary ensemble. Both are FROZEN and are the
    same files the greedy arm reads -- so the ONLY thing that differs between the two
    methods' rates is the network. Compare against Grant's AIB9 MFPT of 4.7 +/- 0.69 ns.

    NB this deliberately uses NO WESTPA weights. Our WE walker weights collapse to
    denormals under the aggressive split/merge (recycled weight ~1e-309 on ADP, where
    the true steady-state flux is ~1e-4/iter), so the WE flux is not a usable rate.
    Grant's estimator needs only q, and is therefore immune to that.

    CAVEAT, and it is a big one: <q> on the A boundary is ~1e-7, so this metric is
    dominated ENTIRELY by the accuracy of q in its near-zero tail -- precisely where a
    WE-trained net is weakest (the replay buffer is enriched in transition-region
    samples by construction). On ADP our net overshoots there and the resulting MFPT is
    wrong by ~1e3 and violates detailed balance, while Pearson r stays ~0.96 and hides
    it completely. Logging the rate every training call is exactly how we watch that.
    """
    if not getattr(cfg, "rate_eval_enabled", False):
        return {}
    ref = getattr(cfg, "rate_ref_dir", None)
    if not ref or not os.path.isdir(ref):
        return {}
    try:
        if not _RATE_REF:
            for s in ("A", "B"):
                _RATE_REF[f"{s}_conf"] = np.load(os.path.join(ref, f"{s}_boundary_confs.npy"))
                _RATE_REF[f"{s}_flux"] = float(np.load(os.path.join(ref, f"{s}_flux.npy"))[0])
            print(f"  [rate] loaded frozen boundary ensemble from {ref}: "
                  f"A={_RATE_REF['A_conf'].shape[0]} B={_RATE_REF['B_conf'].shape[0]} confs, "
                  f"flux_A={_RATE_REF['A_flux']:.4g}/ns flux_B={_RATE_REF['B_flux']:.4g}/ns",
                  flush=True)

        was_training = net.training
        net.eval()

        def _q(confs, chunk=128):
            # Batched: the AIB9 featurizer materializes an (n, 129, 129, 3) difference
            # tensor, which is ~200 MB for all 1001 confs at once -- on a GPU that is
            # ALSO holding the OpenMM context. Chunking keeps that well bounded.
            out = []
            with torch.no_grad():
                for i in range(0, len(confs), chunk):
                    t = torch.tensor(confs[i:i + chunk], dtype=torch.float32, device=device)
                    out.append(net.forward_mean(t).float().cpu().numpy())
            return np.concatenate(out)

        qa = _q(_RATE_REF["A_conf"])
        qb = _q(_RATE_REF["B_conf"])
        if was_training:
            net.train()

        # Grant's compute_rate_estimate (sc_committor_rates AIB9/constructor.py:621-636)
        # averages the committor over n_boundary_samples = len(escape_times) - 1 of the
        # boundary confs -- the FIRST n_a on dA ([:n_a]) and the LAST n_b on dB ([-n_b:]).
        # The frozen ensemble here has confs == times length (1001), so n = len(confs) - 1.
        # Slice identically so our rate equals his to the conf, not just in formula.
        n_a = len(qa) - 1
        n_b = len(qb) - 1
        mean_q_da = float(np.mean(qa[:n_a]))
        mean_q_db = float(np.mean((1.0 - qb)[-n_b:]))
        k_ab = mean_q_da * _RATE_REF["A_flux"]        # 1/ns
        k_ba = mean_q_db * _RATE_REF["B_flux"]
        m = {
            "rate/mean_q_on_dA": mean_q_da,
            "rate/mean_1mq_on_dB": mean_q_db,
            "rate/k_AB_per_ns": k_ab,
            "rate/k_BA_per_ns": k_ba,
            "rate/mfpt_AB_ns": (1.0 / k_ab) if k_ab > 0 else float("inf"),
            "rate/mfpt_BA_ns": (1.0 / k_ba) if k_ba > 0 else float("inf"),
        }
        # Detailed balance: k_AB/k_BA must equal P_B/P_A. A physically wrong committor
        # shows up here even when Pearson r looks fine.
        m["rate/detailed_balance_ratio"] = (k_ab / k_ba) if k_ba > 0 else float("nan")
        print(f"  [rate] <q>_dA={mean_q_da:.3e}  k_AB={k_ab:.4g}/ns  MFPT_AB={m['rate/mfpt_AB_ns']:.4g} ns  |  "
              f"k_BA={k_ba:.4g}/ns  MFPT_BA={m['rate/mfpt_BA_ns']:.4g} ns", flush=True)
        return m
    except Exception as exc:
        print(f"  [rate] failed: {exc}", flush=True)
        return {}


def evaluate_test_set(*args, **kwargs):
    return {}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def _init_wandb(cfg):
    """Initialize the shared wandb run once per process.

    Persists the run id so a WESTPA restart resumes the same run;
    init.sh wipes data_dir to start a fresh experiment.
    """
    os.makedirs(cfg.data_dir, exist_ok=True)
    run_id_file = os.path.join(cfg.data_dir, "wandb_run_id")
    if os.path.exists(run_id_file):
        with open(run_id_file) as f:
            run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(run_id_file, "w") as f:
            f.write(run_id)
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        name=cfg.wandb_run_name,
        config={
            "network_type": cfg.network_type,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "n_layers": cfg.n_layers,
            "embed_dim": cfg.embed_dim,
            "ffn_embed_dim": cfg.ffn_embed_dim,
            "attention_heads": cfg.attention_heads,
            "num_kernel": cfg.num_kernel,
            "grant_n_hidden": cfg.grant_n_hidden,
            "grant_n_residual_layers": cfg.grant_n_residual_layers,
            "grant_leak": cfg.grant_leak,
            "warmup_iters": cfg.warmup_iters,
            "train_interval": cfg.train_interval,
            "per_alpha": cfg.per_alpha,
            "per_beta_start": cfg.per_beta_start,
            "var_loss_weight": cfg.var_loss_weight,
            "n_bootstrap_steps": cfg.n_bootstrap_steps,
            "semigroup_loss_weight": cfg.semigroup_loss_weight,
            "boundary_loss_weight": cfg.boundary_loss_weight,
            "bke_loss_weight": cfg.bke_loss_weight,
            "bke_warmup_iters": cfg.bke_warmup_iters,
            "bke_anneal_start_iter": cfg.bke_anneal_start_iter,
            "bke_anneal_end_iter": cfg.bke_anneal_end_iter,
            "bke_exclude_boundary": cfg.bke_exclude_boundary,
            "boundary_min_per_batch": cfg.boundary_min_per_batch,
        },
        resume="allow",
        id=run_id,
    )
    # Index diagnostic/test/epoch/timing panels by WE iteration so repeat
    # image logs collapse into one panel with an iteration slider, rather
    # than landing on wandb's auto-incremented _step (bumped by per-batch
    # scalar logs) and rendering as separate media cards.
    wandb.define_metric("we_iteration")
    wandb.define_metric("diagnostics/*", step_metric="we_iteration")
    wandb.define_metric("test/*",        step_metric="we_iteration")
    wandb.define_metric("epoch/*",       step_metric="we_iteration")
    wandb.define_metric("timing/*",      step_metric="we_iteration")
    # BKE schedule panel: weight, target, anneal endpoints — all keyed on
    # we_iteration (the lagging-sim WESTPA iteration), so the user can read
    # off "weight at WESTPA iter X" directly. The schedule is in WESTPA-iter
    # units, not training-call units.
    wandb.define_metric("bke/*",         step_metric="we_iteration")
    # Best-checkpoint tracker (lowest test/rmse seen across all train calls).
    # `best/test_rmse_so_far` gives a non-increasing step plot vs we_iteration.
    wandb.define_metric("best/*",        step_metric="we_iteration")


def train(cfg, iteration, westpa_iter=None):
    """Train the committor network on pooled data from all cfg.sim_sources and save checkpoint.

    ``iteration`` is a monotonic training-call counter used for wandb
    step labels and as a nominal ``iter`` for the committor checkpoint.
    ``westpa_iter`` is the actual WESTPA iteration number (lagging sim);
    used as the BKE-anneal x-axis so the schedule is in WESTPA-iter units,
    not training-call units. Falls back to ``iteration`` if not supplied
    (legacy callers).
    Actual WESTPA iteration numbers per source are also discovered by
    reading the h5 files inside collect_training_data / collect_nstep_data.
    """
    if westpa_iter is None:
        westpa_iter = iteration
    os.makedirs(cfg.data_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase timers (CUDA-synced for honest GPU attribution).
    def _t():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    timings = {}
    t_start = _t()
    t = t_start

    # Build system and network (architecture selected by cfg.network_type).
    # Seed NN init reproducibly from the per-run SEED (only affects the very
    # first train call; subsequent calls resume from the saved checkpoint below).
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    system = ADPSystem(cfg.pdb_path)
    net = build_committor_net(cfg, system).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    print(f"  network_type={cfg.network_type} "
          f"sg={cfg.semigroup_loss_weight} bd={cfg.boundary_loss_weight} "
          f"bke_target={cfg.bke_loss_weight} "
          f"anneal=({cfg.bke_anneal_start_iter}->{cfg.bke_anneal_end_iter}) "
          f"bdmin={cfg.boundary_min_per_batch}")

    # Resume from checkpoint if available
    if os.path.exists(cfg.checkpoint):
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["opt"])
        print("  Loaded checkpoint")
    timings["setup"] = _t() - t; t = _t()

    # Collect 1-step parent->child data pooled from all sim_sources
    parent_pos_1step, child_groups_1step, child_pcoords_1step, sources_1step = \
        collect_training_data(cfg)
    if len(parent_pos_1step) == 0:
        print("  No training data yet")
        return
    timings["collect_1step"] = _t() - t; t = _t()

    # n-step bootstrap disabled in this ablation — no tracing performed.
    nstep_pos, nstep_groups, nstep_pcoords, nstep_sources = [], [], [], []
    timings["collect_nstep"] = 0.0; t = _t()

    parent_pos = list(parent_pos_1step)
    child_groups = list(child_groups_1step)
    child_pcoords = list(child_pcoords_1step)
    source_list = list(sources_1step)
    n_1step = len(parent_pos)

    if nstep_pos:
        # Subsample n-step groups to match 1-step count (50/50 mix)
        n_nstep_avail = len(nstep_pos)
        n_sample = min(n_1step, n_nstep_avail)
        # Seeded per (run, train call) so the n-step subsample is reproducible.
        rng_sub = np.random.default_rng([int(cfg.seed), int(iteration)])
        sample_idx = rng_sub.choice(n_nstep_avail, size=n_sample, replace=False)
        for i in sample_idx:
            parent_pos.append(nstep_pos[i])
            child_groups.append(nstep_groups[i])
            child_pcoords.append(nstep_pcoords[i])
            source_list.append(nstep_sources[i])
        print(f"  Mixed {n_1step} 1-step + {n_sample} n-step groups "
              f"(from {n_nstep_avail} available)")

    # Wipe and recreate debug Ramachandran folder
    debug_rama_dir = os.path.join(cfg.root, "output_files", "debugramachandran")
    if os.path.exists(debug_rama_dir):
        shutil.rmtree(debug_rama_dir)
    os.makedirs(debug_rama_dir)

    # Prepare data for epoch-based training
    parent_pos_arr = np.array(parent_pos, dtype=np.float32)
    n_groups = len(parent_pos)
    n_children = sum(len(cg) for cg in child_groups)
    n_batches = max(1, (n_groups + cfg.batch_size - 1) // cfg.batch_size)
    print(f"  {n_groups} total groups, {n_children} total children "
          f"(training call #{iteration}, {n_batches} batches/epoch)")

    # Training loop: multiple epochs over all data.
    # Seeded per (run, train call) so PER/minibatch sampling is reproducible
    # across replicates with the same SEED.
    eps = 1e-6
    rng = np.random.default_rng([int(cfg.seed), int(iteration)])
    loss_sum = 0.0
    n_steps = 0
    prev_loss = None
    # AIB9 central-residue (phi, psi) -- the SAME quadruplets as
    # batched_propagator.PHI_INDICES/PSI_INDICES and Grant's compute_aib9_dihedrals.
    # NB: this constant is duplicated here rather than imported; keep it in sync.
    phi_indices = np.array([[56, 58, 60, 69]])
    psi_indices = np.array([[58, 60, 69, 71]])

    # Compute ALL targets once per training call with frozen network.
    # Matches Grant's pattern: fixed targets for all inner optimization steps.
    net.eval()
    all_indices = np.arange(n_groups)
    all_targets_mean, all_targets_var = compute_targets(
        net, child_groups, child_pcoords, all_indices, cfg, device)
    timings["compute_targets"] = _t() - t; t = _t()

    # --- PER: compute fixed priorities from self-consistency residuals ---
    with torch.no_grad():
        all_q_init = []
        for s in range(0, n_groups, cfg.batch_size):
            e = min(s + cfg.batch_size, n_groups)
            pos = torch.tensor(parent_pos_arr[s:e], device=device)
            q_mean_init, _ = net(pos)
            all_q_init.append(q_mean_init.cpu().numpy())
        all_q_init = np.concatenate(all_q_init)
    residuals = np.abs(all_q_init - all_targets_mean.cpu().numpy())
    priorities = (residuals ** cfg.per_alpha) + 1e-6
    if cfg.grant_loss:
        # Grant drops exact-0/1 targets from the loss; exclude those boundary
        # parents from sampling too (they carry no gradient). Keep them only if
        # the buffer is ALL boundary (early warmup) to avoid a zero-sum.
        interior_par = ((all_targets_mean.cpu().numpy() > 0.0)
                        & (all_targets_mean.cpu().numpy() < 1.0))
        if interior_par.any():
            priorities = priorities * interior_par
    probs = priorities / priorities.sum()
    timings["per_priority_init"] = _t() - t; t = _t()

    # ------- BKE warm-up state for this train call (constant across epochs) ----
    # Schedule is in WESTPA-iter units (anneal_start_iter / anneal_end_iter
    # are WESTPA iterations, not training-call counters). The coordinator
    # passes the lagging sim's latest iter as ``westpa_iter``.
    bke_w_eff = bke_weight_for_iter(cfg, westpa_iter)
    bke_via_features = (
        bke_w_eff > 0
        and hasattr(net, "forward_from_features")
        and hasattr(net, "compute_features")
    )
    bke_via_dihedrals = (
        bke_w_eff > 0
        and not bke_via_features
        and hasattr(net, "forward_from_dihedrals")
    )
    bke_active = bke_via_features or bke_via_dihedrals

    # Dedicated wandb panel for the BKE schedule, plotted vs WESTPA iter
    # (we_iteration step_metric). Logged once per training call — the weight
    # is constant within a call.
    wandb.log({
        "bke/weight":            bke_w_eff,
        "bke/target_weight":     float(cfg.bke_loss_weight),
        "bke/anneal_start_iter": int(cfg.bke_anneal_start_iter),
        "bke/anneal_end_iter":   int(cfg.bke_anneal_end_iter),
        "we_iteration":          int(westpa_iter),
    })
    print(f"  [bke] westpa_iter={westpa_iter} train_call={iteration} "
          f"weight={bke_w_eff:.4g}/{cfg.bke_loss_weight} "
          f"anneal=({cfg.bke_anneal_start_iter}->{cfg.bke_anneal_end_iter})",
          flush=True)

    # ------- Stratified boundary sampling: precompute boundary parent indices
    # once per train call (targets are frozen for the round, so this is fixed).
    targets_mean_np = all_targets_mean.detach().cpu().numpy()
    boundary_indices_np = None
    if cfg.boundary_min_per_batch > 0 and not cfg.grant_loss:
        boundary_mask_full = (
            (targets_mean_np == 0.0) | (targets_mean_np == 1.0)
        )
        if boundary_mask_full.any():
            boundary_indices_np = np.where(boundary_mask_full)[0]
            print(f"  [stratified] boundary parents: "
                  f"{len(boundary_indices_np)}/{n_groups} "
                  f"({100.0*len(boundary_indices_np)/max(n_groups,1):.1f}%); "
                  f"forcing >= {cfg.boundary_min_per_batch} per minibatch")
        else:
            print("  [stratified] no boundary parents in buffer yet "
                  "(stratified sampling inactive this round)")

    step_losses = []
    t_data_xfer = 0.0
    t_fwd_bwd = 0.0
    t_wandb = 0.0

    for epoch in range(cfg.n_epochs):
        # Anneal beta from per_beta_start toward 1.0 over epochs
        beta = cfg.per_beta_start + (1.0 - cfg.per_beta_start) * epoch / max(1, cfg.n_epochs - 1)

        epoch_loss = 0.0
        epoch_steps = 0

        for start in range(0, n_groups, cfg.batch_size):
            # Sample with replacement proportional to priorities
            batch_idx = rng.choice(n_groups, size=min(cfg.batch_size, n_groups),
                                   replace=True, p=probs)

            # Stratified augment: if PER didn't yield enough boundary parents,
            # replace random non-boundary slots with uniform draws from the
            # boundary set so the boundary-loss term always fires.
            forced_slots = None
            if boundary_indices_np is not None:
                in_boundary = (
                    (targets_mean_np[batch_idx] == 0.0)
                    | (targets_mean_np[batch_idx] == 1.0)
                )
                n_have = int(in_boundary.sum())
                n_need = cfg.boundary_min_per_batch - n_have
                if n_need > 0:
                    non_b = np.where(~in_boundary)[0]
                    n_swap = min(n_need, len(non_b))
                    if n_swap > 0:
                        forced_slots = rng.choice(non_b, size=n_swap, replace=False)
                        new_b = rng.choice(boundary_indices_np,
                                           size=n_swap, replace=True)
                        batch_idx[forced_slots] = new_b

            if cfg.live_targets:
                # Target-network ablation: recompute this batch's targets from
                # the CURRENT (live) net instead of the frozen per-call snapshot,
                # so predictions and targets always come from identical weights.
                # compute_targets runs under no_grad (detached bootstrap);
                # net.train() is restored before the prediction forward below.
                net.eval()
                targets_mean, targets_var = compute_targets(
                    net, child_groups, child_pcoords, batch_idx, cfg, device)
                net.train()
            else:
                targets_mean = all_targets_mean[batch_idx]
                targets_var = all_targets_var[batch_idx]

            # IS weights correct for non-uniform sampling bias
            is_weights = (n_groups * probs[batch_idx]) ** (-beta)
            is_weights = is_weights / is_weights.max()
            if forced_slots is not None:
                # Stratified slots: uniform-from-boundary, no PER bias to undo.
                is_weights[forced_slots] = 1.0
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=device)

            # Forward on minibatch parents
            _t0 = _t()
            batch_pos = torch.tensor(parent_pos_arr[batch_idx], device=device)
            t_data_xfer += _t() - _t0

            _t0 = _t()
            net.train()
            optimizer.zero_grad()

            if bke_via_features:
                # Pairwise-distance leaves: ∂q/∂features for the Grant arch.
                with torch.no_grad():
                    feat_vals = net.compute_features(batch_pos)
                feat = feat_vals.detach().clone().requires_grad_(True)
                q_mean, q_var = net.forward_from_features(feat)
                grad_q = torch.autograd.grad(
                    q_mean.sum(), feat, create_graph=True)[0]
                # mean across input dims (= sum / d), normalizes per-dim BKE.
                bke_ps = (grad_q ** 2).mean(dim=-1)
                if cfg.bke_exclude_boundary:
                    interior_mask = (targets_mean > 0.0) & (targets_mean < 1.0)
                    bke_ps = bke_ps * interior_mask.float()
                bke_term = (is_weights_t * bke_ps).mean()
            elif bke_via_dihedrals:
                from committor.torsion_utils import compute_dihedrals as _cd
                with torch.no_grad():
                    dih_values = _cd(batch_pos, net.dihedral_indices)
                dih = dih_values.detach().clone().requires_grad_(True)
                q_mean, q_var = net.forward_from_dihedrals(dih)
                grad_q = torch.autograd.grad(
                    q_mean.sum(), dih, create_graph=True
                )[0]                                        # (B, n_dih)
                bke_ps = (grad_q ** 2).sum(dim=-1)          # (B,)
                if cfg.bke_exclude_boundary:
                    interior_mask = (targets_mean > 0.0) & (targets_mean < 1.0)
                    bke_ps = bke_ps * interior_mask.float()
                bke_term = (is_weights_t * bke_ps).mean()
            else:
                q_mean, q_var = net(batch_pos)
                bke_term = torch.tensor(0.0, device=batch_pos.device)

            # AIB9 nets stash their pre-sigmoid logit; pass it so the loss is computed
            # in logit space (logsigmoid) and retains a gradient even at saturation.
            # ADP nets do not set _last_logit -> logit=None -> unchanged q-space path.
            _logit = getattr(net, "_last_logit", None)
            if cfg.grant_loss and _logit is None:
                raise RuntimeError(
                    "grant_loss requires the net's pre-sigmoid logit (_last_logit), "
                    "which this net did not stash -- the stable log-sigmoid path would "
                    "silently fall back to the clamped q-space loss.")
            mean_losses = log_loss_per_sample(q_mean, targets_mean, logit=_logit,
                                              drop_boundary=cfg.grant_loss)
            v_losses = var_loss_per_sample(q_var, targets_var)
            # Split mean_losses into semi-group (interior target) and
            # boundary (target == 0/1) streams. Sum-then-divide-by-batch_size
            # keeps the (sg=1, bd=1) default equivalent to the old `.mean()`.
            B = mean_losses.shape[0]
            interior_mask_w = ((targets_mean > 0.0) & (targets_mean < 1.0)).float()
            boundary_mask_w = 1.0 - interior_mask_w
            sg_loss = (is_weights_t * mean_losses * interior_mask_w).sum() / B
            bd_loss = (is_weights_t * mean_losses * boundary_mask_w).sum() / B
            var_loss_b = (is_weights_t * cfg.var_loss_weight * v_losses).mean()
            base_loss = (cfg.semigroup_loss_weight * sg_loss
                         + cfg.boundary_loss_weight * bd_loss
                         + var_loss_b)
            loss = base_loss + bke_w_eff * bke_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            loss_val = loss.item()
            t_fwd_bwd += _t() - _t0

            step_losses.append(loss_val)

            # Dump debug Ramachandran plots on loss spikes
            if prev_loss is not None and loss_val > 20 * prev_loss:
                _q = q_mean.detach().cpu().numpy()
                _tgt = targets_mean.detach().cpu().numpy()
                _pos = parent_pos_arr[batch_idx]
                # Compute phi/psi from 3D positions
                n_atoms = system.n_atoms
                xyz = _pos.reshape(len(_pos), 1, n_atoms, 3)
                traj = mdtraj.Trajectory(xyz[:, 0], system.topology)
                _phi = mdtraj.compute_dihedrals(traj, phi_indices)[:, 0]
                _psi = mdtraj.compute_dihedrals(traj, psi_indices)[:, 0]

                tag = f"epoch{epoch}_step{epoch_steps}"
                np.savez(os.path.join(debug_rama_dir, f"{tag}_qs.npz"),
                         q_pred=_q, q_target=_tgt, phi=_phi, psi=_psi)

                for label, vals in [("pred", _q), ("target", _tgt)]:
                    fig_d, ax_d = plt.subplots(figsize=(6, 5))
                    sc = ax_d.scatter(np.degrees(_phi), np.degrees(_psi),
                                      c=vals, cmap="coolwarm", vmin=0, vmax=1,
                                      s=30, edgecolors="k", linewidths=0.3)
                    fig_d.colorbar(sc, ax=ax_d, label="q")
                    ax_d.plot(np.degrees(cfg.basin_a_phi), np.degrees(cfg.basin_a_psi),
                              "g*", ms=15, label="A: right-handed helix (−60,−40)")
                    ax_d.plot(np.degrees(cfg.basin_b_phi), np.degrees(cfg.basin_b_psi),
                              "r*", ms=15, label="B: left-handed helix (+60,+40)")
                    ax_d.set_xlabel("phi (deg)")
                    ax_d.set_ylabel("psi (deg)")
                    ax_d.set_xlim(-180, 180)
                    ax_d.set_ylim(-180, 180)
                    ax_d.set_title(f"Loss spike {label} q  (loss {loss_val:.4f} vs prev {prev_loss:.4f})")
                    ax_d.legend(fontsize=8)
                    fig_d.tight_layout()
                    fig_d.savefig(os.path.join(debug_rama_dir, f"{tag}_{label}.pdf"))
                    plt.close(fig_d)
                print(f"  [DEBUG] loss spike at {tag}: {loss_val:.4f} vs prev {prev_loss:.4f}, "
                      f"dumped to {debug_rama_dir}")
            prev_loss = loss_val

            epoch_loss += loss_val
            epoch_steps += 1
            loss_sum += loss_val
            n_steps += 1

            # Count how many targets are in transition region vs pinned to basins
            n_valid = ((targets_mean > eps) & (targets_mean < 1 - eps)).sum().item()

            _t0 = _t()
            wandb.log({
                "train/loss": loss_val,
                "train/base_loss": base_loss.item(),
                "train/bke_loss_raw": bke_term.item(),
                "train/bke_loss_weighted": bke_w_eff * bke_term.item(),
                "train/bke_w_eff": bke_w_eff,
                "train/semigroup_loss_raw": sg_loss.item(),
                "train/semigroup_loss_weighted": cfg.semigroup_loss_weight * sg_loss.item(),
                "train/boundary_loss_raw": bd_loss.item(),
                "train/boundary_loss_weighted": cfg.boundary_loss_weight * bd_loss.item(),
                "train/mean_loss": mean_losses.mean().item(),
                "train/var_loss": v_losses.mean().item(),
                "train/q_pred_mean": q_mean.mean().item(),
                "train/q_pred_std": q_mean.std().item(),
                "train/q_target_mean": targets_mean.mean().item(),
                "train/q_target_std": targets_mean.std().item(),
                "train/var_pred_mean": q_var.mean().item(),
                "train/var_target_mean": targets_var.mean().item(),
                "train/n_transition_samples": n_valid,
                "train/n_basin_samples": len(batch_idx) - n_valid,
                "train/n_forced_boundary_slots":
                    int(0 if forced_slots is None else len(forced_slots)),
            })
            t_wandb += _t() - _t0

        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"    epoch {epoch+1}/{cfg.n_epochs}  avg_loss={avg_epoch_loss:.6f}  "
              f"batches={epoch_steps}")

    timings["epoch_loop_total"] = _t() - t; t = _t()

    # Per-step loss curve for this training round
    fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
    ax_lc.plot(range(1, len(step_losses) + 1), step_losses, lw=0.8)
    ax_lc.set_xlabel("Optimization step")
    ax_lc.set_ylabel("Loss")
    ax_lc.set_title(f"Training loss curve (WE iter {iteration})")
    ax_lc.set_yscale("log")
    fig_lc.tight_layout()
    fig_lc_img = wandb.Image(fig_lc)
    plt.close(fig_lc)

    # Summary across all epochs
    avg_loss = loss_sum / n_steps
    print(f"  Training summary: avg_loss={avg_loss:.6f}  final_loss={loss_val:.6f}")

    # Post-epoch diagnostics
    diag_metrics = log_diagnostics(net, parent_pos_arr, child_groups, child_pcoords,
                                   source_list, cfg, device, iteration)
    timings["diagnostics"] = _t() - t; t = _t()

    # Ground-truth test-set evaluation
    test_metrics = evaluate_test_set(net, cfg, device, iteration)
    # Rate / MFPT via Grant's estimator on the frozen boundary ensemble. This is the
    # AIB9 headline metric (no AIB9 empirical-committor test set exists), logged every
    # training call so we can watch it converge -- and watch the low-q tail, which is
    # what actually drives it.
    rate_metrics = estimate_rate(net, cfg, device)
    timings["test_set_eval"] = _t() - t; t = _t()

    # Log all per-round metrics in one call to avoid duplicate wandb panels
    round_metrics = {
        "diagnostics/training_loss_curve": fig_lc_img,
        "epoch/we_iteration": iteration,
        "epoch/avg_loss": avg_loss,
        "epoch/final_loss": loss_val,
        "epoch/n_groups": n_groups,
        "epoch/n_children": n_children,
        "epoch/n_batches": n_batches,
        "epoch/bke_w_eff": bke_w_eff,
        "epoch/n_boundary_parents":
            int(0 if boundary_indices_np is None else len(boundary_indices_np)),
    }
    round_metrics.update(diag_metrics)
    round_metrics.update(test_metrics)
    round_metrics.update(rate_metrics)
    for _k, _v in timings.items():
        round_metrics[f"timing/{_k}"] = _v
    round_metrics["timing/loop_data_xfer"] = t_data_xfer
    round_metrics["timing/loop_fwd_bwd"] = t_fwd_bwd
    round_metrics["timing/loop_wandb"] = t_wandb
    # we_iteration is the lagging-sim WESTPA iter (NOT the training-call
    # counter). All diagnostics/test/epoch/timing/bke panels are stepped by
    # this so they read in WESTPA-iter units.
    round_metrics["we_iteration"] = int(westpa_iter)
    round_metrics["train_call"] = int(iteration)

    # Truth-free empirical-committor validation (held-out swarms). Logged
    # alongside the existing metrics; never used to select checkpoints. Runs
    # only every cfg.val_interval training calls to bound the OpenMM cost.
    if getattr(cfg, "val_interval", 0) and cfg.val_interval > 0 \
            and int(iteration) % int(cfg.val_interval) == 0:
        try:
            from committor.validation import validate_committor
            round_metrics.update(validate_committor(net, cfg, device))
        except Exception as exc:
            print(f"  [val] empirical-committor validation skipped: {exc}")

    # Unified compute accounting: cumulative MD integrator steps across both
    # sims (total walkers * ADP_TOTAL_STEPS), logged to wandb + an atomic
    # compute_summary.json so seed runs are comparable on a common MD-sample axis.
    try:
        import json as _json_cs
        total_steps = int(os.environ.get("ADP_TOTAL_STEPS", 100))
        sim_steps = 0
        for _src in cfg.sim_sources:
            _h5 = _src.get("west_h5")
            if _h5 and os.path.exists(_h5):
                with h5py.File(_h5, "r") as _f:
                    if "summary" in _f and _f["summary"].dtype.names \
                            and "n_particles" in _f["summary"].dtype.names:
                        sim_steps += int(np.nansum(
                            _f["summary"]["n_particles"])) * total_steps
        round_metrics["compute/sim_steps_cumulative"] = int(sim_steps)
        _cs_tmp = os.path.join(cfg.data_dir, "compute_summary.json.tmp")
        with open(_cs_tmp, "w") as _cf:
            _json_cs.dump({"train_call": int(iteration),
                           "westpa_iter": int(westpa_iter),
                           "sim_steps_cumulative": int(sim_steps),
                           "n_groups": int(n_groups),
                           "n_children": int(n_children),
                           "train_elapsed_s": _t() - t_start}, _cf)
        os.replace(_cs_tmp, os.path.join(cfg.data_dir, "compute_summary.json"))
    except Exception as exc:
        print(f"  [compute] accounting skipped: {exc}")

    # Throttle diagnostic-image UPLOADS to bound wandb media storage: strip
    # wandb.Image values except every _WANDB_IMAGE_EVERY-th training call. All
    # scalar metrics still log every call, so the comparison curves stay dense.
    if not _log_images_now(iteration):
        round_metrics = {_k: _v for _k, _v in round_metrics.items()
                         if not isinstance(_v, wandb.Image)}
    wandb.log(round_metrics)

    # Atomic checkpoint write: write to tmp then rename. Both sims' pre_we
    # read this file; a partial write would crash torch.load on the reader.
    tmp_ckpt = cfg.checkpoint + ".tmp"
    torch.save(
        {"net": net.state_dict(), "opt": optimizer.state_dict(),
         "iteration": iteration},
        tmp_ckpt,
    )
    os.replace(tmp_ckpt, cfg.checkpoint)
    print(f"  Saved checkpoint to {cfg.checkpoint}")

    # Best-RMSE checkpoint: a *separate* file that we only overwrite when
    # test/rmse improves. The running checkpoint above is consumed by both
    # sims' pre_we, so we leave it alone — this best file is read by
    # downstream analysis only. Persistence across train calls (including
    # SLURM restarts) is via a small sidecar `best_test_rmse.txt`.
    test_rmse = test_metrics.get("test/rmse")
    if test_rmse is not None and np.isfinite(test_rmse):
        best_rmse_file = os.path.join(cfg.data_dir, "best_test_rmse.txt")
        best_ckpt_path = os.path.join(cfg.data_dir, "committor_net_best.pt")
        prev_best = float("inf")
        if os.path.exists(best_rmse_file):
            try:
                with open(best_rmse_file) as _f:
                    prev_best = float(_f.read().strip())
            except Exception:
                prev_best = float("inf")
        # Warm-up floor: don't freeze an early transient dip as "best".
        improved = (test_rmse < prev_best
                    and int(iteration) >= int(cfg.ckpt_warmup_steps))
        if improved:
            tmp_best = best_ckpt_path + ".tmp"
            torch.save(
                {"net": net.state_dict(),
                 "opt": optimizer.state_dict(),
                 "iteration": int(iteration),
                 "westpa_iter": int(westpa_iter),
                 "test_rmse": float(test_rmse),
                 "test_mae": float(test_metrics.get("test/mae", float("nan"))),
                 "test_pearson_r": float(
                     test_metrics.get("test/pearson_r", float("nan"))),
                 "test_accuracy": float(
                     test_metrics.get("test/accuracy", float("nan")))},
                tmp_best,
            )
            os.replace(tmp_best, best_ckpt_path)
            with open(best_rmse_file, "w") as _f:
                _f.write(f"{test_rmse:.10f}\n")
            print(f"  [best-ckpt] NEW BEST test RMSE = {test_rmse:.6f} "
                  f"(prev {prev_best if prev_best != float('inf') else 'inf'}); "
                  f"westpa_iter={westpa_iter} train_call={iteration}; "
                  f"saved to {best_ckpt_path}")
        else:
            print(f"  [best-ckpt] test RMSE = {test_rmse:.6f} "
                  f"(best so far = {prev_best:.6f}, no update)")
        running_best = test_rmse if improved else prev_best
        wandb.log({
            "best/test_rmse_so_far": float(running_best),
            "best/improved_this_call": int(improved),
            "best/test_rmse_current": float(test_rmse),
            "we_iteration": int(westpa_iter),
        })

    # Truth-free "lowloss" checkpoint: a deployable selection criterion that
    # never peeks at the labeled test set (the `best` ckpt above is oracle-
    # selected off test/rmse, so it is NOT deployable). Overwrite only when the
    # round-average training loss hits a new min, gated by the same warm-up
    # floor as best. Read by downstream analysis / aggregate_seeds.py only.
    if np.isfinite(avg_loss):
        lowloss_file = os.path.join(cfg.data_dir, "lowloss_value.txt")
        lowloss_ckpt_path = os.path.join(cfg.data_dir, "committor_net_lowloss.pt")
        prev_low = float("inf")
        if os.path.exists(lowloss_file):
            try:
                with open(lowloss_file) as _f:
                    prev_low = float(_f.read().strip())
            except Exception:
                prev_low = float("inf")
        low_improved = (avg_loss < prev_low
                        and int(iteration) >= int(cfg.ckpt_warmup_steps))
        if low_improved:
            tmp_low = lowloss_ckpt_path + ".tmp"
            torch.save(
                {"net": net.state_dict(),
                 "opt": optimizer.state_dict(),
                 "iteration": int(iteration),
                 "westpa_iter": int(westpa_iter),
                 "avg_loss": float(avg_loss)},
                tmp_low,
            )
            os.replace(tmp_low, lowloss_ckpt_path)
            with open(lowloss_file, "w") as _f:
                _f.write(f"{avg_loss:.10f}\n")
            print(f"  [lowloss-ckpt] NEW LOW avg_loss = {avg_loss:.6f} "
                  f"(prev {prev_low if prev_low != float('inf') else 'inf'}); "
                  f"train_call={iteration}; saved to {lowloss_ckpt_path}")
        else:
            print(f"  [lowloss-ckpt] avg_loss = {avg_loss:.6f} "
                  f"(low so far = {prev_low:.6f}, no update)")
        running_low = avg_loss if low_improved else prev_low
        wandb.log({
            "lowloss/avg_loss_so_far": float(running_low),
            "lowloss/improved_this_call": int(low_improved),
            "lowloss/avg_loss_current": float(avg_loss),
            "we_iteration": int(westpa_iter),
        })
    timings["finalize"] = _t() - t

    total = sum(timings.values())
    print(f"  Timing breakdown (total {total:.1f}s):")
    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"    {k:24s} {v:7.2f}s  {100*v/max(total, 1e-9):5.1f}%")
    print(f"    [in-loop] data_xfer={t_data_xfer:.2f}s  "
          f"fwd_bwd={t_fwd_bwd:.2f}s  wandb={t_wandb:.2f}s")


def main():
    """Entry point called from post_iter.sh."""
    cfg = CommittorConfig()
    iteration = int(os.environ.get("WEST_CURRENT_ITER", 0))

    if iteration < cfg.warmup_iters:
        return
    if (iteration - cfg.warmup_iters) % cfg.train_interval != 0:
        return

    print(f"=== Committor update at iteration {iteration} ===")
    _init_wandb(cfg)
    train(cfg, iteration)
    wandb.finish()
    print("=== Done ===")


if __name__ == "__main__":
    main()
