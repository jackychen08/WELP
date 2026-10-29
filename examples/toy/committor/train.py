"""Train the toy committor using WESTPA parent->child Markov transitions.

Simplified port of adp_test_mab_ss_velreinit_v3/committor/train.py:
    - No mdtraj, no PDB: positions are (x, y) floats from seg.npy.
    - 1-step self-consistent mean-matching (log-space loss as in Rotskoff
      arXiv:2408.05879).
    - PER: priorities from per-parent self-consistency residual; IS-weighted
      log + variance loss (mirrors adp_test_mab_ss_recyclefix_per).
    - Variance head still trained with a cheap MSE against the empirical
      children variance.
    - Test-set eval vs the true committor from solve_true_committor.py.

Data collection pools parent-child transitions from ALL cfg.sim_sources.
"""
from __future__ import annotations

import os
import time

import h5py
import numpy as np
import torch

from committor.config import ToyCommittorConfig
from committor.network import build_committor_net


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _list_iter_keys(h5):
    out = []
    for k in h5["iterations"].keys():
        try:
            out.append(int(k.split("_")[-1]))
        except ValueError:
            pass
    out.sort()
    return out


def _iter_fully_written(h5, iter_key):
    """True iff the iteration is complete (status==2 for all segs) AND iter+1
    exists (which means pre_we already ran and wrote q into pcoord[:,:,0])."""
    if iter_key not in h5["iterations"]:
        return False
    grp = h5["iterations"][iter_key]
    try:
        status = grp["seg_index"]["status"]
    except (KeyError, ValueError):
        return False
    if not (status == 2).all():
        return False
    it_num = int(iter_key.rsplit("_", 1)[-1])
    return f"iter_{it_num + 1:08d}" in h5["iterations"]


def _load_seg_xy(traj_dir: str, n_iter: int, seg_id: int) -> np.ndarray:
    p = os.path.join(traj_dir, f"{n_iter:06d}", f"{seg_id:06d}", "seg.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p).astype(np.float32)     # (11, 2)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_training_data(cfg: ToyCommittorConfig):
    """Return pooled (parent_xy, child_groups, source_labels).

    parent_xy[i]: (2,) — last frame of parent seg
    child_groups[i]: (n_children, 2) — last frames of children born from parent_i
    """
    parents = []
    children = []
    labels = []

    last_iter_by_source = {s["label"]: 1 for s in cfg.sim_sources}

    if os.path.exists(cfg.cache_file):
        cache = np.load(cfg.cache_file, allow_pickle=True)
        # Coerce every element back to float32 arrays of the expected shape.
        # This repairs caches written by the old save path, which turned the
        # list of uniform (2,) arrays into a dissolved (N, 2) object array.
        parents = [np.asarray(p, dtype=np.float32).reshape(2)
                   for p in cache["parent_xy"]]
        children = [np.asarray(c, dtype=np.float32).reshape(-1, 2)
                    for c in cache["child_groups"]]
        labels = list(cache["source_list"]) if "source_list" in cache.files else \
                 ["unknown"] * len(parents)
        cached = cache["last_iter_by_source"].item() if "last_iter_by_source" in cache.files else {}
        for label, it in cached.items():
            if label in last_iter_by_source:
                last_iter_by_source[label] = int(it)

    n_new = 0
    for src in cfg.sim_sources:
        label = src["label"]
        if not os.path.exists(src["west_h5"]):
            continue
        with h5py.File(src["west_h5"], "r", locking=False) as h5:
            iter_nums = _list_iter_keys(h5)
            for it in iter_nums:
                if it <= last_iter_by_source[label]:
                    continue
                key = f"iter_{it:08d}"
                if key not in h5["iterations"]:
                    continue
                if not _iter_fully_written(h5, key):
                    continue

                grp = h5["iterations"][key]
                seg_idx = grp["seg_index"]
                n_seg = seg_idx.shape[0]
                parent_ids = seg_idx["parent_id"][:]           # (n_seg,)

                # Parent iter = it - 1; to load parent xy we need to find those
                # segments' last frames in the previous iter's pcoord/aux if available,
                # OR we can read the parent's seg.npy.
                # Simpler: parent is identified by (prev_iter, parent_id) for children
                # with parent_id >= 0. We use seg.npy last frame on disk.
                prev_iter = it - 1
                # Group children by parent
                from collections import defaultdict
                children_by_parent = defaultdict(list)
                for sid in range(n_seg):
                    pid = int(parent_ids[sid])
                    if pid < 0:
                        continue   # recycled / initial; no markov pair
                    children_by_parent[pid].append(sid)

                for pid, child_sids in children_by_parent.items():
                    try:
                        parent_xy = _load_seg_xy(src["traj_dir"], prev_iter, pid)[-1]
                    except FileNotFoundError:
                        continue
                    child_xys = []
                    for csid in child_sids:
                        try:
                            cx = _load_seg_xy(src["traj_dir"], it, csid)[-1]
                        except FileNotFoundError:
                            continue
                        child_xys.append(cx)
                    if not child_xys:
                        continue
                    parents.append(parent_xy.astype(np.float32))
                    children.append(np.stack(child_xys).astype(np.float32))
                    labels.append(label)
                    n_new += 1

                last_iter_by_source[label] = it

    # FIFO trim to replay buffer size
    if len(parents) > cfg.replay_buffer_size:
        drop = len(parents) - cfg.replay_buffer_size
        parents = parents[drop:]
        children = children[drop:]
        labels = labels[drop:]

    # Save cache
    #
    # IMPORTANT: DO NOT use np.asarray(parents, dtype=object) for a list of
    # uniform-shape arrays. NumPy "dissolves" it into a (N, inner_shape)
    # object array with each scalar wrapped as a Python float, which then
    # fails to round-trip through torch.tensor.
    #
    # Parents have fixed shape (2,) so save them as a real (N, 2) float32.
    # Children have variable first-dim per group, so use element-wise
    # assignment into a 1D object array — the canonical numpy idiom for
    # "list of variable-shape arrays."
    os.makedirs(cfg.data_dir, exist_ok=True)
    if parents:
        parent_arr = np.stack(parents).astype(np.float32)         # (N, 2)
    else:
        parent_arr = np.zeros((0, 2), dtype=np.float32)
    child_obj = np.empty(len(children), dtype=object)
    for i, c in enumerate(children):
        child_obj[i] = np.asarray(c, dtype=np.float32)
    np.savez(cfg.cache_file,
             parent_xy=parent_arr,
             child_groups=child_obj,
             source_list=np.asarray(labels),
             last_iter_by_source=np.asarray(last_iter_by_source, dtype=object))

    print(f"[train] collected: new_groups={n_new}  total_groups={len(parents)}  "
          f"last_iter_by_source={last_iter_by_source}", flush=True)
    return parents, children, labels


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _compute_targets(net, children_flat: torch.Tensor,
                      child_sizes: list[int],
                      in_basin_a_mask: torch.Tensor,
                      in_basin_b_mask: torch.Tensor):
    """Mean committor of each parent's children. Basin hits clamp q to 0/1."""
    with torch.no_grad():
        q_mean, _ = net(children_flat)
        q_mean = torch.where(in_basin_a_mask, torch.zeros_like(q_mean), q_mean)
        q_mean = torch.where(in_basin_b_mask, torch.ones_like(q_mean), q_mean)
    # Split by group
    targets = []
    offset = 0
    for n in child_sizes:
        grp = q_mean[offset:offset + n]
        targets.append(grp.mean())
        offset += n
    return torch.stack(targets)


def _compute_var_targets(children_flat: torch.Tensor, child_sizes: list[int]):
    """Empirical var of children positions; used as variance-head target proxy."""
    var_t = []
    offset = 0
    for n in child_sizes:
        grp = children_flat[offset:offset + n]
        if n > 1:
            var_t.append(grp.var(dim=0).sum())
        else:
            var_t.append(torch.tensor(0.0, device=children_flat.device))
        offset += n
    return torch.stack(var_t)


def _log_loss(q_pred, q_tgt, eps: float = 1e-6):
    a = torch.clamp(q_pred, eps, 1 - eps)
    b = torch.clamp(q_tgt, eps, 1 - eps)
    return ((torch.log(a) - torch.log(b)) ** 2
            + (torch.log(1 - a) - torch.log(1 - b)) ** 2).mean()


def _log_loss_per_sample(q_pred, q_tgt, eps: float = 1e-6):
    """Per-sample symmetrized log-space loss; returns a (B,) tensor (no
    reduction). Used by PER so importance-sampling weights can scale each
    sample's contribution to the minibatch loss."""
    a = torch.clamp(q_pred, eps, 1 - eps)
    b = torch.clamp(q_tgt, eps, 1 - eps)
    return ((torch.log(a) - torch.log(b)) ** 2
            + (torch.log(1 - a) - torch.log(1 - b)) ** 2)


# ---------------------------------------------------------------------------
# wandb setup (shared by coordinator)
# ---------------------------------------------------------------------------

_WANDB_INITED = False


def _init_wandb(cfg: ToyCommittorConfig):
    global _WANDB_INITED
    if _WANDB_INITED:
        return
    try:
        import wandb
        os.makedirs(cfg.data_dir, exist_ok=True)
        run_id_path = os.path.join(cfg.data_dir, "wandb_run_id.txt")
        _run_id = None
        if os.path.exists(run_id_path):
            _run_id = open(run_id_path).read().strip() or None
        init_kwargs = dict(project=cfg.wandb_project, name=cfg.wandb_run_name,
                            id=_run_id, resume="allow",
                            config={"method": "westpa_mab_committor_per",
                                    "alpha": cfg.alpha,
                                    "lr": cfg.lr, "batch_size": cfg.batch_size,
                                    "n_epochs": cfg.n_epochs,
                                    "train_interval": cfg.train_interval,
                                    "warmup_iters": cfg.warmup_iters,
                                    "per_alpha": cfg.per_alpha,
                                    "per_beta_start": cfg.per_beta_start,
                                    "bke_loss_weight": cfg.bke_loss_weight,
                                    "bke_warmup_iters": cfg.bke_warmup_iters,
                                    "var_loss_weight": cfg.var_loss_weight,
                                    "grad_clip": 1.0})
        if cfg.wandb_entity:
            init_kwargs["entity"] = cfg.wandb_entity
        wandb.init(**init_kwargs)
        if _run_id is None:
            with open(run_id_path, "w") as f:
                f.write(wandb.run.id)
        wandb.define_metric("train_step")
        wandb.define_metric("*", step_metric="train_step")
        # Cumulative integration time across both sims (n_segments * n_steps
        # * dt). Same units on Grant's side, so wandb's "compare runs" UI can
        # use this as the shared x-axis.
        wandb.define_metric("time/sim_time")
        wandb.define_metric("time/sim_steps")
        _WANDB_INITED = True
    except Exception as exc:
        print(f"[train] wandb init skipped: {exc}")


def _wandb_log(payload: dict):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ground-truth evaluation
# ---------------------------------------------------------------------------

def _load_truth(cfg: ToyCommittorConfig):
    if not os.path.exists(cfg.test_set_path):
        return None
    data = np.load(cfg.test_set_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _walk_h5_metrics(cfg: ToyCommittorConfig):
    """One pass over every source h5; returns a dict matching Grant's metric
    names so the two methods can be compared on the same wandb axes.

    Counts per-source (label-aware):
        sim_steps              total integrator steps × walkers
        n_samples_total        total walker-segments completed
        n_a_transits           # iters in sim_AB with at least one fresh in_target latch
        n_b_transits           # iters in sim_BA with at least one fresh in_target latch
    """
    n_steps_per_seg = int(os.environ.get("TOY_N_STEPS", 25))
    sim_steps = 0
    n_samples = 0
    transits = {"sim_AB": 0, "sim_BA": 0}
    for src in cfg.sim_sources:
        path = src["west_h5"]
        if not os.path.exists(path):
            continue
        label = src["label"]
        local_transits = 0
        try:
            with h5py.File(path, "r", locking=False) as h5:
                for key in h5["iterations"].keys():
                    grp = h5["iterations"][key]
                    try:
                        n_seg = int(grp["seg_index"].shape[0])
                    except (KeyError, ValueError):
                        continue
                    sim_steps += n_seg * n_steps_per_seg
                    n_samples += n_seg
                    if "pcoord" in grp:
                        try:
                            pc = grp["pcoord"]
                            first = pc[:, 0, 2]
                            last = pc[:, -1, 2]
                            if np.any((first < 0.5) & (last > 0.5)):
                                local_transits += 1
                        except (IndexError, TypeError):
                            pass
        except (OSError, KeyError):
            continue
        if label in transits:
            transits[label] = local_transits
    return {
        "sim_steps": sim_steps,
        "n_samples": n_samples,
        "n_a_transits": transits.get("sim_AB", 0),
        "n_b_transits": transits.get("sim_BA", 0),
    }


def _load_basin_flux(cfg: ToyCommittorConfig):
    """Load the shared basin-flux cache produced by `grant/run_grant.py`. Used
    so both methods evaluate the same rate definition: flux * <sigma(net)>.
    Returns None if the cache hasn't been generated yet.
    """
    p = os.path.join(cfg.common_files, "basin_flux.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _eval_rmse(net, truth, device) -> tuple[float, float]:
    X, Y = np.meshgrid(truth["x"], truth["y"])
    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    with torch.no_grad():
        t = torch.tensor(xy, device=device)
        q = net.forward_mean(t).cpu().numpy().reshape(X.shape)
    err = q - truth["q"]
    return float(np.sqrt((err ** 2).mean())), float(np.abs(err).max())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(cfg: ToyCommittorConfig, train_step: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Deterministic NN init given SEED (only matters on the first call, before a
    # checkpoint exists to load from).
    torch.manual_seed(cfg.seed)
    net = build_committor_net(cfg).to(device)
    if os.path.exists(cfg.checkpoint):
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["net"])

    parents, children, labels = collect_training_data(cfg)
    if len(parents) < 4:
        print(f"[train] only {len(parents)} parent groups — skipping training")
        return

    # Defensive: force float32 dtype even if the cache or collection path
    # somehow produced object-dtype arrays. Prevents the numpy.object_ -> torch
    # crash seen when the old cache format leaked through.
    parent_arr = np.stack(parents).astype(np.float32)            # (N, 2)
    parent_t = torch.tensor(parent_arr, device=device)
    # Flatten children (also coerce to float32)
    child_sizes = [c.shape[0] for c in children]
    children_flat_np = np.concatenate(
        [np.asarray(c, dtype=np.float32) for c in children], axis=0
    )
    children_flat = torch.tensor(children_flat_np, device=device)
    child_cpu_np = children_flat.cpu().numpy()
    in_a = torch.tensor(cfg.in_basin_a(child_cpu_np), device=device, dtype=torch.bool)
    in_b = torch.tensor(cfg.in_basin_b(child_cpu_np), device=device, dtype=torch.bool)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    truth = _load_truth(cfg)

    t0 = time.time()
    last_loss = 0.0
    last_bke = 0.0
    eps_b = 1e-6
    bke_active = (cfg.bke_loss_weight > 0.0
                  and train_step >= cfg.bke_warmup_iters)
    # Seed PER minibatch sampling deterministically per (seed, train_step):
    # reproducible given SEED, but a distinct stream each training call.
    rng = np.random.default_rng([cfg.seed, train_step])
    n_p = len(parents)
    last_per_max_w = 0.0
    last_per_priority_max = 0.0
    for epoch in range(cfg.n_epochs):
        # 1. Compute self-consistent targets with CURRENT net (updated each epoch)
        with torch.no_grad():
            q_children, _ = net(children_flat)
            q_children = torch.where(in_a, torch.zeros_like(q_children), q_children)
            q_children = torch.where(in_b, torch.ones_like(q_children), q_children)

        q_target = torch.zeros(n_p, device=device)
        offset = 0
        for i, n in enumerate(child_sizes):
            q_target[i] = q_children[offset:offset + n].mean()
            offset += n
        q_target = torch.clamp(q_target, 1e-6, 1 - 1e-6)

        var_target = _compute_var_targets(children_flat, child_sizes)

        # 2. PER priorities. residual = |q(parent) - q*(parent)| with current
        # net; matches adp_test_mab_ss_recyclefix_per. Recomputed per-epoch
        # because the toy keeps targets refreshed each epoch (vs the ADP
        # reference which freezes them once per train call).
        with torch.no_grad():
            q_parents_now, _ = net(parent_t)
        residuals = (q_parents_now - q_target).abs().cpu().numpy()
        priorities = (residuals ** cfg.per_alpha) + 1e-6
        probs = priorities / priorities.sum()
        last_per_priority_max = float(priorities.max())

        # 3. Anneal IS-weight exponent beta toward 1.0 across epochs.
        beta = cfg.per_beta_start + (1.0 - cfg.per_beta_start) * epoch / max(1, cfg.n_epochs - 1)

        # 4. Mini-batch SGD with PER sampling (with-replacement, priority-prop).
        n_batches = max(1, (n_p + cfg.batch_size - 1) // cfg.batch_size)
        batch_losses = []
        batch_log_losses = []
        batch_bke = []
        for _b in range(n_batches):
            batch_idx_np = rng.choice(n_p,
                                       size=min(cfg.batch_size, n_p),
                                       replace=True, p=probs)
            is_w = (n_p * probs[batch_idx_np]) ** (-beta)
            is_w = is_w / is_w.max()
            is_w_t = torch.tensor(is_w, dtype=torch.float32, device=device)
            idx = torch.tensor(batch_idx_np, dtype=torch.long, device=device)

            opt.zero_grad()
            if bke_active:
                # Differentiate q_mean w.r.t. (x, y) so we can penalize |∇q|²
                # — the variational form of the backward Kolmogorov equation.
                # Mirror of adp_test_mab_ss_velreinit_v3_bke/committor/train.py
                # (which differentiates w.r.t. dihedrals); for the toy 2D case
                # the network input *is* the coordinate.
                xy_leaf = parent_t[idx].detach().clone().requires_grad_(True)
                q_pred, v_pred = net(xy_leaf)
                grad_q = torch.autograd.grad(
                    q_pred.sum(), xy_leaf, create_graph=True
                )[0]                                       # (B, 2)
                bke_ps = (grad_q ** 2).sum(dim=-1)         # (B,)
                if cfg.bke_exclude_boundary:
                    interior = ((q_target[idx] > eps_b)
                                & (q_target[idx] < 1.0 - eps_b)).float()
                    bke_ps = bke_ps * interior
                bke_term = (is_w_t * bke_ps).mean()
            else:
                q_pred, v_pred = net(parent_t[idx])
                bke_term = torch.zeros((), device=device)
            mean_losses = _log_loss_per_sample(q_pred, q_target[idx])
            v_losses = (v_pred - var_target[idx]) ** 2
            loss_q = (is_w_t * mean_losses).mean()
            loss_v = (is_w_t * v_losses).mean()
            loss = (loss_q
                    + cfg.var_loss_weight * loss_v
                    + cfg.bke_loss_weight * bke_term)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.item()))
            batch_log_losses.append(float(loss_q.item()))
            batch_bke.append(float(bke_term.item()))
            last_per_max_w = max(last_per_max_w, float(is_w.max()))
        last_loss = float(np.mean(batch_losses))
        last_log_loss = float(np.mean(batch_log_losses))
        last_bke = float(np.mean(batch_bke)) if batch_bke else 0.0

    # Save current checkpoint atomically
    os.makedirs(cfg.data_dir, exist_ok=True)
    tmp = cfg.checkpoint + ".tmp"
    torch.save({"net": net.state_dict(), "step": train_step,
                "n_groups": len(parents)}, tmp)
    os.replace(tmp, cfg.checkpoint)

    # Best-rmse checkpoint: if this step beats the best seen so far, also save
    # to committor_net_best.pt and record (step, rmse) in best_ckpt.json so
    # we can recover a low-rmse model after later divergence.
    if truth is not None:
        best_rmse_now, _ = _eval_rmse(net, truth, device)
        import json
        best_meta_path = os.path.join(cfg.data_dir, "best_ckpt.json")
        best_ckpt_path = os.path.join(cfg.data_dir, "committor_net_best.pt")
        prev_best = float("inf")
        if os.path.exists(best_meta_path):
            try:
                prev_best = float(json.load(open(best_meta_path))["rmse"])
            except Exception:
                prev_best = float("inf")
        if (np.isfinite(best_rmse_now) and best_rmse_now < prev_best
                and train_step >= cfg.ckpt_warmup_steps):
            tmp_b = best_ckpt_path + ".tmp"
            torch.save({"net": net.state_dict(), "step": train_step,
                        "n_groups": len(parents),
                        "rmse": best_rmse_now}, tmp_b)
            os.replace(tmp_b, best_ckpt_path)
            with open(best_meta_path, "w") as f:
                json.dump({"step": train_step, "rmse": best_rmse_now,
                           "n_groups": len(parents)}, f)
            print(f"[train] new best ckpt: rmse={best_rmse_now:.4f} "
                  f"(prev={prev_best:.4f}) step={train_step}")

    # Lowest-loss checkpoint: save weights from train_step with min last_loss.
    import json as _json_ll
    lowloss_meta_path = os.path.join(cfg.data_dir, "lowloss_ckpt.json")
    lowloss_ckpt_path = os.path.join(cfg.data_dir, "committor_net_lowloss.pt")
    prev_lowloss = float("inf")
    if os.path.exists(lowloss_meta_path):
        try:
            prev_lowloss = float(_json_ll.load(open(lowloss_meta_path))["loss"])
        except Exception:
            prev_lowloss = float("inf")
    if (np.isfinite(last_loss) and last_loss < prev_lowloss
            and train_step >= cfg.ckpt_warmup_steps):
        tmp_l = lowloss_ckpt_path + ".tmp"
        torch.save({"net": net.state_dict(), "step": train_step,
                    "n_groups": len(parents),
                    "loss": float(last_loss)}, tmp_l)
        os.replace(tmp_l, lowloss_ckpt_path)
        with open(lowloss_meta_path, "w") as f:
            _json_ll.dump({"step": train_step, "loss": float(last_loss),
                           "n_groups": len(parents)}, f)
        print(f"[train] new lowloss ckpt: loss={last_loss:.4f} "
              f"(prev={prev_lowloss:.4f}) step={train_step}")

    dt = float(os.environ.get("TOY_DT", 1e-3))
    h5m = _walk_h5_metrics(cfg)
    sim_steps = h5m["sim_steps"]
    payload = {
        "train_step": train_step,
        "loss/total": last_loss,
        "loss/log_loss": last_log_loss,
        "loss/bke_raw": last_bke,
        "loss/bke_weighted": cfg.bke_loss_weight * last_bke,
        "loss/bke_active": float(bke_active),
        "train/n_parent_groups": len(parents),
        "train/n_children": int(sum(child_sizes)),
        "train/elapsed_s": time.time() - t0,
        "time/sim_steps": sim_steps,
        "time/sim_time": sim_steps * dt,
        "time/elapsed_s": time.time() - t0,
        "sampling/n_samples_total": h5m["n_samples"],
        "sampling/n_a_transits": h5m["n_a_transits"],
        "sampling/n_b_transits": h5m["n_b_transits"],
        "per/is_weight_max_seen": last_per_max_w,
        "per/priority_max_last": last_per_priority_max,
        "per/beta_final_epoch": cfg.per_beta_start
            + (1.0 - cfg.per_beta_start) * (cfg.n_epochs - 1) / max(1, cfg.n_epochs - 1),
    }
    if truth is not None:
        rmse, maxabs = _eval_rmse(net, truth, device)
        payload["committor/rmse_vs_true"] = rmse
        payload["committor/max_abs_err"] = maxabs

    # Rate estimate evaluated on shared basin flux samples (generated by
    # grant/run_grant.py). Uses the SAME definition as Grant: flux * <q>
    # at the basin boundary, so the two methods plot on the same axis.
    flux = _load_basin_flux(cfg)
    if flux is not None:
        with torch.no_grad():
            a_bnd = torch.tensor(flux["a_boundary_confs"].astype(np.float32),
                                 device=device).reshape(-1, 2)
            b_bnd = torch.tensor(flux["b_boundary_confs"].astype(np.float32),
                                 device=device).reshape(-1, 2)
            a_q = float(net.forward_mean(a_bnd).mean().item())
            b_q = float(net.forward_mean(b_bnd).mean().item())
        payload["rate/a_to_b"] = float(flux["a_flux"]) * a_q
        payload["rate/b_to_a"] = float(flux["b_flux"]) * (1.0 - b_q)

    # Channel-occupancy of parent seeds
    from common_files.viz import channel_fractions
    frac_up, frac_lo = channel_fractions(parent_arr)
    payload["sampling/frac_upper_channel"] = frac_up
    payload["sampling/frac_lower_channel"] = frac_lo

    # Unified compute accounting: sim_steps from _walk_h5_metrics is already the
    # cumulative MD-integrator-step count across all iterations of both sims.
    # Expose it as the fixed-budget / RMSE-vs-compute axis and dump a summary
    # json the budget calibrator + aggregator can read without parsing logs.
    payload["compute/sim_steps_cumulative"] = sim_steps
    try:
        import json as _json_cs
        cs_tmp = os.path.join(cfg.data_dir, "compute_summary.json.tmp")
        with open(cs_tmp, "w") as _f:
            _json_cs.dump({"train_step": train_step,
                           "sim_steps_cumulative": int(sim_steps),
                           "n_samples_total": int(h5m["n_samples"]),
                           "n_a_transits": int(h5m["n_a_transits"]),
                           "n_b_transits": int(h5m["n_b_transits"]),
                           "train_elapsed_s": time.time() - t0}, _f)
        os.replace(cs_tmp, os.path.join(cfg.data_dir, "compute_summary.json"))
    except Exception as exc:
        print(f"[train] compute_summary dump skipped: {exc}")

    # Truth-free empirical-committor validation (held-out swarms), periodic.
    # Logged ALONGSIDE the existing metrics (schema unchanged); never used to
    # select checkpoints (that's lowloss). Skipped silently if unavailable.
    if cfg.val_interval > 0 and train_step % cfg.val_interval == 0:
        try:
            from committor.validation import validate_committor
            payload.update(validate_committor(net, cfg, device))
        except Exception as exc:
            print(f"[train] empirical-committor validation skipped: {exc}")

    _wandb_log(payload)
    print(f"[train step {train_step}] loss={last_loss:.4f}  "
          f"bke={last_bke:.4f}{'(on)' if bke_active else '(off)'}  "
          f"rmse={payload.get('committor/rmse_vs_true', float('nan')):.4f}  "
          f"upper={frac_up:.2f} lower={frac_lo:.2f}  "
          f"n_groups={len(parents)}  "
          f"transits={h5m['n_a_transits']}/{h5m['n_b_transits']}  "
          f"elapsed={payload['train/elapsed_s']:.1f}s")
