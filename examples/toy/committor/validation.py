"""Truth-free empirical-committor validation via held-out swarms.

The committor q(x) = P(reach B before A | start at x) IS measurable without the
ground-truth grid: launch many short unbiased trajectories from a held-out seed
and count the fraction that commit to B before A. This module does exactly that
for a fixed set of held-out seeds (common_files/val_seeds.npz, shared across all
runs), then compares the learned net's q(seed) to the empirical q_emp.

Used two ways: (1) periodically during training (logged to wandb alongside the
existing metrics, never used to select checkpoints), and (2) post-hoc on saved
checkpoints by eval scripts. Reuses the same overdamped-Langevin EM dynamics as
the propagator, implemented inline here to avoid importing WESTPA.
"""
import os

import numpy as np
import torch

from common_files.potential import force, ALPHA_DEFAULT, GAMMA_DEFAULT, KT_DEFAULT


def _swarm_qemp(seeds_xy, cfg, device, n_swarms, max_steps, dt, gen):
    """Empirical committor for each seed via M swarms run to commitment.

    seeds_xy: (S, 2) tensor. Returns (q_emp (S,), committed_frac (S,)).
    q_emp = fraction of *committed* swarms that reached B before A.
    """
    S = seeds_xy.shape[0]
    xy = seeds_xy.repeat_interleave(n_swarms, dim=0).clone()      # (S*M, 2)
    N = xy.shape[0]
    well_a = torch.tensor(cfg.well_a_xy, device=device, dtype=xy.dtype)
    well_b = torch.tensor(cfg.well_b_xy, device=device, dtype=xy.dtype)
    r2 = float(cfg.basin_radius) ** 2
    alpha = float(os.environ.get("TOY_ALPHA", ALPHA_DEFAULT))
    beta = 1.0 / KT_DEFAULT
    gamma = GAMMA_DEFAULT
    sigma = (2.0 * dt / (beta * gamma)) ** 0.5

    label = torch.full((N,), -1, device=device, dtype=torch.int8)  # -1=open,0=A,1=B
    for _ in range(int(max_steps)):
        active = label < 0
        if not bool(active.any()):
            break
        noise = torch.randn(N, 2, generator=gen, device=device, dtype=xy.dtype)
        f = force(xy, alpha)
        xy_next = xy + f * dt / gamma + sigma * noise
        # Freeze any walker kicked into a non-finite state (past the potential's
        # exp wall) at its previous position so it can't poison the batch.
        bad = ~torch.isfinite(xy_next).all(dim=-1, keepdim=True)
        xy = torch.where(bad, xy, xy_next)
        da = ((xy - well_a) ** 2).sum(-1) < r2
        db = ((xy - well_b) ** 2).sum(-1) < r2
        label = torch.where(active & db, torch.ones_like(label), label)
        label = torch.where(active & da & ~db, torch.zeros_like(label), label)

    label = label.view(S, n_swarms)
    committed = label >= 0
    reached_b = label == 1
    denom = committed.sum(1).clamp(min=1).float()
    q_emp = reached_b.sum(1).float() / denom
    committed_frac = committed.float().mean(1)
    return q_emp, committed_frac


def validate_committor(net, cfg, device, prefix="committor/val_emp"):
    """Run held-out empirical-committor validation; return a wandb-ready dict.

    Truth-free: needs only the fixed seed set and the shared potential. Returns
    {} (and prints) if the seed file is missing so training never breaks.
    """
    if not os.path.exists(cfg.val_seeds_path):
        print(f"[val] no seed file at {cfg.val_seeds_path}; skipping")
        return {}
    d = np.load(cfg.val_seeds_path)
    seeds = torch.tensor(np.asarray(d["xy"], dtype=np.float32), device=device)
    dt = float(os.environ.get("TOY_DT", 1e-3))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed) * 7919 + 13)

    q_emp, committed_frac = _swarm_qemp(
        seeds, cfg, device, cfg.val_n_swarms, cfg.val_max_steps, dt, gen)
    with torch.no_grad():
        q_net = net.forward_mean(seeds)
    err = q_net - q_emp
    w = q_emp * (1.0 - q_emp)                       # reactive weight (empirical)
    rmse = float(torch.sqrt((err ** 2).mean()))
    wden = float(w.sum())
    wrmse = float(torch.sqrt((w * err ** 2).sum() / wden)) if wden > 0 else float("nan")
    mae = float(err.abs().mean())
    cfrac = float(committed_frac.mean())
    print(f"[val] empirical-committor: rmse={rmse:.4f} wrmse={wrmse:.4f} "
          f"mae={mae:.4f} committed={cfrac:.2f} n_seeds={seeds.shape[0]}")
    return {
        f"{prefix}_rmse": rmse,
        f"{prefix}_wrmse": wrmse,
        f"{prefix}_mae": mae,
        f"{prefix}_committed_frac": cfrac,
        f"{prefix}_n_seeds": int(seeds.shape[0]),
    }
