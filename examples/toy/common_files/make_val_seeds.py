"""Generate a fixed, shared held-out seed set for empirical-committor validation.

Seeds are placed in the transition region (where the committor actually matters)
by sampling grid points whose ground-truth committor q* falls in (lo, hi). NOTE:
truth is used ONLY to PLACE the seeds; the validation signal itself (q_emp from
swarms, in committor/validation.py) never uses truth, so the metric stays
deployable. The output common_files/val_seeds.npz is shared by every run via the
per-run `common_files` symlink, so all seeds/configs validate on identical points.

Usage:
    python common_files/make_val_seeds.py --n 300 --lo 0.1 --hi 0.9 --seed 0
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common_files.potential import U_np, KT_DEFAULT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of seed points")
    ap.add_argument("--lo", type=float, default=0.1, help="lower q* band edge")
    ap.add_argument("--hi", type=float, default=0.9, help="upper q* band edge")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    ap.add_argument("--truth", default=os.path.join(HERE, "true_committor.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "val_seeds.npz"))
    args = ap.parse_args()

    d = np.load(args.truth, allow_pickle=True)
    X, Y = np.meshgrid(d["x"], d["y"])
    q = np.asarray(d["q"])
    mask = (q > args.lo) & (q < args.hi)
    xy_all = np.stack([X[mask], Y[mask]], axis=-1).astype(np.float32)
    q_all = q[mask]
    if len(xy_all) == 0:
        raise SystemExit("no grid points in the requested q* band")

    # Place seeds where MD actually goes: Boltzmann-weight within the band so we
    # avoid high-energy off-pathway points (those blow up against the potential's
    # exponential walls instead of committing, corrupting q_emp). Stratify across
    # q* sub-bins so seeds still span the committor range, not just the saddle.
    U = U_np(xy_all[:, 0], xy_all[:, 1])
    w_all = np.exp(-(U - U.min()) / KT_DEFAULT)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(xy_all))
    nbins = 8
    edges = np.linspace(args.lo, args.hi, nbins + 1)
    picks = []
    per_bin = int(np.ceil(n / nbins))
    for b in range(nbins):
        sel = np.where((q_all >= edges[b]) & (q_all < edges[b + 1]))[0]
        if len(sel) == 0:
            continue
        w = w_all[sel] / w_all[sel].sum()
        k = min(per_bin, len(sel))
        picks.append(rng.choice(sel, size=k, replace=False, p=w))
    idx = np.concatenate(picks)[:n]
    xy = xy_all[idx]

    np.savez(args.out, xy=xy, q_star=q_all[idx].astype(np.float32),
             lo=args.lo, hi=args.hi,
             source="Boltzmann-weighted, q*-stratified true_committor band")
    print(f"wrote {args.out}: {len(idx)} seeds in q*∈({args.lo},{args.hi}) "
          f"(Boltzmann-weighted within {nbins} q* strata; "
          f"from {len(xy_all)} candidate grid points)")


if __name__ == "__main__":
    main()
