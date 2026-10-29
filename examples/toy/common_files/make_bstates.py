"""Generate bstate init_{A,B}_N.npy and pcoord_N.init rows for both sims."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_files.potential import WELL_A, WELL_B, in_basin_a, in_basin_b


def rmsd_to(xy, center):
    return float(np.linalg.norm(np.asarray(xy) - np.asarray(center)))


def generate(n: int, basin: str, out_dir: str, seed: int = 0, jitter: float = 0.05):
    rng = np.random.default_rng(seed)
    well = np.asarray(WELL_A if basin == "A" else WELL_B)
    other = np.asarray(WELL_B if basin == "A" else WELL_A)
    os.makedirs(out_dir, exist_ok=True)
    bstates_txt_lines = []
    prefix = f"init_{basin}"
    for i in range(n):
        # Generate a jittered point inside the basin.
        while True:
            xy = well + rng.normal(scale=jitter, size=2)
            if (basin == "A" and in_basin_a(xy)) or (basin == "B" and in_basin_b(xy)):
                break
        np.save(os.path.join(out_dir, f"{prefix}_{i}.npy"), xy.astype(np.float64))
        rmsd_target = rmsd_to(xy, other)       # for sim_AB target=B, for sim_BA target=A
        # pcoord.init is single-row; columns match the 3D pcoord schema.
        # For the first frame (q_placeholder=0, rmsd_to_target, in_target=0).
        pcoord_row = f"{0.0:.6f} {rmsd_target:.6f} {0.0:.6f}\n"
        with open(os.path.join(out_dir, f"pcoord_{i}.init"), "w") as f:
            f.write(pcoord_row)
        bstates_txt_lines.append(f"{i} {1.0/n:.4f} {prefix}_{i}.npy")

    with open(os.path.join(out_dir, "bstates.txt"), "w") as f:
        f.write("\n".join(bstates_txt_lines) + "\n")
    print(f"Wrote {n} bstates to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim_ab_dir", default=None,
                   help="absolute path to sim_AB/bstates (overrides --root)")
    p.add_argument("--sim_ba_dir", default=None,
                   help="absolute path to sim_BA/bstates (overrides --root)")
    p.add_argument("--root", default=None,
                   help="method directory (e.g. westpa_mab/); resolves "
                        "sim_AB/bstates and sim_BA/bstates under it. Required "
                        "if --sim_ab_dir/--sim_ba_dir not given.")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--jitter", type=float, default=0.03)
    args = p.parse_args()

    if args.sim_ab_dir and args.sim_ba_dir:
        sim_ab, sim_ba = args.sim_ab_dir, args.sim_ba_dir
    else:
        if not args.root:
            p.error("either both --sim_ab_dir and --sim_ba_dir, or --root, "
                    "is required (no implicit westpa_mab default)")
        root = os.path.abspath(args.root)
        sim_ab = args.sim_ab_dir or os.path.join(root, "sim_AB", "bstates")
        sim_ba = args.sim_ba_dir or os.path.join(root, "sim_BA", "bstates")
    generate(args.n, "A", sim_ab, seed=1, jitter=args.jitter)
    generate(args.n, "B", sim_ba, seed=2, jitter=args.jitter)


if __name__ == "__main__":
    main()
