"""Finite-difference solver for the overdamped committor q*(x, y).

The committor satisfies the generator BVP

    L q = -grad U . grad q + kT * Laplacian q = 0    in Omega \ (A U B)
    q = 0   in A
    q = 1   in B
    zero-flux (Neumann) on the outer box boundary

where L is the overdamped generator.  The advection-diffusion PDE is
discretized with centered differences on a uniform (x, y) grid and solved
as one sparse linear system.

Usage
-----
    python -m toy_two_channel.common_files.solve_true_committor \
        [--alpha 72] [--nx 401] [--ny 401] [--kT 1.0] [--out true_committor.npz]

Output: npz with keys `q`, `x`, `y`, `alpha`, `kT`, `well_a`, `well_b`,
`basin_radius`. Loaded by analyze.py for RMSE vs trained networks.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_files.potential import (  # noqa: E402
    ALPHA_DEFAULT, BASIN_RADIUS, KT_DEFAULT, U_np, WELL_A, WELL_B,
)


def gradU_np(x: np.ndarray, y: np.ndarray, alpha: float):
    """Analytic gradient of U (faster than finite-differences and more stable)."""
    r2 = x * x + y * y
    # term1 = alpha * (x^2+y^2-1)^2 * y^2
    t1 = alpha * (r2 - 1.0) ** 2 * y * y
    dt1_dx = alpha * 4.0 * (r2 - 1.0) * x * y * y
    dt1_dy = alpha * (4.0 * (r2 - 1.0) * y * y * y + 2.0 * (r2 - 1.0) ** 2 * y)

    # t2 = -exp(-4[(x-1)^2+y^2])
    e2 = np.exp(-4.0 * ((x - 1.0) ** 2 + y * y))
    dt2_dx = +8.0 * (x - 1.0) * e2
    dt2_dy = +8.0 * y * e2

    # t3 = -exp(-4[(x+1)^2+y^2])
    e3 = np.exp(-4.0 * ((x + 1.0) ** 2 + y * y))
    dt3_dx = +8.0 * (x + 1.0) * e3
    dt3_dy = +8.0 * y * e3

    # t4 = exp(8(x-1.5))
    e4 = np.exp(8.0 * (x - 1.5))
    dt4_dx = 8.0 * e4
    dt4_dy = 0.0

    # t5 = 4*exp(-8(x+1.5))
    e5 = 4.0 * np.exp(-8.0 * (x + 1.5))
    dt5_dx = -8.0 * e5
    dt5_dy = 0.0

    # t6 = exp(-4(y+0.25))
    e6 = np.exp(-4.0 * (y + 0.25))
    dt6_dx = 0.0
    dt6_dy = -4.0 * e6

    # t7 = 16 * exp(-2 x^2)
    e7 = 16.0 * np.exp(-2.0 * x * x)
    dt7_dx = -4.0 * x * e7
    dt7_dy = 0.0

    dUdx = dt1_dx + dt2_dx + dt3_dx + dt4_dx + dt5_dx + dt6_dx + dt7_dx
    dUdy = dt1_dy + dt2_dy + dt3_dy + dt4_dy + dt5_dy + dt6_dy + dt7_dy
    return dUdx, dUdy


def solve_committor(alpha: float = ALPHA_DEFAULT,
                    kT: float = KT_DEFAULT,
                    xrange=(-1.7, 1.7),
                    yrange=(-1.0, 2.0),
                    nx: int = 401,
                    ny: int = 401,
                    well_a=None,
                    well_b=None,
                    basin_radius: float = BASIN_RADIUS):
    """Discretize L q = 0 with Dirichlet inside basins, zero-flux on box.

    Returns dict with `q`, `x`, `y` and metadata.
    """
    wa = np.asarray(well_a if well_a is not None else WELL_A)
    wb = np.asarray(well_b if well_b is not None else WELL_B)

    x = np.linspace(xrange[0], xrange[1], nx)
    y = np.linspace(yrange[0], yrange[1], ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="xy")      # shape (ny, nx)
    N = nx * ny

    dUdx, dUdy = gradU_np(X, Y, alpha)           # drift -grad U
    ax = -dUdx
    ay = -dUdy

    # row(i,j) index = j*nx + i    (i: x index, j: y index)
    def idx(i, j):
        return j * nx + i

    # Build sparse operator via COO lists
    rows, cols, data = [], [], []
    rhs = np.zeros(N)

    in_A = ((X - wa[0]) ** 2 + (Y - wa[1]) ** 2) < basin_radius ** 2
    in_B = ((X - wb[0]) ** 2 + (Y - wb[1]) ** 2) < basin_radius ** 2

    for j in range(ny):
        for i in range(nx):
            k = idx(i, j)

            # Dirichlet in basins
            if in_A[j, i]:
                rows.append(k); cols.append(k); data.append(1.0); rhs[k] = 0.0
                continue
            if in_B[j, i]:
                rows.append(k); cols.append(k); data.append(1.0); rhs[k] = 1.0
                continue

            # Neumann (zero-flux) on outer box: use one-sided derivatives.
            # For interior we use centered differences:
            #   d/dx q ~ (q[i+1,j] - q[i-1,j]) / (2 dx)
            #   Laplacian ~ (q[i+1,j]-2 q[i,j]+q[i-1,j])/dx^2 + (y analog)

            cx = ax[j, i]
            cy = ay[j, i]

            # --- x direction ---
            if 0 < i < nx - 1:
                # centered
                # advection -ax * d/dx q  -> contributes -ax/(2 dx) at (i+1), +ax/(2 dx) at (i-1)... wait sign:
                # PDE: ax * dq/dx + ay * dq/dy + kT * Laplacian q = 0, with ax = -dU/dx (drift).
                # Discretize: ax * (q[i+1]-q[i-1])/(2 dx) contributes:
                #   row k col idx(i+1,j): +ax/(2 dx)
                #   row k col idx(i-1,j): -ax/(2 dx)
                rows.append(k); cols.append(idx(i + 1, j)); data.append(+cx / (2 * dx))
                rows.append(k); cols.append(idx(i - 1, j)); data.append(-cx / (2 * dx))
                # diffusion kT * d^2q/dx^2
                rows.append(k); cols.append(idx(i + 1, j)); data.append(+kT / dx ** 2)
                rows.append(k); cols.append(idx(i,     j)); data.append(-2 * kT / dx ** 2)
                rows.append(k); cols.append(idx(i - 1, j)); data.append(+kT / dx ** 2)
            elif i == 0:
                # Neumann at i=0: q[-1] mirror = q[1]. This eliminates the advection
                # (since q[1] - q[-1] = 0) and the Laplacian becomes 2*(q[1]-q[0])/dx^2.
                rows.append(k); cols.append(idx(1, j));    data.append(+2 * kT / dx ** 2)
                rows.append(k); cols.append(idx(0, j));    data.append(-2 * kT / dx ** 2)
            else:  # i == nx-1
                rows.append(k); cols.append(idx(nx - 2, j)); data.append(+2 * kT / dx ** 2)
                rows.append(k); cols.append(idx(nx - 1, j)); data.append(-2 * kT / dx ** 2)

            # --- y direction ---
            if 0 < j < ny - 1:
                rows.append(k); cols.append(idx(i, j + 1)); data.append(+cy / (2 * dy))
                rows.append(k); cols.append(idx(i, j - 1)); data.append(-cy / (2 * dy))
                rows.append(k); cols.append(idx(i, j + 1)); data.append(+kT / dy ** 2)
                rows.append(k); cols.append(idx(i, j    )); data.append(-2 * kT / dy ** 2)
                rows.append(k); cols.append(idx(i, j - 1)); data.append(+kT / dy ** 2)
            elif j == 0:
                rows.append(k); cols.append(idx(i, 1));      data.append(+2 * kT / dy ** 2)
                rows.append(k); cols.append(idx(i, 0));      data.append(-2 * kT / dy ** 2)
            else:  # j == ny-1
                rows.append(k); cols.append(idx(i, ny - 2)); data.append(+2 * kT / dy ** 2)
                rows.append(k); cols.append(idx(i, ny - 1)); data.append(-2 * kT / dy ** 2)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    q_flat = spla.spsolve(A, rhs)
    q = q_flat.reshape(ny, nx)
    q_raw = q.copy()   # unclipped solution, kept for validation of the discretisation
    # Enforce bounds (rounding can nudge basin boundary cells a hair off)
    q = np.clip(q, 0.0, 1.0)
    return {
        "q": q,
        "q_raw": q_raw,
        "x": x,
        "y": y,
        "alpha": alpha,
        "kT": kT,
        "well_a": wa,
        "well_b": wb,
        "basin_radius": basin_radius,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    p.add_argument("--kT", type=float, default=KT_DEFAULT)
    p.add_argument("--nx", type=int, default=401)
    p.add_argument("--ny", type=int, default=401)
    p.add_argument("--out", type=str,
                   default=os.path.join(os.path.dirname(__file__), "true_committor.npz"))
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    print(f"Solving overdamped committor BVP at alpha={args.alpha}, nx={args.nx}, ny={args.ny} ...")
    out = solve_committor(alpha=args.alpha, kT=args.kT, nx=args.nx, ny=args.ny)
    np.savez(args.out, **out)
    print(f"Wrote {args.out}   q min={out['q'].min():.4f}, max={out['q'].max():.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(out["x"], out["y"])
        Z = U_np(X, Y, args.alpha)
        Zc = np.clip(Z, -20, 25)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.contourf(X, Y, Zc, levels=30, cmap="viridis", alpha=0.5)
        cs = ax.contour(X, Y, out["q"], levels=np.linspace(0.05, 0.95, 11),
                        colors="red", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=8)
        ax.scatter(*out["well_a"], color="white", edgecolor="k", s=60, zorder=3)
        ax.scatter(*out["well_b"], color="black", edgecolor="k", s=60, zorder=3)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"true q at alpha={args.alpha}")
        plot_path = os.path.splitext(args.out)[0] + ".png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
