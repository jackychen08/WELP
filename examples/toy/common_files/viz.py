"""Plot helpers shared by Grant's run, our run, and analyze.py."""
from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common_files.potential import U_np, WELL_A, WELL_B, ALPHA_DEFAULT


def plot_potential_contour(alpha: float = ALPHA_DEFAULT, ax=None,
                            xrange=(-1.7, 1.7), yrange=(-1.0, 2.0), nxy=301):
    xs = np.linspace(*xrange, nxy)
    ys = np.linspace(*yrange, nxy)
    X, Y = np.meshgrid(xs, ys)
    Z = U_np(X, Y, alpha)
    Zc = np.clip(Z, -20, 25)
    if ax is None:
        ax = plt.gca()
    ax.contourf(X, Y, Zc, levels=30, cmap="viridis", alpha=0.6)
    ax.contour(X, Y, Zc, levels=10, colors="k", linewidths=0.3)
    ax.scatter(*WELL_A, color="white", edgecolor="k", s=80, zorder=3, label="A")
    ax.scatter(*WELL_B, color="red",   edgecolor="k", s=80, zorder=3, label="B")
    ax.set_xlim(*xrange); ax.set_ylim(*yrange)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return ax


def channel_label(xy: np.ndarray) -> np.ndarray:
    """Label each (x, y) as upper-channel (+1), direct/lower (-1), or undecided (0).

    On the ZJZ alpha=72 potential the two low-energy channels sit near:
      - direct (lower): y ≈ 0  saddle at (0, 0)    ~16 kBT
      - upper arc:      y ≈ 1  saddle at (0, 1)    ~16 kBT
    The ridge between them peaks around (0, 0.5) at ~26 kBT.

    A point is classified only in the bottleneck region (|x| < 0.7). Using
    y > 0.5 as the upper/direct cutoff lands each labeled point on one side
    of the ridge.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    mid = np.abs(x) < 0.7
    upper = mid & (y > 0.5)
    direct = mid & (y <= 0.5)
    out = np.zeros(x.shape, dtype=np.int8)
    out[upper] = +1
    out[direct] = -1
    return out


def channel_fractions(xy: np.ndarray):
    """Return (fraction_upper, fraction_lower) over points in the mid-region."""
    lab = channel_label(xy)
    total = np.count_nonzero(lab)
    if total == 0:
        return 0.0, 0.0
    up = np.count_nonzero(lab == +1)
    lo = np.count_nonzero(lab == -1)
    return up / total, lo / total


def plot_samples_over_contour(xy: np.ndarray, title: str, outpath: str,
                              alpha: float = ALPHA_DEFAULT, c=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_potential_contour(alpha, ax=ax)
    c = c if c is not None else "red"
    ax.scatter(xy[:, 0], xy[:, 1], c=c, s=4, alpha=0.4, zorder=4)
    ax.set_title(title)
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
