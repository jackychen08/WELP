"""Two-channel 2D potential from Zhang, Jasnow, Zuckerman (arXiv:0902.2772, Eq. 1).

U(x,y)/kBT = alpha*(x^2+y^2-1)^2 * y^2
           - exp(-4*((x-1)^2 + y^2))  - exp(-4*((x+1)^2 + y^2))
           + exp(8*(x-1.5))  + 4*exp(-8*(x+1.5))
           + exp(-4*(y+0.25))
           + 16*exp(-2*x^2)

Wells sit near (-1, 0) and (+1, 0); two channels connect them, one above
(y>0) and one below (y<0). The alpha*(...)*y^2 term controls the inter-
channel barrier: alpha=20 -> ~3 kBT (easy), alpha=72 -> ~10 kBT (hard).

This file imports torch for autograd gradients but everything is usable
on CPU or CUDA. The alpha=72 well positions are hardcoded (computed once
via locate_wells) so every WESTPA segment doesn't re-run an SGD loop at
import. Use locate_wells(alpha) if you want wells at a different alpha.
"""
from __future__ import annotations

import numpy as np
import torch

ALPHA_DEFAULT = 72.0
BASIN_RADIUS = 0.25          # Euclidean, matches paper's "U(x,y) < Umin + 2" ring at alpha=72
KT_DEFAULT = 1.0             # everything in kBT units
GAMMA_DEFAULT = 1.0
MASS_DEFAULT = 1.0


def U(xy: torch.Tensor, alpha: float = ALPHA_DEFAULT) -> torch.Tensor:
    """Potential U(x,y) in units of kBT. Accepts arbitrary leading batch shape."""
    x, y = xy[..., 0], xy[..., 1]
    return (alpha * (x * x + y * y - 1.0) ** 2 * y * y
            - torch.exp(-4.0 * ((x - 1.0) ** 2 + y * y))
            - torch.exp(-4.0 * ((x + 1.0) ** 2 + y * y))
            + torch.exp(8.0 * (x - 1.5))
            + 4.0 * torch.exp(-8.0 * (x + 1.5))
            + torch.exp(-4.0 * (y + 0.25))
            + 16.0 * torch.exp(-2.0 * x * x))


def U_np(x: np.ndarray, y: np.ndarray, alpha: float = ALPHA_DEFAULT) -> np.ndarray:
    """Numpy analog for fast grid evaluation (solver, plotting)."""
    return (alpha * (x * x + y * y - 1.0) ** 2 * y * y
            - np.exp(-4.0 * ((x - 1.0) ** 2 + y * y))
            - np.exp(-4.0 * ((x + 1.0) ** 2 + y * y))
            + np.exp(8.0 * (x - 1.5))
            + 4.0 * np.exp(-8.0 * (x + 1.5))
            + np.exp(-4.0 * (y + 0.25))
            + 16.0 * np.exp(-2.0 * x * x))


def force(xy: torch.Tensor, alpha: float = ALPHA_DEFAULT) -> torch.Tensor:
    """F(x,y) = -grad U. Uses autograd; caller should .detach() the input if needed."""
    xy = xy.detach().clone().requires_grad_(True)
    u = U(xy, alpha).sum()
    g, = torch.autograd.grad(u, xy)
    return -g


def _numeric_minimum(x0: float, y0: float, alpha: float, n_steps: int = 1000,
                     lr: float = 1e-2) -> tuple[float, float]:
    """Gradient descent from (x0, y0) to find nearest well minimum."""
    xy = torch.tensor([[x0, y0]], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([xy], lr=lr, momentum=0.9)
    for _ in range(n_steps):
        opt.zero_grad()
        u = U(xy, alpha).sum()
        u.backward()
        opt.step()
    return float(xy[0, 0].detach()), float(xy[0, 1].detach())


def locate_wells(alpha: float = ALPHA_DEFAULT) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ((x_A, y_A), (x_B, y_B)) at the given alpha."""
    wa = _numeric_minimum(-1.0, 0.0, alpha)
    wb = _numeric_minimum(+1.0, 0.0, alpha)
    return wa, wb


# Hardcoded well positions at ALPHA_DEFAULT=72. Computed once via locate_wells;
# pinned here so each WESTPA segment import doesn't re-run 1000 SGD steps.
# If ALPHA_DEFAULT ever changes, regenerate with `locate_wells(new_alpha)`.
WELL_A: tuple[float, float] = (-1.1965325247864693, 0.0372850185948454)
WELL_B: tuple[float, float] = (1.2791771018663545, 0.021034994407029616)
WELL_A_T = torch.tensor(WELL_A, dtype=torch.float64)
WELL_B_T = torch.tensor(WELL_B, dtype=torch.float64)


def in_basin_a(xy, well_a=None, radius: float = BASIN_RADIUS):
    """True iff (x,y) lies within Euclidean BASIN_RADIUS of WELL_A. Torch OR numpy."""
    wa = well_a if well_a is not None else WELL_A
    if torch.is_tensor(xy):
        wa = torch.as_tensor(wa, dtype=xy.dtype, device=xy.device)
        return ((xy - wa) ** 2).sum(-1) < radius * radius
    else:
        xy = np.asarray(xy)
        return ((xy - np.asarray(wa)) ** 2).sum(-1) < radius * radius


def in_basin_b(xy, well_b=None, radius: float = BASIN_RADIUS):
    wb = well_b if well_b is not None else WELL_B
    if torch.is_tensor(xy):
        wb = torch.as_tensor(wb, dtype=xy.dtype, device=xy.device)
        return ((xy - wb) ** 2).sum(-1) < radius * radius
    else:
        xy = np.asarray(xy)
        return ((xy - np.asarray(wb)) ** 2).sum(-1) < radius * radius


def rmsd_to_a(xy, well_a=None):
    wa = well_a if well_a is not None else WELL_A
    if torch.is_tensor(xy):
        wa = torch.as_tensor(wa, dtype=xy.dtype, device=xy.device)
        return torch.linalg.norm(xy - wa, dim=-1)
    return np.linalg.norm(np.asarray(xy) - np.asarray(wa), axis=-1)


def rmsd_to_b(xy, well_b=None):
    wb = well_b if well_b is not None else WELL_B
    if torch.is_tensor(xy):
        wb = torch.as_tensor(wb, dtype=xy.dtype, device=xy.device)
        return torch.linalg.norm(xy - wb, dim=-1)
    return np.linalg.norm(np.asarray(xy) - np.asarray(wb), axis=-1)


if __name__ == "__main__":
    # Sanity dump — well positions, barrier heights, quick contour.
    import sys
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else ALPHA_DEFAULT
    wa, wb = locate_wells(alpha)
    Ua = U(torch.tensor([list(wa)], dtype=torch.float64), alpha).item()
    Ub = U(torch.tensor([list(wb)], dtype=torch.float64), alpha).item()
    ridge = U(torch.tensor([[0.0, 0.0]], dtype=torch.float64), alpha).item()
    saddle_direct = U(torch.tensor([[0.0, 0.0]], dtype=torch.float64), alpha).item()
    saddle_upper = U(torch.tensor([[0.0, 1.0]], dtype=torch.float64), alpha).item()
    inter_ridge = U(torch.tensor([[0.0, 0.5]], dtype=torch.float64), alpha).item()
    print(f"alpha = {alpha}")
    print(f"WELL_A = ({wa[0]:.6f}, {wa[1]:.6f})  U = {Ua:.4f} kBT")
    print(f"WELL_B = ({wb[0]:.6f}, {wb[1]:.6f})  U = {Ub:.4f} kBT")
    print(f"direct saddle (0, 0)            U = {saddle_direct:.4f} kBT")
    print(f"upper  saddle (0, 1)            U = {saddle_upper:.4f} kBT")
    print(f"inter-channel ridge (0, 0.5)    U = {inter_ridge:.4f} kBT   (barrier height ~ {inter_ridge - min(saddle_direct, saddle_upper):.2f} kBT)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.linspace(-1.7, 1.7, 301)
        ys = np.linspace(-1.0, 2.0, 301)
        X, Y = np.meshgrid(xs, ys)
        Z = U_np(X, Y, alpha)
        plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, Z, levels=np.linspace(-5, 30, 36),
                     cmap="viridis", extend="both")
        plt.colorbar(label=r"$U/k_B T$")
        plt.contour(X, Y, Z, levels=np.linspace(-5, 30, 12), colors="k", linewidths=0.3)
        plt.scatter(*wa, color="white", s=60, edgecolor="k", zorder=3, label="A")
        plt.scatter(*wb, color="red",   s=60, edgecolor="k", zorder=3, label="B")
        plt.xlabel("x"); plt.ylabel("y"); plt.title(f"ZJZ potential, alpha={alpha}")
        plt.legend()
        out = f"contour_alpha{int(alpha)}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Wrote {out}")
    except Exception as exc:
        print(f"(plotting skipped: {exc})")
