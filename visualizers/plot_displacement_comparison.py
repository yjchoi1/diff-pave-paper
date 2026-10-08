#!/usr/bin/env python
"""
Displacement fields and pointwise errors of DiffFEM and XPINN against the ground truth.

Rows: u_r and u_z. Columns: ground truth, DiffFEM, XPINN, DiffFEM error, XPINN error.

Usage:
    python visualizers/plot_displacement_comparison.py --noise_std 0
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
from matplotlib.collections import PolyCollection

# Project root on path
head_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, head_path)

from utils.common_utils import ANN, device, np_to_torch, torch_to_np
from utils.loss_function import solve_forward

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--noise_std", type=int, default=5, choices=list(range(6)))
parser.add_argument("--out", type=str, default=None,
                    help="Output file path (default: figures/displacement_comparison_noise_N.png)")
args = parser.parse_args()

noise_std   = args.noise_std
data_path   = os.path.join(head_path, "Dataset")
result_path = os.path.join(head_path, "Results")
pretrained_path = os.path.join(head_path, "pretrained", "xpinn")

out_path = args.out or os.path.join(head_path, "figures", f"displacement_comparison_noise_{noise_std}.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# =============================================================================
# Load FEM data
# =============================================================================
data = np.load(os.path.join(data_path, "FEM_precomputed.npz"), allow_pickle=True)

v          = data["v"].copy()           # (nnode, 2)  [r, z]
f_elem     = data["f"].copy().astype(np.int64)  # (nel, n_per_elem)
dirichlet  = data["dirichlet"].copy()
free       = data["free"].copy()
F          = data["F"].copy()
nnode      = len(v)

def _to_dense(arr):
    try:
        return arr.item().toarray()
    except AttributeError:
        return arr.toarray()

K_surface_unit  = _to_dense(data["K_surface_unit"])
K_base_unit     = _to_dense(data["K_base_unit"])
K_subgrade_unit = _to_dense(data["K_subgrade_unit"])

# =============================================================================
# Scaling constants
# =============================================================================
F_ref = 1e+3
u_ref = 1e-3
E_ref = F_ref / u_ref**2

K_s = K_surface_unit  / (F_ref / u_ref / E_ref)
K_b = K_base_unit     / (F_ref / u_ref / E_ref)
K_g = K_subgrade_unit / (F_ref / u_ref / E_ref)
F_n = F / F_ref

# True (normalised) elastic moduli
E_surf_true = 1378e6 / E_ref
E_base_true = 206e6  / E_ref
E_sub_true  = 69e6   / E_ref

# =============================================================================
# Helper: FEM solve, returns displacements in m
# =============================================================================
def fem_solve(E_surf, E_base, E_sub):
    ur, uz = solve_forward(
        E_surf, E_base, E_sub,
        K_s, K_b, K_g,
        F_n, free, dirichlet, nnode,
    )
    return np.asarray(ur) * u_ref, np.asarray(uz) * u_ref

# =============================================================================
# 1. Ground Truth
# =============================================================================
ur_gt, uz_gt = fem_solve(E_surf_true, E_base_true, E_sub_true)
print(f"[GT]       E_surf={1378:.0f} MPa  E_base={206:.0f} MPa  E_sub={69:.0f} MPa")

# =============================================================================
# Helper: return the first existing file
# =============================================================================
def _find(filename, *dirs):
    """Return the first existing path, or None."""
    for d in dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

# =============================================================================
# 2. DiffFEM  (load final E from history)
# =============================================================================
diff_hist_path = _find(f"inverse_DiffFEM_history_noise_{noise_std}.npz", result_path)

if diff_hist_path:
    print(f"[DiffFEM] loading from: {diff_hist_path}")
    hist = np.load(diff_hist_path)
    # The history stores E in Pa
    E1 = float(hist["E1"][-1]) / E_ref
    E2 = float(hist["E2"][-1]) / E_ref
    E3 = float(hist["E3"][-1]) / E_ref
    ur_diff, uz_diff = fem_solve(E1, E2, E3)
    print(f"[DiffFEM] E_surf={E1*E_ref/1e6:.1f} MPa  "
          f"E_base={E2*E_ref/1e6:.1f} MPa  E_sub={E3*E_ref/1e6:.1f} MPa")
    has_diff = True
else:
    print("[WARN] DiffFEM history not found; run inverse_difffem.py first")
    ur_diff, uz_diff = np.zeros_like(ur_gt), np.zeros_like(uz_gt)
    has_diff = False

# =============================================================================
# 3. XPINN (trained model)
# =============================================================================
pinn_path = _find(
    f"inverse_PINN_models_noise_{noise_std}.pt",
    os.path.join(result_path, f"noise_{noise_std}"),
    pretrained_path,
)

def predict_pinn(v_torch, NN_surf, NN_base, NN_sub):
    """Reconstruct XPINN displacement field (physical units, [m])."""
    with torch.no_grad():
        u1 = NN_surf(v_torch) * 1e-1
        u2 = NN_base(v_torch) * 1e-1
        u3 = NN_sub(v_torch)  * 1e-1

        eps   = 1e-3
        z     = v_torch[:, 1]
        mask1 = ((z >  1.40 - eps) & (z <  1.50 + eps)).unsqueeze(1)
        mask2 = ((z >= 1.15 - eps) & (z <  1.40 - eps)).unsqueeze(1)
        mask3 = ((z >= 0.00 - eps) & (z <  1.15 - eps)).unsqueeze(1)

        u = u1 * mask1 + u2 * mask2 + u3 * mask3
        ur = u[:, 0] * v_torch[:, 0] * (1.5 - v_torch[:, 0]) * z
        uz = u[:, 1] * z

    return torch_to_np(ur) * u_ref, torch_to_np(uz) * u_ref

if pinn_path:
    print(f"[XPINN]    loading from: {pinn_path}")
    NN_surf = ANN(input_dim=2, output_dim=2, hidden=32, num_layers=3).to(device).double()
    NN_base = ANN(input_dim=2, output_dim=2, hidden=32, num_layers=3).to(device).double()
    NN_sub  = ANN(input_dim=2, output_dim=2, hidden=32, num_layers=3).to(device).double()

    ckpt = torch.load(pinn_path, map_location=device, weights_only=False)
    NN_surf.load_state_dict(ckpt["NN_surface"])
    NN_base.load_state_dict(ckpt["NN_base"])
    NN_sub.load_state_dict(ckpt["NN_subgrade"])
    NN_surf.eval(); NN_base.eval(); NN_sub.eval()

    E_pinn = ckpt["E_list"].detach().cpu().numpy() * E_ref
    print(f"[XPINN]    E_surf={E_pinn[0]/1e6:.1f} MPa  "
          f"E_base={E_pinn[1]/1e6:.1f} MPa  E_sub={E_pinn[2]/1e6:.1f} MPa")

    v_torch = np_to_torch(v)
    ur_pinn, uz_pinn = predict_pinn(v_torch, NN_surf, NN_base, NN_sub)
    has_pinn = True
else:
    print("[WARN] XPINN model not found")
    ur_pinn, uz_pinn = np.zeros_like(ur_gt), np.zeros_like(uz_gt)
    has_pinn = False

# =============================================================================
# Element-averaged fields  (mm for display)
# =============================================================================
def elem_avg(arr):
    """Average nodal values over each element."""
    return arr[f_elem].mean(axis=1) * 1e3  # m to mm

ur_gt_e   = elem_avg(ur_gt)
uz_gt_e   = elem_avg(uz_gt)
ur_diff_e = elem_avg(ur_diff)
uz_diff_e = elem_avg(uz_diff)
ur_pinn_e = elem_avg(ur_pinn)
uz_pinn_e = elem_avg(uz_pinn)

err_ur_diff = np.abs(ur_diff_e - ur_gt_e)
err_uz_diff = np.abs(uz_diff_e - uz_gt_e)
err_ur_pinn = np.abs(ur_pinn_e - ur_gt_e)
err_uz_pinn = np.abs(uz_pinn_e - uz_gt_e)

# Shared color limits from GT and DiffFEM. XPINN values outside the range are
# clipped and marked by the colorbar extensions.
def _symmetric_pad(lo, hi, pad=0.02):
    """Add a small pad so the extrema aren't rendered at the exact colormap edge."""
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1e-12)
    return lo - pad * span, hi + pad * span

vmin_ur, vmax_ur = _symmetric_pad(
    min(ur_gt_e.min(), ur_diff_e.min()),
    max(ur_gt_e.max(), ur_diff_e.max()),
)
vmin_uz, vmax_uz = _symmetric_pad(
    min(uz_gt_e.min(), uz_diff_e.min()),
    max(uz_gt_e.max(), uz_diff_e.max()),
)
# Error color limit: max of the DiffFEM error and the 95th percentile of the XPINN error
def _err_cap(err_diff, err_pinn):
    return max(err_diff.max(), np.percentile(err_pinn, 95)) * 1.05

vmax_err_ur = max(_err_cap(err_ur_diff, err_ur_pinn), 1e-12)
vmax_err_uz = max(_err_cap(err_uz_diff, err_uz_pinn), 1e-12)

def _extend_for(values, vmin, vmax):
    lo = values.min() < vmin
    hi = values.max() > vmax
    if lo and hi: return "both"
    if lo:        return "min"
    if hi:        return "max"
    return "neither"

# Displacements can be negative, so both colorbar tails are marked
ext_ur   = "both"
ext_uz   = "both"
# Errors are non-negative; mark the upper tail only if exceeded
ext_e_ur = _extend_for(err_ur_pinn, 0.0, vmax_err_ur)
ext_e_uz = _extend_for(err_uz_pinn, 0.0, vmax_err_uz)

# =============================================================================
# Figure layout with horizontal colorbars on top
#   rows: [cb_ur | ur_data | spacer | cb_uz | uz_data]
#   cols: [GT | DiffFEM | XPINN | gap | DiffFEM_err | XPINN_err]
# =============================================================================
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.size":         8,
    "axes.titlesize":    8.5,
    "axes.labelsize":    8,
    "xtick.labelsize":   6.5,
    "ytick.labelsize":   6.5,
    "font.family":       "sans-serif",
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
})

CMAP_DISP = "RdBu_r"
CMAP_ERR  = "Oranges"

# clip=True maps out-of-range XPINN values to the colorbar extremes
norm_ur   = mcolors.Normalize(vmin=vmin_ur,   vmax=vmax_ur,      clip=True)
norm_uz   = mcolors.Normalize(vmin=vmin_uz,   vmax=vmax_uz,      clip=True)
norm_e_ur = mcolors.Normalize(vmin=0,         vmax=vmax_err_ur,  clip=True)
norm_e_uz = mcolors.Normalize(vmin=0,         vmax=vmax_err_uz,  clip=True)

polys = v[f_elem]
r_lim = (v[:, 0].min(), v[:, 0].max())
z_lim = (v[:, 1].min(), v[:, 1].max())

# GridSpec
# 5 rows: cb_ur(0) | ur_data(1) | spacer(2) | cb_uz(3) | uz_data(4)
fig = plt.figure(figsize=(8.5, 6.5), dpi=300)
gs  = GridSpec(
    5, 6,
    figure=fig,
    height_ratios=[0.045, 1, 0.20, 0.045, 1],
    width_ratios=[1, 1, 1, 0.14, 1, 1],
    hspace=0.10,
    wspace=0.10,
    left=0.08, right=0.99, top=0.84, bottom=0.09,
)

# Helpers
X_TICKS = [0.0, 0.5, 1.0]   # skip 1.5 to avoid overlap between adjacent plots

def draw_cell(ax, values, norm, cmap,
              xtick_labels=False, xlabel=False, ylabel_text=None, title=None):
    pc = PolyCollection(polys, array=values, cmap=cmap, norm=norm,
                        edgecolors="none", linewidths=0)
    ax.add_collection(pc)
    ax.set_xlim(*r_lim)
    ax.set_ylim(*z_lim)
    ax.set_aspect("equal")
    ax.set_xticks(X_TICKS)
    ax.tick_params(axis="both", labelsize=6.5, width=0.5, length=2.5)
    if xtick_labels:
        ax.set_xticklabels([str(x) for x in X_TICKS])
    else:
        ax.set_xticklabels([])
    if xlabel:
        ax.set_xlabel("r  [m]", fontsize=7.5, labelpad=2)
    if ylabel_text:
        ax.set_ylabel(ylabel_text, fontsize=8, labelpad=4)
    else:
        ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=8.5, pad=3, fontweight="bold")
    return pc


def draw_hcb(cax, norm, cmap, label, ticks_on_top=True, section_title=None,
             extend="neither"):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend=extend,
                      extendfrac=0.03)
    cb.set_label(label, fontsize=8, labelpad=3)
    pos = "top" if ticks_on_top else "bottom"
    cb.ax.xaxis.set_ticks_position(pos)
    cb.ax.xaxis.set_label_position(pos)
    cb.locator = matplotlib.ticker.MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2.5)
    cb.outline.set_linewidth(0.5)
    if section_title:
        cax.set_title(section_title, fontsize=9, fontweight="bold", pad=14)


# Colorbar axes (rows 0 and 3), ticks on top
ax_cb_ur_d = fig.add_subplot(gs[0, 0:3])
ax_cb_ur_e = fig.add_subplot(gs[0, 4:6])
ax_cb_uz_d = fig.add_subplot(gs[3, 0:3])
ax_cb_uz_e = fig.add_subplot(gs[3, 4:6])

draw_hcb(ax_cb_ur_d, norm_ur,   CMAP_DISP, r"$u_r$  [mm]",
         ticks_on_top=True,  section_title="Displacement field", extend=ext_ur)
draw_hcb(ax_cb_ur_e, norm_e_ur, CMAP_ERR,  r"$|e_{u_r}|$  [mm]",
         ticks_on_top=True,  section_title="Error  |pred − GT|", extend=ext_e_ur)
draw_hcb(ax_cb_uz_d, norm_uz,   CMAP_DISP, r"$u_z$  [mm]",
         ticks_on_top=True, extend=ext_uz)
draw_hcb(ax_cb_uz_e, norm_e_uz, CMAP_ERR,  r"$|e_{u_z}|$  [mm]",
         ticks_on_top=True, extend=ext_e_uz)

# Data axes, u_r row (gs row 1)
ax_ur_gt    = fig.add_subplot(gs[1, 0])
ax_ur_diff  = fig.add_subplot(gs[1, 1])
ax_ur_pinn  = fig.add_subplot(gs[1, 2])
ax_eur_diff = fig.add_subplot(gs[1, 4])
ax_eur_pinn = fig.add_subplot(gs[1, 5])

draw_cell(ax_ur_gt,    ur_gt_e,     norm_ur,   CMAP_DISP,
          xtick_labels=True, xlabel=True, ylabel_text="z  [m]", title="Ground Truth")
draw_cell(ax_ur_diff,  ur_diff_e,   norm_ur,   CMAP_DISP,
          xtick_labels=True, xlabel=True, title="DiffFEM")
draw_cell(ax_ur_pinn,  ur_pinn_e,   norm_ur,   CMAP_DISP,
          xtick_labels=True, xlabel=True, title="XPINN")
draw_cell(ax_eur_diff, err_ur_diff, norm_e_ur,  CMAP_ERR,
          xtick_labels=True, xlabel=True, title="DiffFEM")
draw_cell(ax_eur_pinn, err_ur_pinn, norm_e_ur,  CMAP_ERR,
          xtick_labels=True, xlabel=True, title="XPINN")

# Data axes, u_z row (gs row 4)
ax_uz_gt    = fig.add_subplot(gs[4, 0])
ax_uz_diff  = fig.add_subplot(gs[4, 1])
ax_uz_pinn  = fig.add_subplot(gs[4, 2])
ax_euz_diff = fig.add_subplot(gs[4, 4])
ax_euz_pinn = fig.add_subplot(gs[4, 5])

draw_cell(ax_uz_gt,    uz_gt_e,     norm_uz,   CMAP_DISP,
          xtick_labels=True, xlabel=True, ylabel_text="z  [m]", title="Ground Truth")
draw_cell(ax_uz_diff,  uz_diff_e,   norm_uz,   CMAP_DISP,
          xtick_labels=True, xlabel=True, title="DiffFEM")
draw_cell(ax_uz_pinn,  uz_pinn_e,   norm_uz,   CMAP_DISP,
          xtick_labels=True, xlabel=True, title="XPINN")
draw_cell(ax_euz_diff, err_uz_diff, norm_e_uz,  CMAP_ERR,
          xtick_labels=True, xlabel=True, title="DiffFEM")
draw_cell(ax_euz_pinn, err_uz_pinn, norm_e_uz,  CMAP_ERR,
          xtick_labels=True, xlabel=True, title="XPINN")

# Vertical separator between displacement and error sections
fig.canvas.draw()
x_sep = (ax_ur_pinn.get_position().x1 + ax_eur_diff.get_position().x0) / 2
y_bot = ax_uz_gt.get_position().y0
y_top = ax_cb_ur_d.get_position().y1
fig.add_artist(
    plt.Line2D([x_sep, x_sep], [y_bot, y_top],
               transform=fig.transFigure,
               color="#bbbbbb", linewidth=0.8, linestyle="--")
)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
