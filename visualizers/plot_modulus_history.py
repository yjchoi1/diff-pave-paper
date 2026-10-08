"""
Modulus histories of DiffFEM and XPINN for one noise level.

Usage:
    python visualizers/plot_modulus_history.py --noise_std 1
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRUE_MPA = {"E1": 1378, "E2": 206, "E3": 69}
LABELS = {"E1": r"$E_{\mathrm{surface}}$", "E2": r"$E_{\mathrm{base}}$", "E3": r"$E_{\mathrm{subgrade}}$"}
COLORS = {"E1": "C0", "E2": "C1", "E3": "C2"}

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})

parser = argparse.ArgumentParser(description="Plot the modulus histories of DiffFEM and XPINN.")
parser.add_argument("--noise_std", type=int, default=1, choices=list(range(6)))
parser.add_argument("--out", type=str, default=None,
                    help="Output file path (default: figures/modulus_history_noise_N.png)")
args = parser.parse_args()

head_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k = args.noise_std
runs = [
    ("DiffFEM", os.path.join(head_path, "Results", f"inverse_DiffFEM_history_noise_{k}.npz"), 1),
    ("XPINN", os.path.join(head_path, "Results", f"noise_{k}", f"inverse_PINN_history_noise_{k}.npz"), 1e3),
]
out_path = args.out or os.path.join(head_path, "figures", f"modulus_history_noise_{k}.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, layout="constrained")
for ax, (name, path, scale), tag in zip(axes, runs, "ab"):
    hist = np.load(path)
    for key in ("E1", "E2", "E3"):
        it = np.arange(1, len(hist[key]) + 1) / scale
        ax.plot(it, hist[key] / 1e6, color=COLORS[key], lw=1.2, label=LABELS[key])
        ax.axhline(TRUE_MPA[key], color=COLORS[key], ls="--", lw=0.8, alpha=0.7)
    ax.set_title(f"({tag}) {name}")
    ax.set_xlabel("Iteration" if scale == 1 else r"Iteration ($\times 10^3$)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
axes[0].set_ylabel("Elastic modulus (MPa)")
axes[1].legend(loc="center right", title="dashed: true", title_fontsize=8)

fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
