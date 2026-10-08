import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def torch_to_np(A):
    if isinstance(A, np.ndarray):
        return A
    elif isinstance(A, torch.Tensor):
        return A.detach().cpu().numpy()
    else:
        return np.array(A)

def np_to_torch(A):
    if isinstance(A, torch.Tensor):
        return A.to(dtype=torch.float64, device=device).requires_grad_()
    elif isinstance(A, np.ndarray):
        return torch.tensor(A, dtype=torch.float64, device=device, requires_grad=True)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(A)}")


def plot_loss_history(loss_history, save_path=None):
    plt.figure(figsize=(8, 6), dpi=600)

    valid_losses = {}

    for key, values in loss_history.items():
        if values and np.any(np.array(values) != 0):
            valid_losses[key] = values

    if not valid_losses:
        print("No valid loss data to plot.")
        return

    for key, values in valid_losses.items():
        plt.plot(values, label=key, alpha=0.7)

    plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Loss History")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    min_loss = min(min(values) for values in valid_losses.values())
    max_loss = max(max(values) for values in valid_losses.values())

    plt.ylim(min_loss*1e-1, max_loss * 10)

    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.close()



def plot_param_history(
    E_history,
    E_surface_true=1,
    E_base_true=1,
    E_subgrade_true=1,
    title="E modulus history",
    figsize=(6, 4),
    dpi=300,
    save_path=None
):

    plt.figure(figsize=figsize, dpi=dpi)

    plt.plot(E_history["E1"], label="E_surface", linewidth=2.0, color="tab:blue")
    plt.plot(E_history["E2"], label="E_base", linewidth=2.0, color="tab:orange")
    plt.plot(E_history["E3"], label="E_subgrade", linewidth=2.0, color="tab:green")

    plt.axhline(E_surface_true, linestyle="--", linewidth=1.5,
                label="E_surface_true", color="tab:blue")
    plt.axhline(E_base_true, linestyle="--", linewidth=1.5,
                label="E_base_true", color="tab:orange")
    plt.axhline(E_subgrade_true, linestyle="--", linewidth=1.5,
                label="E_subgrade_true", color="tab:green")

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("E value (Pa)", fontsize=12)
    plt.legend(fontsize=8, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_displacement(v, f, ur, uz, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    def as_numpy_copy(x):
        # torch tensor
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().copy()
        # numpy / list / tuple
        return np.asarray(x).copy()

    v_np  = as_numpy_copy(v)
    f_np  = as_numpy_copy(f).astype(np.int64)
    ur_np = as_numpy_copy(ur).reshape(-1)
    uz_np = as_numpy_copy(uz).reshape(-1)

    polys = v_np[f_np]

    ur_elem = ur_np[f_np].mean(axis=1)
    uz_elem = uz_np[f_np].mean(axis=1)

    vmin_ur, vmax_ur = np.nanmin(ur_elem), np.nanmax(ur_elem)
    vmin_uz, vmax_uz = np.nanmin(uz_elem), np.nanmax(uz_elem)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    pc0 = PolyCollection(
        polys,
        array=ur_elem,
        cmap="RdBu_r",
        edgecolors="k",
        linewidths=0.12
    )
    pc0.set_clim(vmin=vmin_ur, vmax=vmax_ur)
    axs[0].add_collection(pc0)
    axs[0].autoscale_view()
    axs[0].set_aspect("equal")
    axs[0].set_title(r"$u_r$", fontsize=14)
    axs[0].set_xlabel("r")
    axs[0].set_ylabel("z")
    plt.colorbar(pc0, ax=axs[0], fraction=0.046, pad=0.04)

    pc1 = PolyCollection(
        polys,
        array=uz_elem,
        cmap="RdBu_r",
        edgecolors="k",
        linewidths=0.12
    )
    pc1.set_clim(vmin=vmin_uz, vmax=vmax_uz)
    axs[1].add_collection(pc1)
    axs[1].autoscale_view()
    axs[1].set_aspect("equal")
    axs[1].set_title(r"$u_z$", fontsize=14)
    axs[1].set_xlabel("r")
    axs[1].set_ylabel("z")
    plt.colorbar(pc1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


class HistoryTracker:
    def __init__(self, keys):
        self.data = {k: [] for k in keys}

    def record(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(float(v))

    def save(self, path, **extra):
        arrays = {k: np.array(v) for k, v in self.data.items()}
        arrays.update(extra)
        np.savez(path, **arrays)


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return nn.Tanh()( x )


class ANN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden=32, num_layers=3):
        """
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output.
            hidden (int): Number of neurons in each hidden layer.
            num_layers (int): Number of hidden (hidden x hidden) layers after the input layer.
        """
        super(ANN, self).__init__()
        self.activation = CustomActivation()

        self.input_layer = nn.Linear(input_dim, hidden)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x_input):
        x = self.activation(self.input_layer(x_input))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
