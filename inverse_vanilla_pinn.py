import argparse
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

head_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(head_path)

from utils.common_utils import *
from utils.common_utils import ANN
from utils.common_utils import plot_loss_history
from utils.common_utils import plot_param_history
from utils.common_utils import plot_displacement

from utils.loss_function import evaluate_loss_vanilla_PINN

# =============================================================================
# Configuration
# =============================================================================
parser = argparse.ArgumentParser(description="Inverse Vanilla PINN with configurable noise level")
parser.add_argument(
    "--noise_std",
    type=int,
    default=0,
    choices=[0, 1, 2, 3, 4, 5],
    help="Noise standard deviation in μm (0 = noise-free)",
)
args = parser.parse_args()

CONFIG = {
    "noise_std": args.noise_std,
    "print_interval": 1000,
    "plot_interval": 1000,
    "checkpoint_interval": 1000,
    "max_iter": 10000,
    "epochs": 100,
    "resume": True,
    "save_dir": f"Results/vanilla_pinn_noise_{args.noise_std}",
    "data_dir": "Dataset",
    "seed": 1,
}

set_seed(CONFIG["seed"])

data_path = os.path.join(head_path, CONFIG["data_dir"])
save_path = os.path.join(head_path, CONFIG["save_dir"])

os.makedirs(save_path, exist_ok=True)

config_path = os.path.join(save_path, "config.json")
with open(config_path, "w") as f:
    json.dump(CONFIG, f, indent=2)

# =============================================================================
# Import data
# =============================================================================

os.chdir(data_path)

noise_std = CONFIG["noise_std"]
if noise_std == 0:
    data = np.load("sensor_data_noise_free.npz", allow_pickle=True)
else:
    data = np.load("sensor_data_noise_{0}μm.npz".format(noise_std), allow_pickle=True)

sensor_node_idx = data["sensor_node_idx"].copy()
sensor_ur = data["sensor_ur"].copy()
sensor_uz = data["sensor_uz"].copy()

data = np.load("FEM_precomputed.npz", allow_pickle=True)

v = data["v"].copy()
f = data["f"].copy()

dirichlet = data["dirichlet"].copy()
free      = data["free"].copy()

Bot_line_idx    = data["Bot_line_idx"].copy()
Left_line_idx   = data["Left_line_idx"].copy()
Right_line_idx  = data["Right_line_idx"].copy()
Top_line_idx    = data["Top_line_idx"].copy()
Force_line_idx  = data["Force_line_idx"].copy()

K_surface_unit  = data["K_surface_unit"].copy()
K_base_unit     = data["K_base_unit"].copy()
K_subgrade_unit = data["K_subgrade_unit"].copy()

F = data["F"].copy()

nnode = len(v)
nel   = len(f)

os.chdir(head_path)

# =============================================================================
#  Scaling (Normalization)
# =============================================================================
F_ref = 1e+3
u_ref = 1e-3
K_ref = F_ref / u_ref
E_ref = F_ref / u_ref**2

K_surface_unit  = K_surface_unit  / (K_ref / E_ref)
K_base_unit     = K_base_unit     / (K_ref / E_ref)
K_subgrade_unit = K_subgrade_unit / (K_ref / E_ref)

F = F / F_ref

sensor_ur = sensor_ur / u_ref
sensor_uz = sensor_uz / u_ref

# =============================================================================
# numpy to torch
# =============================================================================

sensor_ur_torch = np_to_torch(sensor_ur)
sensor_uz_torch = np_to_torch(sensor_uz)

v_torch = np_to_torch(v)
f_torch = np_to_torch(f)

F_torch = np_to_torch(F)

try:
    K_surface_unit_torch  = np_to_torch(K_surface_unit.item().toarray())
    K_base_unit_torch     = np_to_torch(K_base_unit.item().toarray())
    K_subgrade_unit_torch = np_to_torch(K_subgrade_unit.item().toarray())
except:
    K_surface_unit_torch  = np_to_torch(K_surface_unit.toarray())
    K_base_unit_torch     = np_to_torch(K_base_unit.toarray())
    K_subgrade_unit_torch = np_to_torch(K_subgrade_unit.toarray())

dirichlet_torch = np_to_torch(dirichlet).to(int)
free_torch = np_to_torch(free).to(int)

sensor_node_idx_torch = np_to_torch(sensor_node_idx).to(int)

Bot_line_idx_torch   = np_to_torch(Bot_line_idx).to(int)
Left_line_idx_torch  = np_to_torch(Left_line_idx).to(int)
Right_line_idx_torch = np_to_torch(Right_line_idx).to(int)
Top_line_idx_torch   = np_to_torch(Top_line_idx).to(int)
Force_line_idx_torch = np_to_torch(Force_line_idx).to(int)


# =============================================================================
# Model & Optimizer Setup
# =============================================================================

history = HistoryTracker(["E1", "E2", "E3", "loss_data", "loss_physics"])
E_list = nn.Parameter(np_to_torch(np.array([1e-3, 1e-3, 1e-3])))
NN = ANN(input_dim=2, output_dim=2, hidden=64, num_layers=3).to(device)

if (E_list.dtype == torch.float64):
    NN.double()

optimizer = torch.optim.LBFGS(
    list(NN.parameters()) + [E_list],
    lr=1.0,
    max_iter=CONFIG["max_iter"],
    max_eval=CONFIG["max_iter"],
    tolerance_grad=1e-20,
    tolerance_change=1e-20,
    history_size=200,
    line_search_fn="strong_wolfe",
)

iters = 0
start_time = time.time()
checkpoint_path = os.path.join(save_path, f"checkpoint_noise_{CONFIG['noise_std']}.pt")

# =============================================================================
# Resume from Checkpoint
# =============================================================================
if CONFIG["resume"] and os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    NN.load_state_dict(checkpoint['NN'])
    E_list.data = checkpoint['E_list'].data
    history.data = checkpoint['history']
    iters = checkpoint['iters']
    start_time = time.time() - checkpoint.get('elapsed_time', 0)
    print(f"Resumed from iteration {iters}")
else:
    print("Starting fresh training.")

# =============================================================================
# Training Vanilla PINN (L-BFGS)
# =============================================================================

def save_checkpoint_func():
    torch.save({
        'iters': iters,
        'NN': NN.state_dict(),
        'E_list': E_list,
        'optimizer': optimizer.state_dict(),
        'history': history.data,
        'elapsed_time': time.time() - start_time
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    history.save(
        os.path.join(save_path, f"inverse_vanilla_PINN_history_noise_{noise_std}.npz"),
        Elapsed_time=time.time() - start_time,
    )

def closure():
    global iters
    optimizer.zero_grad()

    E_surface  = E_list[0]
    E_base     = E_list[1]
    E_subgrade = E_list[2]

    loss_physics, loss_data, ur_pred_torch, uz_pred_torch, res_x, res_y = evaluate_loss_vanilla_PINN(
        E_surface, E_base, E_subgrade,
        K_surface_unit_torch, K_base_unit_torch, K_subgrade_unit_torch,
        F_torch, free_torch, dirichlet_torch, nnode, nel,
        sensor_ur_torch, sensor_uz_torch, sensor_node_idx_torch,
        NN, v_torch, f_torch,
    )
    total_loss = loss_physics * 1e0 + loss_data * 1e3
    total_loss.backward(retain_graph=False)

    history.record(
        loss_data    = loss_data.item(),
        loss_physics = loss_physics.item(),
        E1           = E_surface.item()  * E_ref,
        E2           = E_base.item()     * E_ref,
        E3           = E_subgrade.item() * E_ref,
    )

    iters += 1

    if iters % CONFIG["print_interval"] == 0:
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"L-BFGS : Iter {iters} | time={elapsed_time_str} | Loss={total_loss.item():.4e}")

    if iters % CONFIG["plot_interval"] == 0:
        ur_np = torch_to_np(ur_pred_torch) * u_ref
        uz_np = torch_to_np(uz_pred_torch) * u_ref

        plot_loss_history({k: history.data[k] for k in ["loss_data", "loss_physics"]},
                          save_path=os.path.join(save_path, "loss_history.png"))
        plot_displacement(v, f, ur_np, uz_np, save_path=os.path.join(save_path, f"displacement_iter_{iters}.png"))
        plot_param_history(history.data,
                            E_surface_true  = 1378 * 1e6,
                            E_base_true     = 206 * 1e6,
                            E_subgrade_true = 69 * 1e6,
                            save_path=os.path.join(save_path, "param_history.png"))
        plot_displacement(v, f, torch_to_np(res_x), torch_to_np(res_y), save_path=os.path.join(save_path, f"residual_iter_{iters}.png"))

    if iters % CONFIG["checkpoint_interval"] == 0:
        save_checkpoint_func()

    return total_loss


for epoch in range(CONFIG["epochs"]):
    print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
    loss = optimizer.step(closure)


# =============================================================================
# Final Save
# =============================================================================

os.chdir(save_path)

history.save(
    f"inverse_vanilla_PINN_history_noise_{noise_std}.npz",
    Elapsed_time=time.time() - start_time,
)

save_name = "inverse_vanilla_PINN_models_noise_{0}.pt".format(noise_std)
torch.save(
    {
        "NN": NN.state_dict(),
        "E_list": E_list.detach().cpu(),
    },
    save_name
)

plot_loss_history({k: history.data[k] for k in ["loss_data", "loss_physics"]},
                  save_path="final_loss_history.png")
plot_param_history(history.data,
                    E_surface_true  = 1378 * 1e6,
                    E_base_true     = 206 * 1e6,
                    E_subgrade_true = 69 * 1e6,
                    save_path="final_param_history.png")

os.chdir(head_path)
print("Training completed.")
