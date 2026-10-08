"""
Inverse PINN training (L-BFGS). Used by inverse_xpinn.py and sweep_inverse_pinn.py.

All hyperparameters come from a config dict (loaded from JSON).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.common_utils import (
    ANN,
    HistoryTracker,
    np_to_torch,
    plot_displacement,
    plot_loss_history,
    plot_param_history,
    set_seed,
    torch_to_np,
)
from utils.loss_function import evaluate_loss_PINN


def _abs_path(head_path: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(head_path, p)


def load_inverse_pinn_arrays(head_path: str, noise_std: int) -> Dict[str, Any]:
    """Load FEM and sensor arrays from Dataset/. Called once per training run."""
    data_dir = os.path.join(head_path, "Dataset")
    if noise_std == 0:
        sensor_path = os.path.join(data_dir, "sensor_data_noise_free.npz")
    else:
        sensor_path = os.path.join(data_dir, "sensor_data_noise_{0}μm.npz".format(noise_std))
    data = np.load(sensor_path, allow_pickle=True)
    sensor_node_idx = data["sensor_node_idx"].copy()
    sensor_ur = data["sensor_ur"].copy()
    sensor_uz = data["sensor_uz"].copy()

    data = np.load(os.path.join(data_dir, "FEM_precomputed.npz"), allow_pickle=True)
    v = data["v"].copy()
    f = data["f"].copy()
    dirichlet = data["dirichlet"].copy()
    free = data["free"].copy()
    Bot_line_idx = data["Bot_line_idx"].copy()
    Left_line_idx = data["Left_line_idx"].copy()
    Right_line_idx = data["Right_line_idx"].copy()
    Top_line_idx = data["Top_line_idx"].copy()
    Force_line_idx = data["Force_line_idx"].copy()
    K_surface_unit = data["K_surface_unit"].copy()
    K_base_unit = data["K_base_unit"].copy()
    K_subgrade_unit = data["K_subgrade_unit"].copy()
    F = data["F"].copy()
    nnode = len(v)
    nel = len(f)

    F_ref = 1e3
    u_ref = 1e-3
    K_ref = F_ref / u_ref
    E_ref = F_ref / u_ref**2

    K_surface_unit = K_surface_unit / (K_ref / E_ref)
    K_base_unit = K_base_unit / (K_ref / E_ref)
    K_subgrade_unit = K_subgrade_unit / (K_ref / E_ref)
    F = F / F_ref
    sensor_ur = sensor_ur / u_ref
    sensor_uz = sensor_uz / u_ref

    try:
        K_surface_arr = K_surface_unit.item().toarray()
        K_base_arr = K_base_unit.item().toarray()
        K_subgrade_arr = K_subgrade_unit.item().toarray()
    except Exception:
        K_surface_arr = K_surface_unit.toarray()
        K_base_arr = K_base_unit.toarray()
        K_subgrade_arr = K_subgrade_unit.toarray()

    return {
        "v": v,
        "f": f,
        "nnode": nnode,
        "nel": nel,
        "dirichlet": dirichlet,
        "free": free,
        "Bot_line_idx": Bot_line_idx,
        "Left_line_idx": Left_line_idx,
        "Right_line_idx": Right_line_idx,
        "Top_line_idx": Top_line_idx,
        "Force_line_idx": Force_line_idx,
        "sensor_node_idx": sensor_node_idx,
        "sensor_ur": sensor_ur,
        "sensor_uz": sensor_uz,
        "F": F,
        "K_surface_arr": K_surface_arr,
        "K_base_arr": K_base_arr,
        "K_subgrade_arr": K_subgrade_arr,
        "F_ref": F_ref,
        "u_ref": u_ref,
        "K_ref": K_ref,
        "E_ref": E_ref,
    }


def _to_torch_tensors(arr: Dict[str, Any]):
    from utils import common_utils as cu

    device = cu.device
    sensor_ur_torch = np_to_torch(arr["sensor_ur"])
    sensor_uz_torch = np_to_torch(arr["sensor_uz"])
    v_torch = np_to_torch(arr["v"])
    f_torch = np_to_torch(arr["f"])
    F_torch = np_to_torch(arr["F"])
    K_surface_unit_torch = np_to_torch(arr["K_surface_arr"])
    K_base_unit_torch = np_to_torch(arr["K_base_arr"])
    K_subgrade_unit_torch = np_to_torch(arr["K_subgrade_arr"])
    dirichlet_torch = torch.tensor(
        np.asarray(arr["dirichlet"]), dtype=torch.long, device=device
    )
    free_torch = torch.tensor(np.asarray(arr["free"]), dtype=torch.long, device=device)
    sensor_node_idx_torch = torch.tensor(
        np.asarray(arr["sensor_node_idx"]), dtype=torch.long, device=device
    )
    return {
        "sensor_ur_torch": sensor_ur_torch,
        "sensor_uz_torch": sensor_uz_torch,
        "v_torch": v_torch,
        "f_torch": f_torch,
        "F_torch": F_torch,
        "K_surface_unit_torch": K_surface_unit_torch,
        "K_base_unit_torch": K_base_unit_torch,
        "K_subgrade_unit_torch": K_subgrade_unit_torch,
        "dirichlet_torch": dirichlet_torch,
        "free_torch": free_torch,
        "sensor_node_idx_torch": sensor_node_idx_torch,
    }


def train_inverse_pinn(
    cfg: Dict[str, Any],
    head_path: Optional[str] = None,
    report_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run inverse PINN optimization.

    Args:
        cfg: Configuration dict (e.g., configs/xpinn_noise_0.json).
        head_path: Project root (directory containing Dataset/). Defaults to this file's parent.
        report_fn: If set, called as report_fn(metrics) with scalars for Ray Tune / logging.

    Returns:
        Dict with final history.data-like summary keys.
    """
    if head_path is None:
        head_path = os.path.dirname(os.path.abspath(__file__))

    noise_std = int(cfg["noise_std"])
    lambda_phys = float(cfg["lambda_phys"])
    lambda_data = float(cfg["lambda_data"])
    hidden = int(cfg["hidden"])
    num_layers = int(cfg["num_layers"])
    print_interval = int(cfg.get("print_interval", 1000))
    plot_interval = int(cfg.get("plot_interval", 1000))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 1000))
    report_interval = int(cfg.get("report_interval", print_interval))
    max_iter = int(cfg["max_iter"])
    epochs = int(cfg["epochs"])
    resume = bool(cfg.get("resume", True))
    seed = int(cfg.get("seed", 1))
    lbfgs_lr = float(cfg.get("lbfgs_lr", 1.0))

    save_path = _abs_path(head_path, cfg["save_dir"])
    os.makedirs(save_path, exist_ok=True)

    tb_writer: Optional[SummaryWriter] = None
    if bool(cfg.get("tensorboard", True)):
        tb_writer = SummaryWriter(log_dir=os.path.join(save_path, "tb"))

    def _log_tensorboard(metrics: Dict[str, Any]) -> None:
        if tb_writer is None:
            return
        step = int(metrics.get("training_iteration", 0))
        for k, v in metrics.items():
            if k in ("training_iteration", "done") or not isinstance(v, (int, float)):
                continue
            tb_writer.add_scalar(k, float(v), step)

    with open(os.path.join(save_path, "config.json"), "w") as fp:
        json.dump(cfg, fp, indent=2)

    set_seed(seed)

    arr = load_inverse_pinn_arrays(head_path, noise_std)
    E_ref = arr["E_ref"]
    u_ref = arr["u_ref"]
    v = arr["v"]
    f = arr["f"]
    nnode = arr["nnode"]
    nel = arr["nel"]

    t = _to_torch_tensors(arr)
    sensor_ur_torch = t["sensor_ur_torch"]
    sensor_uz_torch = t["sensor_uz_torch"]
    v_torch = t["v_torch"]
    f_torch = t["f_torch"]
    F_torch = t["F_torch"]
    K_surface_unit_torch = t["K_surface_unit_torch"]
    K_base_unit_torch = t["K_base_unit_torch"]
    K_subgrade_unit_torch = t["K_subgrade_unit_torch"]
    dirichlet_torch = t["dirichlet_torch"]
    free_torch = t["free_torch"]
    sensor_node_idx_torch = t["sensor_node_idx_torch"]

    true_E = cfg.get(
        "true_moduli",
        {"E_surface": 1378e6, "E_base": 206e6, "E_subgrade": 69e6},
    )

    history = HistoryTracker(["E1", "E2", "E3", "loss_data", "loss_physics"])
    E_list = nn.Parameter(np_to_torch(np.array([1e-3, 1e-3, 1e-3])))
    NN_surface = ANN(input_dim=2, output_dim=2, hidden=hidden, num_layers=num_layers)
    NN_base = ANN(input_dim=2, output_dim=2, hidden=hidden, num_layers=num_layers)
    NN_subgrade = ANN(input_dim=2, output_dim=2, hidden=hidden, num_layers=num_layers)

    from utils import common_utils as cu

    device = cu.device
    NN_surface = NN_surface.to(device)
    NN_base = NN_base.to(device)
    NN_subgrade = NN_subgrade.to(device)

    if E_list.dtype == torch.float64:
        NN_surface.double()
        NN_base.double()
        NN_subgrade.double()

    optimizer = torch.optim.LBFGS(
        list(NN_surface.parameters())
        + list(NN_base.parameters())
        + list(NN_subgrade.parameters())
        + [E_list],
        lr=lbfgs_lr,
        max_iter=max_iter,
        max_eval=max_iter,
        tolerance_grad=1e-20,
        tolerance_change=1e-20,
        history_size=int(cfg.get("lbfgs_history_size", 200)),
        line_search_fn="strong_wolfe",
    )

    iters = 0
    start_time = time.time()
    checkpoint_path = os.path.join(save_path, "checkpoint_inverse_pinn.pt")

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        NN_surface.load_state_dict(checkpoint["NN_surface"])
        NN_base.load_state_dict(checkpoint["NN_base"])
        NN_subgrade.load_state_dict(checkpoint["NN_subgrade"])
        E_list.data = checkpoint["E_list"].data
        history.data = checkpoint["history"]
        iters = checkpoint["iters"]
        start_time = time.time() - checkpoint.get("elapsed_time", 0)
        print(f"Resumed from iteration {iters}")
    else:
        print("Starting fresh training.")

    def save_checkpoint_func():
        torch.save(
            {
                "iters": iters,
                "NN_surface": NN_surface.state_dict(),
                "NN_base": NN_base.state_dict(),
                "NN_subgrade": NN_subgrade.state_dict(),
                "E_list": E_list,
                "optimizer": optimizer.state_dict(),
                "history": history.data,
                "elapsed_time": time.time() - start_time,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")
        history.save(
            os.path.join(save_path, f"inverse_PINN_history_noise_{noise_std}.npz"),
            Elapsed_time=time.time() - start_time,
        )

    def closure():
        nonlocal iters
        optimizer.zero_grad()

        E_surface = E_list[0]
        E_base = E_list[1]
        E_subgrade = E_list[2]

        loss_physics, loss_data, ur_pred_torch, uz_pred_torch, res_x, res_y = evaluate_loss_PINN(
            E_surface,
            E_base,
            E_subgrade,
            K_surface_unit_torch,
            K_base_unit_torch,
            K_subgrade_unit_torch,
            F_torch,
            free_torch,
            dirichlet_torch,
            nnode,
            nel,
            sensor_ur_torch,
            sensor_uz_torch,
            sensor_node_idx_torch,
            NN_surface,
            NN_base,
            NN_subgrade,
            v_torch,
            f_torch,
        )
        total_loss = loss_physics * lambda_phys + loss_data * lambda_data
        total_loss.backward(retain_graph=False)

        E1 = E_surface.item() * E_ref
        E2 = E_base.item() * E_ref
        E3 = E_subgrade.item() * E_ref
        ld = loss_data.item()
        lp = loss_physics.item()

        history.record(
            loss_data=ld,
            loss_physics=lp,
            E1=E1,
            E2=E2,
            E3=E3,
        )

        iters += 1

        if iters % print_interval == 0:
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(
                f"L-BFGS : Iter {iters} | time={elapsed_time_str} | Loss={total_loss.item():.4e}"
            )

        if report_interval > 0 and (iters % report_interval == 0):
            m = {
                "training_iteration": iters,
                "loss_total": float(total_loss.item()),
                "loss_data": ld,
                "loss_physics": lp,
                "E1": E1,
                "E2": E2,
                "E3": E3,
            }
            _log_tensorboard(m)
            if report_fn is not None:
                report_fn(m)

        if iters % plot_interval == 0:
            ur_np = torch_to_np(ur_pred_torch) * u_ref
            uz_np = torch_to_np(uz_pred_torch) * u_ref
            plot_loss_history(
                {k: history.data[k] for k in ["loss_data", "loss_physics"]},
                save_path=os.path.join(save_path, "loss_history.png"),
            )
            plot_displacement(
                v,
                f,
                ur_np,
                uz_np,
                save_path=os.path.join(save_path, f"displacement_iter_{iters}.png"),
            )
            plot_param_history(
                history.data,
                E_surface_true=true_E["E_surface"],
                E_base_true=true_E["E_base"],
                E_subgrade_true=true_E["E_subgrade"],
                save_path=os.path.join(save_path, "param_history.png"),
            )
            plot_displacement(
                v,
                f,
                torch_to_np(res_x),
                torch_to_np(res_y),
                save_path=os.path.join(save_path, f"residual_iter_{iters}.png"),
            )

        if iters % checkpoint_interval == 0:
            save_checkpoint_func()

        return total_loss

    for epoch in range(iters // max_iter, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        optimizer.step(closure)

    history.save(
        os.path.join(save_path, f"inverse_PINN_history_noise_{noise_std}.npz"),
        Elapsed_time=time.time() - start_time,
    )
    torch.save(
        {
            "NN_surface": NN_surface.state_dict(),
            "NN_base": NN_base.state_dict(),
            "NN_subgrade": NN_subgrade.state_dict(),
            "E_list": E_list.detach().cpu(),
        },
        os.path.join(save_path, f"inverse_PINN_models_noise_{noise_std}.pt"),
    )
    plot_loss_history(
        {k: history.data[k] for k in ["loss_data", "loss_physics"]},
        save_path=os.path.join(save_path, "final_loss_history.png"),
    )
    plot_param_history(
        history.data,
        E_surface_true=true_E["E_surface"],
        E_base_true=true_E["E_base"],
        E_subgrade_true=true_E["E_subgrade"],
        save_path=os.path.join(save_path, "final_param_history.png"),
    )

    final_metrics = {
        "training_iteration": iters,
        "loss_total": float(
            history.data["loss_data"][-1] * lambda_data
            + history.data["loss_physics"][-1] * lambda_phys
        ),
        "loss_data": history.data["loss_data"][-1],
        "loss_physics": history.data["loss_physics"][-1],
        "E1": history.data["E1"][-1],
        "E2": history.data["E2"][-1],
        "E3": history.data["E3"][-1],
        "done": 1.0,
    }
    _log_tensorboard(final_metrics)
    if report_fn is not None:
        report_fn(final_metrics)

    if tb_writer is not None:
        tb_writer.close()

    print("Training completed.")
    return {
        "history": history.data,
        "save_path": save_path,
        "final_iters": iters,
    }
