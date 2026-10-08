"""
Ray Tune: exhaustive grid over XPINN hyperparameters (no ASHA / early stopping).

Usage (single machine):
    RAY_AIR_NEW_OUTPUT=0 python sweep_inverse_pinn.py configs/sweep_loss_weight.json
    RAY_AIR_NEW_OUTPUT=0 python sweep_inverse_pinn.py configs/sweep_network.json

Usage (cluster already running):
    RAY_ADDRESS=auto RAY_AIR_NEW_OUTPUT=0 python sweep_inverse_pinn.py configs/sweep_network.json

Config:
    Sweep JSON with experiment name, results_dir, base_config path, and search_space.
    Lists in search_space are combined as a full factorial via tune.grid_search.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import ray
import torch
from ray import tune

from inverse_pinn_train import train_inverse_pinn


def _head_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_json(path: str) -> dict:
    with open(path) as fp:
        return json.load(fp)


def _resolve_path(head_path: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(head_path, p)


def build_param_space(base_cfg: dict, search_space: dict) -> dict:
    """Merge base config with Ray Tune grid_search for each key listed in search_space."""
    merged = copy.deepcopy(base_cfg)
    param_space: dict = {}
    for key in sorted(set(merged.keys()) | set(search_space.keys())):
        if key in search_space:
            ov = search_space[key]
            if isinstance(ov, list):
                param_space[key] = tune.grid_search(ov)
            else:
                param_space[key] = ov
        else:
            param_space[key] = merged[key]
    return param_space


def ray_trainable(config: dict):
    head_path = _head_path()
    cfg = copy.deepcopy(config)
    trial_dir = tune.get_context().get_trial_dir()
    trial_name = os.path.basename(trial_dir)
    experiment_path = cfg.pop("_experiment_path")
    cfg["save_dir"] = os.path.join(experiment_path, trial_name)

    def report_fn(metrics: dict):
        tune.report(metrics)

    train_inverse_pinn(cfg, head_path=head_path, report_fn=report_fn)


def main():
    head_path = _head_path()
    sweep_path = os.environ.get(
        "SWEEP_CONFIG",
        os.path.join(head_path, "configs/sweep_network.json"),
    )
    if len(sys.argv) > 1:
        sweep_path = sys.argv[1]
    if not os.path.isabs(sweep_path):
        sweep_path = os.path.join(head_path, sweep_path)

    sweep_cfg = _load_json(sweep_path)
    base_path = _resolve_path(head_path, sweep_cfg["base_config"])
    base_cfg = _load_json(base_path)

    experiment_name = sweep_cfg["experiment_name"]
    results_dir = _resolve_path(head_path, sweep_cfg["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    experiment_path = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    search_space = sweep_cfg["search_space"]
    param_space = build_param_space(base_cfg, search_space)
    param_space["_experiment_path"] = experiment_path

    ray.init(address=os.environ.get("RAY_ADDRESS") or None)

    default_gpu = "1" if torch.cuda.is_available() else "0"
    ngpu = float(os.environ.get("RAY_NUM_GPUS_PER_TRIAL", default_gpu))
    if ngpu > 0:
        trainable = tune.with_resources(ray_trainable, resources={"gpu": ngpu})
    else:
        ncpu = float(os.environ.get("RAY_CPUS_PER_TRIAL", "4"))
        trainable = tune.with_resources(ray_trainable, resources={"cpu": ncpu})

    tuner_pkl = os.path.join(experiment_path, "tuner.pkl")
    if os.path.exists(tuner_pkl):
        print(f"Resuming experiment from {experiment_path}")
        tuner = tune.Tuner.restore(
            path=experiment_path,
            trainable=trainable,
            resume_errored=True,
            resume_unfinished=True,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric="loss_total",
                mode="min",
            ),
            run_config=ray.tune.RunConfig(
                name=experiment_name,
                storage_path=results_dir,
            ),
        )

    results = tuner.fit()
    best = results.get_best_result(metric="loss_total", mode="min")
    print("\n=== Best trial (by final loss_total) ===")
    print("Config:", best.config)
    print("Metrics:", best.metrics)


if __name__ == "__main__":
    main()
