"""
Inverse XPINN (L-BFGS). Hyperparameters are read from a JSON config file.

Default: configs/xpinn_noise_0.json
Override path: environment variable INVERSE_PINN_CONFIG or --config /path/to/config.json
"""
import argparse
import json
import os

head_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(head_path)

default_cfg = os.path.join(head_path, "configs/xpinn_noise_0.json")
parser = argparse.ArgumentParser(description="Inverse PINN (config JSON only for hyperparameters)")
parser.add_argument(
    "--config",
    type=str,
    default=os.environ.get("INVERSE_PINN_CONFIG", default_cfg),
    help="Path to training config JSON (default: configs/xpinn_noise_0.json or $INVERSE_PINN_CONFIG)",
)
args = parser.parse_args()

config_path = args.config if os.path.isabs(args.config) else os.path.join(head_path, args.config)
with open(config_path) as f:
    CONFIG = json.load(f)

from inverse_pinn_train import train_inverse_pinn

train_inverse_pinn(CONFIG, head_path=head_path)
