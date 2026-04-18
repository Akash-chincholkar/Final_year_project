# module1_fog/config_loader.py
"""
Central config loader for Module 1.
Reads .env for machine-specific paths, config.yaml for hyperparams.
Import this at the top of every script instead of hardcoding paths.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root (works regardless of which subfolder you run from)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Inject paths from .env
    fog_dir  = os.getenv("FOG_DATASET_DIR")
    save_dir = os.getenv("SAVE_DIR")

    if not fog_dir:
        raise EnvironmentError(
            "\nFOG_DATASET_DIR not set.\n"
            "Copy .env.example to .env and fill in your local path.\n"
        )
    if not save_dir:
        raise EnvironmentError(
            "\nSAVE_DIR not set.\n"
            "Copy .env.example to .env and fill in your local path.\n"
        )

    cfg["paths"]["dataset_dir"] = fog_dir
    cfg["paths"]["save_dir"]    = save_dir

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded successfully!")
    print(f"  Dataset : {cfg['paths']['dataset_dir']}")
    print(f"  Save to : {cfg['paths']['save_dir']}")
    print(f"  Epochs  : {cfg['training']['epochs']}")
    print(f"  LR      : {cfg['training']['lr']}")