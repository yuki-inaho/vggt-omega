"""Hydra entry point for supervised VGGT-Omega RGB-D fine-tuning."""

from __future__ import annotations

import json

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from vggt_omega.training.config import validate_training_config
from vggt_omega.training.runner import run_training


@hydra.main(version_base="1.3", config_path="../configs/training", config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate the composed configuration before allocating data or GPU memory."""

    validate_training_config(cfg)
    output_dir = HydraConfig.get().runtime.output_dir
    summary = run_training(cfg, output_dir=output_dir, original_cwd=get_original_cwd())
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
