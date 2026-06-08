"""
Config loading, path expansion, and directory creation for QUEST runs.
Author: Aritra Bal (ETP)
Date: 2026-06-08
"""

from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


def load_config(config_path: str = "configs/base.yaml"):
    """
    Load YAML config, expand paths with run ID, create directories.

    Args:
        config_path: path to the YAML config file

    Returns:
        OmegaConf config with all paths resolved and directories created
    """
    cfg = OmegaConf.load(config_path)
    run_id: str = cfg.general.id

    # append run ID to base dirs
    model_dir = Path(cfg.paths.model_dir) / run_id
    inf_dir   = Path(cfg.paths.inference_results_dir) / run_id
    log_dir   = Path(cfg.paths.log_dir) / run_id

    # model_dir subdirs
    checkpoint_dir = model_dir / "checkpoints"
    plot_dir       = model_dir / "plots"

    for d in [model_dir, inf_dir, log_dir, checkpoint_dir, plot_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ready: {d}")

    # write expanded paths back
    OmegaConf.update(cfg, "paths.model_dir",              str(model_dir))
    OmegaConf.update(cfg, "paths.inference_results_dir",  str(inf_dir))
    OmegaConf.update(cfg, "paths.log_dir",                str(log_dir))
    OmegaConf.update(cfg, "paths.checkpoint_dir",         str(checkpoint_dir))
    OmegaConf.update(cfg, "paths.plot_dir",               str(plot_dir))
    OmegaConf.update(cfg, "paths.run_dir",                str(model_dir))

    return cfg
