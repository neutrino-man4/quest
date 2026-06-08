"""
Master training script for the QUEST CV quantum classifier.
Author: Aritra Bal (ETP)
Date: 2026-06-08
"""

import argparse
import os
import sys

import numpy as np
from loguru import logger

import configs.configs as configs
import data_utils.dataloader as dataloader
import src.circuit as cq
import src.losses as losses
import src.trainer as trn


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str, run_id: str) -> None:
    """Configure loguru: stderr + timestamped file in log_dir."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    log_file = os.path.join(log_dir, f"{run_id}.log")
    logger.add(log_file, level="DEBUG", rotation="50 MB")
    logger.info(f"Logging to {log_file}")


# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------

def _setup_wandb(cfg):
    """
    Initialise wandb if enabled. Warns if API key is missing.
    Returns active wandb run or None.
    """
    if not cfg.wandb.enabled:
        return None

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("wandb enabled but WANDB_API_KEY is not set — skipping wandb init")
        return None

    import wandb  # lazy import; not required if disabled

    run_name = cfg.wandb.run_name or cfg.general.id
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=run_name,
        config=dict(cfg),
    )
    logger.info(f"wandb run initialised: {run.name}")
    return run


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the QUEST CV quantum classifier")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config file (default: configs/base.yaml)",
    )
    args = parser.parse_args()

    # --- config & dirs ---
    cfg = configs.load_config(args.config)
    run_id: str = cfg.general.id

    _setup_logging(cfg.paths.log_dir, run_id)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Config: {args.config}")

    # --- reproducibility ---
    seed: int = int(cfg.general.seed)
    np.random.seed(seed)
    logger.info(f"Global seed: {seed}")

    # --- wandb ---
    wandb_run = _setup_wandb(cfg)

    # --- data ---
    logger.info("Loading train/val dataloaders ...")
    train_loader, val_loader = dataloader.get_dataloader(cfg, scenario="train")
    logger.info(
        f"Train size: {len(train_loader.dataset)} | "
        f"Val size: {len(val_loader.dataset)} | "
        f"Batch size: {cfg.data.batch_size}"
    )

    # --- model ---
    logger.info("Building quantum circuit ...")
    model = cq.QuantumClassifier(cfg)
    weights = model.init_weights(seed=seed)
    logger.info(
        f"Circuit | qumodes={model.qumodes} | layers={model.layers} | "
        f"params_per_state={model.params_per_state} | total_params={model.n_params}"
    )

    # save circuit diagram
    try:
        model._circuit_diagram(cfg.paths.plot_dir)
        logger.info(f"Circuit diagram saved to {cfg.paths.plot_dir}/circuit.png")
    except Exception as e:
        logger.warning(f"Could not save circuit diagram: {e}")

    # --- loss ---
    loss_fn = losses.get_loss_fn(cfg.training.loss)
    logger.info(f"Loss function: {cfg.training.loss}")

    # --- train ---
    trainer = trn.QuantumTrainer(model, loss_fn, cfg, wandb_run=wandb_run)
    history = trainer.train(train_loader, val_loader)

    # --- save history ---
    history_path = os.path.join(cfg.paths.run_dir, "history.npy")
    np.save(history_path, history)
    logger.info(f"History saved to {history_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
