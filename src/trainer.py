"""
Training and validation loops for the CV quantum classifier.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

import os
import time
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from loguru import logger
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.circuit import QuantumClassifier


class QuantumTrainer:
    """
    Trains a QuantumClassifier using PennyLane's AdamOptimizer.

    Args:
        model:     QuantumClassifier with current_weights already initialized
        loss_fn:   callable (weights, inputs, quantum_circuit, labels) -> scalar
        config:    OmegaConf config (reads config.training and config.paths)
        wandb_run: active wandb run or None
    """

    def __init__(
        self,
        model: QuantumClassifier,
        loss_fn: Callable,
        config,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.circuit = model.fetch_circuit()
        self.loss_fn = loss_fn
        self.wandb = wandb_run

        tcfg = config.training
        self.epochs = int(tcfg.epochs)
        self.checkpoint_every = int(tcfg.checkpoint_every)
        self.model_dir = config.paths.model_dir

        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = qml.AdamOptimizer(stepsize=float(tcfg.learning_rate))
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_auc": []
        }

    def _train_step(self, data: pnp.tensor, labels: pnp.tensor) -> float:
        self.model.current_weights, cost = self.optimizer.step_and_cost(
            self.loss_fn,
            self.model.current_weights,
            inputs=data,
            quantum_circuit=self.circuit,
            labels=labels,
        )
        return float(cost)

    def _val_step(self, data: pnp.tensor, labels: pnp.tensor):
        w = pnp.array(self.model.current_weights, requires_grad=False)
        loss = self.loss_fn(w, data, self.circuit, labels)
        score = float(self.circuit(w, data))
        return float(loss), score

    def _run_train_epoch(self, loader) -> float:
        losses = []
        for data, labels in tqdm(loader, desc="  train", leave=False):
            loss = self._train_step(data, labels)
            losses.append(loss)
            if self.wandb:
                self.wandb.log({"train_loss_step": loss})
        return float(np.mean(losses))

    def _run_val(self, loader):
        val_losses, val_scores, val_labels = [], [], []
        for data, labels in tqdm(loader, desc="  val  ", leave=False):
            loss, score = self._val_step(data, labels)
            val_losses.append(loss)
            val_scores.append(score)
            val_labels.append(float(labels))
        return (
            float(np.mean(val_losses)),
            np.array(val_scores, dtype=np.float64),
            np.array(val_labels, dtype=np.float64),
        )

    def _plot_roc_to_wandb(
        self, scores: np.ndarray, labels: np.ndarray, epoch: int
    ) -> None:
        fpr, tpr, _ = roc_curve(labels, scores)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"Validation ROC - Epoch {epoch}")
        # same key each epoch so wandb overwrites rather than accumulating panels
        self.wandb.log({"val_roc": self.wandb.Image(fig)})
        plt.close(fig)

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.checkpoint_dir, f"ep{epoch:03d}.npy")
        np.save(path, np.array(self.model.current_weights))

    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        logger.info(
            f"Starting training | epochs={self.epochs} | n_params={self.model.n_params}"
        )

        for epoch in tqdm(range(1, self.epochs + 1), desc="epoch"):
            t0 = time.time()

            train_loss = self._run_train_epoch(train_loader)
            val_loss, val_scores, val_labels = self._run_val(val_loader)
            val_auc = roc_auc_score(val_labels, val_scores)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            logger.info(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"{elapsed:.1f}s"
            )

            if self.wandb:
                self.wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                })
                self._plot_roc_to_wandb(val_scores, val_labels, epoch)

            if epoch % self.checkpoint_every == 0:
                self._save_checkpoint(epoch)
                logger.info(f"Checkpoint saved at epoch {epoch}")

        final_path = os.path.join(self.model_dir, "final_weights.npy")
        self.model.save_weights(final_path)
        logger.info(f"Training complete. Final weights -> {final_path}")
        return self.history
