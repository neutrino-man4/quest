"""
Loss functions for CV quantum circuit training.
Each function runs the forward pass internally and returns a scalar loss.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

import pennylane.numpy as pnp
from src import mfunc


def bce_loss(
    weights: pnp.tensor,
    inputs: pnp.tensor,
    quantum_circuit,
    labels: pnp.tensor,
) -> pnp.tensor:
    """BCE loss; sigmoid is applied inside mfunc.binary_cross_entropy."""
    scores = pnp.array(quantum_circuit(weights, inputs), requires_grad=True)
    return mfunc.binary_cross_entropy(labels, scores)


def mse_loss(
    weights: pnp.tensor,
    inputs: pnp.tensor,
    quantum_circuit,
    labels: pnp.tensor,
) -> pnp.tensor:
    scores = pnp.array(quantum_circuit(weights, inputs), requires_grad=True)
    return mfunc.mean_squared_error(labels, scores)


# map config string -> loss function
LOSS_REGISTRY = {
    "bce": bce_loss,
    "mse": mse_loss,
}


def get_loss_fn(name: str):
    """Return loss function by config name ('bce' or 'mse')."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]
