"""
Math primitives used by loss functions.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

import pennylane.numpy as pnp

_EPS = 1e-7  # numerical stability for log


def sigmoid(x: pnp.tensor) -> pnp.tensor:
    return 1.0 / (1.0 + pnp.exp(-x))


def binary_cross_entropy(labels: pnp.tensor, scores: pnp.tensor) -> pnp.tensor:
    """BCE from raw scores (applies sigmoid internally)."""
    probs = sigmoid(scores)
    return -(labels * pnp.log(probs + _EPS) + (1.0 - labels) * pnp.log(1.0 - probs + _EPS))


def mean_squared_error(labels: pnp.tensor, scores: pnp.tensor) -> pnp.tensor:
    return (scores - labels) ** 2
