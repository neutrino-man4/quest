"""
QuantumClassifier: CV QNode for binary jet classification on default.gaussian.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

from itertools import combinations
from typing import Optional

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from src import operations as ops
from src.mfunc import sigmoid


class QuantumClassifier:
    """
    CV quantum circuit for binary jet classification.

    Weight layout: [3 * layers * qumodes variational params | 1 scale factor]
    Total trainable parameters: 3 * layers * qumodes + 1
    """

    def __init__(
        self,
        qumodes: int,
        layers: int,
        device_name: str = "default.gaussian",
        shots: Optional[int] = None,
        backend: str = "autograd",
        diff_method: str = "parameter_shift",
        measurement: str = "homodyne",
        measurement_mode: int = 0,
    ) -> None:
        self.qumodes = qumodes
        self.layers = layers
        self.backend = backend
        self.measurement = measurement
        self.measurement_mode = measurement_mode
        self.current_weights: Optional[pnp.tensor] = None

        self._wires = list(range(qumodes))
        self._wire_pairs = list(combinations(self._wires, 2))

        self.device = qml.device(device_name, wires=qumodes, shots=shots)
        self._qnode = qml.QNode(
            self._circuit,
            self.device,
            interface=backend,
            diff_method=diff_method,
        )

    def _circuit(self, weights: pnp.tensor, inputs: pnp.tensor):
        # learnable input scale factor, bounded to (0.01, 10.01)
        sf = 10.0 * sigmoid(weights[-1]) + 0.01
        ops.state_preparation(inputs, self._wires, sf)
        for L in range(self.layers):
            ops.entanglement_layer(self._wire_pairs)
            ops.variational_layer(weights, self._wires, L)

        if self.measurement == "homodyne":
            return qml.expval(qml.QuadX(self.measurement_mode))
        return qml.expval(qml.NumberOperator(self.measurement_mode))

    @property
    def n_params(self) -> int:
        """3 params per mode per layer, plus 1 global scale factor."""
        return 3 * self.layers * self.qumodes + 1

    def init_weights(self, seed: Optional[int] = None) -> pnp.tensor:
        rng = np.random.default_rng(seed)
        self.current_weights = pnp.array(
            rng.uniform(-np.pi, np.pi, self.n_params).astype(np.float64),
            requires_grad=True,
        )
        return self.current_weights

    def fetch_circuit(self) -> qml.QNode:
        return self._qnode

    def load_weights(self, path: str) -> None:
        self.current_weights = pnp.array(np.load(path), requires_grad=True)

    def save_weights(self, path: str) -> None:
        np.save(path, np.array(self.current_weights))
