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


class QuantumClassifier:
    """
    CV quantum circuit for binary jet classification.

    Weight shape: (layers, qumodes, params_per_state)
    Total trainable parameters: layers * qumodes * params_per_state
    """

    def __init__(
        self,
        qumodes: int,
        layers: int,
        params_per_state: int,
        device_name: str = "default.gaussian",
        shots: Optional[int] = None,
        backend: str = "autograd",
        diff_method: str = "parameter_shift",
        measurement: str = "homodyne",
        measurement_mode: int = 0,
        input_scale: float = 1.0,
    ) -> None:
        self.qumodes = qumodes
        self.layers = layers
        self.params_per_state = params_per_state
        self.input_scale = input_scale
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
        # weights shape: (layers, qumodes, params_per_state)
        ops.state_preparation(inputs, self._wires, self.input_scale)
        for L in range(self.layers):
            ops.entanglement_layer(self._wire_pairs)
            ops.variational_layer(weights[L], self._wires)

        if self.measurement == "homodyne":
            return qml.expval(qml.QuadX(self.measurement_mode))
        return qml.expval(qml.NumberOperator(self.measurement_mode))

    @property
    def n_params(self) -> int:
        return self.layers * self.qumodes * self.params_per_state

    def init_weights(self, seed: Optional[int] = None) -> pnp.tensor:
        rng = np.random.default_rng(seed)
        shape = (self.layers, self.qumodes, self.params_per_state)
        self.current_weights = pnp.array(
            rng.uniform(-np.pi, np.pi, shape).astype(np.float64),
            requires_grad=True,
        )
        return self.current_weights

    def fetch_circuit(self) -> qml.QNode:
        return self._qnode

    def load_weights(self, path: str) -> None:
        self.current_weights = pnp.array(np.load(path), requires_grad=True)

    def save_weights(self, path: str) -> None:
        np.save(path, np.array(self.current_weights))
