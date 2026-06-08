"""
QuantumClassifier: CV QNode for binary jet classification on default.gaussian.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

import os
from itertools import combinations
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    def __init__(self, config) -> None:
        ccfg = config.circuit
        self.qumodes          = int(ccfg.qumodes)
        self.layers           = int(ccfg.layers)
        self.params_per_state = int(ccfg.params_per_state)
        self.input_scale      = float(getattr(ccfg, "input_scale", 1.0))
        self.backend          = str(config.training.backend)
        self.measurement      = str(ccfg.measurement)
        self.measurement_mode = int(ccfg.measurement_mode)
        self.current_weights: Optional[pnp.tensor] = None

        self._wires      = list(range(self.qumodes))
        self._wire_pairs = list(combinations(self._wires, 2))

        self.device = qml.device(ccfg.device, wires=self.qumodes, shots=ccfg.shots or None)
        self._qnode = qml.QNode(
            self._circuit,
            self.device,
            interface=self.backend,
            diff_method=str(ccfg.diff_method),
        )

    def _circuit(self, weights: pnp.tensor, inputs: pnp.tensor):
        """Run the CV quantum circuit."""
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
        """Total number of trainable parameters."""
        return self.layers * self.qumodes * self.params_per_state

    def init_weights(self, seed: Optional[int] = None) -> pnp.tensor:
        """Randomly initialise weights in [-pi, pi]."""
        rng = np.random.default_rng(seed)
        shape = (self.layers, self.qumodes, self.params_per_state)
        self.current_weights = pnp.array(
            rng.uniform(-np.pi, np.pi, shape).astype(np.float64),
            requires_grad=True,
        )
        return self.current_weights

    def fetch_circuit(self) -> qml.QNode:
        """Return the compiled QNode."""
        return self._qnode

    def load_weights(self, path: str) -> None:
        """Load weights from a .npy file."""
        self.current_weights = pnp.array(np.load(path), requires_grad=True)

    def save_weights(self, path: str) -> None:
        """Save current weights to a .npy file."""
        np.save(path, np.array(self.current_weights))

    def _circuit_diagram(self, plot_dir: str) -> None:
        """Draw circuit via qml.draw_mpl and save to plot_dir/circuit.png."""
        w = self.current_weights if self.current_weights is not None else pnp.array(
            np.zeros((self.layers, self.qumodes, self.params_per_state)), requires_grad=False
        )
        dummy_inputs = np.zeros((self.qumodes, 3), dtype=np.float64)
        fig, _ = qml.draw_mpl(self._qnode)(w, dummy_inputs)
        fig.savefig(os.path.join(plot_dir, "circuit.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
