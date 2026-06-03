"""
CV gate operations: encoding, entanglement, variational layers.
Author: Aritra Bal (ETP)
Date: 2026-06-03
"""

import pennylane as qml
import pennylane.numpy as pnp

# jetConstituentsList column order from h5_maker_slim: [deta, dphi, pt]
_DETA, _DPHI, _PT = 0, 1, 2


def state_preparation(inputs: pnp.tensor, wires: list, scale_factor: float) -> None:
    """Encode jet constituents via displacement + squeezing per mode."""
    for w in wires:
        deta = float(inputs[w, _DETA])
        dphi = float(inputs[w, _DPHI])
        pt   = float(inputs[w, _PT])
        qml.Displacement(scale_factor * pt, deta, wires=w)
        qml.Squeezing(deta, pt * dphi / 2.0, wires=w)


def entanglement_layer(wire_pairs: list) -> None:
    """ControlledAddition on all qumode pairs."""
    for pair in wire_pairs:
        qml.ControlledAddition(1.0, wires=pair)


def variational_layer(weights: pnp.tensor, wires: list, layer_idx: int) -> None:
    """Parameterized displacement + squeezing per mode for one layer."""
    N = len(wires)
    start = 3 * layer_idx * N
    for i, w in enumerate(wires):
        phi   = weights[start + i]
        theta = weights[start + N + i]
        omega = weights[start + 2 * N + i]
        qml.Displacement(theta, phi, wires=w)
        qml.Squeezing(omega, pnp.pi / 4.0, wires=w)
