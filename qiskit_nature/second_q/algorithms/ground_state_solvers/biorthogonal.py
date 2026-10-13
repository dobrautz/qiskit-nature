# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Biorthogonal expectation values of non-Hermitian operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import CRZGate, PhaseGate, RZGate
from qiskit.primitives import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_nature import QiskitNatureError

_BASIS_GATES = ["rz", "sx", "x", "cx"]


def transition_circuit(
    ansatz: QuantumCircuit,
    left_point: Sequence[float],
    right_point: Sequence[float],
) -> QuantumCircuit:
    r"""Returns a circuit which prepares the superposition of a left and a right state.

    For an ansatz :math:`U(\theta)`, the circuit prepares the state

    .. math::

        \frac{1}{\sqrt{2}}\left(|0\rangle \otimes U(\theta_L)|0\rangle
        + |1\rangle \otimes U(\theta_R)|0\rangle\right)

    with an ancilla qubit, which is the last qubit of the circuit. The ansatz is decomposed into
    ``rz``, ``sx``, ``x`` and ``cx`` gates, such that all parameters enter as angles of ``rz``
    gates, and every parameterized gate :math:`R_Z(\phi(\theta_L))` is followed by an
    ancilla-controlled :math:`R_Z(\phi(\theta_R) - \phi(\theta_L))`. Hence, compared to the
    ansatz, the circuit contains only one controlled rotation (two CNOT gates) per
    parameterized rotation whose angles differ, and none for the parameters with equal values.

    The expectation values of :math:`X \otimes P` and :math:`Y \otimes P` of this state, where
    :math:`X` and :math:`Y` act on the ancilla, are the real and imaginary parts of the transition
    matrix element :math:`\langle L|P|R\rangle` of a Pauli operator :math:`P` (see
    :func:`biorthogonal_expectation_value`).

    Args:
        ansatz: The parameterized circuit.
        left_point: The parameter values :math:`\theta_L` of the left state.
        right_point: The parameter values :math:`\theta_R` of the right state.

    Raises:
        QiskitNatureError: If the numbers of parameters do not match or if the ansatz cannot be
            decomposed into the gates named above.

    Returns:
        The circuit of ``ansatz.num_qubits + 1`` qubits.
    """
    left_values = np.asarray(left_point, dtype=float)
    right_values = np.asarray(right_point, dtype=float)
    if len(left_values) != ansatz.num_parameters or len(right_values) != ansatz.num_parameters:
        raise QiskitNatureError(
            f"The ansatz has {ansatz.num_parameters} parameters, but {len(left_values)} and "
            f"{len(right_values)} values were given."
        )
    left = dict(zip(ansatz.parameters, left_values))
    right = dict(zip(ansatz.parameters, right_values))
    decomposed = transpile(ansatz, basis_gates=_BASIS_GATES, optimization_level=0)

    num_qubits = ansatz.num_qubits
    ancilla = num_qubits
    circuit = QuantumCircuit(num_qubits + 1, name="transition")
    circuit.h(ancilla)
    for instruction in decomposed.data:
        operation = instruction.operation
        qubits = [decomposed.find_bit(qubit).index for qubit in instruction.qubits]
        parameterized = any(
            isinstance(param, ParameterExpression) and param.parameters
            for param in operation.params
        )
        if not parameterized:
            circuit.append(operation, qubits)
            continue
        if operation.name != "rz":
            raise QiskitNatureError(
                f"The parameterized gate '{operation.name}' could not be decomposed into rz gates."
            )
        angle_left = _bind(operation.params[0], left)
        angle_right = _bind(operation.params[0], right)
        circuit.append(RZGate(angle_left), qubits)
        if not np.isclose(angle_left, angle_right, rtol=0.0, atol=1e-14):
            circuit.append(CRZGate(angle_right - angle_left), [ancilla] + qubits)

    # a parameter-dependent global phase of the ansatz is a relative phase of the two branches
    phase_left = _bind(decomposed.global_phase, left)
    phase_right = _bind(decomposed.global_phase, right)
    circuit.global_phase = phase_left
    if not np.isclose(phase_left, phase_right, rtol=0.0, atol=1e-14):
        circuit.append(PhaseGate(phase_right - phase_left), [ancilla])
    return circuit


def biorthogonal_expectation_value(
    operator: SparsePauliOp,
    ansatz: QuantumCircuit,
    left_point: Sequence[float],
    right_point: Sequence[float],
    estimator: BaseEstimatorV2 | None = None,
    *,
    transpiler: Any | None = None,
    transpiler_options: dict[str, Any] | None = None,
    precision: float | None = None,
) -> complex:
    r"""Computes the biorthogonal expectation value of an operator.

    For a non-Hermitian Hamiltonian :math:`H`, such as a transcorrelated one, the right and left
    eigenvectors :math:`|R\rangle` and :math:`|L\rangle` of an eigenvalue :math:`E` differ. The
    expectation value :math:`\langle R|H|R\rangle` of an approximate right eigenvector, as
    computed by the :class:`.VarQITEMinimumEigensolver`, is not variational, and its error is of
    first order in the error of the state. In contrast, the error of the biorthogonal expectation
    value

    .. math::

        E_{LR} = \frac{\langle L|H|R\rangle}{\langle L|R\rangle}

    is of second order: :math:`E_{LR} - E = \langle\delta L|H - E|\delta R\rangle /
    \langle L|R\rangle` for :math:`|L\rangle = |L_0\rangle + |\delta L\rangle` and
    :math:`|R\rangle = |R_0\rangle + |\delta R\rangle`. An approximate left eigenvector is
    obtained by a VarQITE of the adjoint operator :math:`H^\dagger`, whose right eigenvectors are
    the left eigenvectors of :math:`H`.

    Both states are given by the same ansatz :math:`U(\theta)` with different parameter values,
    :math:`|L\rangle = U(\theta_L)|0\rangle` and :math:`|R\rangle = U(\theta_R)|0\rangle`. If an
    estimator is given, the transition matrix elements are measured with the
    :func:`transition_circuit`, using

    .. math::

        \langle L|P|R\rangle = \langle X \otimes P\rangle + i\langle Y \otimes P\rangle

    for the Pauli operators :math:`P` of the decomposition :math:`H = A + iB` into Hermitian
    operators :math:`A` and :math:`B`, and for :math:`P = I`. These are four observables,
    :math:`X \otimes A - Y \otimes B`, :math:`Y \otimes A + X \otimes B`, :math:`X \otimes I` and
    :math:`Y \otimes I`, which the estimator can measure jointly in the qubit-wise commuting
    groups of the Pauli terms of :math:`H` with the ancilla in the :math:`X` and :math:`Y` basis,
    i.e. with twice the number of measurement bases of :math:`\langle H\rangle`. This measurement
    is required only once, after both evolutions.

    Args:
        operator: The (non-Hermitian) qubit operator.
        ansatz: The parameterized circuit of both states.
        left_point: The parameter values of the left state.
        right_point: The parameter values of the right state.
        estimator: The estimator. If ``None``, the expectation value is computed exactly from the
            statevectors.
        transpiler: An optional object with a ``run`` method (e.g. a pass manager), which compiles
            the transition circuit for the backend of the estimator. The layout of the compiled
            circuit is applied to the observables.
        transpiler_options: Keyword arguments passed to the ``run`` method of ``transpiler``.
        precision: The precision passed to the estimator.

    Raises:
        QiskitNatureError: If the overlap of the left and right states vanishes.

    Returns:
        The biorthogonal expectation value :math:`\langle L|H|R\rangle / \langle L|R\rangle`.
    """
    operator = SparsePauliOp(operator).simplify(atol=0.0)
    if estimator is None:
        left_state = Statevector(ansatz.assign_parameters(left_point)).data
        right_state = Statevector(ansatz.assign_parameters(right_point)).data
        numerator = np.vdot(left_state, operator.to_matrix(sparse=True) @ right_state)
        denominator = np.vdot(left_state, right_state)
    else:
        circuit = transition_circuit(ansatz, left_point, right_point)
        if transpiler is not None:
            circuit = transpiler.run(circuit, **(transpiler_options or {}))
        hermitian = SparsePauliOp(operator.paulis, np.real(operator.coeffs))
        anti_hermitian = SparsePauliOp(operator.paulis, np.imag(operator.coeffs))
        identity = SparsePauliOp("I" * ansatz.num_qubits)
        x_part, y_part = SparsePauliOp("X"), SparsePauliOp("Y")
        observables = [
            (x_part ^ hermitian) - (y_part ^ anti_hermitian),
            (y_part ^ hermitian) + (x_part ^ anti_hermitian),
            x_part ^ identity,
            y_part ^ identity,
        ]
        observables = [observable.simplify() for observable in observables]
        if circuit.layout is not None:
            observables = [observable.apply_layout(circuit.layout) for observable in observables]
        result = estimator.run([(circuit, observables)], precision=precision).result()
        values = np.asarray(result[0].data.evs, dtype=float)
        numerator = values[0] + 1j * values[1]
        denominator = values[2] + 1j * values[3]
    if abs(denominator) < 1e-12:
        raise QiskitNatureError("The left and right states are orthogonal.")
    return complex(numerator / denominator)


def _bind(value: ParameterExpression | float, values: dict) -> float:
    """Binds the parameters of an angle and returns its (real) value."""
    if isinstance(value, ParameterExpression):
        value = value.bind({param: values[param] for param in value.parameters})
        return float(np.real(complex(value)))
    return float(value)
