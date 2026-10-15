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

"""An exact statevector implementation of McLachlan's imaginary-time variational principle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, Parameter, ParameterExpression
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparseObservable, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.synthesis import LieTrotter
from qiskit_algorithms.time_evolvers.variational import ImaginaryVariationalPrinciple

from qiskit_nature import QiskitNatureError

_ROTATIONS = {"rx": "X", "ry": "Y", "rz": "Z", "rxx": "XX", "ryy": "YY", "rzz": "ZZ"}
"""Gates of the form exp(-i theta/2 P) for a Pauli string P (acting on the gate's qubits)."""

_SKIPPED = {"barrier", "delay"}

_MAX_MATRIX_QUBITS = 3
"""Parameter-free instructions acting on more qubits are decomposed instead of being applied as a
matrix."""


@dataclass
class _LinearForm:
    """An angle of the form ``constant + sum_k coeffs[k] * theta_k``."""

    constant: float = 0.0
    coeffs: dict[int, float] = field(default_factory=dict)

    def evaluate(self, values: np.ndarray) -> float:
        """Evaluates the angle for the given parameter values."""
        return self.constant + sum(c * values[k] for k, c in self.coeffs.items())

    def scaled(self, factor: float) -> _LinearForm:
        """Returns the angle multiplied by a constant factor."""
        return _LinearForm(self.constant * factor, {k: c * factor for k, c in self.coeffs.items()})


@dataclass
class _Rotation:
    """The operation ``exp(-i angle P)`` for a Pauli string ``P = factor X^x Z^z``."""

    x_mask: int
    z_mask: int
    factor: complex
    angle: _LinearForm


@dataclass
class _Phase:
    """The operation ``exp(i angle)``."""

    angle: _LinearForm


@dataclass
class _Matrix:
    """A fixed unitary acting on some qubits."""

    qubits: tuple[int, ...]
    matrix: np.ndarray


class StatevectorImaginaryMcLachlanPrinciple(ImaginaryVariationalPrinciple):
    r"""An exact statevector implementation of McLachlan's imaginary-time variational principle.

    This variational principle can be used with :class:`~qiskit_algorithms.VarQITE` (for example
    through the :class:`.VarQITEMinimumEigensolver`) as a fast alternative to
    :class:`~qiskit_algorithms.time_evolvers.variational.ImaginaryMcLachlanPrinciple` for exact
    (noise-free) simulations. Instead of evaluating gradient and quantum geometric tensor circuits
    with estimator primitives, it propagates the state :math:`|\psi\rangle` together with all its
    parameter derivatives :math:`|\partial_i\psi\rangle` through the circuit in a single pass and
    evaluates

    .. math::

        A_{ij} = \mathrm{Re}\left[\langle\partial_i\psi|\partial_j\psi\rangle
        - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right],
        \qquad
        C_i = -\mathrm{Re}\left[\langle\partial_i\psi|H|\psi\rangle
        - \langle\partial_i\psi|\psi\rangle\langle\psi|H|\psi\rangle\right],

    which holds for Hermitian and non-Hermitian (e.g. transcorrelated) Hamiltonians alike. The
    derivatives are exact (no finite differences are used).

    The circuit must consist of Pauli rotations (``rx``, ``ry``, ``rz``, ``rxx``, ``ryy``,
    ``rzz``), phase gates, :class:`~qiskit.circuit.library.PauliEvolutionGate` instances and
    parameter-free gates. Other instructions are decomposed. The gate angles must depend linearly
    on the parameters. A :class:`~qiskit.circuit.library.PauliEvolutionGate` is simulated as the
    product of the exponentials of its terms, which is exact if its terms commute (as for unitary
    coupled cluster ansätze) and otherwise corresponds to its first-order Lie-Trotter synthesis.

    The memory and runtime requirements scale as :math:`(P+1) 2^n` and
    :math:`N_\mathrm{gates} (P+1) 2^n`, respectively, for :math:`n` qubits and :math:`P`
    parameters.
    """

    def __init__(self, phase_fix: bool = True) -> None:
        """
        Args:
            phase_fix: Whether to include the terms accounting for the global phase. This should
                be ``True`` unless the ansatz states are real-valued.
        """
        super().__init__(qgt=None, gradient=None)
        self.phase_fix = phase_fix
        self._compiled: dict[int, tuple[QuantumCircuit, list]] = {}
        self._state_cache: tuple[int, tuple[float, ...], np.ndarray] | None = None
        self._matrix_cache: tuple[BaseOperator, object] | None = None

    @staticmethod
    def supports(circuit: QuantumCircuit) -> bool:
        """Checks whether a circuit can be simulated by this variational principle.

        Args:
            circuit: The parametrized circuit.

        Returns:
            Whether the circuit is supported.
        """
        try:
            _compile(circuit)
        except QiskitNatureError:
            return False
        return True

    def statevector(self, ansatz: QuantumCircuit, param_values: Sequence[float]) -> np.ndarray:
        """Simulates the state prepared by the ansatz.

        Args:
            ansatz: The parametrized circuit.
            param_values: Values of the parameters of ``ansatz`` (in the order of
                ``ansatz.parameters``).

        Returns:
            The statevector (in Qiskit's little-endian qubit ordering).
        """
        values = np.asarray(param_values, dtype=float)
        return _simulate(self._compile(ansatz), ansatz.num_qubits, values, derivatives=False)[0]

    def expectation_value(self, operator: BaseOperator, state: np.ndarray) -> complex:
        """Computes the (complex) expectation value of a (possibly non-Hermitian) operator.

        Args:
            operator: The operator.
            state: The statevector, e.g. obtained from :meth:`statevector`.

        Returns:
            The expectation value.
        """
        return complex(np.vdot(state, self._hamiltonian_matrix(operator) @ state))

    def metric_tensor(self, ansatz: QuantumCircuit, param_values: Sequence[float]) -> np.ndarray:
        """
        Calculates the metric tensor (the real part of the quantum geometric tensor).

        Args:
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of the parameters of ``ansatz`` (in the order of
                ``ansatz.parameters``).

        Returns:
            The metric tensor.
        """
        states = self._states(ansatz, param_values)
        psi, derivatives = states[0], states[1:]
        metric = derivatives.conj() @ derivatives.T
        if self.phase_fix:
            overlaps = derivatives.conj() @ psi
            metric -= np.outer(overlaps, overlaps.conj())
        return np.real(metric)

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence[Parameter] | None = None,
    ) -> np.ndarray:
        """
        Calculates the right-hand side of the linear system of McLachlan's principle.

        Args:
            hamiltonian: The (possibly non-Hermitian) Hamiltonian.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of the parameters of ``ansatz`` (in the order of
                ``ansatz.parameters``).
            gradient_params: The parameters with respect to which the gradient is computed. If
                ``None``, all parameters of ``ansatz`` are used.

        Returns:
            The right-hand side of the linear system.
        """
        states = self._states(ansatz, param_values)
        psi, derivatives = states[0], states[1:]
        h_psi = self._hamiltonian_matrix(hamiltonian) @ psi
        rhs = derivatives.conj() @ h_psi
        if self.phase_fix:
            rhs -= (derivatives.conj() @ psi) * np.vdot(psi, h_psi)
        rhs = -np.real(rhs)
        if gradient_params is not None:
            indices = {param: i for i, param in enumerate(ansatz.parameters)}
            rhs = rhs[[indices[param] for param in gradient_params]]
        return rhs

    def tangent_hamiltonian(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
    ) -> np.ndarray:
        r"""
        Calculates the Hamiltonian in the tangent space, as required by implicit integration
        steps (see the ``implicit`` option of the :class:`.VarQITEMinimumEigensolver`),

        .. math::

            K_{ij} = \mathrm{Re}\langle T_i|H - E|T_j\rangle, \qquad
            |T_i\rangle = |\partial_i\psi\rangle - |\psi\rangle\langle\psi|\partial_i\psi\rangle,
            \qquad E = \langle\psi|H|\psi\rangle,

        where the projection of the tangent vectors is omitted if ``phase_fix`` is ``False``.
        This also holds for non-Hermitian Hamiltonians, for which :math:`E` is complex and
        :math:`K` is not symmetric.

        Args:
            hamiltonian: The (possibly non-Hermitian) Hamiltonian.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of the parameters of ``ansatz`` (in the order of
                ``ansatz.parameters``).

        Returns:
            The matrix :math:`K`.
        """
        states = self._states(ansatz, param_values)
        psi, derivatives = states[0], states[1:]
        if self.phase_fix:
            derivatives = derivatives - np.outer(derivatives @ psi.conj(), psi)
        matrix = self._hamiltonian_matrix(hamiltonian)
        energy = np.vdot(psi, matrix @ psi)
        h_derivatives = (matrix @ derivatives.T).T - energy * derivatives
        return np.real(derivatives.conj() @ h_derivatives.T)

    def _states(self, ansatz: QuantumCircuit, param_values: Sequence[float]) -> np.ndarray:
        """Returns the state and its derivatives (as rows), caching the last evaluation."""
        values = tuple(float(v) for v in param_values)
        if (
            self._state_cache is not None
            and self._state_cache[0] == id(ansatz)
            and self._state_cache[1] == values
        ):
            return self._state_cache[2]

        states = _simulate(self._compile(ansatz), ansatz.num_qubits, np.asarray(values))
        self._state_cache = (id(ansatz), values, states)
        return states

    def _compile(self, ansatz: QuantumCircuit) -> list:
        """Returns the compiled operations of the ansatz, caching the result."""
        compiled = self._compiled.get(id(ansatz))
        if compiled is None or compiled[0] is not ansatz:
            compiled = (ansatz, _compile(ansatz))
            self._compiled[id(ansatz)] = compiled
        return compiled[1]

    def _hamiltonian_matrix(self, hamiltonian: BaseOperator):
        """Returns the (sparse) matrix of the Hamiltonian, caching the last one."""
        if self._matrix_cache is not None and self._matrix_cache[0] is hamiltonian:
            return self._matrix_cache[1]
        if isinstance(hamiltonian, SparseObservable):
            matrix = SparsePauliOp.from_sparse_observable(hamiltonian).to_matrix(sparse=True)
        elif isinstance(hamiltonian, SparsePauliOp):
            matrix = hamiltonian.to_matrix(sparse=True)
        else:
            matrix = Operator(hamiltonian).data
        self._matrix_cache = (hamiltonian, matrix)
        return matrix


def _compile(circuit: QuantumCircuit) -> list[_Rotation | _Phase | _Matrix]:
    """Translates a circuit into a list of rotations, phases and fixed matrices."""
    parameter_indices = {param: i for i, param in enumerate(circuit.parameters)}
    operations: list[_Rotation | _Phase | _Matrix] = []
    _compile_circuit(circuit, list(range(circuit.num_qubits)), parameter_indices, operations)
    return operations


def _compile_circuit(
    circuit: QuantumCircuit,
    qubit_map: list[int],
    parameter_indices: dict[Parameter, int],
    operations: list,
) -> None:
    if circuit.global_phase != 0:
        operations.append(_Phase(_linear_form(circuit.global_phase, parameter_indices)))
    for instruction in circuit.data:
        qubits = [qubit_map[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        _compile_instruction(instruction.operation, qubits, parameter_indices, operations)


def _compile_instruction(
    operation: Instruction,
    qubits: list[int],
    parameter_indices: dict[Parameter, int],
    operations: list,
) -> None:
    name = operation.name
    if name in _SKIPPED:
        return
    if name in ("measure", "reset", "initialize") or operation.num_clbits:
        raise QiskitNatureError(f"The instruction '{name}' is not supported.")

    parametrized = any(
        isinstance(param, ParameterExpression) and param.parameters for param in operation.params
    )

    if name in _ROTATIONS:
        angle = _linear_form(operation.params[0], parameter_indices).scaled(0.5)
        operations.append(_rotation(_ROTATIONS[name], qubits, 1.0, angle))
        return
    if name in ("p", "u1"):
        # diag(1, exp(i lambda)) = exp(i lambda/2) rz(lambda)
        angle = _linear_form(operation.params[0], parameter_indices).scaled(0.5)
        operations.append(_Phase(angle))
        operations.append(_rotation("Z", qubits, 1.0, angle))
        return
    if isinstance(operation, PauliEvolutionGate) and _compile_pauli_evolution(
        operation, qubits, parameter_indices, operations
    ):
        return
    if not parametrized and len(qubits) <= _MAX_MATRIX_QUBITS:
        try:
            matrix = Operator(operation).data
        except Exception:  # pylint: disable=broad-except
            matrix = None
        if matrix is not None:
            operations.append(_Matrix(tuple(qubits), matrix))
            return
    definition = operation.definition
    if definition is None:
        raise QiskitNatureError(f"The instruction '{name}' cannot be simulated.")
    _compile_circuit(definition, qubits, parameter_indices, operations)


def _compile_pauli_evolution(
    gate: PauliEvolutionGate,
    qubits: list[int],
    parameter_indices: dict[Parameter, int],
    operations: list,
) -> bool:
    """Compiles a PauliEvolutionGate into Pauli rotations, if this is exact.

    Returns:
        Whether the gate was compiled.
    """
    operators = gate.operator if isinstance(gate.operator, list) else [gate.operator]
    terms: list[tuple[str, float]] = []
    for operator in operators:
        if isinstance(operator, SparseObservable):
            operator = SparsePauliOp.from_sparse_observable(operator)
        if not isinstance(operator, SparsePauliOp):
            return False
        coeffs = np.real_if_close(operator.coeffs)
        if np.iscomplexobj(coeffs) or coeffs.dtype == object:
            return False
        terms.extend(zip(operator.paulis.to_labels(), coeffs.tolist()))

    masks = [_rotation(label, qubits, 1.0, _LinearForm()) for label, _ in terms]
    # two Pauli strings commute if their symplectic inner product is even
    commuting = all(
        (int(a.x_mask & b.z_mask) ^ int(a.z_mask & b.x_mask)).bit_count() % 2 == 0
        for i, a in enumerate(masks)
        for b in masks[i + 1 :]
    )
    synthesis = gate.synthesis
    reps = 1
    if not commuting:
        # the product of the exponentials is only exact for commuting terms; otherwise, mimic
        # the first-order Lie-Trotter synthesis with preserved term order
        if not isinstance(synthesis, LieTrotter) or not getattr(synthesis, "preserve_order", True):
            return False
        reps = synthesis.reps

    time = _linear_form(gate.params[0], parameter_indices)
    for _ in range(reps):
        for label, coeff in terms:
            operations.append(_rotation(label, qubits, 1.0, time.scaled(coeff / reps)))
    return True


def _rotation(label: str, qubits: list[int], factor: complex, angle: _LinearForm) -> _Rotation:
    """Creates the rotation exp(-i angle P) for a Pauli label acting on the given qubits.

    The label is little-endian, i.e. its last character acts on ``qubits[0]``.
    """
    x_mask = z_mask = 0
    num_y = 0
    for position, char in enumerate(reversed(label)):
        bit = 1 << qubits[position]
        if char in "XY":
            x_mask |= bit
        if char in "ZY":
            z_mask |= bit
        num_y += char == "Y"
    # with Y = i X Z, the Pauli string equals i^num_y X^x Z^z
    return _Rotation(x_mask, z_mask, factor * 1j**num_y, angle)


def _linear_form(value, parameter_indices: dict[Parameter, int]) -> _LinearForm:
    """Converts a (parametrized) angle into a linear form of the circuit parameters."""
    if not isinstance(value, ParameterExpression) or not value.parameters:
        return _LinearForm(float(value))
    coeffs = {}
    for param in value.parameters:
        gradient = value.gradient(param)
        if isinstance(gradient, ParameterExpression) and gradient.parameters:
            raise QiskitNatureError(
                f"The angle {value} does not depend linearly on the parameter {param}."
            )
        coeffs[parameter_indices[param]] = float(gradient)
    constant = float(value.bind({param: 0.0 for param in value.parameters}))
    return _LinearForm(constant, coeffs)


def _simulate(
    operations: list, num_qubits: int, values: np.ndarray, derivatives: bool = True
) -> np.ndarray:
    """Propagates the state and its parameter derivatives through the compiled operations.

    Returns:
        An array of shape ``(num_parameters + 1, 2**num_qubits)`` holding the state in its first
        row and the derivatives with respect to the parameters in the following rows. If
        ``derivatives`` is ``False``, only the state is computed.
    """
    dim = 2**num_qubits
    states = np.zeros((len(values) + 1 if derivatives else 1, dim), dtype=complex)
    states[0, 0] = 1.0
    basis = np.arange(dim)

    for operation in operations:
        if isinstance(operation, _Rotation):
            angle = operation.angle.evaluate(values)
            pauli_states = _apply_pauli(states, operation, basis)
            states = np.cos(angle) * states - 1j * np.sin(angle) * pauli_states
            if derivatives and operation.angle.coeffs:
                # d/dtheta exp(-i a theta P) = a (-i P) exp(-i a theta P)
                pauli_psi = _apply_pauli(states[:1], operation, basis)[0]
                for k, coeff in operation.angle.coeffs.items():
                    states[k + 1] -= 1j * coeff * pauli_psi
        elif isinstance(operation, _Phase):
            states *= np.exp(1j * operation.angle.evaluate(values))
            if derivatives:
                for k, coeff in operation.angle.coeffs.items():
                    states[k + 1] += 1j * coeff * states[0]
        else:
            states = _apply_matrix(states, operation, num_qubits)
    return states


def _apply_pauli(states: np.ndarray, rotation: _Rotation, basis: np.ndarray) -> np.ndarray:
    """Applies the Pauli string of a rotation to every row of ``states``."""
    # (P psi)[k] = factor (-1)^popcount((k ^ x) & z) psi[k ^ x]
    signs = np.where(np.bitwise_count(basis & rotation.z_mask) & 1, -1.0, 1.0)
    return rotation.factor * (states * signs)[:, basis ^ rotation.x_mask]


def _apply_matrix(states: np.ndarray, operation: _Matrix, num_qubits: int) -> np.ndarray:
    """Applies a fixed matrix to every row of ``states``."""
    num_rows = states.shape[0]
    num_targets = len(operation.qubits)
    tensor = states.reshape((num_rows,) + (2,) * num_qubits)
    matrix = operation.matrix.reshape((2,) * (2 * num_targets))
    # the matrix index of qubits[j] is bit j; tensor axis 1 + (n - 1 - q) belongs to qubit q
    targets = [
        1 + num_qubits - 1 - operation.qubits[num_targets - 1 - j] for j in range(num_targets)
    ]
    result = np.tensordot(matrix, tensor, axes=(list(range(num_targets, 2 * num_targets)), targets))
    result = np.moveaxis(result, list(range(num_targets)), targets)
    return result.reshape(num_rows, -1)
