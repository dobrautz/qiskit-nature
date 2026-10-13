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

"""A VarQITE metric from state overlaps, normalized by a built-in reference circuit."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms import AlgorithmError
from qiskit_algorithms.time_evolvers.variational import ImaginaryVariationalPrinciple

from qiskit_nature import QiskitNatureError

# rotation gates exp(-i theta P / 2) with a Pauli operator P, and phase gates diag(..., e^{i theta})
_ROTATIONS = {"rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx"}
_CONTROLLED_ROTATIONS = {"crx", "cry", "crz"}
_PHASES = {"p", "u1", "cp", "cu1"}


class OverlapMetricPrinciple(ImaginaryVariationalPrinciple):
    r"""A VarQITE principle whose metric is measured from state overlaps.

    The metric of the McLachlan equations of motion is the Fubini-Study metric, i.e. the second
    order of the overlap :math:`F(s) = |\langle\psi(\theta)|\psi(\theta + s)\rangle|^2 = 1 -
    \sum_{kl} A_{kl} s_k s_l + O(s^3)`. This principle measures :math:`F` as the probability of
    the all-zero outcome of the compute-uncompute circuit :math:`U(\theta + s)^\dagger U(\theta)`
    and evaluates the second derivatives exactly with parameter-shift rules. If a parameter enters
    a single gate :math:`e^{-i\theta G}`, :math:`F` is a trigonometric polynomial in :math:`s_k`
    whose frequencies are the differences of the eigenvalues of :math:`G`, i.e. multiples
    :math:`1, \ldots, R` of a base frequency (see :func:`parameter_frequencies`; e.g.
    :math:`R = 1` for Pauli rotations and :math:`R = 2` for the excitations of a :class:`.UCC`
    ansatz). Hence, :math:`2R - 1` shifts determine :math:`A_{kk}` and :math:`4 R_k R_l` shifts
    determine :math:`A_{kl}` (generalized parameter-shift rule).

    The unshifted circuit (:math:`s = 0`) has the exact overlap :math:`F = 1`. Hence, its measured
    value is a reference which is built into every evaluation: all measured overlaps of the same
    evaluation are divided by it (``normalize=True``). This removes the damping of the overlaps by
    gate noise and readout errors, which is to a good approximation the same for all circuits of
    an evaluation since they share their structure. In noisy simulations, the metric obtained in
    this way gave VarQITE trajectories as accurate as with the exact metric, whereas the metric of
    the :class:`~qiskit_algorithms.gradients.LinCombQGT` was sometimes several times less accurate.
    The compute-uncompute circuits are twice as deep as the ansatz, which limits the method to
    shallow circuits, and the number of circuits per evaluation, :math:`1 + \sum_k (2R_k - 1) +
    \sum_{k<l} 4R_kR_l`, is about four times that of the LinComb metric for :math:`R = 1`. Only the
    diagonal blocks given by ``blocks`` can be evaluated to reduce it, which however changes the
    trajectory.

    The gradient is taken from the wrapped ``principle`` (e.g. an
    :class:`~qiskit_algorithms.time_evolvers.variational.ImaginaryMcLachlanPrinciple` or a
    :class:`.ReferenceStateMitigatedPrinciple`). With shot noise, the linear equations should be
    solved with a singular value cutoff of about :math:`10^{-2}`, as done by default by the
    :class:`.VarQITEMinimumEigensolver` for estimators with shot noise.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        principle: ImaginaryVariationalPrinciple,
        sampler: BaseSamplerV2 | None = None,
        *,
        shots: int = 10000,
        normalize: bool = True,
        blocks: Sequence[Sequence[int]] | None = None,
        transpiler: Any | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            principle: The principle which evaluates the gradient.
            sampler: The sampler which measures the overlaps. If ``None``, the overlaps are
                computed exactly from statevectors.
            shots: The number of shots per overlap circuit.
            normalize: Whether to divide the measured overlaps by that of the unshifted circuit.
            blocks: Lists of parameter indices. If given, only the metric elements within these
                blocks are evaluated (the others are zero).
            transpiler: An optional object with a ``run`` method (e.g. a pass manager), which
                compiles the overlap circuits for the backend of the sampler.
            transpiler_options: Keyword arguments passed to the ``run`` method of ``transpiler``.
        """
        super().__init__(principle.qgt, principle.gradient)
        self.principle = principle
        self.sampler = sampler
        self.shots = shots
        self.normalize = normalize
        self.blocks = blocks
        self.transpiler = transpiler
        self.transpiler_options = transpiler_options
        self.last_reference_overlap: float | None = None
        """The measured overlap of the unshifted circuit in the last evaluation (exactly 1
        without noise), which indicates the noise level of the overlap circuits."""
        self._cache: dict[str, Any] = {}

    def num_circuits(self, ansatz: QuantumCircuit) -> int:
        """The number of overlap circuits per evaluation of the metric.

        Args:
            ansatz: The ansatz.

        Returns:
            The number of circuits.
        """
        return len(self._setup(ansatz)["shifts"])

    def metric_tensor(  # type: ignore[override]
        self, ansatz: QuantumCircuit, param_values: Sequence[float]
    ) -> np.ndarray:
        setup = self._setup(ansatz)
        theta = np.asarray(param_values, dtype=float)
        overlaps = self._overlaps(ansatz, setup, theta)
        self.last_reference_overlap = float(overlaps[0])
        if self.normalize:
            if overlaps[0] <= 0:
                raise AlgorithmError("The overlap of the unshifted circuit vanishes.")
            overlaps = overlaps / overlaps[0]
        return _metric_from_overlaps(overlaps, setup)

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self.principle.evolution_gradient(hamiltonian, ansatz, param_values, gradient_params)
        )

    def _setup(self, ansatz: QuantumCircuit) -> dict[str, Any]:
        if self._cache.get("ansatz") is ansatz:
            return self._cache
        num_params = ansatz.num_parameters
        omegas, levels = parameter_frequencies(ansatz)
        if self.blocks is None:
            pairs = list(itertools.combinations(range(num_params), 2))
        else:
            pairs = sorted(
                {
                    (min(i, j), max(i, j))
                    for block in self.blocks
                    for i in block
                    for j in block
                    if i != j
                }
            )
        unit = np.eye(num_params)
        shifts = [np.zeros(num_params)]
        diagonal = []
        for k in range(num_params):
            start = len(shifts)
            for m in range(1, levels[k] + 1):
                delta = m * np.pi / (levels[k] * omegas[k])
                shifts.append(delta * unit[k])
                if m < levels[k]:
                    shifts.append(-delta * unit[k])
            diagonal.append(slice(start, len(shifts)))
        off_diagonal = []
        for k, q in pairs:
            start = len(shifts)
            for (x_k, _), (x_q, _) in itertools.product(
                _shift_rule(levels[k]), _shift_rule(levels[q])
            ):
                shifts.append(x_k / omegas[k] * unit[k] + x_q / omegas[q] * unit[q])
            off_diagonal.append(slice(start, len(shifts)))
        self._cache = {
            "ansatz": ansatz,
            "omegas": omegas,
            "levels": levels,
            "pairs": pairs,
            "shifts": np.array(shifts),
            "diagonal": diagonal,
            "off_diagonal": off_diagonal,
            "circuit": None,
        }
        return self._cache

    def _overlaps(
        self, ansatz: QuantumCircuit, setup: dict[str, Any], theta: np.ndarray
    ) -> np.ndarray:
        shifted = theta + setup["shifts"]
        if self.sampler is None:
            reference = Statevector(ansatz.assign_parameters(theta)).data
            return np.array(
                [
                    abs(np.vdot(reference, Statevector(ansatz.assign_parameters(x)).data)) ** 2
                    for x in shifted
                ]
            )
        if setup["circuit"] is None:
            num_params = ansatz.num_parameters
            first, second = ParameterVector("a", num_params), ParameterVector("b", num_params)
            circuit = ansatz.assign_parameters(first).compose(
                ansatz.assign_parameters(second).inverse()
            )
            circuit.measure_all()
            if self.transpiler is not None:
                circuit = self.transpiler.run(circuit, **(self.transpiler_options or {}))
            setup["circuit"] = circuit
        values = np.hstack([np.tile(theta, (len(shifted), 1)), shifted])
        try:
            result = self.sampler.run([(setup["circuit"], values)], shots=self.shots).result()
        except Exception as exc:
            raise AlgorithmError("The sampler job failed!") from exc
        bits = result[0].join_data()
        return np.array(
            [bits[i].get_int_counts().get(0, 0) / bits.num_shots for i in range(len(shifted))]
        )


def parameter_frequencies(circuit: QuantumCircuit) -> tuple[np.ndarray, np.ndarray]:
    """The frequencies of the overlaps as functions of the parameters of a circuit.

    If a parameter enters a single gate :math:`e^{-i a\\theta G}` linearly, the overlap
    :math:`|\\langle\\psi(\\theta)|\\psi(\\theta + s)\\rangle|^2` is a trigonometric polynomial in
    :math:`s` whose frequencies are the differences of the eigenvalues of :math:`aG`. They are
    returned as multiples :math:`1, \\ldots, R` of a base frequency :math:`\\omega`. Supported are
    Pauli rotations, controlled Pauli rotations, phase gates and
    :class:`~qiskit.circuit.library.PauliEvolutionGate` instances; composite gates are inspected
    recursively.

    Args:
        circuit: The parameterized circuit.

    Raises:
        QiskitNatureError: If a parameter enters several gates or an unsupported gate, or if the
            eigenvalue differences of a generator are not commensurate.

    Returns:
        The base frequencies and the numbers of multiples, in the order of ``circuit.parameters``.
    """
    spectra: dict = {}

    def walk(current: QuantumCircuit) -> None:
        for instruction in current.data:
            operation = instruction.operation
            expressions = [
                p for p in operation.params if isinstance(p, ParameterExpression) and p.parameters
            ]
            if not expressions:
                continue
            if isinstance(operation, PauliEvolutionGate):
                operator = operation.operator
                if isinstance(operator, list):
                    operator = sum(operator[1:], operator[0])
                eigenvalues = _extreme_and_all_eigenvalues(SparsePauliOp(operator))
            elif operation.name in _ROTATIONS:
                eigenvalues = np.array([-0.5, 0.5])
            elif operation.name in _CONTROLLED_ROTATIONS:
                eigenvalues = np.array([-0.5, 0.0, 0.5])
            elif operation.name in _PHASES:
                eigenvalues = np.array([0.0, 1.0])
            elif operation.definition is not None:
                walk(operation.definition)
                continue
            else:
                raise QiskitNatureError(f"The gate {operation.name} is not supported.")
            if len(expressions) != 1 or len(expressions[0].parameters) != 1:
                raise QiskitNatureError(f"The gate {operation.name} depends on several parameters.")
            (parameter,) = expressions[0].parameters
            if parameter in spectra:
                raise QiskitNatureError(
                    f"The parameter {parameter} enters several gates, which is not supported."
                )
            gradient = expressions[0].gradient(parameter)
            if isinstance(gradient, ParameterExpression):
                raise QiskitNatureError(f"The parameter {parameter} does not enter linearly.")
            spectra[parameter] = _frequencies(abs(float(gradient)) * eigenvalues, parameter)

    walk(circuit)
    missing = [p for p in circuit.parameters if p not in spectra]
    if missing:
        raise QiskitNatureError(f"The parameters {missing} were not found in supported gates.")
    omegas = np.array([spectra[p][0] for p in circuit.parameters], dtype=float)
    levels = np.array([spectra[p][1] for p in circuit.parameters], dtype=int)
    return omegas, levels


def _frequencies(eigenvalues: np.ndarray, parameter: Any) -> tuple[float, int]:
    """The base frequency and the number of its multiples among the eigenvalue differences."""
    distinct = np.unique(np.round(eigenvalues, 9))
    differences = np.unique(np.round(np.abs(distinct[:, None] - distinct[None, :]), 9))
    differences = differences[differences > 1e-9]
    if len(differences) == 0:
        raise QiskitNatureError(f"The parameter {parameter} does not change the state.")
    omega = differences.min()
    multiples = differences / omega
    if np.any(np.abs(multiples - np.round(multiples)) > 1e-6):
        raise QiskitNatureError(
            f"The eigenvalue differences of the generator of {parameter} are not commensurate."
        )
    return float(omega), int(round(multiples.max()))


def _extreme_and_all_eigenvalues(operator: SparsePauliOp) -> np.ndarray:
    """The eigenvalues of a Hermitian Pauli operator, restricted to the qubits it acts on."""
    operator = operator.simplify()
    support = np.flatnonzero(np.any(operator.paulis.x | operator.paulis.z, axis=0))
    if len(support) == 0:
        return np.zeros(1)
    reduced = SparsePauliOp(
        [label[::-1][support[0] : support[-1] + 1][::-1] for label in operator.paulis.to_labels()],
        operator.coeffs,
    ).simplify()
    if reduced.num_qubits > 14:
        raise QiskitNatureError(
            "The generator of a PauliEvolutionGate acts on more than 14 qubits, whose spectrum "
            "is not computed."
        )
    return np.linalg.eigvalsh(reduced.to_matrix())


def _shift_rule(level: int) -> list[tuple[float, float]]:
    """Shifts and coefficients of the first derivative of a trigonometric polynomial of degree R."""
    rule = []
    for mu in range(1, 2 * level + 1):
        shift = (2 * mu - 1) * np.pi / (2 * level)
        rule.append((shift, (-1) ** (mu - 1) / (4 * level * np.sin(shift / 2) ** 2)))
    return rule


def _metric_from_overlaps(overlaps: np.ndarray, setup: dict[str, Any]) -> np.ndarray:
    omegas, levels = setup["omegas"], setup["levels"]
    num_params = len(levels)
    metric = np.zeros((num_params, num_params))
    for k in range(num_params):
        level = levels[k]
        values = overlaps[setup["diagonal"][k]]
        # the even part E(m pi / R) = a_0 + sum_r b_r cos(r m pi / R), m = 0..R, with E(0) = 1
        even = [1.0]
        index = 0
        for m in range(1, level + 1):
            if m < level:
                even.append((values[index] + values[index + 1]) / 2)
                index += 2
            else:
                even.append(values[index])
                index += 1
        matrix = np.array(
            [
                [1.0] + [np.cos(r * m * np.pi / level) for r in range(1, level + 1)]
                for m in range(level + 1)
            ]
        )
        coefficients = np.linalg.solve(matrix, np.array(even))
        # A_kk = -1/2 F''(0) = 1/2 omega^2 sum_r r^2 b_r
        metric[k, k] = (
            0.5 * omegas[k] ** 2 * sum(r**2 * coefficients[r] for r in range(1, level + 1))
        )
    for (k, q), part in zip(setup["pairs"], setup["off_diagonal"]):
        weights = [
            c_k * c_q
            for (_, c_k), (_, c_q) in itertools.product(
                _shift_rule(levels[k]), _shift_rule(levels[q])
            )
        ]
        scale = omegas[k] * omegas[q]
        metric[k, q] = metric[q, k] = -0.5 * scale * float(np.dot(weights, overlaps[part]))
    return metric
