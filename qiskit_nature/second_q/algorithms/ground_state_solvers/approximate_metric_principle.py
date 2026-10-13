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

"""A McLachlan principle with an approximated metric tensor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms import AlgorithmError
from qiskit_algorithms.gradients import DerivativeType
from qiskit_algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
    ImaginaryVariationalPrinciple,
)

from .statevector_mc_lachlan_principle import StatevectorImaginaryMcLachlanPrinciple


class ApproximateMetricPrinciple(ImaginaryVariationalPrinciple):
    r"""An imaginary-time McLachlan principle with an approximated metric tensor.

    The VarQITE equations of motion :math:`A\dot{\theta} = C` require the metric tensor
    :math:`A` (the real part of the quantum geometric tensor) at every time step, whose
    evaluation with :math:`n(n+1)/2` circuits for :math:`n` parameters dominates the cost of
    hardware runs. This principle wraps another principle, e.g. an
    :class:`~qiskit_algorithms.time_evolvers.variational.ImaginaryMcLachlanPrinciple`, uses its
    evolution gradient :math:`C` unchanged and approximates the metric in two ways:

    * ``metric`` selects which part of the metric is evaluated: the ``"full"`` metric, its
      ``"block_diagonal"`` part (by default with the blocks of parameters in the same layer of
      the circuit, as for the quantum natural gradient [1]), its ``"diagonal"`` or no part at all
      (``"identity"``, replaced by :math:`\mathbb{1}/4`, the metric of a single Pauli rotation on
      a state without overlap with its eigenstates). Only the circuits of the selected entries
      are evaluated, if the wrapped principle evaluates the metric with a
      :class:`~qiskit_algorithms.gradients.BaseQGT`.
    * ``update`` selects how the metric is updated between its evaluations, which take place
      at the first time step and every ``recompute_interval`` time steps: it is either kept
      (``None``) or updated with the qBroyden rule of [2],

      .. math::

          A_{k+1} = (1 - \varepsilon_k) A_k + \varepsilon_k C_k C_k^T, \qquad
          \varepsilon_k = \frac{\varepsilon_0}{k + 1},

      (``"qbroyden"``), which is the rank-one update
      :math:`B_{k+1} = (1 - \varepsilon_k) B_k + \varepsilon_k \nabla E \nabla E^T` of the quantum
      Fisher information :math:`B = 4A` for the gradient :math:`\nabla E = -2C` of a Hermitian
      Hamiltonian, without any further circuit evaluations. For non-Hermitian Hamiltonians,
      :math:`C` is not a gradient, and the update is a heuristic.

    A small multiple ``regularization`` of the identity is added to every evaluated metric.

    The time step counter assumes one evaluation of the equations of motion per time step, as
    for the forward Euler integration used by default by the :class:`.VarQITEMinimumEigensolver`.
    The state is reset with :meth:`reset` or when a different ansatz is passed.

    **References:**

    [1] J. Stokes, J. Izaac, N. Killoran and G. Carleo, Quantum Natural Gradient,
    `Quantum 4, 269 (2020) <https://doi.org/10.22331/q-2020-05-25-269>`_.

    [2] D. Fitzek, R. S. Jonsson, W. Dobrautz and C. Schäfer, Optimizing variational quantum
    algorithms with qBang: Efficiently interweaving metric and momentum to navigate flat energy
    landscapes, `Quantum 8, 1313 (2024) <https://doi.org/10.22331/q-2024-04-09-1313>`_.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        principle: ImaginaryVariationalPrinciple | None = None,
        *,
        metric: Literal["full", "block_diagonal", "diagonal", "identity"] = "full",
        blocks: Sequence[Sequence[int]] | None = None,
        update: Literal["qbroyden"] | None = "qbroyden",
        epsilon: float = 0.2,
        recompute_interval: int | None = None,
        regularization: float = 1e-7,
    ) -> None:
        """
        Args:
            principle: The wrapped principle. If ``None``, an
                :class:`~qiskit_algorithms.time_evolvers.variational.ImaginaryMcLachlanPrinciple`
                is used.
            metric: The part of the metric which is evaluated.
            blocks: The blocks of parameter indices for ``metric="block_diagonal"``. If ``None``,
                the parameters in the same layer of the circuit form a block.
            update: The update of the metric between its evaluations.
            epsilon: The initial filter parameter :math:`\\varepsilon_0` of the qBroyden update.
            recompute_interval: The number of time steps after which the metric is evaluated
                anew. If ``None``, it is only evaluated at the first time step.
            regularization: The multiple of the identity added to every evaluated metric.

        Raises:
            ValueError: If the arguments are invalid.
        """
        if principle is None:
            principle = ImaginaryMcLachlanPrinciple()
        if metric not in ("full", "block_diagonal", "diagonal", "identity"):
            raise ValueError(f"Invalid metric '{metric}'.")
        if update not in (None, "qbroyden"):
            raise ValueError(f"Invalid update '{update}'.")
        if recompute_interval is not None and recompute_interval < 1:
            raise ValueError("The recompute interval must be positive.")
        super().__init__(principle.qgt, principle.gradient)
        self.principle = principle
        self.metric = metric
        self.blocks = blocks
        self.update = update
        self.epsilon = epsilon
        self.recompute_interval = recompute_interval
        self.regularization = regularization
        self.num_metric_entries = 0
        """The number of evaluated entries of the metric (i.e. of quantum geometric tensor
        circuits) since the last reset."""
        self.num_metric_evaluations = 0
        """The number of metric evaluations since the last reset."""
        self._ansatz: QuantumCircuit | None = None
        self._metric: np.ndarray | None = None
        self._gradient: np.ndarray | None = None
        self._step = 0
        self._num_updates = 0

    def reset(self) -> None:
        """Resets the metric, the time step counter and the evaluation counters."""
        self._ansatz = None
        self._metric = None
        self._gradient = None
        self._step = 0
        self._num_updates = 0
        self.num_metric_entries = 0
        self.num_metric_evaluations = 0
        reset = getattr(self.principle, "reset", None)
        if callable(reset):
            reset()  # pylint: disable=not-callable

    def metric_tensor(  # type: ignore[override]
        self, ansatz: QuantumCircuit, param_values: Sequence[float]
    ) -> np.ndarray:
        if ansatz is not self._ansatz:
            self.reset()
            self._ansatz = ansatz
        due = self.recompute_interval is not None and self._step % self.recompute_interval == 0
        if self._metric is None or due:
            metric = self._evaluate_metric(ansatz, param_values)
            self._metric = metric + self.regularization * np.eye(len(metric))
            self._num_updates = 0
            self.num_metric_evaluations += 1
        elif self.update == "qbroyden" and self._gradient is not None:
            epsilon = self.epsilon / (self._num_updates + 1)
            self._metric = (1 - epsilon) * self._metric + epsilon * np.outer(
                self._gradient, self._gradient
            )
            self._num_updates += 1
        self._step += 1
        return self._metric

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence | None = None,
    ) -> np.ndarray:
        gradient = np.asarray(
            self.principle.evolution_gradient(hamiltonian, ansatz, param_values, gradient_params)
        )
        self._gradient = np.real(gradient)
        return gradient

    def _evaluate_metric(self, ansatz: QuantumCircuit, param_values: Sequence[float]) -> np.ndarray:
        """Evaluates the selected part of the metric."""
        num_params = ansatz.num_parameters
        if self.metric == "identity":
            return np.eye(num_params) / 4
        if self.metric == "full":
            self.num_metric_entries += num_params * (num_params + 1) // 2
            return np.real(np.asarray(self.principle.metric_tensor(ansatz, param_values)))
        if self.metric == "diagonal":
            blocks = [[index] for index in range(num_params)]
        else:
            blocks = [list(block) for block in (self.blocks or parameter_layers(ansatz))]
        covered = sorted(index for block in blocks for index in block)
        if covered != list(range(num_params)):
            raise ValueError("The blocks must contain every parameter index exactly once.")
        self.num_metric_entries += sum(len(block) * (len(block) + 1) // 2 for block in blocks)

        metric = np.zeros((num_params, num_params))
        if self.principle.qgt is None or isinstance(
            self.principle, StatevectorImaginaryMcLachlanPrinciple
        ):
            # the wrapped principle evaluates the full metric at once (e.g. exactly)
            full = np.real(np.asarray(self.principle.metric_tensor(ansatz, param_values)))
            for block in blocks:
                metric[np.ix_(block, block)] = full[np.ix_(block, block)]
            return metric
        parameters = list(ansatz.parameters)
        self.principle.qgt.derivative_type = DerivativeType.REAL
        try:
            qgts = (
                self.principle.qgt.run(
                    [ansatz] * len(blocks),
                    [param_values] * len(blocks),
                    [[parameters[index] for index in block] for block in blocks],
                )
                .result()
                .qgts
            )
        except Exception as exc:
            raise AlgorithmError("The QGT primitive job failed!") from exc
        for block, qgt in zip(blocks, qgts):
            metric[np.ix_(block, block)] = np.real(qgt)
        return metric


def parameter_layers(circuit: QuantumCircuit) -> list[list[int]]:
    """Groups the parameters of a circuit by the layers in which they first occur.

    Args:
        circuit: The parameterized circuit. Composite gates (e.g. the evolution of an operator in
            a :class:`.UCC` ansatz) are decomposed by one level.

    Returns:
        The lists of parameter indices (with respect to ``circuit.parameters``) of the layers.
    """
    indices = {param: index for index, param in enumerate(circuit.parameters)}
    dag = circuit_to_dag(circuit.decompose())
    layers: list[list[int]] = []
    seen: set[int] = set()
    for layer in dag.layers():
        block = []
        for node in layer["graph"].op_nodes():
            for value in node.op.params:
                if not isinstance(value, ParameterExpression):
                    continue
                for param in sorted(value.parameters, key=lambda p: indices[p]):
                    if indices[param] not in seen:
                        seen.add(indices[param])
                        block.append(indices[param])
        if block:
            layers.append(sorted(block))
    return layers
