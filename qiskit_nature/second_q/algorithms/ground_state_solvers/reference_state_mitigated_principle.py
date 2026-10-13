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

"""Reference-state error mitigation of the VarQITE equations of motion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.time_evolvers.variational import ImaginaryVariationalPrinciple

from .statevector_mc_lachlan_principle import StatevectorImaginaryMcLachlanPrinciple


class ReferenceStateMitigatedPrinciple(ImaginaryVariationalPrinciple):
    r"""Reference-state error mitigation (REM) of the VarQITE equations of motion.

    Reference-state error mitigation [1] and its multireference extension (MREM) [2] correct a
    noisy energy with the noise-induced deviation of a reference state, whose exact energy is
    known classically. This principle applies the same idea to the quantities measured at every
    VarQITE time step: the gradient :math:`C` and, optionally, the metric :math:`A` of the wrapped
    (noisy) principle are corrected as

    .. math::

        C_\mathrm{mit}(\theta) = C(\theta) - \left[C(\theta_\mathrm{ref})
        - C_\mathrm{exact}(\theta_\mathrm{ref})\right],

    where the exact values at the reference parameters :math:`\theta_\mathrm{ref}` (by default
    zero, i.e. the reference state prepared by the initial-state part of the ansatz) are computed
    by ``exact_principle``. The offsets are measured once per evolution.

    Since the evolution converges to the parameters at which :math:`C` vanishes, the noise bias of
    :math:`C` determines how far the converged state is shifted by the noise. The metric does not
    affect the converged state, but it determines the trajectory, which a more accurate metric
    improves considerably. However, the noise of the metric depends on the parameters, such that
    an offset measured at a single reference state does not capture it (it can even degrade the
    trajectory), which is why ``mitigate_metric`` is disabled by default. The gradient correction
    is most reliable if the reference is a stationary point of the exact equations of
    motion, :math:`C_\mathrm{exact}(\theta_\mathrm{ref}) = 0`, as is the case for Hartree-Fock or
    symmetric multireference states at :math:`\theta = 0` for many ansätze: then the offset is
    the pure noise bias of the gradient. Otherwise, a damping :math:`C \to fC` of the gradient by
    noise leaves a residual :math:`(1 - f) C_\mathrm{exact}(\theta_\mathrm{ref})` at every step.
    The statistical error of the offsets enters every step, so they should be measured precisely,
    e.g. by averaging ``num_reference_evaluations`` evaluations.

    **References:**

    [1] P. Lolur et al., Reference-state error mitigation: A strategy for high accuracy quantum
    computation of chemistry, `J. Chem. Theory Comput. 19, 783 (2023)
    <https://doi.org/10.1021/acs.jctc.2c00807>`_.

    [2] H. Zou, E. Magnusson, H. Brunander, W. Dobrautz and M. Rahm, Multireference error
    mitigation for quantum computation of chemistry, Digital Discovery 4, 2521 (2025).
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        principle: ImaginaryVariationalPrinciple,
        exact_principle: ImaginaryVariationalPrinciple | None = None,
        reference_point: Sequence[float] | None = None,
        *,
        mitigate_gradient: bool = True,
        mitigate_metric: bool = False,
        num_reference_evaluations: int = 1,
    ) -> None:
        """
        Args:
            principle: The wrapped principle, which evaluates the noisy quantities.
            exact_principle: The principle which evaluates the exact quantities at the reference
                point. If ``None``, a :class:`.StatevectorImaginaryMcLachlanPrinciple` is used.
            reference_point: The parameter values of the reference state. If ``None``, all
                parameters are zero.
            mitigate_gradient: Whether to correct the gradient.
            mitigate_metric: Whether to correct the metric.
            num_reference_evaluations: The number of evaluations of the noisy quantities at the
                reference point, whose average is used.

        Raises:
            ValueError: If ``num_reference_evaluations`` is not positive.
        """
        if num_reference_evaluations < 1:
            raise ValueError("The number of reference evaluations must be positive.")
        super().__init__(principle.qgt, principle.gradient)
        self.principle = principle
        self.exact_principle = (
            exact_principle
            if exact_principle is not None
            else StatevectorImaginaryMcLachlanPrinciple()
        )
        self.reference_point = reference_point
        self.mitigate_gradient = mitigate_gradient
        self.mitigate_metric = mitigate_metric
        self.num_reference_evaluations = num_reference_evaluations
        self.gradient_offset: np.ndarray | None = None
        """The offset subtracted from the gradient, i.e. its noise bias at the reference point."""
        self.metric_offset: np.ndarray | None = None
        """The offset subtracted from the metric, i.e. its noise bias at the reference point."""
        self._ansatz: QuantumCircuit | None = None

    def reset(self) -> None:
        """Discards the offsets, such that they are measured anew."""
        self.gradient_offset = None
        self.metric_offset = None
        self._ansatz = None

    def metric_tensor(  # type: ignore[override]
        self, ansatz: QuantumCircuit, param_values: Sequence[float]
    ) -> np.ndarray:
        self._check_ansatz(ansatz)
        metric = np.real(np.asarray(self.principle.metric_tensor(ansatz, param_values)))
        if not self.mitigate_metric:
            return metric
        if self.metric_offset is None:
            reference = self._reference(ansatz)
            noisy = np.mean(
                [
                    np.real(np.asarray(self.principle.metric_tensor(ansatz, reference)))
                    for _ in range(self.num_reference_evaluations)
                ],
                axis=0,
            )
            exact = np.real(np.asarray(self.exact_principle.metric_tensor(ansatz, reference)))
            self.metric_offset = noisy - exact
        return metric - self.metric_offset

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence | None = None,
    ) -> np.ndarray:
        self._check_ansatz(ansatz)
        gradient = np.real(
            np.asarray(
                self.principle.evolution_gradient(
                    hamiltonian, ansatz, param_values, gradient_params
                )
            )
        )
        if not self.mitigate_gradient:
            return gradient
        if self.gradient_offset is None:
            reference = self._reference(ansatz)
            noisy = np.mean(
                [
                    np.real(
                        np.asarray(
                            self.principle.evolution_gradient(
                                hamiltonian, ansatz, reference, gradient_params
                            )
                        )
                    )
                    for _ in range(self.num_reference_evaluations)
                ],
                axis=0,
            )
            exact = np.real(
                np.asarray(
                    self.exact_principle.evolution_gradient(
                        hamiltonian, ansatz, reference, gradient_params
                    )
                )
            )
            self.gradient_offset = noisy - exact
        return gradient - self.gradient_offset

    def _check_ansatz(self, ansatz: QuantumCircuit) -> None:
        if ansatz is not self._ansatz:
            self.reset()
            self._ansatz = ansatz

    def _reference(self, ansatz: QuantumCircuit) -> list[float]:
        if self.reference_point is None:
            return [0.0] * ansatz.num_parameters
        return [float(value) for value in self.reference_point]
