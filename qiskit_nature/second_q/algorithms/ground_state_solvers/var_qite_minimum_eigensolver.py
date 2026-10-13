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

"""A minimum eigensolver based on variational quantum imaginary time evolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import scipy.integrate
from scipy.integrate import OdeSolver

from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms import (
    MinimumEigensolver,
    MinimumEigensolverResult,
    TimeEvolutionProblem,
    VarQITE,
    VarQTEResult,
    VariationalResult,
)
from qiskit_algorithms.gradients import LinCombEstimatorGradient, LinCombQGT
from qiskit_algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
    ImaginaryVariationalPrinciple,
)

from .statevector_mc_lachlan_principle import StatevectorImaginaryMcLachlanPrinciple


class VarQITEMinimumEigensolver(MinimumEigensolver):
    r"""A minimum eigensolver based on variational quantum imaginary time evolution (VarQITE).

    The normalized imaginary time evolution :math:`e^{-H\tau}|\psi_0\rangle/\|\cdot\|` converges to
    the (right) eigenvector of :math:`H` whose eigenvalue has the lowest real part, provided the
    initial state has a non-zero overlap with it. VarQITE approximates this evolution within the
    manifold of states spanned by a parametrized ansatz, using McLachlan's variational principle
    (see :class:`~qiskit_algorithms.VarQITE`).

    In contrast to the variational quantum eigensolver (VQE), this also works for
    **non-Hermitian** Hamiltonians, such as the transcorrelated Hamiltonians of electronic
    structure theory [1, 2, 3], for which the expectation value is neither real nor an upper bound
    to the ground state energy. Non-Hermitian Hamiltonians are detected automatically by the
    complex coefficients of their Pauli representation. For exact simulations, the
    :class:`.StatevectorImaginaryMcLachlanPrinciple` is used, which supports non-Hermitian
    Hamiltonians directly. Otherwise, this requires an installation of ``qiskit-algorithms`` whose
    :class:`~qiskit_algorithms.time_evolvers.variational.ImaginaryMcLachlanPrinciple` supports
    non-Hermitian Hamiltonians.

    The returned eigenvalue is the expectation value :math:`\langle\psi(\tau)|H|\psi(\tau)\rangle`
    at the final imaginary time, which is complex for non-Hermitian Hamiltonians. The energies
    along the trajectory are available through :attr:`.VarQITEMinimumEigensolverResult.energies`.

    .. code-block:: python

        from qiskit_nature.second_q.algorithms import (
            GroundStateEigensolver,
            VarQITEMinimumEigensolver,
        )
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        mapper = JordanWignerMapper()
        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=HartreeFock(
                problem.num_spatial_orbitals, problem.num_particles, mapper
            ),
        )
        solver = GroundStateEigensolver(mapper, VarQITEMinimumEigensolver(ansatz, time=10.0))
        result = solver.solve(problem)

    **References:**

    [1] S. McArdle and D. P. Tew, Improving the accuracy of quantum computational chemistry
    using the transcorrelated method, `arXiv:2006.11181 <https://arxiv.org/abs/2006.11181>`_.

    [2] I. O. Sokolov, W. Dobrautz, H. Luo, A. Alavi and I. Tavernelli, Orders of magnitude
    increased accuracy for quantum many-body problems on quantum computers via an exact
    transcorrelated method, `Phys. Rev. Research 5, 023174 (2023)
    <https://doi.org/10.1103/PhysRevResearch.5.023174>`_.

    [3] W. Dobrautz, I. O. Sokolov, K. Liao, P. López Ríos, M. Rahm, A. Alavi and I. Tavernelli,
    Toward real chemical accuracy on current quantum hardware through the transcorrelated method,
    `J. Chem. Theory Comput. 20, 4146 (2024) <https://doi.org/10.1021/acs.jctc.4c00070>`_.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        ansatz: QuantumCircuit,
        initial_point: Sequence[float] | None = None,
        *,
        time: float = 10.0,
        estimator: BaseEstimatorV2 | None = None,
        variational_principle: ImaginaryVariationalPrinciple | None = None,
        ode_solver: type[OdeSolver] | str = "RK45",
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        num_timesteps: int | None = None,
        rtol: float | None = 1e-8,
        atol: float | None = 1e-10,
    ) -> None:
        """
        Args:
            ansatz: The parametrized ansatz circuit.
            initial_point: The initial parameter values. If ``None``, all parameters are
                initialized to zero, which, for example, corresponds to the reference state of
                unitary coupled cluster ansätze.
            time: The total imaginary time of the evolution.
            estimator: The estimator used to evaluate the expectation values. If ``None``, a
                :class:`~qiskit.primitives.StatevectorEstimator` is used.
            variational_principle: The variational principle. If ``None`` and the evolution is
                simulated exactly (``estimator`` is ``None`` or a
                :class:`~qiskit.primitives.StatevectorEstimator` without shot noise), the fast
                :class:`.StatevectorImaginaryMcLachlanPrinciple` is used if it supports the
                ansatz. Otherwise, an
                :class:`~qiskit_algorithms.time_evolvers.variational.ImaginaryMcLachlanPrinciple`
                is used, whose gradients and quantum geometric tensor are evaluated with
                ``estimator`` (if provided).
            ode_solver: The ODE solver, see :class:`~qiskit_algorithms.VarQITE`. Defaults to the
                adaptive ``"RK45"`` method.
            lse_solver: The solver of the linear system of equations, see
                :class:`~qiskit_algorithms.VarQITE`.
            num_timesteps: The number of time steps of the forward Euler solver, see
                :class:`~qiskit_algorithms.VarQITE`.
            rtol: The relative tolerance of the adaptive ODE solver, if ``ode_solver`` is the
                name of a SciPy method. Note that, for non-Hermitian Hamiltonians, the error of
                the expectation value is of first order in the error of the state, which requires
                tighter tolerances than the SciPy defaults. If ``None``, the SciPy default is used.
            atol: The absolute tolerance of the adaptive ODE solver, see ``rtol``.
        """
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.time = time
        self.estimator = estimator
        self.variational_principle = variational_principle
        self.ode_solver = ode_solver
        self.lse_solver = lse_solver
        self.num_timesteps = num_timesteps
        self.rtol = rtol
        self.atol = atol

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: list[BaseOperator | None] | dict[str, BaseOperator] | None = None,
    ) -> VarQITEMinimumEigensolverResult:
        initial_point = (
            np.zeros(self.ansatz.num_parameters)
            if self.initial_point is None
            else np.asarray(self.initial_point, dtype=float)
        )
        estimator = self.estimator if self.estimator is not None else StatevectorEstimator()
        variational_principle = self.variational_principle
        if variational_principle is None:
            exact = isinstance(estimator, StatevectorEstimator) and not estimator.default_precision
            if exact and StatevectorImaginaryMcLachlanPrinciple.supports(self.ansatz):
                variational_principle = StatevectorImaginaryMcLachlanPrinciple()
            elif self.estimator is None:
                variational_principle = ImaginaryMcLachlanPrinciple()
            else:
                variational_principle = ImaginaryMcLachlanPrinciple(
                    LinCombQGT(self.estimator), LinCombEstimatorGradient(self.estimator)
                )

        # the energy is evaluated as the first auxiliary operator along the trajectory
        aux_keys: list[Any] = []
        aux_list: list[BaseOperator] = []
        if isinstance(aux_operators, dict):
            aux_keys = list(aux_operators.keys())
            aux_list = list(aux_operators.values())
        elif aux_operators is not None:
            aux_keys = [i for i, op in enumerate(aux_operators) if op is not None]
            aux_list = [op for op in aux_operators if op is not None]

        ode_solver = self.ode_solver
        if isinstance(ode_solver, str) and (self.rtol is not None or self.atol is not None):
            ode_solver = _with_tolerances(ode_solver, self.rtol, self.atol)

        var_qite = VarQITE(
            self.ansatz,
            initial_point,
            variational_principle,
            estimator,
            ode_solver=ode_solver,
            lse_solver=self.lse_solver,
            num_timesteps=self.num_timesteps,
        )
        exact = isinstance(variational_principle, StatevectorImaginaryMcLachlanPrinciple)
        evolution_result = var_qite.evolve(
            TimeEvolutionProblem(
                operator, self.time, aux_operators=None if exact else [operator, *aux_list]
            )
        )
        if exact:
            # evaluate the observables with the fast statevector simulation
            observables = []
            for values in evolution_result.parameter_values:
                state = variational_principle.statevector(self.ansatz, values)
                observables.append(
                    [
                        (_as_value(variational_principle.expectation_value(op, state)), {})
                        for op in [operator, *aux_list]
                    ]
                )
            evolution_result.observables = observables
            evolution_result.aux_ops_evaluated = observables[-1]

        result = VarQITEMinimumEigensolverResult()
        result.evolution_result = evolution_result
        result.energies = np.asarray([values[0][0] for values in evolution_result.observables])
        result.times = np.asarray(evolution_result.times)
        result.eigenvalue = result.energies[-1]
        result.optimal_value = float(np.real(result.eigenvalue))
        result.optimal_point = np.asarray(evolution_result.parameter_values[-1])
        result.optimal_parameters = dict(zip(self.ansatz.parameters, result.optimal_point))
        result.optimal_circuit = self.ansatz

        final_values = evolution_result.observables[-1][1:]
        if isinstance(aux_operators, dict):
            result.aux_operators_evaluated = dict(zip(aux_keys, final_values))
        elif aux_operators is not None:
            evaluated: list[tuple[complex, dict[str, Any]] | None] = [None] * len(aux_operators)
            for key, value in zip(aux_keys, final_values):
                evaluated[key] = value
            result.aux_operators_evaluated = evaluated

        return result


class VarQITEMinimumEigensolverResult(VariationalResult, MinimumEigensolverResult):
    """The result of the :class:`.VarQITEMinimumEigensolver`."""

    def __init__(self) -> None:
        super().__init__()
        self._energies: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._evolution_result: VarQTEResult | None = None

    @property
    def energies(self) -> np.ndarray | None:
        """The expectation values of the Hamiltonian along the imaginary time evolution."""
        return self._energies

    @energies.setter
    def energies(self, energies: np.ndarray) -> None:
        self._energies = energies

    @property
    def times(self) -> np.ndarray | None:
        """The imaginary times at which :attr:`energies` were evaluated."""
        return self._times

    @times.setter
    def times(self, times: np.ndarray) -> None:
        self._times = times

    @property
    def evolution_result(self) -> VarQTEResult | None:
        """The full result of the imaginary time evolution."""
        return self._evolution_result

    @evolution_result.setter
    def evolution_result(self, evolution_result: VarQTEResult) -> None:
        self._evolution_result = evolution_result


def _with_tolerances(method: str, rtol: float | None, atol: float | None) -> type[OdeSolver]:
    """Returns the SciPy ODE solver class of the given method with modified default tolerances."""
    base = getattr(scipy.integrate, method)

    class _OdeSolver(base):  # type: ignore[misc,valid-type]
        # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            if rtol is not None:
                kwargs.setdefault("rtol", rtol)
            if atol is not None:
                kwargs.setdefault("atol", atol)
            super().__init__(*args, **kwargs)

    _OdeSolver.__name__ = base.__name__
    return _OdeSolver


def _as_value(value: complex) -> float | complex:
    """Returns a real number if the imaginary part vanishes."""
    return value.real if abs(value.imag) < 1e-14 else value
