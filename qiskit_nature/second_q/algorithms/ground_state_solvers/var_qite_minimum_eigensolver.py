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
from scipy.integrate._ivp.base import ConstantDenseOutput

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
from qiskit_algorithms.observables_evaluator import estimate_observables
from qiskit_algorithms.custom_types import Transpiler
from qiskit_algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
    ImaginaryVariationalPrinciple,
)

from .approximate_metric_principle import ApproximateMetricPrinciple
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
        time: float | None = None,
        timestep: float = 0.05,
        convergence_threshold: float | None = 1e-6,
        estimator: BaseEstimatorV2 | None = None,
        variational_principle: ImaginaryVariationalPrinciple | None = None,
        ode_solver: type[OdeSolver] | str | None = None,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        num_timesteps: int | None = None,
        rtol: float | None = 1e-8,
        atol: float | None = 1e-10,
        num_trajectory_points: int | None = 200,
        transpiler: Transpiler | None = None,
        transpiler_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            ansatz: The parametrized ansatz circuit.
            initial_point: The initial parameter values. If ``None``, all parameters are
                initialized to zero, which, for example, corresponds to the reference state of
                unitary coupled cluster ansätze.
            time: The maximal imaginary time of the evolution. If ``None``, it is ``1000`` for
                exact simulations, which typically stop much earlier because the parameters
                converge, and ``10`` otherwise, since shot noise prevents the convergence of the
                parameters.
            timestep: The imaginary time step of the default forward Euler integration.
            convergence_threshold: The default forward Euler integration stops once the norm of
                the parameter derivative :math:`\\|\\dot{\\theta}\\|` falls below this threshold. If
                ``None``, the evolution always proceeds until ``time``.
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
            ode_solver: The ODE solver, see :class:`~qiskit_algorithms.VarQITE`. If ``None``, a
                forward Euler integration with the given ``timestep`` and
                ``convergence_threshold`` is used.
            lse_solver: The solver of the linear system of equations, see
                :class:`~qiskit_algorithms.VarQITE`. If ``None``, ``numpy.linalg.lstsq`` is used
                with a relative cutoff for the singular values of ``1e-7`` for exact simulations
                and of ``1e-2`` otherwise, which prevents the amplification of shot noise.
            num_timesteps: The number of time steps of the
                :class:`~qiskit_algorithms.time_evolvers.variational.ForwardEulerSolver`, if it is
                given as ``ode_solver``, see :class:`~qiskit_algorithms.VarQITE`.
            rtol: The relative tolerance of the adaptive ODE solver, if ``ode_solver`` is the
                name of a SciPy method. Note that, for non-Hermitian Hamiltonians, the error of
                the expectation value is of first order in the error of the state, which requires
                tighter tolerances than the SciPy defaults. If ``None``, the SciPy default is used.
            atol: The absolute tolerance of the adaptive ODE solver, see ``rtol``.
            num_trajectory_points: The maximal number of time points at which the observables
                along the trajectory are evaluated (always including the final one). If ``None``,
                they are evaluated at all time points.
            transpiler: An optional object with a ``run`` method (e.g. a pass manager), which
                transpiles the circuits evaluated by ``estimator`` for its backend, as required by
                the V2 primitives of hardware backends and simulators. It is applied to the gradient
                and quantum geometric tensor circuits of the default variational principle (which
                contain an additional ancilla qubit) and to the ansatz for the evaluation of the
                observables, whose layout is taken into account. The evolution itself operates on
                the logical ``ansatz``.
            transpiler_options: Keyword arguments passed to the ``run`` method of ``transpiler``.
        """
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.time = time
        self.timestep = timestep
        self.convergence_threshold = convergence_threshold
        self.estimator = estimator
        self.variational_principle = variational_principle
        self.ode_solver = ode_solver
        self.lse_solver = lse_solver
        self.num_timesteps = num_timesteps
        self.rtol = rtol
        self.atol = atol
        self.num_trajectory_points = num_trajectory_points
        self.transpiler = transpiler
        self.transpiler_options = transpiler_options

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
            elif self.estimator is None and self.transpiler is None:
                variational_principle = ImaginaryMcLachlanPrinciple()
            else:
                variational_principle = ImaginaryMcLachlanPrinciple(
                    LinCombQGT(
                        estimator,
                        transpiler=self.transpiler,
                        transpiler_options=self.transpiler_options,
                    ),
                    LinCombEstimatorGradient(
                        estimator,
                        transpiler=self.transpiler,
                        transpiler_options=self.transpiler_options,
                    ),
                )
        reset = getattr(variational_principle, "reset", None)
        if callable(reset):
            # e.g. the ApproximateMetricPrinciple and the ReferenceStateMitigatedPrinciple
            reset()  # pylint: disable=not-callable
        statevector_principle = _statevector_principle(variational_principle)

        # the energy is evaluated as the first auxiliary operator along the trajectory
        aux_keys: list[Any] = []
        aux_list: list[BaseOperator] = []
        if isinstance(aux_operators, dict):
            aux_keys = list(aux_operators.keys())
            aux_list = list(aux_operators.values())
        elif aux_operators is not None:
            aux_keys = [i for i, op in enumerate(aux_operators) if op is not None]
            aux_list = [op for op in aux_operators if op is not None]

        exact_simulation = statevector_principle is not None or (
            isinstance(estimator, StatevectorEstimator) and not estimator.default_precision
        )
        time = self.time if self.time is not None else (1000.0 if exact_simulation else 10.0)

        ode_solver = self.ode_solver
        if ode_solver is None:
            ode_solver = _converging_euler_solver(self.timestep, self.convergence_threshold)
        elif isinstance(ode_solver, str) and (self.rtol is not None or self.atol is not None):
            ode_solver = _with_tolerances(ode_solver, self.rtol, self.atol)
        lse_solver = self.lse_solver
        if lse_solver is None:
            # a strong regularization is required to prevent the amplification of shot noise
            lse_solver = _lstsq if exact_simulation else _lstsq_regularized

        var_qite = VarQITE(
            self.ansatz,
            initial_point,
            variational_principle,
            estimator,
            ode_solver=ode_solver,
            lse_solver=lse_solver,
            num_timesteps=self.num_timesteps,
        )
        evolution_result = var_qite.evolve(TimeEvolutionProblem(operator, time))

        # evaluate the observables at (a subset of) the time points of the trajectory
        num_points = len(evolution_result.times)
        if self.num_trajectory_points is None or num_points <= self.num_trajectory_points:
            indices = np.arange(num_points)
        else:
            indices = np.unique(
                np.linspace(0, num_points - 1, self.num_trajectory_points).astype(int)
            )
        exact = statevector_principle is not None
        observables_to_evaluate = [operator, *aux_list]
        circuit = self.ansatz
        if self.transpiler is not None and not exact:
            circuit = self.transpiler.run(self.ansatz, **(self.transpiler_options or {}))
        if circuit.layout is not None:
            # take the layout of a transpiled ansatz into account (as VarQITE does for the
            # Hamiltonian)
            observables_to_evaluate = [
                op.apply_layout(circuit.layout) for op in observables_to_evaluate
            ]
        observables: list[list[tuple[Any, dict[str, Any]]]] = []
        for index in indices:
            values = evolution_result.parameter_values[index]
            if exact:
                state = statevector_principle.statevector(self.ansatz, values)
                observables.append(
                    [
                        (_as_value(statevector_principle.expectation_value(op, state)), {})
                        for op in observables_to_evaluate
                    ]
                )
            else:
                observables.append(
                    estimate_observables(estimator, circuit, observables_to_evaluate, values)
                )
        evolution_result.observables = observables
        evolution_result.aux_ops_evaluated = observables[-1]

        result = VarQITEMinimumEigensolverResult()
        result.evolution_result = evolution_result
        result.energies = np.asarray([values[0][0] for values in observables])
        result.times = np.asarray(evolution_result.times)[indices]
        result.converged = self.ode_solver is None and bool(
            evolution_result.times[-1] < time * (1 - 1e-12)
        )
        result.eigenvalue = result.energies[-1]
        result.optimal_value = float(np.real(result.eigenvalue))
        result.optimal_point = np.asarray(evolution_result.parameter_values[-1])
        result.optimal_parameters = dict(zip(self.ansatz.parameters, result.optimal_point))
        result.optimal_circuit = self.ansatz

        final_values = observables[-1][1:]
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
        self._converged: bool = False

    @property
    def converged(self) -> bool:
        """Whether the default forward Euler integration stopped because the convergence
        threshold was reached (before the maximal imaginary time)."""
        return self._converged

    @converged.setter
    def converged(self, converged: bool) -> None:
        self._converged = converged

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


def _lstsq(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system in the least-squares sense with a relative singular value cutoff
    of 1e-7."""
    return np.linalg.lstsq(matrix, rhs, rcond=1e-7)[0]


def _lstsq_regularized(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solves the linear system in the least-squares sense with a relative singular value cutoff
    of 1e-2, suitable for matrices and right-hand sides subject to shot noise."""
    return np.linalg.lstsq(matrix, rhs, rcond=1e-2)[0]


def _converging_euler_solver(timestep: float, threshold: float | None) -> type[OdeSolver]:
    """Returns a forward Euler ODE solver class with a fixed time step, which finishes once the norm
    of the derivative falls below the given threshold."""

    class _ConvergingEulerSolver(OdeSolver):
        # pylint: disable=too-few-public-methods
        def __init__(self, fun, t0, y0, t_bound, vectorized=False, **kwargs):
            # pylint: disable=unused-argument
            super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=False)
            self._y_old = None

        def _step_impl(self):
            remaining = self.t_bound - self.t
            # avoid a tiny final step due to floating-point accumulation
            step = remaining if remaining < timestep * (1 + 1e-9) else timestep
            derivative = np.asarray(self.fun(self.t, self.y))
            self._y_old = self.y
            self.y = self.y + step * derivative
            self.t = self.t + step
            if threshold is not None and np.linalg.norm(derivative) < threshold:
                # converged: finish the integration at the current time
                self.t_bound = self.t
            return True, None

        def _dense_output_impl(self):
            return ConstantDenseOutput(self.t_old, self.t, self._y_old)

    return _ConvergingEulerSolver


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


def _statevector_principle(
    principle: ImaginaryVariationalPrinciple,
) -> StatevectorImaginaryMcLachlanPrinciple | None:
    """The (possibly wrapped) exact statevector principle, if any."""
    if isinstance(principle, ApproximateMetricPrinciple):
        principle = principle.principle
    if isinstance(principle, StatevectorImaginaryMcLachlanPrinciple):
        return principle
    return None
