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

"""VarQITE for unitary coupled cluster ansätze in the fixed particle-number sector."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse
from qiskit.quantum_info import SparsePauliOp, Statevector

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp
from qiskit_nature.second_q.problems import (
    BaseProblem,
    EigenstateResult,
    ElectronicStructureProblem,
)

from .ground_state_solver import GroundStateSolver
from .var_qite_integration import integrate, step_options
from .var_qite_minimum_eigensolver import (
    VarQITEMinimumEigensolverResult,
    _lstsq,
    _trajectory_indices,
)


class FermionicVarQITESolver(GroundStateSolver):
    r"""Variational quantum imaginary time evolution (VarQITE) for unitary coupled cluster (UCC)
    ansätze, simulated exactly in the sector of fixed particle numbers.

    Instead of simulating the qubit circuit of the ansatz, the state

    .. math::

        |\psi(\theta)\rangle = \prod_k e^{\theta_k (T_k - T_k^\dagger)} |\mathrm{HF}\rangle

    is represented by its coefficients in the basis of Slater determinants with the particle
    numbers of the problem, using the `ffsim <https://github.com/qiskit-community/ffsim>`_
    library. The dimension of this space, :math:`\binom{n}{N_\alpha}\binom{n}{N_\beta}` for
    :math:`n` spatial orbitals, is much smaller than the :math:`2^{2n}` amplitudes of the qubit
    statevector, which enables exact simulations of larger systems. The parameters agree with
    those of the qubit circuit of the :class:`.UCC` ansatz, such that they can be transferred to
    it, e.g. for an execution on quantum hardware.

    As for the :class:`.VarQITEMinimumEigensolver`, the McLachlan equations of motion

    .. math::

        \sum_j \mathrm{Re}\left[\langle\partial_i\psi|\partial_j\psi\rangle
        - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right]
        \dot{\theta}_j
        = -\mathrm{Re}\left[\langle\partial_i\psi|H|\psi\rangle
        - \langle\partial_i\psi|\psi\rangle\langle\psi|H|\psi\rangle\right]

    are integrated with forward Euler steps until the parameters converge (or, optionally, with
    line-search or shifted implicit steps, see :class:`.VarQITEMinimumEigensolver`). This also
    holds for non-Hermitian Hamiltonians, such as transcorrelated ones including three-body terms,
    in which case the evolution converges to the right eigenvector whose eigenvalue has the lowest
    real part.

    The ansatz must be a :class:`.UCC` instance (e.g. :class:`.UCCSD`) without
    ``include_imaginary``. Its initial state is usually the :class:`.HartreeFock` state; other
    initial states, such as a :class:`.SlaterDeterminant` of delocalized orbitals, are supported
    for the :class:`.JordanWignerMapper` if they lie in the sector of fixed particle numbers (their
    statevector is computed once). The Hamiltonian is
    built from ``problem.hamiltonian.second_q_op()`` and stored as a dense matrix of the
    dimension of the particle-number sector. Besides an :class:`.ElectronicStructureProblem`,
    any problem whose Hamiltonian is a :class:`.FermionicOp` in the block-ordered spin-orbital
    basis (spin-up orbitals first) can be solved, e.g. a :class:`.LatticeModelProblem` of a
    :class:`.TranscorrelatedFermiHubbardModel`; the numbers of orbitals and particles are then
    taken from the ansatz.

    With ``adjoint=True``, the evolution is performed for the adjoint Hamiltonian
    :math:`H^\dagger`, which yields an approximation of the left eigenvector of :math:`H`. Together
    with the right eigenvector, it gives the :meth:`biorthogonal_energy`, whose error is of second
    order in the errors of the states (see :func:`.biorthogonal_expectation_value`).
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        ansatz: UCC,
        initial_point: Sequence[float] | None = None,
        *,
        time: float = 1000.0,
        timestep: float | None = None,
        convergence_threshold: float | None = 1e-6,
        implicit: bool = False,
        line_search: bool = False,
        trust_region: float | None = None,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        num_trajectory_points: int | None = 200,
        adjoint: bool = False,
    ) -> None:
        """
        Args:
            ansatz: The UCC ansatz.
            initial_point: The initial parameter values. If ``None``, all parameters are
                initialized to zero, i.e. the evolution starts from the Hartree-Fock state.
            time: The maximal imaginary time of the evolution.
            timestep: The imaginary time step: of the forward Euler integration (default
                ``0.05``), the maximal one with ``line_search=True`` (default ``1.0``), or that of
                the implicit steps (default ``5.0``).
            convergence_threshold: The integration stops once the norm of the parameter
                derivative :math:`\\|\\dot{\\theta}\\|` falls below this threshold. If ``None``,
                the evolution always proceeds until ``time``.
            implicit: Whether to integrate with shifted semi-implicit (Galerkin backward Euler)
                steps, see :class:`.VarQITEMinimumEigensolver`.
            line_search: Whether to shorten every step until the energy decreases sufficiently
                (Hermitian Hamiltonians only), see :class:`.VarQITEMinimumEigensolver`.
            trust_region: The maximal Fubini-Study length of a step. If ``None``, it is ``0.2``
                for implicit steps and unrestricted otherwise; use ``numpy.inf`` to disable it
                for implicit steps.
            lse_solver: The solver of the linear system of equations. If ``None``,
                ``numpy.linalg.lstsq`` with a relative cutoff of ``1e-7`` for the singular values
                is used.
            num_trajectory_points: The maximal number of time points at which the energy along
                the trajectory is recorded (always including the final one). If ``None``, it is
                recorded at all time points.
            adjoint: Whether to evolve with the adjoint Hamiltonian :math:`H^\\dagger` to obtain
                the left eigenvector. The recorded energies are then those of :math:`H^\\dagger`.

        Raises:
            QiskitNatureError: If the ansatz is not a :class:`.UCC` instance.
            ValueError: If ``num_trajectory_points`` is not positive.
        """
        if not isinstance(ansatz, UCC):
            raise QiskitNatureError("The ansatz must be a UCC instance.")
        if num_trajectory_points is not None and num_trajectory_points < 1:
            raise ValueError("The number of trajectory points must be positive.")
        super().__init__(ansatz.qubit_mapper)
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.time = time
        self.timestep = timestep
        self.convergence_threshold = convergence_threshold
        self.implicit = implicit
        self.line_search = line_search
        self.trust_region = trust_region
        self.lse_solver = lse_solver
        self.num_trajectory_points = num_trajectory_points
        self.adjoint = adjoint

    @property
    def solver(self) -> None:
        return None

    def supports_aux_operators(self) -> bool:
        return False

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        main_second_q_op, _ = problem.second_q_ops()
        return self._qubit_mapper.map(main_second_q_op), None

    @_optionals.HAS_FFSIM.require_in_call
    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> EigenstateResult:
        """Computes the ground state.

        Args:
            problem: The problem, e.g. an :class:`.ElectronicStructureProblem` or a
                :class:`.LatticeModelProblem`.
            aux_operators: Not supported.

        Raises:
            QiskitNatureError: If the problem, the ansatz or ``aux_operators`` are not supported.

        Returns:
            The result interpreted by the problem, e.g. an :class:`.ElectronicStructureResult`.
        """
        if aux_operators:
            raise QiskitNatureError("Auxiliary operators are not supported by this solver.")
        norb, nelec, hamiltonian = self._sector_hamiltonian(problem)
        if self.adjoint:
            hamiltonian = hamiltonian.conj().T
        generators = self._generators(norb, nelec)

        num_params = self.ansatz.num_parameters
        values = (
            np.zeros(num_params)
            if self.initial_point is None
            else np.asarray(self.initial_point, dtype=float)
        )
        reference = self._reference_state(norb, nelec)
        lse_solver = self.lse_solver if self.lse_solver is not None else _lstsq

        if self.implicit or self.line_search or self.trust_region is not None:
            times, parameter_values, converged = self._step_integration(
                hamiltonian, generators, reference, values, lse_solver
            )
        else:
            times, parameter_values, converged = self._euler_integration(
                hamiltonian, generators, reference, values, lse_solver
            )
        return self._finish(
            problem,
            hamiltonian,
            generators,
            reference,
            times,
            parameter_values,
            converged,
            values=parameter_values[-1],
        )

    # pylint: disable=too-many-positional-arguments
    def _euler_integration(self, hamiltonian, generators, reference, values, lse_solver):
        """The forward Euler integration with a fixed time step."""
        timestep = self.timestep if self.timestep is not None else 0.05
        times = [0.0]
        parameter_values = [values.copy()]
        converged = False
        time = 0.0
        while time < self.time:
            states = _states_and_derivatives(generators, values, reference)
            psi, derivatives = states[0], states[1:]
            h_psi = hamiltonian @ psi
            energy = np.vdot(psi, h_psi)
            overlaps = derivatives.conj() @ psi
            metric = np.real(
                derivatives.conj() @ derivatives.T - np.outer(overlaps, overlaps.conj())
            )
            rhs = -np.real(derivatives.conj() @ h_psi - overlaps * energy)
            derivative = np.real(lse_solver(metric, rhs))

            remaining = self.time - time
            # avoid a tiny final step due to floating-point accumulation
            step = remaining if remaining < timestep * (1 + 1e-9) else timestep
            values = values + step * derivative
            time += step
            times.append(time)
            parameter_values.append(values.copy())
            if self.convergence_threshold is not None and (
                np.linalg.norm(derivative) < self.convergence_threshold
            ):
                converged = True
                break
        return times, parameter_values, converged

    # pylint: disable=too-many-positional-arguments
    def _step_integration(self, hamiltonian, generators, reference, values, lse_solver):
        """The line-search and/or implicit integration (see ``var_qite_integration``)."""
        if self.line_search and np.abs(hamiltonian - hamiltonian.conj().T).max() > 1e-12:
            raise QiskitNatureError(
                "The line search requires a Hermitian Hamiltonian, whose energy decreases "
                "monotonically along the imaginary time evolution."
            )

        def evaluate(point: np.ndarray, implicit: bool):
            states = _states_and_derivatives(generators, point, reference)
            psi, derivatives = states[0], states[1:]
            h_psi = hamiltonian @ psi
            energy = np.vdot(psi, h_psi)
            overlaps = derivatives.conj() @ psi
            metric = np.real(
                derivatives.conj() @ derivatives.T - np.outer(overlaps, overlaps.conj())
            )
            rhs = -np.real(derivatives.conj() @ h_psi - overlaps * energy)
            tangent = None
            if implicit:
                tangents = derivatives - np.outer(overlaps.conj(), psi)
                h_tangents = (hamiltonian @ tangents.T).T - energy * tangents
                tangent = np.real(tangents.conj() @ h_tangents.T)
            return metric, rhs, tangent

        def energy(point: np.ndarray) -> float:
            psi = _states_and_derivatives(generators, point, reference, derivatives=False)[0]
            return float(np.real(np.vdot(psi, hamiltonian @ psi)))

        timestep, trust_region = step_options(
            self.timestep, self.implicit, self.line_search, self.trust_region
        )
        return integrate(
            evaluate,
            energy,
            values,
            time=self.time,
            timestep=timestep,
            implicit=self.implicit,
            line_search=self.line_search,
            trust_region=trust_region,
            convergence_threshold=self.convergence_threshold,
            lse_solver=lse_solver,
        )

    # pylint: disable=too-many-positional-arguments
    def _finish(
        self,
        problem,
        hamiltonian,
        generators,
        reference,
        times,
        parameter_values,
        converged,
        values,
    ):
        """Evaluates the energies along the trajectory and interprets the result."""
        # energies along the trajectory at (a subset of) the time points
        indices = _trajectory_indices(len(times), self.num_trajectory_points)
        energies = []
        for index in indices:
            psi = _states_and_derivatives(
                generators, parameter_values[index], reference, derivatives=False
            )[0]
            energies.append(np.vdot(psi, hamiltonian @ psi))

        raw_result = VarQITEMinimumEigensolverResult()
        raw_result.energies = np.asarray([_as_value(e) for e in energies])
        raw_result.times = np.asarray(times)[indices]
        raw_result.converged = converged
        raw_result.eigenvalue = raw_result.energies[-1]
        raw_result.optimal_value = float(np.real(raw_result.eigenvalue))
        raw_result.optimal_point = values
        raw_result.optimal_parameters = dict(zip(self.ansatz.parameters, values))
        raw_result.optimal_circuit = self.ansatz
        return problem.interpret(EigenstateResult.from_result(raw_result))

    @_optionals.HAS_FFSIM.require_in_call
    def biorthogonal_energy(
        self,
        problem: BaseProblem,
        left_point: Sequence[float],
        right_point: Sequence[float],
    ) -> complex:
        r"""Computes the biorthogonal energy :math:`\langle L|H|R\rangle / \langle L|R\rangle`.

        Args:
            problem: The problem.
            left_point: The parameter values of the left state, e.g. the optimal point of a
                solver with ``adjoint=True``.
            right_point: The parameter values of the right state.

        Raises:
            QiskitNatureError: If the left and right states are orthogonal.

        Returns:
            The biorthogonal energy (without constant energy offsets of the problem).
        """
        norb, nelec, hamiltonian = self._sector_hamiltonian(problem)
        generators = self._generators(norb, nelec)
        reference = self._reference_state(norb, nelec)
        left, right = (
            _states_and_derivatives(
                generators, np.asarray(point, dtype=float), reference, derivatives=False
            )[0]
            for point in (left_point, right_point)
        )
        overlap = np.vdot(left, right)
        if abs(overlap) < 1e-12:
            raise QiskitNatureError("The left and right states are orthogonal.")
        return complex(np.vdot(left, hamiltonian @ right) / overlap)

    def _sector_hamiltonian(self, problem: BaseProblem) -> tuple[int, tuple[int, int], np.ndarray]:
        """The numbers of orbitals and particles and the Hamiltonian matrix in the sector."""
        if isinstance(problem, ElectronicStructureProblem):
            norb = problem.num_spatial_orbitals
            num_alpha, num_beta = problem.num_particles
        else:
            norb = self.ansatz.num_spatial_orbitals
            num_alpha, num_beta = self.ansatz.num_particles
        nelec = (num_alpha, num_beta)
        operator = problem.hamiltonian.second_q_op()
        if not isinstance(operator, FermionicOp) or operator.register_length != 2 * norb:
            raise QiskitNatureError(
                f"The Hamiltonian must be a FermionicOp of {2 * norb} spin orbitals."
            )
        return norb, nelec, _sector_matrix(operator, norb, nelec)

    def _generators(self, norb: int, nelec: tuple[int, int]) -> list[scipy.sparse.csr_matrix]:
        """The sector matrices of the generators exp(theta (T - T^dagger)), in parameter order."""
        ansatz = self.ansatz
        if not isinstance(ansatz, UCC):
            raise QiskitNatureError("The ansatz must be a UCC instance.")
        if getattr(ansatz, "_include_imaginary", False):
            raise QiskitNatureError("UCC ansätze with include_imaginary=True are not supported.")
        if ansatz.num_spatial_orbitals != norb or tuple(ansatz.num_particles) != nelec:
            raise QiskitNatureError(
                "The numbers of orbitals and particles of the ansatz and the problem differ."
            )
        _ = ansatz.operators  # builds (and possibly filters) the excitations
        excitations = ansatz.excitation_list
        generators = []
        for occupied, unoccupied in excitations:
            label = " ".join([f"+_{i}" for i in occupied] + [f"-_{i}" for i in unoccupied])
            excitation = FermionicOp({label: 1.0}, num_spin_orbitals=2 * norb)
            generator = scipy.sparse.csr_matrix(
                np.real_if_close(_sector_matrix(excitation - excitation.adjoint(), norb, nelec))
            )
            cube = generator @ (generator @ generator)
            if abs(cube + generator).max() > 1e-12:
                raise QiskitNatureError(
                    f"The excitation {(occupied, unoccupied)} does not generate a rotation with "
                    "G^3 = -G, which is not supported."
                )
            generators.append(generator)
        reps = ansatz.num_parameters // len(generators)
        if reps * len(generators) != ansatz.num_parameters:
            raise QiskitNatureError("The parameters of the ansatz do not match its excitations.")
        return generators * reps

    def _reference_state(self, norb: int, nelec: tuple[int, int]) -> np.ndarray:
        """The initial state of the ansatz in the sector of fixed particle numbers."""
        import ffsim  # pylint: disable=import-error

        initial_state = self.ansatz.initial_state
        if isinstance(initial_state, HartreeFock):
            return ffsim.hartree_fock_state(norb, nelec)
        if initial_state is None or not isinstance(self.ansatz.qubit_mapper, JordanWignerMapper):
            raise QiskitNatureError(
                "The ansatz must have a HartreeFock initial state or, for the "
                "JordanWignerMapper, an initial state circuit."
            )
        # ffsim orders the spin orbitals as the Jordan-Wigner mapping of qiskit-nature (spin-up
        # orbitals first), such that the conversion only selects the amplitudes of the sector
        reference = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            Statevector(initial_state).data, norb, nelec
        )
        if abs(np.vdot(reference, reference) - 1) > 1e-8:
            raise QiskitNatureError(
                f"The initial state does not lie in the sector of {nelec} particles."
            )
        return reference


def _sector_matrix(operator: FermionicOp, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """The dense matrix of a fermionic operator in the sector of fixed particle numbers."""
    import ffsim  # pylint: disable=import-error

    terms: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for label, coeff in operator.items():
        key = tuple(
            (action == "+", int(index) >= norb, int(index) % norb)
            for action, index in (token.split("_") for token in label.split())
        )
        terms[key] = terms.get(key, 0.0) + complex(coeff)
    linear_operator = ffsim.linear_operator(ffsim.FermionOperator(terms), norb=norb, nelec=nelec)
    dim = linear_operator.shape[0]
    return linear_operator @ np.eye(dim, dtype=complex)


def _states_and_derivatives(
    generators: list,
    values: np.ndarray,
    reference: np.ndarray,
    derivatives: bool = True,
) -> np.ndarray:
    """Propagates the state (first row) and its parameter derivatives (further rows).

    Every generator G = T - T^dagger of a single or double excitation obeys G^3 = -G, such that
    exp(theta G) = 1 + sin(theta) G + (1 - cos(theta)) G^2.
    """
    states = np.zeros((len(values) + 1 if derivatives else 1, reference.shape[0]), dtype=complex)
    states[0] = reference
    for k, (generator, theta) in enumerate(zip(generators, values)):
        g_states = (generator @ states.T).T
        gg_states = (generator @ g_states.T).T
        states = states + np.sin(theta) * g_states + (1 - np.cos(theta)) * gg_states
        if derivatives:
            states[k + 1] += generator @ states[0]
    return states


def _as_value(value: complex) -> float | complex:
    """Returns a real number if the imaginary part vanishes."""
    return value.real if abs(value.imag) < 1e-14 else value
