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
from qiskit.quantum_info import SparsePauliOp

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp
from qiskit_nature.second_q.problems import (
    BaseProblem,
    EigenstateResult,
    ElectronicStructureProblem,
)

from .ground_state_solver import GroundStateSolver
from .var_qite_minimum_eigensolver import VarQITEMinimumEigensolverResult, _lstsq


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

    are integrated with forward Euler steps until the parameters converge. This also holds for
    non-Hermitian Hamiltonians, such as transcorrelated ones including three-body terms, in which
    case the evolution converges to the right eigenvector whose eigenvalue has the lowest real
    part.

    The ansatz must be a :class:`.UCC` instance (e.g. :class:`.UCCSD`) with a
    :class:`.HartreeFock` initial state and without ``include_imaginary``. The Hamiltonian is
    built from ``problem.hamiltonian.second_q_op()`` and stored as a dense matrix of the
    dimension of the particle-number sector.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        ansatz: UCC,
        initial_point: Sequence[float] | None = None,
        *,
        time: float = 1000.0,
        timestep: float = 0.05,
        convergence_threshold: float | None = 1e-6,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        num_trajectory_points: int | None = 200,
    ) -> None:
        """
        Args:
            ansatz: The UCC ansatz.
            initial_point: The initial parameter values. If ``None``, all parameters are
                initialized to zero, i.e. the evolution starts from the Hartree-Fock state.
            time: The maximal imaginary time of the evolution.
            timestep: The imaginary time step of the forward Euler integration.
            convergence_threshold: The integration stops once the norm of the parameter
                derivative :math:`\\|\\dot{\\theta}\\|` falls below this threshold. If ``None``,
                the evolution always proceeds until ``time``.
            lse_solver: The solver of the linear system of equations. If ``None``,
                ``numpy.linalg.lstsq`` with a relative cutoff of ``1e-7`` for the singular values
                is used.
            num_trajectory_points: The maximal number of time points at which the energy along
                the trajectory is recorded (always including the final one). If ``None``, it is
                recorded at all time points.

        Raises:
            QiskitNatureError: If the ansatz is not a :class:`.UCC` instance.
        """
        if not isinstance(ansatz, UCC):
            raise QiskitNatureError("The ansatz must be a UCC instance.")
        super().__init__(ansatz.qubit_mapper)
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.time = time
        self.timestep = timestep
        self.convergence_threshold = convergence_threshold
        self.lse_solver = lse_solver
        self.num_trajectory_points = num_trajectory_points

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
            problem: The electronic structure problem.
            aux_operators: Not supported.

        Raises:
            QiskitNatureError: If the problem, the ansatz or ``aux_operators`` are not supported.

        Returns:
            An interpreted :class:`.ElectronicStructureResult`.
        """
        if aux_operators:
            raise QiskitNatureError("Auxiliary operators are not supported by this solver.")
        if not isinstance(problem, ElectronicStructureProblem):
            raise QiskitNatureError("Only ElectronicStructureProblem instances are supported.")
        norb = problem.num_spatial_orbitals
        nelec = tuple(problem.num_particles)
        generators = self._generators(norb, nelec)
        hamiltonian = _sector_matrix(problem.hamiltonian.second_q_op(), norb, nelec)

        import ffsim  # pylint: disable=import-error

        num_params = self.ansatz.num_parameters
        values = (
            np.zeros(num_params)
            if self.initial_point is None
            else np.asarray(self.initial_point, dtype=float)
        )
        reference = ffsim.hartree_fock_state(norb, nelec)
        lse_solver = self.lse_solver if self.lse_solver is not None else _lstsq

        times = [0.0]
        parameter_values = [values.copy()]
        energies = [np.vdot(reference, hamiltonian @ reference)]
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
            step = remaining if remaining < self.timestep * (1 + 1e-9) else self.timestep
            values = values + step * derivative
            time += step
            times.append(time)
            parameter_values.append(values.copy())
            if self.convergence_threshold is not None and (
                np.linalg.norm(derivative) < self.convergence_threshold
            ):
                converged = True
                break

        # energies along the trajectory at (a subset of) the time points
        num_points = len(times)
        if self.num_trajectory_points is None or num_points <= self.num_trajectory_points:
            indices = np.arange(num_points)
        else:
            indices = np.unique(
                np.linspace(0, num_points - 1, self.num_trajectory_points).astype(int)
            )
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

    def _generators(self, norb: int, nelec: tuple[int, int]) -> list[scipy.sparse.csr_matrix]:
        """The sector matrices of the generators exp(theta (T - T^dagger)), in parameter order."""
        ansatz = self.ansatz
        if not isinstance(ansatz, UCC):
            raise QiskitNatureError("The ansatz must be a UCC instance.")
        if getattr(ansatz, "_include_imaginary", False):
            raise QiskitNatureError("UCC ansätze with include_imaginary=True are not supported.")
        if not isinstance(ansatz.initial_state, HartreeFock):
            raise QiskitNatureError("The ansatz must have a HartreeFock initial state.")
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


def _sector_matrix(operator: FermionicOp, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """The dense matrix of a fermionic operator in the sector of fixed particle numbers."""
    import ffsim  # pylint: disable=import-error

    terms = {}
    for label, coeff in operator.items():
        key = []
        for token in label.split():
            action, index = token.split("_")
            index = int(index)
            key.append((action == "+", index >= norb, index % norb))
        key = tuple(key)
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
