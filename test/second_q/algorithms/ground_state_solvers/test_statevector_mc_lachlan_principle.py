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

"""Test the StatevectorImaginaryMcLachlanPrinciple."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, efficient_su2
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

from qiskit_nature.second_q.algorithms import (
    StatevectorImaginaryMcLachlanPrinciple,
    VarQITEMinimumEigensolver,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.tcdump import TCDump
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

DATA = Path(__file__).parents[2] / "formats" / "tcdump"


def non_hermitian_operator(num_qubits: int, seed: int = 0) -> SparsePauliOp:
    """A random real non-Hermitian operator with a real spectrum."""
    rng = np.random.default_rng(seed)
    dim = 2**num_qubits
    similarity = np.eye(dim) + 0.3 * rng.normal(size=(dim, dim))
    eigenvalues = np.sort(rng.normal(size=dim))
    return SparsePauliOp.from_operator(
        Operator(similarity @ np.diag(eigenvalues) @ np.linalg.inv(similarity))
    )


def finite_difference_system(ansatz, hamiltonian, values, eps=1e-6):
    """The metric tensor and evolution gradient from finite differences of statevectors."""
    matrix = hamiltonian.to_matrix()

    def state(x):
        return Statevector(ansatz.assign_parameters(x)).data

    psi = state(values)
    derivatives = []
    for i in range(len(values)):
        shift = np.zeros(len(values))
        shift[i] = eps
        derivatives.append((state(values + shift) - state(values - shift)) / (2 * eps))
    derivatives = np.asarray(derivatives)
    overlaps = derivatives.conj() @ psi
    metric = np.real(derivatives.conj() @ derivatives.T - np.outer(overlaps, overlaps.conj()))
    energy = np.vdot(psi, matrix @ psi)
    gradient = -np.real(derivatives.conj() @ (matrix @ psi) - overlaps * energy)
    return metric, gradient


def mixed_circuit() -> QuantumCircuit:
    """A circuit with various supported instructions, shared parameters and a global phase."""
    theta = ParameterVector("t", 4)
    circuit = QuantumCircuit(3, global_phase=0.3 * theta[0])
    circuit.h([0, 1, 2])
    circuit.p(theta[0], 1)
    circuit.rzz(2 * theta[1] + 0.1, 0, 2)
    circuit.cx(1, 2)
    # non-commuting terms, synthesized with two first-order Trotter steps
    evolution = PauliEvolutionGate(
        SparsePauliOp(["XYI", "ZZX", "IYZ"], [0.5, -0.3, 0.8]),
        theta[2],
        synthesis=LieTrotter(reps=2),
    )
    circuit.append(evolution, [0, 1, 2])
    circuit.crz(theta[3], 0, 1)
    circuit.ry(theta[1], 2)
    return circuit


@ddt
class TestStatevectorImaginaryMcLachlanPrinciple(QiskitNatureTestCase):
    """Test the StatevectorImaginaryMcLachlanPrinciple."""

    def _uccsd_case(self, mapper_name: str):
        fcidump = FCIDump.from_file(DATA / "lih_4nos_tc.fcidump")
        problem = fcidump_to_problem(fcidump, TCDump.from_file(DATA / "lih_4nos_tc.tcdump"))
        if mapper_name == "parity":
            mapper = ParityMapper(num_particles=problem.num_particles)
        else:
            mapper = JordanWignerMapper()
        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper),
        )
        return ansatz, mapper.map(problem.hamiltonian.second_q_op())

    @data("efficient_su2", "mixed", "uccsd_parity", "uccsd_jordan_wigner")
    def test_against_reverse_mode(self, case):
        """Test the metric tensor and evolution gradient against finite differences and the
        reverse-mode gradients of qiskit-algorithms."""
        if case == "efficient_su2":
            ansatz, hamiltonian = efficient_su2(2, reps=2), non_hermitian_operator(2)
        elif case == "mixed":
            ansatz, hamiltonian = mixed_circuit(), non_hermitian_operator(3, seed=1)
        else:
            ansatz, hamiltonian = self._uccsd_case(case.split("_", 1)[1])
        values = np.random.default_rng(2).uniform(-1, 1, ansatz.num_parameters)

        principle = StatevectorImaginaryMcLachlanPrinciple()
        metric = principle.metric_tensor(ansatz, values)
        gradient = principle.evolution_gradient(hamiltonian, ansatz, values)

        fd_metric, fd_gradient = finite_difference_system(ansatz, hamiltonian, values)
        np.testing.assert_allclose(metric, fd_metric, atol=1e-7)
        np.testing.assert_allclose(gradient, fd_gradient, atol=1e-7)
        if case != "mixed":
            # the reverse-mode gradients of qiskit-algorithms do not support all gates
            reference = ImaginaryMcLachlanPrinciple(ReverseQGT(), ReverseEstimatorGradient())
            np.testing.assert_allclose(metric, reference.metric_tensor(ansatz, values), atol=1e-12)
            np.testing.assert_allclose(
                gradient, reference.evolution_gradient(hamiltonian, ansatz, values), atol=1e-12
            )
        np.testing.assert_allclose(
            principle.statevector(ansatz, values),
            Statevector(ansatz.assign_parameters(values)).data,
            atol=1e-12,
        )

    def test_gradient_parameters(self):
        """Test computing the gradient with respect to a subset of the parameters."""
        ansatz = efficient_su2(2, reps=1)
        hamiltonian = non_hermitian_operator(2)
        values = np.linspace(0.1, 0.8, ansatz.num_parameters)
        principle = StatevectorImaginaryMcLachlanPrinciple()
        full = principle.evolution_gradient(hamiltonian, ansatz, values)
        subset = [ansatz.parameters[3], ansatz.parameters[1]]
        np.testing.assert_allclose(
            principle.evolution_gradient(hamiltonian, ansatz, values, subset), full[[3, 1]]
        )

    def test_supports(self):
        """Test the check for supported circuits."""
        self.assertTrue(StatevectorImaginaryMcLachlanPrinciple.supports(mixed_circuit()))
        circuit = efficient_su2(2, reps=1)
        circuit.measure_all()
        self.assertFalse(StatevectorImaginaryMcLachlanPrinciple.supports(circuit))

    def test_var_qite_h2(self):
        """Test that the VarQITE converges to the transcorrelated H2 ground state."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"),
            TCDump.from_file(DATA / "h2_sto6g_tc.tcdump", 2),
        )
        mapper = ParityMapper(num_particles=problem.num_particles)
        ansatz = UCCSD(
            2, problem.num_particles, mapper, initial_state=HartreeFock(2, (1, 1), mapper)
        )
        hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
        constant = problem.hamiltonian.nuclear_repulsion_energy

        # default: forward Euler steps until the parameters converge
        result = VarQITEMinimumEigensolver(ansatz).compute_minimum_eigenvalue(
            hamiltonian, {"hamiltonian": hamiltonian}
        )
        self.assertTrue(result.converged)
        self.assertLess(result.times[-1], 1000.0)
        self.assertAlmostEqual(result.eigenvalue + constant, -1.1717073880, delta=1e-6)
        self.assertAlmostEqual(result.aux_operators_evaluated["hamiltonian"][0], result.eigenvalue)

        # a tighter convergence threshold
        result = VarQITEMinimumEigensolver(
            ansatz, convergence_threshold=1e-9
        ).compute_minimum_eigenvalue(hamiltonian)
        self.assertAlmostEqual(result.eigenvalue + constant, -1.1717073880, places=8)

        # an adaptive ODE solver up to a fixed imaginary time
        result = VarQITEMinimumEigensolver(
            ansatz, time=10.0, ode_solver="RK45"
        ).compute_minimum_eigenvalue(hamiltonian)
        self.assertFalse(result.converged)
        self.assertAlmostEqual(result.times[-1], 10.0)
        self.assertAlmostEqual(result.eigenvalue + constant, -1.1717073880, places=7)


if __name__ == "__main__":
    unittest.main()
