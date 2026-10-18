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

"""Test ground state solvers for non-Hermitian (transcorrelated) Hamiltonians."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from qiskit import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import (
    MinimumEigensolver,
    MinimumEigensolverResult,
    NumPyMinimumEigensolver,
    VQE,
)
from qiskit_algorithms.optimizers import SLSQP

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.algorithms import (
    GroundStateEigensolver,
    VarQITEMinimumEigensolver,
    VarQITEMinimumEigensolverResult,
    is_hermitian_qubit_operator,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.tcdump import TCDump
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.problems import EigenstateResult

DATA = Path(__file__).parents[2] / "formats" / "tcdump"

# lowest eigenvalue of the published transcorrelated H2 qubit Hamiltonian, including the core energy
H2_TC_ENERGY = -1.1717073880


class _MinimumEigensolverWithoutAnsatz(MinimumEigensolver):
    """A minimum eigensolver which neither supports non-Hermitian operators nor has an ansatz."""

    def compute_minimum_eigenvalue(self, operator, aux_operators=None):
        return MinimumEigensolverResult()


class _MinimumEigensolverWithoutAux(MinimumEigensolver):
    """A minimum eigensolver with an ansatz which does not support auxiliary operators."""

    def __init__(self, ansatz):
        self.ansatz = ansatz

    @classmethod
    def supports_aux_operators(cls):
        return False

    def compute_minimum_eigenvalue(self, operator, aux_operators=None):
        return MinimumEigensolverResult()


class TestNonHermitianGroundState(QiskitNatureTestCase):
    """Test ground state solvers for non-Hermitian (transcorrelated) Hamiltonians."""

    def setUp(self):
        super().setUp()
        self.problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"),
            TCDump.from_file(DATA / "h2_sto6g_tc.tcdump", 2),
        )
        self.mapper = ParityMapper(num_particles=self.problem.num_particles)
        self.ansatz = UCCSD(
            self.problem.num_spatial_orbitals,
            self.problem.num_particles,
            self.mapper,
            initial_state=HartreeFock(
                self.problem.num_spatial_orbitals, self.problem.num_particles, self.mapper
            ),
        )

    def test_is_hermitian_qubit_operator(self):
        """Test the hermiticity check of qubit operators."""
        self.assertTrue(is_hermitian_qubit_operator(SparsePauliOp(["XY", "ZZ"], [0.5, -1.0])))
        self.assertTrue(is_hermitian_qubit_operator(SparsePauliOp(["XY"], [0.5 + 1e-17j])))
        self.assertFalse(is_hermitian_qubit_operator(SparsePauliOp(["XY", "ZZ"], [0.5, 0.1j])))
        qubit_op = self.mapper.map(self.problem.hamiltonian.second_q_op())
        self.assertFalse(is_hermitian_qubit_operator(qubit_op))

    def test_numpy_minimum_eigensolver(self):
        """Test the exact solver with the default filter criterion of the problem."""
        solver = GroundStateEigensolver(
            self.mapper,
            NumPyMinimumEigensolver(filter_criterion=self.problem.get_default_filter_criterion()),
        )
        result = solver.solve(self.problem)
        self.assertAlmostEqual(result.total_energies[0], H2_TC_ENERGY, places=8)

    def test_automatic_var_qite(self):
        """Test that the VQE is replaced by the VarQITE for non-Hermitian Hamiltonians."""
        solver = GroundStateEigensolver(
            self.mapper, VQE(StatevectorEstimator(), self.ansatz, SLSQP())
        )
        with self.assertWarns(UserWarning):
            result = solver.solve(self.problem)
        self.assertIsInstance(result.raw_result, VarQITEMinimumEigensolverResult)
        self.assertAlmostEqual(result.total_energies[0], H2_TC_ENERGY, delta=1e-4)
        self.assertAlmostEqual(result.num_particles[0], 2.0, places=6)
        # the energies along the imaginary time evolution are available
        self.assertEqual(len(result.raw_result.energies), len(result.raw_result.times))

    def test_explicit_non_hermitian_solver(self):
        """Test that a provided solver for non-Hermitian operators is used."""
        non_hermitian_solver = VarQITEMinimumEigensolver(self.ansatz, time=8.0)
        solver = GroundStateEigensolver(
            self.mapper,
            VQE(StatevectorEstimator(), self.ansatz, SLSQP()),
            non_hermitian_solver=non_hermitian_solver,
        )
        result = solver.solve(self.problem)
        self.assertIsInstance(result.raw_result, VarQITEMinimumEigensolverResult)
        self.assertAlmostEqual(result.total_energies[0], H2_TC_ENERGY, delta=1e-5)

    def test_shot_based_var_qite(self):
        """Test the VarQITE with an estimator subject to shot noise."""
        estimator = StatevectorEstimator(default_precision=1e-3, seed=7)
        result = VarQITEMinimumEigensolver(
            self.ansatz, time=3.0, estimator=estimator, convergence_threshold=None
        ).compute_minimum_eigenvalue(self.mapper.map(self.problem.hamiltonian.second_q_op()))
        energy = result.eigenvalue + self.problem.hamiltonian.nuclear_repulsion_energy
        self.assertAlmostEqual(energy, H2_TC_ENERGY, delta=5e-3)

    def test_var_qite_with_transpiler(self):
        """Test the VarQITE with circuits transpiled for a backend with a non-trivial layout."""
        backend = GenericBackendV2(num_qubits=4, coupling_map=[[0, 3], [3, 2], [2, 1]], seed=3)
        pass_manager = generate_preset_pass_manager(
            backend=backend, optimization_level=1, seed_transpiler=5
        )
        # the circuits are not placed on the physical qubits [0, 1]
        self.assertNotEqual(pass_manager.run(self.ansatz).layout.final_index_layout(), [0, 1])
        hamiltonian = self.mapper.map(self.problem.hamiltonian.second_q_op())
        kwargs = {"time": 0.5, "convergence_threshold": None}
        reference = VarQITEMinimumEigensolver(self.ansatz, **kwargs).compute_minimum_eigenvalue(
            hamiltonian
        )
        estimator = StatevectorEstimator(default_precision=1e-5, seed=3)
        result = VarQITEMinimumEigensolver(
            self.ansatz, estimator=estimator, transpiler=pass_manager, **kwargs
        ).compute_minimum_eigenvalue(hamiltonian)
        np.testing.assert_allclose(result.optimal_point, reference.optimal_point, atol=1e-3)
        self.assertAlmostEqual(result.eigenvalue, reference.eigenvalue, delta=1e-3)

    def test_unsupported_solver_raises(self):
        """Test that a solver without support for non-Hermitian operators raises."""
        solver = GroundStateEigensolver(self.mapper, _MinimumEigensolverWithoutAnsatz())
        with self.assertRaises(QiskitNatureError):
            solver.solve(self.problem)

    def test_hermitian_problem_keeps_solver(self):
        """Test that the original solver is used for Hermitian Hamiltonians."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA.parent / "fcidump" / "test_fcidump_h2.fcidump")
        )
        mapper = ParityMapper(num_particles=problem.num_particles)
        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper),
        )
        self.assertTrue(is_hermitian_qubit_operator(mapper.map(problem.hamiltonian.second_q_op())))
        solver = GroundStateEigensolver(mapper, VQE(StatevectorEstimator(), ansatz, SLSQP()))
        result = solver.solve(problem)
        self.assertNotIsInstance(result.raw_result, VarQITEMinimumEigensolverResult)

    def test_aux_operators_with_replacement_solver(self):
        """Test that the auxiliary operators are evaluated by the VarQITE which replaces a solver
        without support for them."""
        solver = GroundStateEigensolver(self.mapper, _MinimumEigensolverWithoutAux(self.ansatz))
        with self.assertWarns(UserWarning):
            result = solver.solve(self.problem)
        self.assertIsNotNone(result.aux_operators_evaluated)
        self.assertAlmostEqual(result.num_particles[0], sum(self.problem.num_particles), places=6)

    def test_hermiticity_check_of_other_types(self):
        """Test that operators which cannot be inspected count as Hermitian."""
        self.assertTrue(is_hermitian_qubit_operator(SparsePauliOp(["X"], [Parameter("a")])))
        self.assertTrue(is_hermitian_qubit_operator(None))

    def test_interpret_warns_about_complex_eigenvalues(self):
        """Test that discarding non-negligible imaginary parts of eigenvalues is reported."""
        result = EigenstateResult()
        result.eigenvalues = np.array([-1.0 + 0.01j])
        with self.assertLogs(
            "qiskit_nature.second_q.problems.electronic_structure_problem", "WARNING"
        ):
            interpreted = self.problem.interpret(result)
        self.assertAlmostEqual(interpreted.computed_energies[0], -1.0)

    def test_var_qite_aux_operators(self):
        """Test the evaluation of auxiliary operators by the VarQITE minimum eigensolver."""
        qubit_op = self.mapper.map(self.problem.hamiltonian.second_q_op())
        aux = {"identity": SparsePauliOp("I" * qubit_op.num_qubits), "hamiltonian": qubit_op}
        result = VarQITEMinimumEigensolver(self.ansatz, time=2.0).compute_minimum_eigenvalue(
            qubit_op, aux
        )
        self.assertAlmostEqual(result.aux_operators_evaluated["identity"][0], 1.0)
        self.assertAlmostEqual(result.aux_operators_evaluated["hamiltonian"][0], result.eigenvalue)

        result = VarQITEMinimumEigensolver(self.ansatz, time=2.0).compute_minimum_eigenvalue(
            qubit_op, [None, aux["identity"]]
        )
        self.assertIsNone(result.aux_operators_evaluated[0])
        self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 1.0)


if __name__ == "__main__":
    unittest.main()
