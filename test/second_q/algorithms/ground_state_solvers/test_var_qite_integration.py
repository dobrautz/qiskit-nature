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

"""Test the line-search and shifted implicit integration of VarQITE."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.algorithms import (
    FermionicVarQITESolver,
    StatevectorImaginaryMcLachlanPrinciple,
    VarQITEMinimumEigensolver,
)
from qiskit_nature.second_q.algorithms.ground_state_solvers.var_qite_integration import (
    step_options,
    tangent_space_shift,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.tcdump import TCDump
from qiskit_nature.second_q.hamiltonians import TranscorrelatedFermiHubbardModel
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.problems import LatticeModelProblem

DATA = Path(__file__).parents[2] / "formats" / "tcdump"

LABELS = ["IX", "IY", "IZ", "XI", "YI", "ZI", "XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]


def random_operator(hermitian: bool, seed: int = 3) -> SparsePauliOp:
    """A random two-qubit operator, Hermitian or not."""
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=len(LABELS))
    if not hermitian:
        coeffs = coeffs + 0.3j * rng.normal(size=len(LABELS))
    return SparsePauliOp(LABELS, coeffs)


@ddt
class TestVarQITEIntegration(QiskitNatureTestCase):
    """Test the line-search and shifted implicit integration of VarQITE."""

    def setUp(self):
        super().setUp()
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"),
            TCDump.from_file(DATA / "h2_sto6g_tc.tcdump", 2),
        )
        mapper = ParityMapper(num_particles=problem.num_particles)
        self.ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper),
        )
        # the transcorrelated (non-Hermitian) qubit Hamiltonian and its Hermitian part
        self.tc_operator = mapper.map(problem.hamiltonian.second_q_op())
        self.hermitian_operator = SparsePauliOp(
            self.tc_operator.paulis, np.real(self.tc_operator.coeffs)
        ).simplify()

    @data(True, False)
    def test_tangent_hamiltonian(self, hermitian):
        """Test the tangent-space Hamiltonian against finite-difference tangent vectors."""
        operator = random_operator(hermitian)
        ansatz = efficient_su2(2, su2_gates=["ry", "rz"], reps=1)
        values = np.random.default_rng(5).normal(size=ansatz.num_parameters)
        principle = StatevectorImaginaryMcLachlanPrinciple()
        tangent = principle.tangent_hamiltonian(operator, ansatz, values)

        def state(point):
            return Statevector(ansatz.assign_parameters(point)).data

        eps = 1e-6
        psi = state(values)
        derivatives = np.array(
            [
                (state(values + eps * unit) - state(values - eps * unit)) / (2 * eps)
                for unit in np.eye(len(values))
            ]
        )
        tangents = derivatives - np.outer(derivatives @ psi.conj(), psi)
        matrix = operator.to_matrix()
        energy = np.vdot(psi, matrix @ psi)
        expected = np.real(tangents.conj() @ (matrix @ tangents.T - energy * tangents.T))
        np.testing.assert_allclose(tangent, expected, atol=1e-7)

    def test_tangent_space_shift(self):
        """Test that the shifted implicit matrix is bounded below by the metric."""
        principle = StatevectorImaginaryMcLachlanPrinciple()
        operator = random_operator(True)
        ansatz = efficient_su2(2, su2_gates=["ry", "rz"], reps=2)
        values = np.random.default_rng(1).normal(size=ansatz.num_parameters)
        metric = principle.metric_tensor(ansatz, values)
        tangent = principle.tangent_hamiltonian(operator, ansatz, values)
        shift = tangent_space_shift(tangent, metric)
        self.assertLess(shift, 0.0)
        for step in (0.5, 5.0, 500.0):
            matrix = metric * (1 - step * shift) + step * tangent
            # (matrix - metric) is positive semidefinite on the tangent space
            eigenvalues, vectors = np.linalg.eigh(metric)
            basis = vectors[:, eigenvalues > 1e-7 * eigenvalues[-1]]
            self.assertGreater(np.linalg.eigvalsh(basis.T @ (matrix - metric) @ basis)[0], -1e-9)
        self.assertEqual(tangent_space_shift(tangent - shift * metric, metric), 0.0)

    @data(True, False)
    def test_implicit(self, hermitian):
        """Test that implicit steps reach the result of the Euler integration in fewer steps."""
        operator = self.hermitian_operator if hermitian else self.tc_operator
        euler = VarQITEMinimumEigensolver(self.ansatz).compute_minimum_eigenvalue(operator)
        implicit = VarQITEMinimumEigensolver(self.ansatz, implicit=True).compute_minimum_eigenvalue(
            operator
        )
        self.assertTrue(implicit.converged)
        self.assertAlmostEqual(complex(implicit.eigenvalue), complex(euler.eigenvalue), delta=1e-6)
        self.assertLess(
            len(implicit.evolution_result.times), len(euler.evolution_result.times) / 10
        )

    def test_line_search(self):
        """Test that the line search gives monotonically decreasing energies in fewer steps."""
        euler = VarQITEMinimumEigensolver(self.ansatz).compute_minimum_eigenvalue(
            self.hermitian_operator
        )
        result = VarQITEMinimumEigensolver(
            self.ansatz, line_search=True, num_trajectory_points=None
        ).compute_minimum_eigenvalue(self.hermitian_operator)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.eigenvalue, euler.eigenvalue, delta=1e-6)
        self.assertTrue(np.all(np.diff(result.energies) <= 1e-12))
        self.assertLess(len(result.evolution_result.times), len(euler.evolution_result.times))

    def test_trust_region(self):
        """Test that the trust region bounds the Fubini-Study length of the steps."""
        result = VarQITEMinimumEigensolver(
            self.ansatz, timestep=1.0, trust_region=0.05, time=5.0
        ).compute_minimum_eigenvalue(self.hermitian_operator)
        principle = StatevectorImaginaryMcLachlanPrinciple()
        points = result.evolution_result.parameter_values
        for start, end in zip(points[:-1], points[1:]):
            metric = principle.metric_tensor(self.ansatz, start)
            delta = np.asarray(end) - np.asarray(start)
            self.assertLessEqual(np.sqrt(delta @ metric @ delta), 0.05 * (1 + 1e-9))

    def test_line_search_with_estimator(self):
        """Test the line search with an estimator-based principle, a transpiler and step
        halving."""
        # a device layout, such that the observable has to follow the layout of the ansatz
        pass_manager = generate_preset_pass_manager(
            optimization_level=1, backend=GenericBackendV2(4, seed=3), seed_transpiler=7
        )
        exact = VarQITEMinimumEigensolver(
            self.ansatz, line_search=True, timestep=50.0, time=200.0
        ).compute_minimum_eigenvalue(self.hermitian_operator)
        result = VarQITEMinimumEigensolver(
            self.ansatz,
            line_search=True,
            timestep=50.0,
            time=200.0,
            estimator=StatevectorEstimator(),
            variational_principle=ImaginaryMcLachlanPrinciple(),
            transpiler=pass_manager,
        ).compute_minimum_eigenvalue(self.hermitian_operator)
        # the first trial step of 50 is too long and gets halved (six times)
        self.assertLess(result.times[1] - result.times[0], 1.0)
        self.assertTrue(np.all(np.diff(np.real(result.energies)) <= 1e-12))
        np.testing.assert_allclose(result.times, exact.times, atol=1e-8)
        self.assertAlmostEqual(result.eigenvalue, exact.eigenvalue, places=8)

        # steps which fail the sufficient-decrease test after all halvings are taken anyway
        long_steps = VarQITEMinimumEigensolver(
            self.ansatz, line_search=True, timestep=1000.0, time=4000.0
        ).compute_minimum_eigenvalue(self.hermitian_operator)
        self.assertAlmostEqual(long_steps.eigenvalue, exact.eigenvalue, places=6)

    def test_default_principle_for_unsupported_ansatz(self):
        """Test that ansätze not supported by the statevector principle use the estimator-based
        principle."""
        theta = Parameter("t")
        ansatz = QuantumCircuit(2)
        ansatz.ry(theta * theta, 0)
        ansatz.cx(0, 1)
        self.assertFalse(StatevectorImaginaryMcLachlanPrinciple.supports(ansatz))
        operator = SparsePauliOp(["ZZ", "XX", "ZI"], [0.5, 0.3, -0.7])
        result = VarQITEMinimumEigensolver(ansatz, [0.4], time=1.0).compute_minimum_eigenvalue(
            operator
        )
        reference = VarQITEMinimumEigensolver(
            ansatz, [0.4], time=1.0, variational_principle=ImaginaryMcLachlanPrinciple()
        ).compute_minimum_eigenvalue(operator)
        np.testing.assert_allclose(result.optimal_point, reference.optimal_point, atol=1e-12)

    def test_options_and_shift_edge_cases(self):
        """Test the default options and the shift for a vanishing metric."""
        self.assertEqual(step_options(None, True, False, None), (5.0, 0.2))
        self.assertEqual(step_options(None, True, False, np.inf), (5.0, None))
        self.assertEqual(step_options(None, False, True, None), (1.0, None))
        self.assertEqual(step_options(0.1, False, False, 0.5), (0.1, 0.5))
        self.assertEqual(tangent_space_shift(np.eye(2), np.zeros((2, 2))), 0.0)

    def test_unsupported_options(self):
        """Test that unsupported combinations raise."""
        with self.assertRaises(QiskitNatureError):
            VarQITEMinimumEigensolver(self.ansatz, line_search=True).compute_minimum_eigenvalue(
                self.tc_operator
            )
        with self.assertRaises(QiskitNatureError):
            VarQITEMinimumEigensolver(
                self.ansatz, implicit=True, variational_principle=ImaginaryMcLachlanPrinciple()
            ).compute_minimum_eigenvalue(self.hermitian_operator)
        with self.assertRaises(QiskitNatureError):
            VarQITEMinimumEigensolver(
                self.ansatz, implicit=True, ode_solver="RK45"
            ).compute_minimum_eigenvalue(self.hermitian_operator)

    @data(1, 2, 5)
    def test_trajectory_points(self, num_points):
        """Test that the subset of trajectory points always includes the final one."""
        full = VarQITEMinimumEigensolver(
            self.ansatz, time=1.0, convergence_threshold=None
        ).compute_minimum_eigenvalue(self.tc_operator)
        result = VarQITEMinimumEigensolver(
            self.ansatz, time=1.0, convergence_threshold=None, num_trajectory_points=num_points
        ).compute_minimum_eigenvalue(self.tc_operator)
        self.assertEqual(len(result.energies), num_points)
        self.assertAlmostEqual(result.times[-1], full.times[-1])
        self.assertAlmostEqual(result.eigenvalue, full.eigenvalue, places=12)
        with self.assertRaises(ValueError):
            VarQITEMinimumEigensolver(self.ansatz, num_trajectory_points=0)
        if _optionals.HAS_FFSIM:
            problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
            mapper = JordanWignerMapper()
            norb, nelec = problem.num_spatial_orbitals, problem.num_particles
            ansatz = UCCSD(norb, nelec, mapper, initial_state=HartreeFock(norb, nelec, mapper))
            fermionic = FermionicVarQITESolver(
                ansatz, time=1.0, convergence_threshold=None, num_trajectory_points=num_points
            ).solve(problem)
            fermionic_full = FermionicVarQITESolver(
                ansatz, time=1.0, convergence_threshold=None
            ).solve(problem)
            self.assertEqual(len(fermionic.raw_result.energies), num_points)
            self.assertAlmostEqual(
                fermionic.raw_result.eigenvalue, fermionic_full.raw_result.eigenvalue, places=12
            )
            with self.assertRaises(ValueError):
                FermionicVarQITESolver(ansatz, num_trajectory_points=0)

    @unittest.skipUnless(_optionals.HAS_FFSIM, "ffsim not available.")
    def test_fermionic_solver_agrees_with_qubit_simulation(self):
        """Test the implicit steps of the FermionicVarQITESolver against the qubit simulation."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "lih_4nos_tc.fcidump"),
            TCDump.from_file(DATA / "lih_4nos_tc.tcdump"),
        )
        mapper = JordanWignerMapper()
        norb, nelec = problem.num_spatial_orbitals, problem.num_particles
        ansatz = UCCSD(norb, nelec, mapper, initial_state=HartreeFock(norb, nelec, mapper))
        result = FermionicVarQITESolver(ansatz, implicit=True, time=20.0).solve(problem)
        reference = VarQITEMinimumEigensolver(
            ansatz, implicit=True, time=20.0
        ).compute_minimum_eigenvalue(mapper.map(problem.hamiltonian.second_q_op()))
        np.testing.assert_allclose(result.raw_result.times, reference.times)
        np.testing.assert_allclose(result.raw_result.energies, reference.energies, atol=1e-10)
        np.testing.assert_allclose(
            result.raw_result.optimal_point, reference.optimal_point, atol=1e-10
        )

    @unittest.skipUnless(_optionals.HAS_FFSIM, "ffsim not available.")
    def test_fermionic_solver_line_search(self):
        """Test the line search of the FermionicVarQITESolver (Hermitian LiH)."""
        problem = fcidump_to_problem(
            FCIDump.from_file(
                Path(__file__).parents[2] / "formats" / "fcidump" / "test_fcidump_lih.fcidump"
            )
        )
        mapper = JordanWignerMapper()
        norb, nelec = problem.num_spatial_orbitals, problem.num_particles
        ansatz = UCCSD(norb, nelec, mapper, initial_state=HartreeFock(norb, nelec, mapper))
        implicit = FermionicVarQITESolver(ansatz, implicit=True).solve(problem)
        result = FermionicVarQITESolver(ansatz, line_search=True, num_trajectory_points=None).solve(
            problem
        )
        self.assertTrue(result.raw_result.converged and implicit.raw_result.converged)
        self.assertAlmostEqual(result.total_energies[0], implicit.total_energies[0], delta=1e-8)
        self.assertTrue(np.all(np.diff(result.raw_result.energies) <= 1e-12))
        with self.assertRaises(QiskitNatureError):
            FermionicVarQITESolver(ansatz, line_search=True).solve(
                LatticeModelProblem(
                    TranscorrelatedFermiHubbardModel(4, correlation=-0.73, basis="momentum")
                )
            )


if __name__ == "__main__":
    unittest.main()
