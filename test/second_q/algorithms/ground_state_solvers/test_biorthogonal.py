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

"""Test the biorthogonal expectation values."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import generate_preset_pass_manager

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.algorithms import (
    FermionicVarQITESolver,
    biorthogonal_expectation_value,
    transition_circuit,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.hamiltonians import TranscorrelatedFermiHubbardModel
from qiskit_nature.second_q.mappers import JordanWignerMapper

DATA = Path(__file__).parents[2] / "formats" / "tcdump"


@ddt
class TestBiorthogonal(QiskitNatureTestCase):
    """Test the biorthogonal expectation values."""

    def setUp(self):
        super().setUp()
        self.mapper = JordanWignerMapper()
        self.operator = self.mapper.map(
            TranscorrelatedFermiHubbardModel(2, correlation=-0.48).second_q_op()
        )
        self.rng = np.random.default_rng(5)

    def ansatz(self, name):
        """The ansatz of the given name."""
        if name == "uccsd":
            return UCCSD(2, (1, 1), self.mapper, initial_state=HartreeFock(2, (1, 1), self.mapper))
        return efficient_su2(4, reps=1)

    @data("uccsd", "efficient_su2")
    def test_transition_circuit(self, name):
        """Test the state prepared by the transition circuit."""
        ansatz = self.ansatz(name)
        left, right = self.rng.normal(size=(2, ansatz.num_parameters))
        circuit = transition_circuit(ansatz, left, right)
        expected = np.concatenate(
            [
                Statevector(ansatz.assign_parameters(left)).data,
                Statevector(ansatz.assign_parameters(right)).data,
            ]
        ) / np.sqrt(2)
        state = Statevector(circuit).data
        # the overall phase is irrelevant
        phase = np.vdot(expected, state)
        self.assertAlmostEqual(abs(phase), 1.0)
        np.testing.assert_allclose(state, phase * expected, atol=1e-10)
        # one controlled rotation per parameterized rotation
        self.assertNotIn("crz", transition_circuit(ansatz, left, left).count_ops())

    @data("uccsd", "efficient_su2")
    def test_estimator(self, name):
        """Test the measurement of the biorthogonal expectation value with an estimator."""
        ansatz = self.ansatz(name)
        left, right = self.rng.normal(size=(2, ansatz.num_parameters))
        left_state = Statevector(ansatz.assign_parameters(left)).data
        right_state = Statevector(ansatz.assign_parameters(right)).data
        expected = np.vdot(left_state, self.operator.to_matrix() @ right_state) / np.vdot(
            left_state, right_state
        )
        exact = biorthogonal_expectation_value(self.operator, ansatz, left, right)
        self.assertAlmostEqual(exact, expected)
        estimated = biorthogonal_expectation_value(
            self.operator, ansatz, left, right, StatevectorEstimator()
        )
        self.assertAlmostEqual(estimated, expected)
        # for equal states, it is the expectation value
        self.assertAlmostEqual(
            biorthogonal_expectation_value(self.operator, ansatz, right, right),
            np.vdot(right_state, self.operator.to_matrix() @ right_state),
        )

    def test_transpiler(self):
        """Test the measurement of a circuit compiled for a backend."""
        ansatz = self.ansatz("uccsd")
        left, right = self.rng.normal(size=(2, ansatz.num_parameters))
        backend = GenericBackendV2(6, coupling_map=[[i, i + 1] for i in range(5)], seed=2)
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        estimated = biorthogonal_expectation_value(
            self.operator,
            ansatz,
            left,
            right,
            StatevectorEstimator(),
            transpiler=pass_manager,
        )
        self.assertAlmostEqual(
            estimated, biorthogonal_expectation_value(self.operator, ansatz, left, right)
        )

    def test_parameter_dependent_global_phase(self):
        """Test an ansatz whose decomposition has a parameter-dependent global phase, which is a
        relative phase of the two branches of the transition circuit."""
        theta = ParameterVector("t", 3)
        circuit = QuantumCircuit(2)
        circuit.ry(theta[0], 0)
        circuit.cx(0, 1)
        circuit.p(theta[1], 1)
        circuit.ry(theta[2], 1)
        operator = SparsePauliOp(["XZ", "ZI", "YY"], [0.4, 1.0 + 0.3j, -0.2])
        left, right = np.array([0.3, 0.9, -0.4]), np.array([0.5, -0.6, 0.2])
        exact = biorthogonal_expectation_value(operator, circuit, left, right)
        measured = biorthogonal_expectation_value(
            operator, circuit, left, right, StatevectorEstimator()
        )
        self.assertAlmostEqual(measured, exact, places=10)

    def test_orthogonal_states_raise(self):
        """Test that orthogonal left and right states raise."""
        circuit = QuantumCircuit(1)
        circuit.ry(Parameter("t"), 0)
        with self.assertRaises(QiskitNatureError):
            biorthogonal_expectation_value(SparsePauliOp("Z"), circuit, [0.0], [np.pi])

    def test_invalid_input(self):
        """Test that invalid input raises."""
        ansatz = self.ansatz("efficient_su2")
        with self.assertRaises(QiskitNatureError):
            transition_circuit(ansatz, np.zeros(3), np.zeros(ansatz.num_parameters))

    @unittest.skipUnless(_optionals.HAS_FFSIM, "ffsim not available.")
    def test_left_eigenvector(self):
        """Test that the biorthogonal energy of the VarQITE states is of second order."""
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        norb, nelec = problem.num_spatial_orbitals, problem.num_particles
        ansatz = UCCSD(
            norb, nelec, self.mapper, initial_state=HartreeFock(norb, nelec, self.mapper)
        )
        right = FermionicVarQITESolver(ansatz, convergence_threshold=1e-5).solve(problem)
        left = FermionicVarQITESolver(ansatz, convergence_threshold=1e-5, adjoint=True).solve(
            problem
        )
        operator = self.mapper.map(problem.hamiltonian.second_q_op())
        eigenvalues = np.linalg.eigvals(operator.to_matrix())
        exact = min(eigenvalues, key=np.real).real
        right_point = right.raw_result.optimal_point
        left_point = left.raw_result.optimal_point
        energy = FermionicVarQITESolver(ansatz).biorthogonal_energy(
            problem, left_point, right_point
        )
        self.assertAlmostEqual(
            energy, biorthogonal_expectation_value(operator, ansatz, left_point, right_point)
        )
        right_error = abs(right.raw_result.eigenvalue - exact)
        self.assertGreater(right_error, 1e-9)
        self.assertLess(abs(energy - exact), right_error**1.5)


if __name__ == "__main__":
    unittest.main()
