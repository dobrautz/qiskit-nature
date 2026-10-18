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

"""Test the FermionicVarQITESolver."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2

import qiskit_nature.optionals as _optionals
from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.algorithms import FermionicVarQITESolver, VarQITEMinimumEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, SlaterDeterminant, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.tcdump import TCDump
from qiskit_nature.second_q.hamiltonians import TranscorrelatedFermiHubbardModel
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitMapper
from qiskit_nature.second_q.problems import LatticeModelProblem, LatticeModelResult

DATA = Path(__file__).parents[2] / "formats" / "tcdump"


def uccsd(problem, mapper):
    """The UCCSD ansatz with a Hartree-Fock initial state."""
    norb, nelec = problem.num_spatial_orbitals, problem.num_particles
    return UCCSD(norb, nelec, mapper, initial_state=HartreeFock(norb, nelec, mapper))


@ddt
@unittest.skipUnless(_optionals.HAS_FFSIM, "ffsim not available.")
class TestFermionicVarQITESolver(QiskitNatureTestCase):
    """Test the FermionicVarQITESolver."""

    @data("jordan_wigner", "parity")
    def test_agrees_with_qubit_simulation(self, mapper_name):
        """Test that the trajectory agrees with the simulation of the qubit circuit."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "lih_4nos_tc.fcidump"),
            TCDump.from_file(DATA / "lih_4nos_tc.tcdump"),
        )
        mapper: QubitMapper
        if mapper_name == "parity":
            mapper = ParityMapper(num_particles=problem.num_particles)
        else:
            mapper = JordanWignerMapper()
        ansatz = uccsd(problem, mapper)

        result = FermionicVarQITESolver(ansatz, time=2.0, convergence_threshold=None).solve(problem)
        reference = VarQITEMinimumEigensolver(
            ansatz, time=2.0, convergence_threshold=None
        ).compute_minimum_eigenvalue(mapper.map(problem.hamiltonian.second_q_op()))

        np.testing.assert_allclose(result.raw_result.times, reference.times)
        np.testing.assert_allclose(result.raw_result.energies, reference.energies, atol=1e-12)
        np.testing.assert_allclose(
            result.raw_result.optimal_point, reference.optimal_point, atol=1e-12
        )
        self.assertAlmostEqual(
            result.total_energies[0],
            reference.eigenvalue + problem.hamiltonian.nuclear_repulsion_energy,
        )

    def test_three_body(self):
        """Test the transcorrelated Be Hamiltonian with three-body terms."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "be_sto6g_tc.fcidump"),
            TCDump.from_file(DATA / "be_sto6g_tc.tcdump"),
        )
        result = FermionicVarQITESolver(uccsd(problem, JordanWignerMapper())).solve(problem)
        self.assertTrue(result.raw_result.converged)
        # the UCCSD ansatz is within a few microhartree of the TC-FCI energy of -14.66465358
        self.assertAlmostEqual(result.total_energies[0], -14.66465358, delta=5e-6)

    def test_lattice_model(self):
        """Test the transcorrelated Hubbard model in the momentum basis."""
        problem = LatticeModelProblem(
            TranscorrelatedFermiHubbardModel(4, correlation=-0.73, basis="momentum")
        )
        mapper = JordanWignerMapper()
        ansatz = UCCSD(4, (2, 2), mapper, initial_state=HartreeFock(4, (2, 2), mapper))
        result = FermionicVarQITESolver(ansatz).solve(problem)
        self.assertIsInstance(result, LatticeModelResult)
        self.assertTrue(result.raw_result.converged)
        # the exact ground-state energy is -2.10274848 (PRR 5, 023174, Fig. 4)
        self.assertAlmostEqual(result.eigenvalues[0], -2.10274848, delta=2e-4)

    def test_slater_determinant(self):
        """Test an initial state of delocalized orbitals against the qubit simulation."""
        problem = LatticeModelProblem(TranscorrelatedFermiHubbardModel(2, correlation=-0.48))
        orbitals = np.zeros((2, 4))
        orbitals[0, :2] = [np.cos(0.3), np.sin(0.3)]
        orbitals[1, 2:] = [np.cos(0.7), -np.sin(0.7)]
        mapper = JordanWignerMapper()
        ansatz = UCCSD(2, (1, 1), mapper, initial_state=SlaterDeterminant(orbitals, mapper))

        result = FermionicVarQITESolver(ansatz, time=1.0, convergence_threshold=None).solve(problem)
        reference = VarQITEMinimumEigensolver(
            ansatz, time=1.0, convergence_threshold=None
        ).compute_minimum_eigenvalue(mapper.map(problem.hamiltonian.second_q_op()))

        np.testing.assert_allclose(result.raw_result.energies, reference.energies, atol=1e-12)
        np.testing.assert_allclose(
            result.raw_result.optimal_point, reference.optimal_point, atol=1e-12
        )

    def test_solver_interface(self):
        """Test the methods of the ground state solver interface."""
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        mapper = JordanWignerMapper()
        solver = FermionicVarQITESolver(uccsd(problem, mapper))
        self.assertIsNone(solver.solver)
        self.assertFalse(solver.supports_aux_operators())
        operator, aux_operators = solver.get_qubit_operators(problem)
        self.assertIsNone(aux_operators)
        self.assertTrue(operator.equiv(mapper.map(problem.hamiltonian.second_q_op())))

    def test_invalid_ansatz(self):
        """Test that unsupported UCC ansätze raise."""
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        mapper = JordanWignerMapper()
        cases = {
            "complex": UCCSD(
                2,
                (1, 1),
                mapper,
                initial_state=HartreeFock(2, (1, 1), mapper),
                include_imaginary=True,
            ),
            "orbitals": UCCSD(3, (1, 1), mapper, initial_state=HartreeFock(3, (1, 1), mapper)),
        }
        for name, ansatz in cases.items():
            with self.subTest(name):
                with self.assertRaises(QiskitNatureError):
                    FermionicVarQITESolver(ansatz).solve(problem)

    def test_unsupported_input(self):
        """Test that unsupported input raises."""
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        with self.assertRaises(QiskitNatureError):
            FermionicVarQITESolver(efficient_su2(4))
        mapper = JordanWignerMapper()
        solver = FermionicVarQITESolver(uccsd(problem, mapper))
        with self.assertRaises(QiskitNatureError):
            solver.solve(problem, {"op": mapper.map(problem.hamiltonian.second_q_op())})
        # an initial state with the wrong number of particles
        initial_state = QuantumCircuit(4)
        initial_state.x(0)
        ansatz = UCCSD(2, (1, 1), mapper, initial_state=initial_state)
        with self.assertRaises(QiskitNatureError):
            FermionicVarQITESolver(ansatz).solve(problem)


if __name__ == "__main__":
    unittest.main()
