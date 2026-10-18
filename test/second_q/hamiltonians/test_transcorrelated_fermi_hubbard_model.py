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

"""Test TranscorrelatedFermiHubbardModel."""

import unittest
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt, unpack

from qiskit_nature.second_q.hamiltonians import (
    FermiHubbardModel,
    TranscorrelatedFermiHubbardModel,
)
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import LatticeModelProblem


def matrix(model):
    """The dense Jordan-Wigner matrix of the model."""
    return JordanWignerMapper().map(model.second_q_op()).to_matrix()


def sector_eigenvalues(mat, num_sites, num_alpha, num_beta):
    """The eigenvalues in the sector of the given numbers of spin-up and spin-down particles."""
    mask = (1 << num_sites) - 1
    indices = [
        index
        for index in range(mat.shape[0])
        if (index & mask).bit_count() == num_alpha and (index >> num_sites).bit_count() == num_beta
    ]
    return np.linalg.eigvals(mat[np.ix_(indices, indices)])


@ddt
class TestTranscorrelatedFermiHubbardModel(QiskitNatureTestCase):
    """Test TranscorrelatedFermiHubbardModel."""

    @data(("periodic", 2, -0.48), ("periodic", 3, 0.4), ("open", 3, -0.7))
    @unpack
    def test_similarity_transformation(self, boundary, num_sites, correlation):
        """Test that the model is the similarity transformed Hubbard Hamiltonian."""
        hubbard = matrix(TranscorrelatedFermiHubbardModel(num_sites, 1.3, 2.5, boundary=boundary))
        model = TranscorrelatedFermiHubbardModel(
            num_sites, 1.3, 2.5, correlation, boundary=boundary
        )
        occupations = (np.arange(4**num_sites)[:, None] >> np.arange(2 * num_sites)) & 1
        correlator = correlation * np.sum(
            occupations[:, :num_sites] * occupations[:, num_sites:], axis=1
        )
        expected = np.exp(-correlator)[:, None] * hubbard * np.exp(correlator)[None, :]
        np.testing.assert_allclose(matrix(model), expected, atol=1e-12)
        self.assertEqual(model.register_length, 2 * num_sites)

    def test_hubbard_model(self):
        """Test that the model without correlation agrees with the FermiHubbardModel."""
        lattice = LineLattice(3, edge_parameter=-1.0, boundary_condition=BoundaryCondition.PERIODIC)
        expected = np.linalg.eigvalsh(matrix(FermiHubbardModel(lattice, onsite_interaction=4.0)))
        eigenvalues = np.linalg.eigvalsh(matrix(TranscorrelatedFermiHubbardModel(3)))
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-12)

    @data(2, 3, 4)
    def test_momentum_basis(self, num_sites):
        """Test the momentum-space representation."""
        real = matrix(TranscorrelatedFermiHubbardModel(num_sites, correlation=-0.6))
        momentum = matrix(
            TranscorrelatedFermiHubbardModel(num_sites, correlation=-0.6, basis="momentum")
        )
        np.testing.assert_allclose(
            np.sort_complex(np.round(np.linalg.eigvals(momentum), 8)),
            np.sort_complex(np.round(np.linalg.eigvals(real), 8)),
            atol=1e-7,
        )
        # without interaction, the momentum orbitals are ordered by their band energy
        kinetic = TranscorrelatedFermiHubbardModel(
            num_sites, onsite_interaction=0.0, basis="momentum"
        )
        # (for two sites, the doubled bond is included in the band energies -2t cos(k))
        orbital_energies = -2 * np.cos(kinetic.momenta)
        occupations = (np.arange(4**num_sites)[:, None] >> np.arange(2 * num_sites)) & 1
        expected = occupations @ np.concatenate([orbital_energies, orbital_energies])
        np.testing.assert_allclose(matrix(kinetic), np.diag(expected), atol=1e-12)
        self.assertTrue(np.all(np.diff(orbital_energies) >= -1e-12))

    @data((2, -0.48, -2.472), (4, -0.73, -2.103))
    @unpack
    def test_ground_state_energies(self, num_sites, correlation, expected):
        """Test the ground-state energies at half filling (Sokolov et al., PRR 5, 023174)."""
        for basis in ("real", "momentum"):
            model = TranscorrelatedFermiHubbardModel(num_sites, 1.0, 4.0, correlation, basis=basis)
            eigenvalues = sector_eigenvalues(
                matrix(model), num_sites, num_sites // 2, num_sites // 2
            )
            ground_state_energy = eigenvalues[np.argmin(eigenvalues.real)]
            self.assertAlmostEqual(ground_state_energy, expected, places=3)

    def test_problem(self):
        """Test the model in a LatticeModelProblem."""
        model = TranscorrelatedFermiHubbardModel(4, correlation=-0.73, basis="momentum")
        problem = LatticeModelProblem(model)
        self.assertIs(problem.hamiltonian, model)
        self.assertEqual(model.lattice.num_nodes, 4)
        self.assertFalse(model.second_q_op().is_hermitian())

    def test_invalid_input(self):
        """Test that invalid input raises."""
        with self.assertRaises(ValueError):
            TranscorrelatedFermiHubbardModel(1)
        with self.assertRaises(ValueError):
            TranscorrelatedFermiHubbardModel(4, boundary="open", basis="momentum")
        with self.assertRaises(ValueError):
            TranscorrelatedFermiHubbardModel(4, basis="position")


if __name__ == "__main__":
    unittest.main()
