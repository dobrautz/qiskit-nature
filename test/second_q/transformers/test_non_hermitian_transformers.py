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

"""Test non-Hermitian (e.g. transcorrelated) electronic integrals and their transformation."""

import itertools
import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt

from qiskit_nature.second_q.formats import MoleculeInfo, fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.tcdump import TCDump
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer

DATA = Path(__file__).parents[1] / "formats" / "tcdump"


def sector_matrix(matrix: np.ndarray, norb: int, occupied: list[int], nalpha: int, nbeta: int):
    """Restricts a block-spin-ordered Jordan-Wigner matrix to the determinants with the given
    particle numbers in which the given spatial orbitals are doubly occupied."""
    indices = np.arange(matrix.shape[0])
    bits = (indices[:, None] >> np.arange(2 * norb)[None, :]) & 1
    mask = (bits[:, :norb].sum(1) == nalpha) & (bits[:, norb:].sum(1) == nbeta)
    for orbital in occupied:
        mask &= (bits[:, orbital] == 1) & (bits[:, norb + orbital] == 1)
    sector = np.flatnonzero(mask)
    return matrix[np.ix_(sector, sector)]


def sorted_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """The eigenvalues sorted by real and imaginary part."""
    eigenvalues = np.linalg.eigvals(matrix)
    return eigenvalues[np.lexsort((eigenvalues.imag.round(8), eigenvalues.real.round(8)))]


@ddt
class TestNonHermitianIntegrals(QiskitNatureTestCase):
    """Test non-Hermitian electronic integrals."""

    @data("physicist", "chemist")
    def test_unrestricted_second_q_op(self, ordering):
        """Test the operator of non-Hermitian unrestricted-spin integrals against a direct
        construction."""
        # pylint: disable=invalid-name
        norb = 2
        rng = np.random.default_rng(0)
        h1_a, h1_b = rng.normal(size=(2, norb, norb))
        # two-body integrals in physicist' order with the electron exchange symmetry
        h2_aa, h2_bb = rng.normal(size=(2,) + (norb,) * 4)
        h2_aa = 0.5 * (h2_aa + h2_aa.transpose(1, 0, 3, 2))
        h2_bb = 0.5 * (h2_bb + h2_bb.transpose(1, 0, 3, 2))
        h2_ba = rng.normal(size=(norb,) * 4)

        data_ = {}
        for p, q in itertools.product(range(norb), repeat=2):
            data_[f"+_{p} -_{q}"] = h1_a[p, q]
            data_[f"+_{p + norb} -_{q + norb}"] = h1_b[p, q]
        for p, q, r, s in itertools.product(range(norb), repeat=4):
            for label, value in (
                (f"+_{p} +_{q} -_{r} -_{s}", 0.5 * h2_aa[p, q, r, s]),
                (f"+_{p + norb} +_{q + norb} -_{r + norb} -_{s + norb}", 0.5 * h2_bb[p, q, r, s]),
                # the beta-alpha and alpha-beta terms are identical operators
                (f"+_{p + norb} +_{q} -_{r} -_{s + norb}", h2_ba[p, q, r, s]),
            ):
                data_[label] = data_.get(label, 0.0) + value
        expected = FermionicOp(data_, num_spin_orbitals=2 * norb)

        if ordering == "physicist":
            energy = ElectronicEnergy.from_raw_integrals(
                h1_a, h2_aa, h1_b, h2_bb, h2_ba, auto_index_order=False
            )
        else:
            # chemist' order (ps|qr) = <pq|rs>
            energy = ElectronicEnergy.from_raw_integrals(
                h1_a,
                S1Integrals(h2_aa.transpose(0, 3, 1, 2)),
                h1_b,
                S1Integrals(h2_bb.transpose(0, 3, 1, 2)),
                S1Integrals(h2_ba.transpose(0, 3, 1, 2)),
            )
        mapper = JordanWignerMapper()
        np.testing.assert_allclose(
            mapper.map(energy.second_q_op()).to_matrix(),
            mapper.map(expected).to_matrix(),
            atol=1e-12,
        )


class TestNonHermitianTransformers(QiskitNatureTestCase):
    """Test the active-space reduction of non-Hermitian (transcorrelated) Hamiltonians."""

    def setUp(self):
        super().setUp()
        # the one- and two-body part of the transcorrelated Be Hamiltonian
        self.problem = fcidump_to_problem(FCIDump.from_file(DATA / "be_sto6g_tc.fcidump"))
        mapper = JordanWignerMapper()
        self.full_matrix = mapper.map(self.problem.hamiltonian.second_q_op()).to_matrix()

    def _check(self, transformer, frozen, norb_active, num_active):
        reduced = transformer.transform(self.problem)
        reduced_matrix = JordanWignerMapper().map(reduced.hamiltonian.second_q_op()).to_matrix()
        shift = sum(reduced.hamiltonian.constants.values())
        actual = sorted_eigenvalues(
            sector_matrix(reduced_matrix, norb_active, [], num_active, num_active)
        )
        expected = sorted_eigenvalues(sector_matrix(self.full_matrix, 5, frozen, 2, 2))
        np.testing.assert_allclose(actual + shift, expected, atol=1e-10)

    def test_active_space_transformer(self):
        """Test that freezing the 1s orbital is exact for the non-Hermitian Hamiltonian."""
        self._check(ActiveSpaceTransformer(2, 4), [0], 4, 1)

    def test_freeze_core_transformer(self):
        """Test the FreezeCoreTransformer for the non-Hermitian Hamiltonian."""
        self.problem.molecule = MoleculeInfo(symbols=["Be"], coords=[(0.0, 0.0, 0.0)])
        self._check(FreezeCoreTransformer(), [0], 4, 1)

    def test_three_body_raises(self):
        """Test that the reduction of three-body terms raises."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "be_sto6g_tc.fcidump"),
            TCDump.from_file(DATA / "be_sto6g_tc.tcdump"),
        )
        with self.assertRaises(NotImplementedError):
            ActiveSpaceTransformer(2, 4).transform(problem)


if __name__ == "__main__":
    unittest.main()
