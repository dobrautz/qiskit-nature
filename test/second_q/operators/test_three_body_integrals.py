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

"""Test ThreeBodyIntegrals."""

import itertools
import unittest
from test import QiskitNatureTestCase

import numpy as np

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.operators.three_body_integrals import (
    S48_PERMUTATIONS,
    ThreeBodyIntegrals,
)


def symmetric_three_body_tensor(norb: int, seed: int = 0) -> np.ndarray:
    """A random three-body tensor with the 48-fold permutational symmetry."""
    tensor = np.random.default_rng(seed).normal(size=(norb,) * 6)
    return sum(tensor.transpose(perm) for perm in S48_PERMUTATIONS) / len(S48_PERMUTATIONS)


def reference_operator(tensor: np.ndarray) -> FermionicOp:
    """The operator 1/2 sum L_pqrstu a+_p a+_q a+_r a_u a_t a_s summed over all indices and spins."""
    # pylint: disable=invalid-name
    norb = tensor.shape[0]
    data = {}
    for index in itertools.product(range(norb), repeat=6):
        value = tensor[index]
        if value == 0.0:
            continue
        p, q, r, s, t, u = index
        for sigma, tau, lam in itertools.product((0, norb), repeat=3):
            label = f"+_{p + sigma} +_{q + tau} +_{r + lam} -_{u + lam} -_{t + tau} -_{s + sigma}"
            data[label] = data.get(label, 0.0) + 0.5 * value
    return FermionicOp(data, num_spin_orbitals=2 * norb)


class TestThreeBodyIntegrals(QiskitNatureTestCase):
    """Test ThreeBodyIntegrals."""

    def test_symmetry_group(self):
        """Test the 48-fold symmetry group."""
        self.assertEqual(len(S48_PERMUTATIONS), 48)
        self.assertIn((0, 1, 2, 3, 4, 5), S48_PERMUTATIONS)
        # electron permutation and exchange of the indices of the first electron
        self.assertIn((1, 0, 2, 4, 3, 5), S48_PERMUTATIONS)
        self.assertIn((3, 1, 2, 0, 4, 5), S48_PERMUTATIONS)
        # the bra-ket pairing of a chemist-like ordering is not a symmetry
        self.assertNotIn((1, 0, 2, 3, 4, 5), S48_PERMUTATIONS)

    def test_reduction(self):
        """Test that only the symmetry-unique integrals are stored."""
        tensor = symmetric_three_body_tensor(3)
        indices = np.asarray(list(np.ndindex(tensor.shape)))
        integrals = ThreeBodyIntegrals(indices, tensor[tuple(indices.T)], 3)
        self.assertEqual(integrals.symmetry, "s48")
        # number of multisets of three electron index pairs (p <= s)
        num_pairs = 3 * 4 // 2
        self.assertEqual(len(integrals), num_pairs * (num_pairs + 1) * (num_pairs + 2) // 6)
        np.testing.assert_allclose(integrals.to_dense(), tensor, atol=1e-14)
        self.assertAlmostEqual(integrals[0, 1, 2, 2, 0, 1], tensor[0, 1, 2, 2, 0, 1])
        self.assertTrue(integrals.is_hermitian())

        # the reduced listing yields the same integrals
        reduced = ThreeBodyIntegrals(integrals.indices, integrals.values, 3, symmetry="s48")
        np.testing.assert_allclose(reduced.to_dense(), tensor, atol=1e-14)

    def test_fallback_without_symmetry(self):
        """Test that integrals without the 48-fold symmetry are stored as given."""
        tensor = np.random.default_rng(1).normal(size=(2,) * 6)
        indices = np.asarray(list(np.ndindex(tensor.shape)))
        with self.assertRaises(ValueError):
            ThreeBodyIntegrals(indices, tensor[tuple(indices.T)], 2, symmetry="s48")
        with self.assertLogs("qiskit_nature", level="WARNING"):
            integrals = ThreeBodyIntegrals(indices, tensor[tuple(indices.T)], 2)
        self.assertEqual(integrals.symmetry, "s1")
        np.testing.assert_allclose(integrals.to_dense(), tensor)
        self.assertFalse(integrals.is_hermitian())

    def test_second_q_op(self):
        """Test the second-quantized operator against a direct construction."""
        mapper = JordanWignerMapper()
        for symmetry, tensor in (
            ("s48", symmetric_three_body_tensor(3, seed=2)),
            ("s1", np.random.default_rng(3).normal(size=(2,) * 6)),
        ):
            with self.subTest(symmetry):
                norb = tensor.shape[0]
                indices = np.asarray(list(np.ndindex(tensor.shape)))
                integrals = ThreeBodyIntegrals(
                    indices, tensor[tuple(indices.T)], norb, symmetry=symmetry
                )
                operator = integrals.second_q_op()
                self.assertEqual(operator.num_spin_orbitals, 2 * norb)
                expected = reference_operator(tensor)
                np.testing.assert_allclose(
                    mapper.map(operator).to_matrix(),
                    mapper.map(expected).to_matrix(),
                    atol=1e-12,
                )

    def test_invalid_input(self):
        """Test invalid input."""
        with self.assertRaises(ValueError):
            ThreeBodyIntegrals(np.array([[0, 0, 0, 0, 0, 2]]), np.array([1.0]), 2)
        with self.assertRaises(ValueError):
            ThreeBodyIntegrals(np.zeros((2, 6), dtype=int), np.array([1.0]), 2)
        with self.assertRaises(ValueError):
            ThreeBodyIntegrals(np.zeros((1, 6), dtype=int), np.array([1.0]), 2, symmetry="s8")


if __name__ == "__main__":
    unittest.main()
