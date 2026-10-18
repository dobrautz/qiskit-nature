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

"""Test the ApproximateMetricPrinciple."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

from qiskit_nature.second_q.algorithms import (
    ApproximateMetricPrinciple,
    StatevectorImaginaryMcLachlanPrinciple,
    VarQITEMinimumEigensolver,
    parameter_layers,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.mappers import JordanWignerMapper

DATA = Path(__file__).parents[2] / "formats" / "tcdump"


@ddt
class TestApproximateMetricPrinciple(QiskitNatureTestCase):
    """Test the ApproximateMetricPrinciple."""

    def setUp(self):
        super().setUp()
        self.ansatz = efficient_su2(2, reps=1)
        self.values = np.random.default_rng(3).normal(size=self.ansatz.num_parameters)
        self.hamiltonian = SparsePauliOp(["ZZ", "XI", "IX", "YY"], [0.5, 0.3 + 0.1j, 0.3, -0.2j])

    def test_parameter_layers(self):
        """Test the layers of parameters."""
        self.assertEqual(parameter_layers(self.ansatz), [[0, 1], [2, 3], [4, 5], [6, 7]])

    @data("block_diagonal", "diagonal")
    def test_partial_metric(self, metric):
        """Test that the block-diagonal and diagonal metrics are parts of the full metric."""
        full = np.asarray(ImaginaryMcLachlanPrinciple().metric_tensor(self.ansatz, self.values))
        blocks = parameter_layers(self.ansatz) if metric == "block_diagonal" else None
        mask = np.zeros_like(full, dtype=bool)
        for block in blocks or [[index] for index in range(len(full))]:
            mask[np.ix_(block, block)] = True
        for inner in (ImaginaryMcLachlanPrinciple(), StatevectorImaginaryMcLachlanPrinciple()):
            principle = ApproximateMetricPrinciple(inner, metric=metric, regularization=0.0)
            approximation = principle.metric_tensor(self.ansatz, self.values)
            np.testing.assert_allclose(approximation, np.where(mask, full, 0.0), atol=1e-10)
            self.assertEqual(principle.num_metric_entries, 12 if blocks else 8)

    def test_qbroyden_update(self):
        """Test the qBroyden update of the metric."""
        principle = ApproximateMetricPrinciple(
            StatevectorImaginaryMcLachlanPrinciple(), epsilon=0.2, regularization=0.0
        )
        metric = principle.metric_tensor(self.ansatz, self.values)
        gradient = principle.evolution_gradient(self.hamiltonian, self.ansatz, self.values)
        expected = (1 - 0.2) * metric + 0.2 * np.outer(gradient, gradient)
        updated = principle.metric_tensor(self.ansatz, self.values + 0.1)
        np.testing.assert_allclose(updated, expected)
        gradient = principle.evolution_gradient(self.hamiltonian, self.ansatz, self.values + 0.1)
        expected = (1 - 0.1) * expected + 0.1 * np.outer(gradient, gradient)
        np.testing.assert_allclose(principle.metric_tensor(self.ansatz, self.values), expected)
        self.assertEqual(principle.num_metric_evaluations, 1)
        # a different ansatz resets the principle
        principle.metric_tensor(efficient_su2(2, reps=1), self.values)
        self.assertEqual(principle.num_metric_evaluations, 1)
        self.assertEqual(principle.num_metric_entries, 36)

    def test_recompute_interval(self):
        """Test that the metric is evaluated anew after the given number of steps."""
        principle = ApproximateMetricPrinciple(
            StatevectorImaginaryMcLachlanPrinciple(), update=None, recompute_interval=2
        )
        for _ in range(5):
            principle.metric_tensor(self.ansatz, self.values)
        self.assertEqual(principle.num_metric_evaluations, 3)

    def test_full_metric(self):
        """Test that the full metric of every step reproduces the wrapped principle."""
        inner = StatevectorImaginaryMcLachlanPrinciple()
        principle = ApproximateMetricPrinciple(
            inner, update=None, recompute_interval=1, regularization=0.0
        )
        reference = VarQITEMinimumEigensolver(
            self.ansatz, self.values, time=0.5, variational_principle=inner
        ).compute_minimum_eigenvalue(self.hamiltonian)
        result = VarQITEMinimumEigensolver(
            self.ansatz, self.values, time=0.5, variational_principle=principle
        ).compute_minimum_eigenvalue(self.hamiltonian)
        np.testing.assert_allclose(result.energies, reference.energies, atol=1e-12)
        self.assertEqual(principle.num_metric_evaluations, 10)

    @data("full", "identity")
    def test_same_solution(self, metric):
        """Test that the qBroyden evolution converges to the same state (C = 0)."""
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        mapper = JordanWignerMapper()
        ansatz = UCCSD(2, (1, 1), mapper, initial_state=HartreeFock(2, (1, 1), mapper))
        operator = mapper.map(problem.hamiltonian.second_q_op())
        reference = VarQITEMinimumEigensolver(ansatz).compute_minimum_eigenvalue(operator)
        principle = ApproximateMetricPrinciple(
            StatevectorImaginaryMcLachlanPrinciple(), metric=metric
        )
        result = VarQITEMinimumEigensolver(
            ansatz, variational_principle=principle
        ).compute_minimum_eigenvalue(operator)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.eigenvalue, reference.eigenvalue, delta=1e-6)
        self.assertLess(len(result.evolution_result.times), len(reference.evolution_result.times))

    def test_invalid_input(self):
        """Test that invalid input raises."""
        with self.assertRaises(ValueError):
            ApproximateMetricPrinciple(metric="tridiagonal")
        with self.assertRaises(ValueError):
            ApproximateMetricPrinciple(update="bfgs")
        with self.assertRaises(ValueError):
            ApproximateMetricPrinciple(recompute_interval=0)
        principle = ApproximateMetricPrinciple(metric="block_diagonal", blocks=[[0, 1], [1]])
        with self.assertRaises(ValueError):
            principle.metric_tensor(self.ansatz, self.values)


if __name__ == "__main__":
    unittest.main()
