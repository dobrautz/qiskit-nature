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

"""Test the ReferenceStateMitigatedPrinciple."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import LinCombEstimatorGradient, LinCombQGT
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

from qiskit_nature.second_q.algorithms import (
    ApproximateMetricPrinciple,
    ReferenceStateMitigatedPrinciple,
    StatevectorImaginaryMcLachlanPrinciple,
    VarQITEMinimumEigensolver,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.mappers import JordanWignerMapper

DATA = Path(__file__).parents[2] / "formats" / "tcdump"


class BiasedPrinciple(StatevectorImaginaryMcLachlanPrinciple):
    """An exact principle with constant biases of the gradient and the metric."""

    def __init__(self, gradient_bias, metric_bias):
        super().__init__()
        self.gradient_bias = gradient_bias
        self.metric_bias = metric_bias
        self.num_gradient_calls = 0

    def metric_tensor(self, ansatz, param_values):
        return super().metric_tensor(ansatz, param_values) + self.metric_bias

    def evolution_gradient(self, hamiltonian, ansatz, param_values, gradient_params=None):
        self.num_gradient_calls += 1
        return (
            super().evolution_gradient(hamiltonian, ansatz, param_values, gradient_params)
            + self.gradient_bias
        )


class TestReferenceStateMitigatedPrinciple(QiskitNatureTestCase):
    """Test the ReferenceStateMitigatedPrinciple."""

    def setUp(self):
        super().setUp()
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        mapper = JordanWignerMapper()
        self.operator = mapper.map(problem.hamiltonian.second_q_op())
        self.ansatz = UCCSD(2, (1, 1), mapper, initial_state=HartreeFock(2, (1, 1), mapper))
        self.values = np.array([0.1, -0.2, 0.3])
        self.gradient_bias = np.array([0.02, -0.01, 0.03])
        self.metric_bias = 0.01 * np.ones((3, 3))

    def test_constant_bias(self):
        """Test that constant biases of the gradient and the metric are removed."""
        exact = StatevectorImaginaryMcLachlanPrinciple()
        biased = BiasedPrinciple(self.gradient_bias, self.metric_bias)
        principle = ReferenceStateMitigatedPrinciple(biased, mitigate_metric=True)
        np.testing.assert_allclose(
            principle.evolution_gradient(self.operator, self.ansatz, self.values),
            np.real(exact.evolution_gradient(self.operator, self.ansatz, self.values)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            principle.metric_tensor(self.ansatz, self.values),
            exact.metric_tensor(self.ansatz, self.values),
            atol=1e-12,
        )
        np.testing.assert_allclose(principle.gradient_offset, self.gradient_bias, atol=1e-12)
        np.testing.assert_allclose(principle.metric_offset, self.metric_bias, atol=1e-12)
        # without mitigation of the metric, it is passed through
        principle = ReferenceStateMitigatedPrinciple(biased)
        np.testing.assert_allclose(
            principle.metric_tensor(self.ansatz, self.values),
            exact.metric_tensor(self.ansatz, self.values) + self.metric_bias,
        )

    def test_converged_state(self):
        """Test that the biased evolution converges to the exact state after mitigation."""
        reference = VarQITEMinimumEigensolver(self.ansatz).compute_minimum_eigenvalue(self.operator)
        results = {}
        for mitigate in (False, True):
            biased = BiasedPrinciple(self.gradient_bias, 0.0)
            principle = ReferenceStateMitigatedPrinciple(biased) if mitigate else biased
            results[mitigate] = VarQITEMinimumEigensolver(
                self.ansatz, time=50.0, variational_principle=principle
            ).compute_minimum_eigenvalue(self.operator)
        self.assertGreater(abs(results[False].eigenvalue - reference.eigenvalue), 1e-4)
        self.assertAlmostEqual(results[True].eigenvalue, reference.eigenvalue, delta=1e-6)

    def test_noiseless_estimator(self):
        """Test that the offsets vanish for an exact estimator."""
        estimator = StatevectorEstimator()
        noisy = ImaginaryMcLachlanPrinciple(
            LinCombQGT(estimator), LinCombEstimatorGradient(estimator)
        )
        principle = ReferenceStateMitigatedPrinciple(
            noisy, reference_point=[0.2, 0.1, -0.1], mitigate_metric=True
        )
        principle.evolution_gradient(self.operator, self.ansatz, self.values)
        principle.metric_tensor(self.ansatz, self.values)
        np.testing.assert_allclose(principle.gradient_offset, 0.0, atol=1e-10)
        np.testing.assert_allclose(principle.metric_offset, 0.0, atol=1e-10)

    def test_reference_evaluations_and_reset(self):
        """Test the averaging of the reference evaluations and the reset of the offsets."""
        biased = BiasedPrinciple(self.gradient_bias, 0.0)
        principle = ReferenceStateMitigatedPrinciple(biased, num_reference_evaluations=3)
        principle.evolution_gradient(self.operator, self.ansatz, self.values)
        principle.evolution_gradient(self.operator, self.ansatz, self.values)
        self.assertEqual(biased.num_gradient_calls, 3 + 2)
        # a different ansatz and a reset discard the offsets
        principle.evolution_gradient(self.operator, self.ansatz.copy(), self.values)
        self.assertEqual(biased.num_gradient_calls, 3 + 2 + 3 + 1)
        principle.reset()
        self.assertIsNone(principle.gradient_offset)
        # the ApproximateMetricPrinciple resets a wrapped ReferenceStateMitigatedPrinciple
        wrapper = ApproximateMetricPrinciple(principle)
        principle.evolution_gradient(self.operator, self.ansatz, self.values)
        wrapper.reset()
        self.assertIsNone(principle.gradient_offset)
        with self.assertRaises(ValueError):
            ReferenceStateMitigatedPrinciple(biased, num_reference_evaluations=0)


if __name__ == "__main__":
    unittest.main()
