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
from qiskit.circuit.library import efficient_su2
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
    """An exact principle with a constant bias of the gradient and the metric and a damping of the
    gradient."""

    def __init__(self, gradient_bias, metric_bias, damping=1.0):
        super().__init__()
        self.gradient_bias = gradient_bias
        self.metric_bias = metric_bias
        self.damping = damping
        self.num_gradient_calls = 0

    def metric_tensor(self, ansatz, param_values):
        return super().metric_tensor(ansatz, param_values) + self.metric_bias

    def evolution_gradient(self, hamiltonian, ansatz, param_values, gradient_params=None):
        self.num_gradient_calls += 1
        exact = super().evolution_gradient(hamiltonian, ansatz, param_values, gradient_params)
        return self.damping * exact + self.gradient_bias


class TestReferenceStateMitigatedPrinciple(QiskitNatureTestCase):
    """Test the ReferenceStateMitigatedPrinciple."""

    def setUp(self):
        super().setUp()
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        mapper = JordanWignerMapper()
        self.operator = mapper.map(problem.hamiltonian.second_q_op())
        # the Hartree-Fock state is a stationary point for the hardware-efficient layers ...
        self.ansatz = HartreeFock(2, (1, 1), mapper).compose(
            efficient_su2(4, su2_gates=["ry"], reps=1, entanglement="linear")
        )
        # ... but not for the UCCSD ansatz (the double excitation has a non-zero gradient)
        self.uccsd = UCCSD(2, (1, 1), mapper, initial_state=HartreeFock(2, (1, 1), mapper))
        self.values = np.random.default_rng(3).normal(scale=0.3, size=self.ansatz.num_parameters)
        self.gradient_bias = 0.02 * np.random.default_rng(4).normal(size=self.ansatz.num_parameters)
        self.metric_bias = 0.01 * np.ones((self.ansatz.num_parameters,) * 2)

    def test_stationary_references(self):
        """Test the classification of the reference states."""
        exact = StatevectorImaginaryMcLachlanPrinciple()
        zero = np.zeros(self.ansatz.num_parameters)
        self.assertLess(
            np.linalg.norm(exact.evolution_gradient(self.operator, self.ansatz, zero)), 1e-12
        )
        zero = np.zeros(self.uccsd.num_parameters)
        self.assertGreater(
            np.linalg.norm(exact.evolution_gradient(self.operator, self.uccsd, zero)), 0.1
        )

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
        self.assertTrue(principle.gradient_correction_active)
        self.assertLess(principle.reference_gradient_norm, 1e-12)
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
            biased = BiasedPrinciple(self.gradient_bias, 0.0, damping=0.8)
            principle = ReferenceStateMitigatedPrinciple(biased) if mitigate else biased
            results[mitigate] = VarQITEMinimumEigensolver(
                self.ansatz, time=50.0, variational_principle=principle
            ).compute_minimum_eigenvalue(self.operator)
        self.assertGreater(abs(results[False].eigenvalue - reference.eigenvalue), 1e-4)
        self.assertAlmostEqual(results[True].eigenvalue, reference.eigenvalue, delta=1e-6)

    def test_non_stationary_reference(self):
        """Test that the correction is disabled, with a warning, for a non-stationary reference."""
        values = np.array([0.1, -0.2, 0.3])
        bias = np.array([0.02, -0.01, 0.03])
        biased = BiasedPrinciple(bias, 0.0, damping=0.7)
        principle = ReferenceStateMitigatedPrinciple(biased)
        with self.assertWarns(UserWarning):
            gradient = principle.evolution_gradient(self.operator, self.uccsd, values)
        self.assertFalse(principle.gradient_correction_active)
        self.assertGreater(principle.reference_gradient_norm, 0.1)
        np.testing.assert_allclose(
            gradient, biased.evolution_gradient(self.operator, self.uccsd, values)
        )
        # the noisy reference gradient is not measured in this case
        self.assertEqual(biased.num_gradient_calls, 2)
        # the check can be disabled, in which case the offset at the reference is subtracted
        forced = ReferenceStateMitigatedPrinciple(biased, reference_gradient_tolerance=np.inf)
        forced.evolution_gradient(self.operator, self.uccsd, values)
        self.assertTrue(forced.gradient_correction_active)
        exact = StatevectorImaginaryMcLachlanPrinciple()
        expected = bias + (0.7 - 1.0) * np.real(
            exact.evolution_gradient(self.operator, self.uccsd, np.zeros(3))
        )
        np.testing.assert_allclose(forced.gradient_offset, expected, atol=1e-12)

    def test_stationary_reference_point(self):
        """Test a UCCSD ansatz with the (stationary) converged point as reference."""
        reference = VarQITEMinimumEigensolver(
            self.uccsd, convergence_threshold=1e-8
        ).compute_minimum_eigenvalue(self.operator)
        bias = np.array([0.02, -0.01, 0.03])
        principle = ReferenceStateMitigatedPrinciple(
            BiasedPrinciple(bias, 0.0, damping=0.8), reference_point=reference.optimal_point
        )
        result = VarQITEMinimumEigensolver(
            self.uccsd, time=50.0, variational_principle=principle
        ).compute_minimum_eigenvalue(self.operator)
        self.assertTrue(principle.gradient_correction_active)
        self.assertAlmostEqual(result.eigenvalue, reference.eigenvalue, delta=1e-6)

    def test_noiseless_estimator(self):
        """Test that the offsets vanish for an exact estimator."""
        estimator = StatevectorEstimator()
        noisy = ImaginaryMcLachlanPrinciple(
            LinCombQGT(estimator), LinCombEstimatorGradient(estimator)
        )
        principle = ReferenceStateMitigatedPrinciple(noisy, mitigate_metric=True)
        principle.evolution_gradient(self.operator, self.ansatz, self.values)
        principle.metric_tensor(self.ansatz, self.values)
        self.assertTrue(principle.gradient_correction_active)
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
        self.assertIsNone(principle.gradient_correction_active)
        # the ApproximateMetricPrinciple resets a wrapped ReferenceStateMitigatedPrinciple
        wrapper = ApproximateMetricPrinciple(principle)
        principle.evolution_gradient(self.operator, self.ansatz, self.values)
        wrapper.reset()
        self.assertIsNone(principle.gradient_offset)
        with self.assertRaises(ValueError):
            ReferenceStateMitigatedPrinciple(biased, num_reference_evaluations=0)
        with self.assertRaises(ValueError):
            ReferenceStateMitigatedPrinciple(biased, reference_gradient_tolerance=-1.0)


if __name__ == "__main__":
    unittest.main()
