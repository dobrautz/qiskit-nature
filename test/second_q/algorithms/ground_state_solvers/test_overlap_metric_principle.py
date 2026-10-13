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

"""Test the OverlapMetricPrinciple."""

import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from ddt import data, ddt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.algorithms import (
    OverlapMetricPrinciple,
    StatevectorImaginaryMcLachlanPrinciple,
    VarQITEMinimumEigensolver,
    parameter_frequencies,
)
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.mappers import JordanWignerMapper

DATA = Path(__file__).parents[2] / "formats" / "tcdump"


def mixed_circuit():
    """A circuit with Pauli evolutions, rotations, phase and controlled gates."""
    theta = ParameterVector("t", 6)
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.ry(theta[0], 1)
    hop = SparsePauliOp(["XX", "YY"], [0.5, 0.5])
    circuit.append(PauliEvolutionGate(hop, time=theta[1]), [0, 1])
    circuit.append(PauliEvolutionGate(hop, time=theta[2]), [1, 2])
    circuit.rzz(2 * theta[3], 0, 2)
    circuit.p(theta[4], 1)
    circuit.crz(theta[5], 2, 0)
    circuit.h(2)
    return circuit


@ddt
class TestOverlapMetricPrinciple(QiskitNatureTestCase):
    """Test the OverlapMetricPrinciple."""

    def setUp(self):
        super().setUp()
        mapper = JordanWignerMapper()
        self.ansatze = {
            "hea": HartreeFock(2, (1, 1), mapper).compose(
                efficient_su2(4, su2_gates=["ry"], reps=1, entanglement="linear")
            ),
            "uccsd": UCCSD(2, (1, 1), mapper, initial_state=HartreeFock(2, (1, 1), mapper)),
            "mixed": mixed_circuit(),
        }
        self.exact = StatevectorImaginaryMcLachlanPrinciple()

    @data("hea", "uccsd", "mixed")
    def test_exact_overlaps(self, name):
        """Test that the metric agrees with the exact metric."""
        ansatz = self.ansatze[name]
        values = np.random.default_rng(3).normal(size=ansatz.num_parameters)
        principle = OverlapMetricPrinciple(self.exact)
        np.testing.assert_allclose(
            principle.metric_tensor(ansatz, values),
            self.exact.metric_tensor(ansatz, values),
            atol=1e-12,
        )
        self.assertAlmostEqual(principle.last_reference_overlap, 1.0)

    def test_parameter_frequencies(self):
        """Test the frequencies of the overlaps."""
        omegas, levels = parameter_frequencies(self.ansatze["hea"])
        np.testing.assert_allclose(omegas, [1] * 8)
        np.testing.assert_array_equal(levels, [1] * 8)
        omegas, levels = parameter_frequencies(self.ansatze["uccsd"])
        np.testing.assert_allclose(omegas, [1] * 3)
        np.testing.assert_array_equal(levels, [2] * 3)
        # ry, two XX+YY evolutions, rzz(2 t), p, crz
        omegas, levels = parameter_frequencies(self.ansatze["mixed"])
        np.testing.assert_allclose(omegas, [1, 1, 1, 2, 1, 0.5])
        np.testing.assert_array_equal(levels, [1, 2, 2, 1, 1, 2])
        theta = Parameter("t")
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)
        circuit.rz(theta, 0)
        with self.assertRaises(QiskitNatureError):
            parameter_frequencies(circuit)
        circuit = QuantumCircuit(1)
        circuit.ry(theta * theta, 0)
        with self.assertRaises(QiskitNatureError):
            parameter_frequencies(circuit)

    def test_normalization(self):
        """Test that the normalization removes a uniform damping of the overlaps."""

        class Damped(OverlapMetricPrinciple):
            """Damps all overlaps by the same factor."""

            def _overlaps(self, ansatz, setup, theta):
                return 0.8 * super()._overlaps(ansatz, setup, theta)

        ansatz = self.ansatze["mixed"]
        values = np.random.default_rng(4).normal(size=ansatz.num_parameters)
        expected = self.exact.metric_tensor(ansatz, values)
        np.testing.assert_allclose(
            Damped(self.exact).metric_tensor(ansatz, values), expected, atol=1e-12
        )
        unnormalized = Damped(self.exact, normalize=False).metric_tensor(ansatz, values)
        self.assertGreater(np.abs(unnormalized - expected).max(), 1e-2)

    def test_sampler(self):
        """Test the measurement of the overlaps with a sampler, also of compiled circuits."""
        ansatz = efficient_su2(2, su2_gates=["ry"], reps=1)
        values = np.random.default_rng(5).normal(size=ansatz.num_parameters)
        expected = self.exact.metric_tensor(ansatz, values)
        backend = GenericBackendV2(3, coupling_map=[[0, 1], [1, 2]], seed=1)
        for transpiler in (
            None,
            generate_preset_pass_manager(1, backend=backend, seed_transpiler=2),
        ):
            principle = OverlapMetricPrinciple(
                self.exact, StatevectorSampler(seed=7), shots=20000, transpiler=transpiler
            )
            np.testing.assert_allclose(principle.metric_tensor(ansatz, values), expected, atol=0.02)

    def test_blocks(self):
        """Test that only the given blocks are evaluated."""
        ansatz = self.ansatze["hea"]
        values = np.random.default_rng(6).normal(size=ansatz.num_parameters)
        blocks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        principle = OverlapMetricPrinciple(self.exact, blocks=blocks)
        metric = principle.metric_tensor(ansatz, values)
        expected = self.exact.metric_tensor(ansatz, values)
        for block in blocks:
            np.testing.assert_allclose(
                metric[np.ix_(block, block)], expected[np.ix_(block, block)], atol=1e-12
            )
        self.assertTrue(np.all(metric[np.ix_(blocks[0], blocks[1])] == 0))
        self.assertEqual(principle.num_circuits(ansatz), 1 + 8 + 2 * 6 * 4)
        self.assertEqual(OverlapMetricPrinciple(self.exact).num_circuits(ansatz), 1 + 8 + 28 * 4)

    def test_var_qite(self):
        """Test the VarQITE with the overlap metric."""
        problem = fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))
        operator = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
        ansatz = self.ansatze["uccsd"]
        reference = VarQITEMinimumEigensolver(ansatz, time=0.5).compute_minimum_eigenvalue(operator)
        result = VarQITEMinimumEigensolver(
            ansatz,
            time=0.5,
            variational_principle=OverlapMetricPrinciple(self.exact),
            lse_solver=lambda a, b: np.linalg.lstsq(a, b, rcond=1e-7)[0],
        ).compute_minimum_eigenvalue(operator)
        np.testing.assert_allclose(result.optimal_point, reference.optimal_point, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
