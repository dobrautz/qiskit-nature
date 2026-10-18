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

"""Whether the installed qiskit-algorithms supports non-Hermitian operators.

Some tests need features of qiskit-algorithms that are newer than its latest release: the
estimator-based VarQITE with non-Hermitian operators and the hermiticity check of the dense NumPy
eigensolver. They are detected by what the installed version computes, not by its version number.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.gradients import LinCombEstimatorGradient, LinCombQGT
from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

from qiskit_nature.second_q.algorithms import StatevectorImaginaryMcLachlanPrinciple

_OPERATOR = SparsePauliOp(["Z", "X", "Y"], [1.0, 0.5, 0.3j])


def _non_hermitian_var_qite() -> bool:
    parameters = ParameterVector("t", 2)
    circuit = QuantumCircuit(1)
    circuit.ry(parameters[0], 0)
    circuit.rz(parameters[1], 0)
    values = [0.4, 0.7]
    estimator = StatevectorEstimator()
    principle = ImaginaryMcLachlanPrinciple(
        LinCombQGT(estimator), LinCombEstimatorGradient(estimator)
    )
    exact = StatevectorImaginaryMcLachlanPrinciple().evolution_gradient(_OPERATOR, circuit, values)
    try:
        gradient = principle.evolution_gradient(_OPERATOR, circuit, values)
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return bool(np.allclose(gradient, exact, atol=1e-8))


def _non_hermitian_eigensolver() -> bool:
    exact = min(np.linalg.eigvals(_OPERATOR.to_matrix()), key=lambda value: value.real)
    result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(_OPERATOR)
    return bool(abs(result.eigenvalue - exact) < 1e-8)


HAS_NON_HERMITIAN_VARQITE = _non_hermitian_var_qite()
"""Whether the estimator-based VarQITE of qiskit-algorithms supports non-Hermitian operators."""

HAS_NON_HERMITIAN_EIGENSOLVER = _non_hermitian_eigensolver()
"""Whether the dense NumPy eigensolver of qiskit-algorithms treats non-Hermitian matrices."""

SKIP_VARQITE = "the installed qiskit-algorithms has no non-Hermitian VarQITE"
SKIP_EIGENSOLVER = "the installed qiskit-algorithms treats non-Hermitian matrices as Hermitian"
