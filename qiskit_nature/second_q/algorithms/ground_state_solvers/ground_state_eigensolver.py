# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Ground state computation using a minimum eigensolver."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from qiskit_algorithms import MinimumEigensolver, NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators import SparseLabelOp
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import BaseProblem
from qiskit_nature.second_q.problems import EigenstateResult

from .ground_state_solver import GroundStateSolver
from .var_qite_minimum_eigensolver import VarQITEMinimumEigensolver

LOGGER = logging.getLogger(__name__)


class GroundStateEigensolver(GroundStateSolver):
    """Ground state computation using a minimum eigensolver.

    Non-Hermitian Hamiltonians, such as transcorrelated Hamiltonians, are detected automatically by
    the complex coefficients of the Pauli representation of the mapped qubit operator. Since
    estimator-based variational eigensolvers such as the VQE cannot treat non-Hermitian
    Hamiltonians, the minimum eigenvalue is then computed with the ``non_hermitian_solver`` instead.
    If no such solver is provided and ``solver`` does not support non-Hermitian Hamiltonians, a
    :class:`.VarQITEMinimumEigensolver` with the ansatz, initial point and estimator of ``solver``
    is used. The :class:`~qiskit_algorithms.NumPyMinimumEigensolver` and the
    :class:`.VarQITEMinimumEigensolver` support non-Hermitian Hamiltonians and are used as is.
    """

    def __init__(
        self,
        qubit_mapper: QubitMapper,
        solver: MinimumEigensolver,
        non_hermitian_solver: MinimumEigensolver | None = None,
    ) -> None:
        # pylint: disable=unused-argument
        """
        Args:
            qubit_mapper: The :class:`~qiskit_nature.second_q.mappers.QubitMapper` instance that
                converts a second quantized operator to qubit operators.
            solver: Minimum Eigensolver object.
            non_hermitian_solver: An optional minimum eigensolver which is used instead of
                ``solver`` if the qubit operator is not Hermitian.
        """
        super().__init__(qubit_mapper)
        self._solver = solver
        self._non_hermitian_solver = non_hermitian_solver

    @property
    def solver(self) -> MinimumEigensolver:
        return self._solver

    def supports_aux_operators(self):
        return self.solver.supports_aux_operators()

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> EigenstateResult:
        """Compute Ground State properties.

        Args:
            problem: A class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """
        main_operator, aux_ops = self.get_qubit_operators(problem, aux_operators)
        solver = self.solver
        if not is_hermitian_qubit_operator(main_operator):
            solver = self._get_non_hermitian_solver()
            if solver.supports_aux_operators() and not self.solver.supports_aux_operators():
                # the auxiliary operators were discarded because of the original solver
                main_operator, aux_ops = self._get_qubit_operators(problem, aux_operators, True)
        raw_mes_result = solver.compute_minimum_eigenvalue(main_operator, aux_ops)

        eigenstate_result = EigenstateResult.from_result(raw_mes_result)
        result = problem.interpret(eigenstate_result)
        return result

    def _get_non_hermitian_solver(self) -> MinimumEigensolver:
        """Returns the minimum eigensolver to be used for non-Hermitian qubit operators.

        Raises:
            QiskitNatureError: if no solver supporting non-Hermitian operators can be determined.
        """
        if self._non_hermitian_solver is not None:
            return self._non_hermitian_solver
        if isinstance(self.solver, (NumPyMinimumEigensolver, VarQITEMinimumEigensolver)):
            return self.solver
        ansatz = getattr(self.solver, "ansatz", None)
        if ansatz is None:
            raise QiskitNatureError(
                "The qubit operator is not Hermitian (its Pauli representation has complex "
                f"coefficients), which is not supported by the {type(self.solver).__name__}. "
                "Provide a solver for non-Hermitian operators, e.g., a VarQITEMinimumEigensolver, "
                "via the `non_hermitian_solver` argument."
            )
        warnings.warn(
            "The qubit operator is not Hermitian (its Pauli representation has complex "
            f"coefficients), which is not supported by the {type(self.solver).__name__}. Using a "
            "VarQITEMinimumEigensolver with the same ansatz instead, which converges to the right "
            "eigenvector whose eigenvalue has the lowest real part. Use the `non_hermitian_solver` "
            "argument to configure this solver.",
            stacklevel=3,
        )
        return VarQITEMinimumEigensolver(
            ansatz,
            initial_point=getattr(self.solver, "initial_point", None),
            estimator=getattr(self.solver, "estimator", None),
        )

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None = None,
    ) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        return self._get_qubit_operators(
            problem, aux_operators, self.solver.supports_aux_operators()
        )

    def _get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: dict[str, SparseLabelOp | SparsePauliOp] | None,
        supports_aux_operators: bool,
    ) -> tuple[SparsePauliOp, dict[str, SparsePauliOp] | None]:
        # Note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation
        main_second_q_op, aux_second_q_ops = problem.second_q_ops()

        main_operator = self._qubit_mapper.map(main_second_q_op)
        aux_ops = self._qubit_mapper.map(aux_second_q_ops)

        if aux_operators is not None:
            for name_aux, aux_op in aux_operators.items():
                if isinstance(aux_op, SparseLabelOp):
                    converted_aux_op = self._qubit_mapper.map(aux_op)
                else:
                    converted_aux_op = aux_op

                if name_aux in aux_ops.keys():
                    LOGGER.warning(
                        "The key '%s' was already taken by an internally constructed auxiliary "
                        "operator! The internal operator was overridden by the one provided manually. "
                        "If this was not the intended behavior, please consider renaming "
                        "this operator.",
                        name_aux,
                    )
                if converted_aux_op is not None:
                    # The custom op overrides the default op if the key is already taken.
                    aux_ops[name_aux] = converted_aux_op
                else:
                    LOGGER.warning(
                        "The manually provided operator '%s' got reduced to `None` in the mapping "
                        "process. This can occur for example when it does not commute with the "
                        "hamiltonian after applying the determined symmetry reductions. Thus, this "
                        "operator will not be used!",
                        name_aux,
                    )

        # if the eigensolver does not support auxiliary operators, reset them
        if not supports_aux_operators:
            aux_ops = None
        return main_operator, aux_ops


def is_hermitian_qubit_operator(operator: SparsePauliOp) -> bool:
    """Checks whether a qubit operator is Hermitian.

    Since Pauli strings are Hermitian, a simplified :class:`~qiskit.quantum_info.SparsePauliOp`
    is Hermitian if and only if all of its coefficients are real. Imaginary parts which would be
    accepted by the estimator primitives are ignored.

    Args:
        operator: the qubit operator.

    Returns:
        Whether the operator is Hermitian.
    """
    if not isinstance(operator, SparsePauliOp) or operator.coeffs.dtype == object:
        return True
    return not np.iscomplexobj(np.real_if_close(operator.simplify().coeffs))
