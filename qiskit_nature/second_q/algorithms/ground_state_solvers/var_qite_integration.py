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

r"""Integration schemes of the VarQITE equations of motion beyond forward Euler.

The McLachlan equations of motion :math:`A\dot{\theta} = C` with the metric
:math:`A_{jk} = \mathrm{Re}\langle T_j|T_k\rangle` and the right-hand side
:math:`C_j = -\mathrm{Re}\langle T_j|H - E|\psi\rangle` (:math:`|T_k\rangle` the tangent vectors
projected orthogonally to the state) are integrated with one of the following steps of size
:math:`h`:

* forward Euler, :math:`\theta \leftarrow \theta + h A^{-1} C`, optionally with a backtracking
  line search on the energy;
* the semi-implicit Galerkin step of the backward Euler method with the generator frozen at the
  current state, :math:`(A + h K)\,d = C` with :math:`K_{jk} = \mathrm{Re}\langle T_j|H - E|T_k
  \rangle`, followed by :math:`\theta \leftarrow \theta + h d`. The matrix :math:`A + hK` becomes
  indefinite once :math:`h` exceeds the inverse distance of the energy to the lowest Ritz value of
  :math:`H - E` in the tangent space. It is made positive definite by shifting the generator by
  :math:`g = \min(0, \lambda_\mathrm{min})`, where :math:`\lambda_\mathrm{min}` is the lowest
  eigenvalue of the pencil :math:`(K, A)`: :math:`M = A(1 - hg) + hK \geq A`. This is the
  unshifted implicit step with the effective time step :math:`h / (1 - hg) < 1/|g|`, i.e. the shift
  caps the step, and the tangent-space Ritz value is the smallest shift that guarantees
  positivity (it is obtained from :math:`A` and :math:`K` alone). For large :math:`h`, the step
  approaches the Gauss-Newton-like step :math:`(K - gA)^{-1} C`.

Optionally, every step is shortened to a maximal Fubini-Study length
:math:`h\sqrt{d^T A d} \leq r` (trust region), which keeps large steps within the region of
validity of the linearization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

ARMIJO = 1e-4
"""Sufficient-decrease parameter of the line search."""

MAX_TRIALS = 8
"""Maximal number of energy evaluations (halvings of the step) per line-search step."""

GROWTH = 1.5
"""Factor by which the line-search step grows after an accepted step (up to the maximal step)."""

RITZ_RCOND = 1e-7
"""Relative cutoff of the eigenvalues of the metric defining the tangent space of the Ritz shift."""


def step_options(
    timestep: float | None, implicit: bool, line_search: bool, trust_region: float | None
) -> tuple[float, float | None]:
    """The time step and the trust region with their defaults: a time step of ``5.0`` for implicit
    steps, ``1.0`` (the maximal one) for the line search and ``0.05`` otherwise, and a trust
    region of ``0.2`` for implicit steps (``numpy.inf`` disables it)."""
    if timestep is None:
        timestep = 5.0 if implicit else (1.0 if line_search else 0.05)
    if trust_region is None and implicit:
        trust_region = 0.2
    if trust_region is not None and np.isinf(trust_region):
        trust_region = None
    return timestep, trust_region


def tangent_space_shift(tangent_matrix: np.ndarray, metric: np.ndarray) -> float:
    r"""The positivity shift :math:`\min(0, \lambda_\mathrm{min})` of the implicit step, where
    :math:`\lambda_\mathrm{min}` is the lowest eigenvalue of the pencil
    :math:`(\frac{1}{2}(K + K^T), A)` on the range of the metric :math:`A` (the lowest Ritz value of
    :math:`H - E` in the tangent space; for a non-Hermitian :math:`H`, of its Hermitian part).

    Args:
        tangent_matrix: The matrix :math:`K`.
        metric: The metric :math:`A`.

    Returns:
        The shift (zero or negative).
    """
    eigenvalues, vectors = np.linalg.eigh(0.5 * (metric + metric.T))
    if eigenvalues.size == 0 or eigenvalues[-1] <= 0:
        return 0.0
    keep = eigenvalues > RITZ_RCOND * eigenvalues[-1]
    basis = vectors[:, keep] / np.sqrt(eigenvalues[keep])
    projected = basis.T @ (0.5 * (tangent_matrix + tangent_matrix.T)) @ basis
    return min(float(np.linalg.eigvalsh(projected)[0]), 0.0)


def integrate(
    evaluate: Callable[[np.ndarray, bool], tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    energy: Callable[[np.ndarray], float],
    initial_point: Sequence[float] | np.ndarray,
    *,
    time: float,
    timestep: float,
    implicit: bool,
    line_search: bool,
    trust_region: float | None,
    convergence_threshold: float | None,
    lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[list[float], list[np.ndarray], bool]:
    """Integrates the equations of motion with explicit or implicit steps.

    Args:
        evaluate: Returns the metric, the right-hand side and, if its second argument is ``True``,
            the matrix ``K`` at the given parameter values.
        energy: Returns the (real) energy at the given parameter values (used by the line search).
        initial_point: The initial parameter values.
        time: The maximal imaginary time (the sum of the steps taken).
        timestep: The step size (the maximal one with a line search).
        implicit: Whether to take the shifted semi-implicit Galerkin steps.
        line_search: Whether to accept a step only if the energy decreases sufficiently
            (Armijo condition), halving it otherwise, and to grow it after accepted steps.
        trust_region: The maximal Fubini-Study length of a step, or ``None``.
        convergence_threshold: The integration stops once the norm of the explicit parameter
            derivative :math:`A^{-1}C` falls below this threshold (if not ``None``).
        lse_solver: The solver of the linear systems.

    Returns:
        The times, the parameter values at these times, and whether the convergence threshold was
        reached.
    """
    values = np.asarray(initial_point, dtype=float).copy()
    times, parameter_values = [0.0], [values.copy()]
    current_time = 0.0
    step_size = timestep
    current_energy = energy(values) if line_search else 0.0
    converged = False
    while current_time < time * (1 - 1e-12):
        metric, rhs, tangent_matrix = evaluate(values, implicit)
        derivative = np.real(lse_solver(metric, rhs))
        shift = tangent_space_shift(tangent_matrix, metric) if implicit else 0.0

        def step(size: float) -> tuple[float, np.ndarray]:
            """The step taken for the step size ``size``: the size after the trust region and the
            parameter update."""
            direction = derivative
            if implicit:
                matrix = metric * (1.0 - size * shift) + size * tangent_matrix
                direction = np.real(lse_solver(matrix, rhs))
            length = size * np.sqrt(max(float(direction @ metric @ direction), 0.0))
            if trust_region is not None and length > trust_region:
                size *= trust_region / length
            return size, size * direction

        remaining = time - current_time
        # avoid a tiny final step due to floating-point accumulation
        size = remaining if remaining < step_size * (1 + 1e-9) else step_size
        taken, update = step(size)
        if line_search:
            for _ in range(MAX_TRIALS - 1):
                # the energy changes as -2 C.d per unit step at h = 0 for a Hermitian Hamiltonian
                new_energy = energy(values + update)
                if new_energy <= current_energy - 2 * ARMIJO * max(float(rhs @ update), 0.0):
                    break
                size *= 0.5
                taken, update = step(size)
            else:
                new_energy = energy(values + update)
            current_energy = new_energy
            step_size = min(timestep, GROWTH * size)

        values = values + update
        current_time += taken
        times.append(current_time)
        parameter_values.append(values.copy())
        if convergence_threshold is not None and np.linalg.norm(derivative) < convergence_threshold:
            converged = True
            break
    return times, parameter_values, converged
