# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Translator methods for the FCIDump."""

from __future__ import annotations

import logging

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.properties import ParticleNumber

from .fcidump import FCIDump
from .tcdump import TCDump

LOGGER = logging.getLogger(__name__)


def fcidump_to_problem(
    fcidump: FCIDump, tcdump: TCDump | None = None
) -> ElectronicStructureProblem:
    """Builds out an :class:`.ElectronicStructureProblem` from a :class:`.FCIDump` instance.

    This method centralizes the construction of an :class:`.ElectronicStructureProblem` from a
    :class:`.FCIDump`. The three-body integrals of a transcorrelated Hamiltonian can be provided
    through an additional :class:`.TCDump`. Approximations which avoid explicit three-body terms,
    such as the xTC method, only require an :class:`.FCIDump` holding the (non-Hermitian) one-
    and two-body integrals.

    .. code-block:: python

        from qiskit_nature.second_q.formats import fcidump_to_problem
        from qiskit_nature.second_q.formats.fcidump import FCIDump
        from qiskit_nature.second_q.formats.tcdump import TCDump

        problem = fcidump_to_problem(FCIDump.from_file("FCIDUMP"), TCDump.from_file("TCDUMP"))

    Args:
        fcidump: the :class:`.FCIDump` object from which to build the problem.
        tcdump: the optional :class:`.TCDump` object holding three-body integrals.

    Raises:
        QiskitNatureError: if the numbers of orbitals of ``fcidump`` and ``tcdump`` do not match.

    Returns:
        An :class:`.ElectronicStructureProblem` instance.
    """
    if tcdump is None and fcidump.metadata.get("III", 0) == 1:
        LOGGER.warning(
            "The FCIDump header indicates explicit three-body integrals (III=1), but no TCDump was "
            "provided. The three-body terms of the Hamiltonian are omitted."
        )
    if tcdump is not None and tcdump.num_orbitals != fcidump.num_orbitals:
        raise QiskitNatureError(
            f"The number of orbitals of the FCIDump ({fcidump.num_orbitals}) and of the TCDump "
            f"({tcdump.num_orbitals}) do not match."
        )

    num_beta = (fcidump.num_electrons - (fcidump.multiplicity - 1)) // 2
    num_alpha = fcidump.num_electrons - num_beta

    particle_number = ParticleNumber(fcidump.num_orbitals)

    electronic_energy = ElectronicEnergy.from_raw_integrals(
        fcidump.hij,
        fcidump.hijkl,
        fcidump.hij_b,
        fcidump.hijkl_bb,
        fcidump.hijkl_ba,
        three_body=None if tcdump is None else tcdump.three_body,
    )
    electronic_energy.nuclear_repulsion_energy = fcidump.constant_energy

    problem = ElectronicStructureProblem(electronic_energy)
    problem.basis = ElectronicBasis.MO
    problem.num_particles = (num_alpha, num_beta)
    problem.num_spatial_orbitals = fcidump.num_orbitals
    problem.properties.particle_number = particle_number

    return problem
