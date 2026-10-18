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

"""
TCDump (:mod:`qiskit_nature.second_q.formats.tcdump`)
=====================================================

Contains tools to parse and dump TCDUMP files, which hold the three-body integrals of
transcorrelated Hamiltonians.

.. currentmodule:: qiskit_nature.second_q.formats.tcdump

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TCDump

"""

from .tcdump import TCDump

__all__ = ["TCDump"]
