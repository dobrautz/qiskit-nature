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

"""TCDump"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators.three_body_integrals import ThreeBodyIntegrals


@dataclass
class TCDump:
    """Qiskit Nature dataclass for representing the TCDUMP format.

    The TCDUMP format stores the three-body integrals of transcorrelated Hamiltonians, as written
    by the TCHInt library [1] and read by NECI [2]. It complements an FCIDUMP file (see
    :class:`~qiskit_nature.second_q.formats.fcidump.FCIDump`), which holds the one- and two-body
    integrals of the same Hamiltonian.

    The file optionally starts with a line holding the number of orbitals. Every other line holds
    one integral in the format

    .. code-block:: text

        value  p  q  r  s  t  u

    with 1-based spatial orbital indices, where ``(p, s)``, ``(q, t)`` and ``(r, u)`` are the index
    pairs of the three electrons. The corresponding operator is defined in
    :class:`~qiskit_nature.second_q.operators.three_body_integrals.ThreeBodyIntegrals`.

    Since the three-body operator is local and Hermitian, its integrals obey a 48-fold
    permutational symmetry for real orbitals. Files may list either all non-zero integrals (often
    named ``TCDUMP-nosym``) or only the symmetry-unique ones. Both variants are parsed, and by
    default only the symmetry-unique integrals are written, which reduces the file size by up to a
    factor of 48.

    References:
        [1]: P. López Ríos, et al., TCHInt: transcorrelated Hamiltonian integrals,
            https://www2.fkf.mpg.de/alavi/tchint/stable/run.html

        [2]: K. Guther, et al., NECI: N-Electron Configuration Interaction with an emphasis on
            state-of-the-art stochastic methods, J. Chem. Phys. 153, 034107 (2020).
    """

    three_body: ThreeBodyIntegrals
    """The three-body integrals."""

    @property
    def num_orbitals(self) -> int:
        """The number of orbitals."""
        return self.three_body.num_orbitals

    @classmethod
    def from_file(
        cls,
        tcdump: str | Path,
        num_orbitals: int | None = None,
        *,
        symmetry: Literal["s48", "s1"] | None = None,
    ) -> TCDump:
        """Constructs a TCDump object from a file.

        Args:
            tcdump: Path to the input file.
            num_orbitals: The number of orbitals. Must be provided if the file has no header line
                holding the number of orbitals and the largest orbital index does not occur in
                any non-zero integral.
            symmetry: The permutational symmetry of the integrals, see
                :class:`~qiskit_nature.second_q.operators.three_body_integrals.ThreeBodyIntegrals`.
                By default, the 48-fold symmetry is used unless it contradicts the listed
                integrals.

        Raises:
            QiskitNatureError: If the input file cannot be read, if an invalid line is
                encountered or if the number of orbitals is inconsistent.

        Returns:
            A :class:`.TCDump` instance.
        """
        path = tcdump if isinstance(tcdump, Path) else Path(tcdump)
        try:
            with path.open("r", encoding="utf8") as file:
                lines = file.read().splitlines()
        except OSError as ex:
            raise QiskitNatureError(f"Input file '{path}' cannot be read!") from ex

        header_norb = None
        indices: list[list[int]] = []
        values: list[float] = []
        for line_number, line in enumerate(lines, start=1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) == 1 and not indices and header_norb is None:
                header_norb = int(tokens[0])
                continue
            if len(tokens) != 7:
                raise QiskitNatureError(
                    f"Invalid TCDUMP entry encountered in line {line_number}: '{line}'"
                )
            values.append(float(tokens[0].replace("D", "E").replace("d", "e")))
            indices.append([int(t) - 1 for t in tokens[1:]])

        index_array = np.asarray(indices, dtype=np.int64).reshape(-1, 6)
        max_index = int(index_array.max()) + 1 if index_array.size else 0

        if header_norb is not None and num_orbitals is not None and header_norb != num_orbitals:
            raise QiskitNatureError(
                f"The number of orbitals in the TCDUMP header ({header_norb}) does not match the "
                f"requested one ({num_orbitals})."
            )
        norb = num_orbitals if num_orbitals is not None else header_norb
        if norb is None:
            norb = max_index
        if max_index > norb or (index_array.size and index_array.min() < 0):
            raise QiskitNatureError(
                f"The TCDUMP file holds orbital indices outside of the range [1, {norb}]."
            )

        return cls(ThreeBodyIntegrals(index_array, np.asarray(values), norb, symmetry=symmetry))

    def to_file(self, tcdump: str | Path, *, symmetry_reduced: bool = True) -> None:
        """Dumps a TCDump object to a file.

        Args:
            tcdump: Path to the output file.
            symmetry_reduced: Whether to only write the symmetry-unique integrals. This only takes
                effect if the integrals are stored with their 48-fold symmetry. Otherwise, all
                non-zero integrals are written.
        """
        path = tcdump if isinstance(tcdump, Path) else Path(tcdump)
        if symmetry_reduced and self.three_body.symmetry == "s48":
            chunks = [(self.three_body.indices, self.three_body.values)]
        else:
            chunks = list(self.three_body.iter_expanded())

        with path.open("w", encoding="utf8") as outfile:
            outfile.write(f"{self.num_orbitals:12d}\n")
            for indices, values in chunks:
                for index, value in zip((indices + 1).tolist(), values.tolist()):
                    if value == 0.0:
                        continue
                    outfile.write(f"{value:24.16E}" + "".join(f"{i:5d}" for i in index) + "\n")
