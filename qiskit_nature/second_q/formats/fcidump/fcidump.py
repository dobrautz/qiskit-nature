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

"""FCIDump"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators.symmetric_two_body import (
    S8Integrals,
    SymmetricTwoBodyIntegrals,
    unfold,
)

from .dumper import _dump_1e_ints, _dump_2e_ints, _write_to_outfile


@dataclass
class FCIDump:
    """
    Qiskit Nature dataclass for representing the FCIDump format.

    The FCIDump format is partially defined in Knowles1989.

    Non-Hermitian integrals, as they arise for example in the transcorrelated method, are supported
    as well. In this case, the 2-electron integrals only obey the electron exchange symmetry
    :math:`(ij|kl) = (kl|ij)` and are stored in an
    :class:`~qiskit_nature.second_q.operators.symmetric_two_body.S1Integrals` container. When
    writing such integrals, the ``ST=1`` header entry is added (following the convention of the
    TCHInt library [1]), which instructs the parser to only exploit the electron exchange symmetry.
    When parsing a file without this header entry, non-Hermitian integrals are detected from the
    data. Explicit 3-electron integrals are handled by the separate
    :class:`~qiskit_nature.second_q.formats.tcdump.TCDump` format.

    References:
        Knowles1989: Peter J. Knowles, Nicholas C. Handy,
            A determinant based full configuration interaction program,
            Computer Physics Communications, Volume 54, Issue 1, 1989, Pages 75-83,
            ISSN 0010-4655, https://doi.org/10.1016/0010-4655(89)90033-7.

        [1]: P. López Ríos, et al., TCHInt: transcorrelated Hamiltonian integrals,
            https://www2.fkf.mpg.de/alavi/tchint/stable/run.html
    """

    num_electrons: int
    """The number of electrons."""
    hij: np.ndarray
    """The alpha 1-electron integrals."""
    hijkl: SymmetricTwoBodyIntegrals
    """The alpha/alpha 2-electron integrals ordered in chemist' notation."""
    hij_b: np.ndarray | None = None
    """The beta 1-electron integrals."""
    hijkl_bb: SymmetricTwoBodyIntegrals | None = None
    """The beta/beta 2-electron integrals ordered in chemist' notation."""
    hijkl_ba: SymmetricTwoBodyIntegrals | None = None
    """The beta/alpha 2-electron integrals ordered in chemist' notation."""
    constant_energy: float | None = None
    """The constant energy comprising (for example) the nuclear repulsion energy and inactive
    energies."""
    multiplicity: int = 1
    """The multiplicity."""
    orbsym: Sequence[int] | None = None
    """A list of spatial symmetries of the orbitals."""
    isym: int = 1
    """The spatial symmetry of the wave function."""
    metadata: dict[str, int] = field(default_factory=dict)
    """Additional (integer) header entries, such as ``ST``, ``III`` and ``ADJOINT`` written by the
    TCHInt library for transcorrelated integrals. These are not written by :meth:`to_file`, which
    determines the ``ST`` entry from the integrals."""

    @property
    def num_orbitals(self) -> int:
        """The number of orbitals."""
        return self.hij.shape[0]

    @property
    def is_hermitian(self) -> bool:
        """Whether the integrals are Hermitian.

        This is the case if the 1-electron integrals are symmetric and the same-spin 2-electron
        integrals obey the 8-fold permutational symmetry.
        """
        one_body = [self.hij] + ([] if self.hij_b is None else [self.hij_b])
        two_body = [self.hijkl] + ([] if self.hijkl_bb is None else [self.hijkl_bb])
        return all(np.allclose(h, h.T, rtol=0.0, atol=1e-10) for h in one_body) and all(
            _has_s8_symmetry(eri) for eri in two_body
        )

    @classmethod
    def from_file(cls, fcidump: str | Path) -> FCIDump:
        """Constructs an FCIDump object from a file.

        Args:
            fcidump: Path to the input file.

        Returns:
            A :class:`.FCIDump` instance.
        """
        # pylint: disable=cyclic-import
        from .parser import _parse

        return _parse(fcidump if isinstance(fcidump, Path) else Path(fcidump))

    def to_file(self, fcidump: str | Path) -> None:
        """Dumps an FCIDump object to a file.

        Args:
            fcidump: Path to the output file.
        Raises:
            QiskitNatureError: not all beta-spin related matrices are either None or not None.
            QiskitNatureError: if the dimensions of the provided integral matrices do not match.
        """
        outpath = fcidump if isinstance(fcidump, Path) else Path(fcidump)
        # either all beta variables are None or all of them are not
        if not all(h is None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]) and not all(
            h is not None for h in [self.hij_b, self.hijkl_ba, self.hijkl_bb]
        ):
            raise QiskitNatureError("Invalid beta variables.")
        if set(self.hij.shape) != set(self.hijkl.shape):
            raise QiskitNatureError(
                "The number of orbitals of the 1- and 2-body matrices do not match: "
                f"{set(self.hij.shape)} vs. {set(self.hijkl.shape)}"
            )
        norb = self.hij.shape[0]
        nelec = self.num_electrons
        einact = self.constant_energy
        ms2 = self.multiplicity - 1
        hermitian_two_body = all(
            _has_s8_symmetry(eri) for eri in [self.hijkl, self.hijkl_bb] if eri is not None
        )

        mos = range(norb)
        with outpath.open("w", encoding="utf8") as outfile:
            # print header
            outfile.write(f"&FCI NORB={norb:4d},NELEC={nelec:4d},MS2={ms2:4d},\n")
            if self.orbsym is None:
                outfile.write(" ORBSYM=" + "1," * norb + ",\n")
            else:
                if len(self.orbsym) != norb:
                    raise QiskitNatureError(f"Invalid number of orbitals {norb} {len(self.orbsym)}")
                outfile.write(" ORBSYM=" + ",".join(str(o) for o in self.orbsym) + ",\n")
            outfile.write(f" ISYM={self.isym:d},\n")
            if not hermitian_two_body:
                # non-Hermitian 2-electron integrals which only obey the electron exchange symmetry
                outfile.write(" ST=1,\n")
            outfile.write("&END\n")
            # append 2e integrals
            _dump_2e_ints(self.hijkl, mos, outfile, exchange_only=not hermitian_two_body)
            if self.hijkl_ba is not None:
                # the alpha-beta integrals are related to the beta-alpha ones by electron exchange
                hijkl_ab = _to_dense(self.hijkl_ba).transpose(2, 3, 0, 1)
                _dump_2e_ints(hijkl_ab, mos, outfile, beta=1, exchange_only=not hermitian_two_body)
            if self.hijkl_bb is not None:
                _dump_2e_ints(
                    self.hijkl_bb, mos, outfile, beta=2, exchange_only=not hermitian_two_body
                )
            # append 1e integrals
            _dump_1e_ints(self.hij, mos, outfile)
            if self.hij_b is not None:
                _dump_1e_ints(self.hij_b, mos, outfile, beta=True)
            # TODO append MO energies (last three indices are 0)
            # append inactive energy
            if einact is not None:
                _write_to_outfile(outfile, einact, (0, 0, 0, 0))


def _to_dense(eri: SymmetricTwoBodyIntegrals | np.ndarray) -> np.ndarray:
    """Returns the full 4-dimensional array of 2-electron integrals."""
    if isinstance(eri, SymmetricTwoBodyIntegrals):
        array = unfold(eri).array
        return np.asarray(array.todense() if hasattr(array, "todense") else array)
    return np.asarray(eri)


def _has_s8_symmetry(eri: SymmetricTwoBodyIntegrals | np.ndarray) -> bool:
    """Checks whether 2-electron integrals in chemist' order obey the 8-fold symmetry."""
    if isinstance(eri, S8Integrals):
        return True
    dense = _to_dense(eri)
    return all(
        np.allclose(dense, dense.transpose(perm), rtol=0.0, atol=1e-10)
        for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1))
    )
