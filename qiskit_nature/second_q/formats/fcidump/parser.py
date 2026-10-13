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

"""FCIDump parser."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.operators.symmetric_two_body import (
    S1Integrals,
    S4Integrals,
    S8Integrals,
    SymmetricTwoBodyIntegrals,
)
from .fcidump import FCIDump

LOGGER = logging.getLogger(__name__)

_SYMMETRY_TOLERANCE = 1e-10
"""Absolute tolerance below which two symmetry-related integrals are considered equal."""

_S8_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)
"""The index permutations of the 8-fold symmetry of real, Hermitian 2-electron integrals in
chemist' notation."""

_EXCHANGE = (2, 3, 0, 1)
"""The index permutation exchanging the two electrons, ``(ij|kl) = (kl|ij)``. This is the only
permutational symmetry of the non-Hermitian 2-electron integrals of transcorrelated
Hamiltonians."""


def _parse(fcidump: Path) -> FCIDump:
    """Parses a FCIDump output.

    In addition to the fields defined in Knowles1989, the following (optional) header fields are
    supported:

    - ``ST``: indicates whether the integrals have been similarity transformed, for example by the
      transcorrelated method. If ``ST=1``, the 2-electron integrals are non-Hermitian and only obey
      the electron exchange symmetry ``(ij|kl) = (kl|ij)``.
    - ``III``: indicates whether the transcorrelated Hamiltonian has explicit 3-electron terms
      (``III=1``), which are provided in a separate TCDUMP file, or whether these have been
      approximated by lower-order terms (``III=0``), as in the xTC method.
    - ``ADJOINT``: indicates that the file holds the integrals of the adjoint Hamiltonian.
    - ``IUHF``: indicates that the integrals are given for spin orbitals (``IUHF=1``).
    - ``ENUC``: the constant energy, which is used if no ``0 0 0 0`` entry is present.

    If ``ST`` is not present, the permutational symmetry of the 2-electron integrals is detected
    from the data: if symmetry-related integrals with differing values are encountered, the
    integrals are treated as non-Hermitian. Hermitian 2-electron integrals are stored in an
    :class:`.S8Integrals` container, non-Hermitian ones in an :class:`.S1Integrals` container.
    Likewise, the 1-electron integrals are only symmetrized when this does not contradict the data.

    Args:
        fcidump: Path to the FCIDump file.
    Raises:
        QiskitNatureError: If the input file cannot be found, if a required field in the FCIDump
            file is missing, if wrong integral indices are encountered, if the alpha/beta or
            beta/alpha 2-electron integrals are mixed, if the 2-electron integrals contradict an
            ``ST=0`` header entry, or if 6-index (3-electron) entries are encountered.
    Returns:
        A dictionary storing the parsed data.
    """
    try:
        with fcidump.open("r", encoding="utf8") as file:
            fcidump_str = file.read()
    except OSError as ex:
        raise QiskitNatureError(f"Input file '{fcidump}' cannot be read!") from ex

    output: dict[str, Any] = {}

    # FCIDump starts with a Fortran namelist of meta data
    namelist_end = re.search("(/|&END)", fcidump_str, flags=re.IGNORECASE)
    if namelist_end is None:
        raise QiskitNatureError("The end of the FCIDump header (`/` or `&END`) is missing!")
    metadata = fcidump_str[: namelist_end.start(0)]
    metadata = " ".join(metadata.split())  # replace duplicate whitespace and newlines
    # we know what elements to look for so we don't get too fancy with the parsing
    # pattern explanation:
    #  .*?      any text
    #  (*|*),   match either part of this group followed by a comma
    #  [-+]?    match up to a single - or +
    #  \d*.\d+  number format
    pattern = r".*?([-+]?\d*\.\d+|[-+]?\d+),"
    # we parse the values in the order in which they are listed in Knowles1989
    _norb = re.search("NORB" + pattern, metadata)
    if _norb is None:
        raise QiskitNatureError("The required NORB entry of the FCIDump format is missing!")
    norb = int(_norb.groups()[0])
    output["NORB"] = norb
    _nelec = re.search("NELEC" + pattern, metadata)
    if _nelec is None:
        raise QiskitNatureError("The required NELEC entry of the FCIDump format is missing!")
    output["NELEC"] = int(_nelec.groups()[0])
    # the rest of these values may occur and are set to their defaults otherwise
    _ms2 = re.search("MS2" + pattern, metadata)
    output["MS2"] = int(_ms2.groups()[0]) if _ms2 else 0
    _isym = re.search("ISYM" + pattern, metadata)
    output["ISYM"] = int(_isym.groups()[0]) if _isym else 1
    # ORBSYM holds a list, thus it requires a little different treatment
    _orbsym = re.search(r"ORBSYM.*?" + r"(\d+)\s*,\s*" * norb, metadata)
    output["ORBSYM"] = [int(s) for s in _orbsym.groups()] if _orbsym else [1] * norb
    _iprtim = re.search("IPRTIM" + pattern, metadata)
    output["IPRTIM"] = int(_iprtim.groups()[0]) if _iprtim else -1
    _int = re.search("INT" + pattern, metadata)
    output["INT"] = int(_int.groups()[0]) if _int else 5
    _memory = re.search("MEMORY" + pattern, metadata)
    output["MEMORY"] = int(_memory.groups()[0]) if _memory else 10000
    _core = re.search("CORE" + pattern, metadata)
    output["CORE"] = float(_core.groups()[0]) if _core else 0.0
    _maxit = re.search("MAXIT" + pattern, metadata)
    output["MAXIT"] = int(_maxit.groups()[0]) if _maxit else 25
    _thr = re.search("THR" + pattern, metadata)
    output["THR"] = float(_thr.groups()[0]) if _thr else 1e-5
    _thrres = re.search("THRRES" + pattern, metadata)
    output["THRRES"] = float(_thrres.groups()[0]) if _thrres else 0.1
    _nroot = re.search("NROOT" + pattern, metadata)
    output["NROOT"] = int(_nroot.groups()[0]) if _nroot else 1
    # header fields written, for example, by the TCHInt library for transcorrelated integrals
    output["ST"] = _parse_int_field(metadata, "ST")
    output["IUHF"] = _parse_int_field(metadata, "IUHF")
    output["III"] = _parse_int_field(metadata, "III")
    output["ADJOINT"] = _parse_int_field(metadata, "ADJOINT")
    _enuc = re.search(r"\bENUC\s*=\s*([-+]?[\d.]+(?:[eEdD][-+]?\d+)?)", metadata, flags=re.I)
    if _enuc is not None:
        output["ecore"] = float(_enuc.groups()[0].replace("D", "E").replace("d", "e"))
    if output["ADJOINT"] == 1:
        LOGGER.info("The FCIDump file holds the integrals of the adjoint Hamiltonian (ADJOINT=1).")

    # If the FCIDump file resulted from an unrestricted spin calculation the indices will label spin
    # rather than molecular orbitals. This means, that a line must exist which encodes the
    # coefficient for the spin orbital with index (norb*2, norb*2). By checking for such a line we
    # can distinguish between unrestricted and restricted FCIDump files.
    _uhf = output["IUHF"] == 1 or bool(
        re.search(
            rf".*(\s+{norb * 2}\s+{norb * 2}\s+0\s+0)",
            fcidump_str[namelist_end.start(0) :],
        )
    )

    # the rest of the FCIDump will hold lines of the form x i a j b
    # a few cases have to be treated differently:
    # i, a, j and b are all zero: x is the core energy
    # TODO: a, j and b are all zero: x is the energy of the i-th MO  (often not supported)
    # j and b are both zero: x is the 1e-integral between i and a (x = <i|h|a>)
    # otherwise: x is the Coulomb integral ( x = (ia|jb) )
    one_body: list[tuple[int, int, float]] = []
    two_body: list[tuple[int, int, int, int, float]] = []
    for line in fcidump_str[namelist_end.end(0) :].splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) == 7:
            raise QiskitNatureError(
                "Encountered a 6-index (3-electron) entry in the FCIDump file. 3-electron "
                "integrals must be provided in a separate TCDUMP file (see TCDump)."
            )
        if len(tokens) != 5:
            raise QiskitNatureError(f"Invalid FCIDump entry encountered: '{line}'")
        x = float(tokens[0].replace("D", "E").replace("d", "e"))
        # Note: differing naming than ijkl due to E741 and this iajb is inline with this:
        # https://hande.readthedocs.io/en/latest/manual/integrals.html#fcidump-format
        i, a, j, b = (int(t) for t in tokens[1:])
        if i == a == j == b == 0:
            output["ecore"] = x
        elif a == j == b == 0:
            # TODO: x is the energy of the i-th MO
            continue
        elif j == b == 0:
            one_body.append((i, a, x))
        else:
            two_body.append((i, a, j, b, x))

    if not one_body and not two_body and re.search(r"\bNPY1", metadata, flags=re.IGNORECASE):
        raise QiskitNatureError(
            "The integrals of this FCIDump file are stored in separate .npy files (NPY1/NPY2), "
            "which is not supported. Write the integrals in ASCII format, e.g. with the "
            "`npy_and_ascii` option of the TCHInt library."
        )

    hermitian = None if output["ST"] is None else output["ST"] == 0

    hij_b = hijkl_ba = hijkl_bb = None
    if not _uhf:
        if any(max(i, a) > norb for i, a, _ in one_body):
            index = next((i, a) for i, a, _ in one_body if max(i, a) > norb)
            raise QiskitNatureError(f"Unknown 1-electron integral indices encountered in '{index}'")
        hij = _build_one_body(one_body, norb)
        hijkl = _build_two_body(two_body, norb, hermitian)
    else:
        hij = _build_one_body([e for e in one_body if max(e[:2]) <= norb], norb)
        hij_b = _build_one_body(
            [(i - norb, a - norb, x) for i, a, x in one_body if min(i, a) > norb], norb
        )
        aa_entries = [e for e in two_body if max(e[:4]) <= norb]
        bb_entries = [
            (i - norb, a - norb, j - norb, b - norb, x)
            for i, a, j, b, x in two_body
            if min(i, a, j, b) > norb
        ]
        mixed_entries = [e for e in two_body if max(e[:4]) > norb >= min(e[:4])]
        if hermitian is None:
            hermitian = _is_consistent(aa_entries, norb, _S8_PERMUTATIONS) and _is_consistent(
                bb_entries, norb, _S8_PERMUTATIONS
            )
        hijkl = _build_two_body(aa_entries, norb, hermitian)
        hijkl_bb = _build_two_body(bb_entries, norb, hermitian)
        hijkl_ba = _build_two_body_beta_alpha(mixed_entries, norb, hermitian)

    return FCIDump(
        num_electrons=output.get("NELEC"),
        hij=hij,
        hijkl=hijkl,
        hij_b=hij_b,
        hijkl_ba=hijkl_ba,
        hijkl_bb=hijkl_bb,
        multiplicity=output.get("MS2", 0) + 1,
        constant_energy=output.get("ecore", None),
        orbsym=output.get("ORBSYM", None),
        isym=output.get("ISYM"),
        metadata={
            key: output[key] for key in ("ST", "III", "ADJOINT", "IUHF") if output[key] is not None
        },
    )


def _parse_int_field(metadata: str, key: str) -> int | None:
    """Parses an optional integer field of the FCIDump header."""
    match = re.search(rf"\b{key}\s*=\s*([-+]?\d+)", metadata, flags=re.IGNORECASE)
    return int(match.groups()[0]) if match else None


def _build_one_body(entries: Sequence[tuple[int, int, float]], norb: int) -> np.ndarray:
    """Builds the 1-electron integral matrix.

    Missing entries are complemented by their transposed counterpart unless the listed entries
    show that the matrix is not symmetric.
    """
    hij = np.zeros((norb, norb))
    listed = np.zeros((norb, norb), dtype=bool)
    for i, a, x in entries:
        hij[i - 1, a - 1] = x
        listed[i - 1, a - 1] = True

    both = listed & listed.T
    if np.any(np.abs(hij - hij.T)[both] > _SYMMETRY_TOLERANCE):
        LOGGER.info("The 1-electron integrals are not symmetric and will not be symmetrized.")
        return hij

    fill = listed.T & ~listed
    hij[fill] = hij.T[fill]
    # remove numerical noise from symmetric matrices
    return 0.5 * (hij + hij.T)


def _build_two_body(
    entries: Sequence[tuple[int, int, int, int, float]], norb: int, hermitian: bool | None
) -> SymmetricTwoBodyIntegrals:
    """Builds the same-spin 2-electron integrals.

    Args:
        entries: the listed integrals (1-based indices in chemist' notation).
        norb: the number of orbitals.
        hermitian: whether the integrals are Hermitian. If ``None``, this is detected from the
            data.

    Raises:
        QiskitNatureError: if unknown indices are encountered or if ``hermitian=True`` but the
            entries contradict the 8-fold symmetry.

    Returns:
        An :class:`.S8Integrals` container for Hermitian and an :class:`.S1Integrals` container
        for non-Hermitian integrals.
    """
    for entry in entries:
        if max(entry[:4]) > norb:
            raise QiskitNatureError(
                f"Unknown 2-electron integral indices encountered in '{tuple(entry[:4])}'"
            )

    consistent = _is_consistent(entries, norb, _S8_PERMUTATIONS)
    if hermitian is None:
        hermitian = consistent
        if not hermitian:
            LOGGER.info(
                "The 2-electron integrals do not obey the 8-fold permutational symmetry of "
                "Hermitian integrals and are treated as non-Hermitian."
            )
    elif hermitian and not consistent:
        raise QiskitNatureError(
            "The FCIDump header indicates Hermitian integrals (ST=0) but the 2-electron integrals "
            "do not obey the 8-fold permutational symmetry."
        )

    if hermitian:
        s8_hijkl = S8Integrals.zero(norb)
        for i, a, j, b, x in entries:
            s8_hijkl[i - 1, a - 1, j - 1, b - 1] = x  # type: ignore[assignment]
        return s8_hijkl

    eri = np.zeros((norb, norb, norb, norb))
    listed = np.zeros(eri.shape, dtype=bool)
    for i, a, j, b, x in entries:
        eri[i - 1, a - 1, j - 1, b - 1] = x
        listed[i - 1, a - 1, j - 1, b - 1] = True
    # complement the entries by the electron exchange symmetry (ij|kl) = (kl|ij)
    exchanged = eri.transpose(_EXCHANGE)
    exchanged_listed = listed.transpose(_EXCHANGE)
    fill = exchanged_listed & ~listed
    eri[fill] = exchanged[fill]
    if np.any(np.abs(eri - eri.transpose(_EXCHANGE)) > _SYMMETRY_TOLERANCE):
        LOGGER.warning(
            "The 2-electron integrals do not obey the electron exchange symmetry (ij|kl) = (kl|ij) "
            "and get symmetrized. This does not change the 2-electron operator."
        )
        eri = 0.5 * (eri + eri.transpose(_EXCHANGE))
    return S1Integrals(eri)


def _build_two_body_beta_alpha(
    entries: Sequence[tuple[int, int, int, int, float]], norb: int, hermitian: bool
) -> SymmetricTwoBodyIntegrals:
    """Builds the beta-alpha 2-electron integrals from the mixed-spin entries of an unrestricted
    FCIDump file, which may list either of ``(ia|jb)`` with beta ``i, a`` and alpha ``j, b`` or the
    electron-exchanged ``(jb|ia)``."""
    eri: S4Integrals | np.ndarray = (
        S4Integrals.zero(norb) if hermitian else np.zeros((norb, norb, norb, norb))
    )
    for i, a, j, b, x in entries:
        first_beta = norb < min(i, a) and max(i, a) <= 2 * norb
        second_beta = norb < min(j, b) and max(j, b) <= 2 * norb
        if first_beta and max(j, b) <= norb:
            index = (i - 1 - norb, a - 1 - norb, j - 1, b - 1)
        elif second_beta and max(i, a) <= norb:
            index = (j - 1 - norb, b - 1 - norb, i - 1, a - 1)
        else:
            raise QiskitNatureError(
                f"Unknown 2-electron integral indices encountered in '{(i, a, j, b)}'"
            )
        eri[index] = x  # type: ignore[index]
    if hermitian:
        return eri  # type: ignore[return-value]
    return S1Integrals(eri)


def _is_consistent(
    entries: Sequence[tuple[int, int, int, int, float]],
    norb: int,
    permutations: Sequence[tuple[int, int, int, int]],
) -> bool:
    """Checks whether all listed entries which are related by the given index permutations have
    the same value.

    The entries are grouped by their canonical index (the smallest index among all permuted ones),
    such that the memory requirement scales with the number of entries only.
    """
    if not entries:
        return True
    data = np.asarray(entries)
    indices = data[:, :4].astype(np.int64) - 1
    values = data[:, 4]
    keys = np.min(
        np.stack(
            [
                ((indices[:, p] * norb + indices[:, q]) * norb + indices[:, r]) * norb
                + indices[:, s]
                for p, q, r, s in permutations
            ]
        ),
        axis=0,
    )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_values = values[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1])))
    spread = np.maximum.reduceat(sorted_values, starts) - np.minimum.reduceat(sorted_values, starts)
    return bool(np.all(spread <= _SYMMETRY_TOLERANCE))
