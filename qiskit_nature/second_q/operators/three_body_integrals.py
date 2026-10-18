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
Three-body electronic integrals (:mod:`qiskit_nature.second_q.operators.three_body_integrals`)
==============================================================================================

.. currentmodule:: qiskit_nature.second_q.operators.three_body_integrals

This module provides a container for spin-independent three-body electronic integrals, as they
arise for example in the transcorrelated method.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ThreeBodyIntegrals

.. autodata:: S48_PERMUTATIONS
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from typing import Literal

import numpy as np

from .fermionic_op import FermionicOp

LOGGER = logging.getLogger(__name__)

_SYMMETRY_TOLERANCE = 1e-10
"""Absolute tolerance below which two symmetry-related integrals are considered equal."""

_CHUNK_SIZE = 10_000
"""The number of unique integrals which are expanded at once."""


def _s48_permutations() -> tuple[tuple[int, ...], ...]:
    """Returns the 48 index permutations of the symmetry group of three-body integrals.

    The indices ``(p, q, r, s, t, u)`` are ordered such that ``(p, s)``, ``(q, t)`` and ``(r, u)``
    are the orbital index pairs of the three electrons. The group consists of the 3! permutations
    of the electrons and of the 2^3 exchanges of the two indices of each electron.
    """
    permutations = set()
    for electrons in itertools.permutations(range(3)):
        for flips in itertools.product((False, True), repeat=3):
            perm = [0] * 6
            for new, old in enumerate(electrons):
                perm[new], perm[new + 3] = (old + 3, old) if flips[new] else (old, old + 3)
            permutations.add(tuple(perm))
    return tuple(sorted(permutations))


S48_PERMUTATIONS = _s48_permutations()
"""The 48 index permutations of the symmetry group of real, spin-independent three-body
integrals of local three-body operators."""


class ThreeBodyIntegrals:
    r"""A container for spin-independent three-body electronic integrals.

    Such integrals arise, for example, in the transcorrelated method, where the similarity
    transformation of the electronic Hamiltonian with a Jastrow factor introduces a three-body
    operator. The container represents the operator

    .. math::

        \frac{1}{2} \sum_{pqrstu} \sum_{\sigma\tau\lambda} L_{pqrstu}\,
        a^\dagger_{p\sigma} a^\dagger_{q\tau} a^\dagger_{r\lambda}
        a_{u\lambda} a_{t\tau} a_{s\sigma},

    where :math:`(p, s)`, :math:`(q, t)` and :math:`(r, u)` are the orbital index pairs of the
    three electrons and the sum runs over all spatial orbital indices and spins. The prefactor of
    :math:`1/2` follows the convention of the TCDUMP files written by the TCHInt library [1] and
    read by NECI [2], such that :math:`L_{pqrstu}` are exactly the values stored in such files (see
    also :class:`~qiskit_nature.second_q.formats.tcdump.TCDump`).

    For real orbitals, the three-body operator of the transcorrelated method is Hermitian and its
    integrals obey a 48-fold permutational symmetry: they are invariant under the 3! permutations
    of the electrons and under the exchange of the two indices of each electron (see
    :data:`S48_PERMUTATIONS`). In this case (``symmetry="s48"``), only the symmetry-unique
    integrals are stored, which reduces the memory requirement by up to a factor of 48. Integrals
    without this symmetry are stored as given (``symmetry="s1"``).

    The integrals are provided as a list of entries (indices and values). Entries which are not
    listed (directly or, in the ``"s48"`` case, through a symmetry-related entry) are zero. Hence,
    both the symmetry-reduced and the full listing of the integrals are accepted.

    References:
        [1]: P. López Ríos, et al., TCHInt: transcorrelated Hamiltonian integrals,
            https://www2.fkf.mpg.de/alavi/tchint/stable/run.html

        [2]: K. Guther, et al., NECI: N-Electron Configuration Interaction with an emphasis on
            state-of-the-art stochastic methods, J. Chem. Phys. 153, 034107 (2020).
    """

    def __init__(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        num_orbitals: int,
        *,
        symmetry: Literal["s48", "s1"] | None = None,
    ) -> None:
        """
        Args:
            indices: an integer array of shape ``(num_entries, 6)`` holding the 0-based spatial
                orbital indices ``(p, q, r, s, t, u)`` of the listed integrals.
            values: the values of the listed integrals.
            num_orbitals: the number of spatial orbitals.
            symmetry: the permutational symmetry of the integrals. If ``"s48"``, the integrals
                are assumed to obey the 48-fold symmetry and listed entries which are related by
                symmetry must agree. If ``"s1"``, no symmetry is assumed. If ``None``, the
                48-fold symmetry is used unless it contradicts the listed entries.

        Raises:
            ValueError: if the indices are out of bounds, if the shapes of ``indices`` and
                ``values`` do not match, if an invalid symmetry is requested, or if
                ``symmetry="s48"`` contradicts the listed entries.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, 6)
        values = np.asarray(values, dtype=float).reshape(-1)
        if indices.shape[0] != values.shape[0]:
            raise ValueError(
                f"The number of indices ({indices.shape[0]}) and values ({values.shape[0]}) "
                "do not match."
            )
        if indices.size and (indices.min() < 0 or indices.max() >= num_orbitals):
            raise ValueError(
                f"The orbital indices must lie in the range [0, {num_orbitals}) but the range "
                f"[{indices.min()}, {indices.max()}] was encountered."
            )
        if symmetry not in (None, "s48", "s1"):
            raise ValueError(f"Invalid symmetry '{symmetry}', expected 's48', 's1' or None.")

        self._num_orbitals = num_orbitals

        if symmetry in (None, "s48"):
            unique = self._reduce_s48(indices, values, num_orbitals)
            if unique is not None:
                self._symmetry: Literal["s48", "s1"] = "s48"
                self._indices, self._values = unique
                return
            if symmetry == "s48":
                raise ValueError(
                    "The listed three-body integrals contradict the 48-fold permutational "
                    "symmetry."
                )
            LOGGER.warning(
                "The listed three-body integrals do not obey the 48-fold permutational symmetry "
                "and are stored without exploiting any symmetry."
            )

        self._symmetry = "s1"
        # remove duplicate entries, keeping the last one
        keys = _encode(indices, num_orbitals)
        _, last = np.unique(keys[::-1], return_index=True)
        keep = np.sort(len(keys) - 1 - last)
        self._indices, self._values = indices[keep], values[keep]

    @staticmethod
    def _reduce_s48(
        indices: np.ndarray, values: np.ndarray, num_orbitals: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Reduces the listed entries to the symmetry-unique ones.

        Returns:
            The unique indices and values or ``None`` if the listed entries contradict the
            symmetry.
        """
        canonical, keys = _canonicalize_s48(indices, num_orbitals)
        if not keys.size:
            return canonical, values
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_values = values[order]
        starts = np.flatnonzero(np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1])))
        spread = np.maximum.reduceat(sorted_values, starts) - np.minimum.reduceat(
            sorted_values, starts
        )
        if np.any(spread > _SYMMETRY_TOLERANCE):
            return None
        return canonical[order[starts]], sorted_values[starts]

    @property
    def num_orbitals(self) -> int:
        """The number of spatial orbitals."""
        return self._num_orbitals

    @property
    def register_length(self) -> int:
        """The number of spin orbitals."""
        return 2 * self._num_orbitals

    @property
    def symmetry(self) -> Literal["s48", "s1"]:
        """The permutational symmetry exploited by this container."""
        return self._symmetry

    @property
    def indices(self) -> np.ndarray:
        """The 0-based orbital indices of the stored (in the ``"s48"`` case, symmetry-unique)
        integrals."""
        return self._indices

    @property
    def values(self) -> np.ndarray:
        """The values of the stored (in the ``"s48"`` case, symmetry-unique) integrals."""
        return self._values

    def __len__(self) -> int:
        return self._values.shape[0]

    def __getitem__(self, index: tuple[int, int, int, int, int, int]) -> float:
        """Returns the integral with the given 0-based orbital indices ``(p, q, r, s, t, u)``."""
        indices = np.asarray(index, dtype=np.int64).reshape(1, 6)
        if self._symmetry == "s48":
            _, key = _canonicalize_s48(indices, self._num_orbitals)
            stored_keys = _canonicalize_s48(self._indices, self._num_orbitals)[1]
        else:
            key = _encode(indices, self._num_orbitals)
            stored_keys = _encode(self._indices, self._num_orbitals)
        match = np.flatnonzero(stored_keys == key[0])
        return float(self._values[match[0]]) if match.size else 0.0

    def iter_expanded(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterates over all (symmetry-expanded) non-zero integrals in chunks.

        Yields:
            Tuples of an index array of shape ``(num_entries, 6)`` and the corresponding values.
            Every index tuple occurs exactly once.
        """
        if self._symmetry == "s1":
            mask = self._values != 0.0
            yield self._indices[mask], self._values[mask]
            return

        permutations = np.asarray(S48_PERMUTATIONS)
        for start in range(0, len(self), _CHUNK_SIZE):
            indices = self._indices[start : start + _CHUNK_SIZE]
            values = self._values[start : start + _CHUNK_SIZE]
            mask = values != 0.0
            indices, values = indices[mask], values[mask]
            # all symmetry-related index tuples, shape (num_entries, 48, 6)
            images = indices[:, permutations].reshape(-1, 6)
            image_values = np.repeat(values, len(permutations))
            # the orbits of distinct unique entries are disjoint
            _, first = np.unique(_encode(images, self._num_orbitals), return_index=True)
            yield images[first], image_values[first]

    def to_dense(self) -> np.ndarray:
        """Returns the full 6-dimensional array of the integrals.

        .. warning::

           The memory requirement of this array scales with the sixth power of the number of
           orbitals.

        Returns:
            The array of shape ``(num_orbitals,) * 6`` indexed by ``(p, q, r, s, t, u)``.
        """
        dense = np.zeros((self._num_orbitals,) * 6)
        for indices, values in self.iter_expanded():
            dense[tuple(indices.T)] = values
        return dense

    def is_hermitian(self) -> bool:
        """Checks whether the represented three-body operator is Hermitian.

        Returns:
            Whether the integrals are invariant under exchanging the creation and annihilation
            indices, :math:`L_{pqrstu} = L_{stupqr}`.
        """
        if self._symmetry == "s48":
            return True
        stored = dict(zip(_encode(self._indices, self._num_orbitals).tolist(), self._values))
        adjoint_keys = _encode(self._indices[:, [3, 4, 5, 0, 1, 2]], self._num_orbitals)
        return all(
            abs(value - stored.get(key, 0.0)) <= _SYMMETRY_TOLERANCE
            for key, value in zip(adjoint_keys.tolist(), self._values)
        )

    def second_q_op(self) -> FermionicOp:
        """Constructs the second-quantized operator of the three-body integrals.

        The spin orbitals are block-ordered: the first half of the register holds the alpha-spin
        orbitals, the second half the beta-spin orbitals.

        For integrals with the 48-fold symmetry, the six permutations of the electrons yield
        identical terms. Each set of three distinct (creation, annihilation) spin-orbital pairs is
        therefore included only once, with a prefactor of :math:`6 \\cdot 1/2 = 3`.

        Returns:
            The :class:`.FermionicOp` of the three-body operator.
        """
        norb = self._num_orbitals
        spins = np.asarray(list(itertools.product((0, 1), repeat=3)))  # shape (8, 3)
        data: dict[str, float] = {}
        for indices, values in self.iter_expanded():
            # all spin assignments of all index tuples, shape (num_entries, 8, 3)
            creators = indices[:, None, :3] + norb * spins[None, :, :]
            annihilators = indices[:, None, 3:] + norb * spins[None, :, :]
            creators = creators.reshape(-1, 3)
            annihilators = annihilators.reshape(-1, 3)
            coeffs = np.repeat(values, len(spins))

            # terms with repeated creation or annihilation operators vanish
            mask = _all_distinct(creators) & _all_distinct(annihilators)
            if self._symmetry == "s48":
                # keep only the ordering in which the (creator, annihilator) pairs are sorted
                pairs = creators * 2 * norb + annihilators
                mask &= (pairs[:, 0] < pairs[:, 1]) & (pairs[:, 1] < pairs[:, 2])
                factor = 3.0
            else:
                factor = 0.5

            for (c_1, c_2, c_3), (a_1, a_2, a_3), coeff in zip(
                creators[mask].tolist(), annihilators[mask].tolist(), coeffs[mask].tolist()
            ):
                label = f"+_{c_1} +_{c_2} +_{c_3} -_{a_3} -_{a_2} -_{a_1}"
                data[label] = data.get(label, 0.0) + factor * coeff

        return FermionicOp(data, num_spin_orbitals=2 * norb, copy=False)

    def __repr__(self) -> str:
        return (
            f"ThreeBodyIntegrals(num_orbitals={self._num_orbitals}, symmetry='{self._symmetry}', "
            f"num_stored={len(self)})"
        )


def _encode(indices: np.ndarray, num_orbitals: int) -> np.ndarray:
    """Encodes 6-tuples of orbital indices into unique integers."""
    key = np.zeros(indices.shape[0], dtype=np.int64)
    for column in range(6):
        key = key * num_orbitals + indices[:, column]
    return key


def _canonicalize_s48(indices: np.ndarray, num_orbitals: int) -> tuple[np.ndarray, np.ndarray]:
    """Maps index tuples to the canonical representative of their 48-fold symmetry orbit.

    The two indices of every electron are sorted and the three electrons are ordered, following
    the indexing of the NECI code.

    Returns:
        The canonical index tuples and an integer key which uniquely identifies the orbit.
    """
    low = np.minimum(indices[:, :3], indices[:, 3:])
    high = np.maximum(indices[:, :3], indices[:, 3:])
    pairs = low * num_orbitals + high
    order = np.argsort(pairs, axis=1, kind="stable")
    low = np.take_along_axis(low, order, axis=1)
    high = np.take_along_axis(high, order, axis=1)
    pairs = np.take_along_axis(pairs, order, axis=1)
    num_pairs = num_orbitals * num_orbitals
    keys = (pairs[:, 0] * num_pairs + pairs[:, 1]) * num_pairs + pairs[:, 2]
    return np.concatenate([low, high], axis=1), keys


def _all_distinct(triples: np.ndarray) -> np.ndarray:
    """Checks for every row of a ``(num_rows, 3)`` array whether its entries are distinct."""
    return (
        (triples[:, 0] != triples[:, 1])
        & (triples[:, 0] != triples[:, 2])
        & (triples[:, 1] != triples[:, 2])
    )
