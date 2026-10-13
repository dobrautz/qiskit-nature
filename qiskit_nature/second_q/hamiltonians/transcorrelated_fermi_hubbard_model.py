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

"""The transcorrelated Fermi-Hubbard model of a one-dimensional chain."""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np

from qiskit_nature.second_q.operators import FermionicOp

from .lattice_model import LatticeModel
from .lattices import BoundaryCondition, LineLattice

# A translation-invariant term sum_r prod_m o_m(r + offset_m) is described by its coefficient and a
# tuple of (is_creator, spin, offset) for the ladder operators o_m.
_Term = tuple[complex, tuple[tuple[bool, int, int], ...]]


class TranscorrelatedFermiHubbardModel(LatticeModel):
    r"""The transcorrelated Fermi-Hubbard model of a one-dimensional chain.

    The Hubbard Hamiltonian

    .. math::

        H = -t \sum_{\langle i,j\rangle,\sigma} a^\dagger_{i\sigma} a_{j\sigma}
        + U \sum_i n_{i\uparrow} n_{i\downarrow}

    (where the sum runs over both orientations of all bonds) is similarity transformed with the
    Gutzwiller correlator :math:`\hat{g} = J \sum_i n_{i\uparrow} n_{i\downarrow}` [1, 2],

    .. math::

        H_{TC} = e^{-\hat{g}} H e^{\hat{g}} = U \sum_i n_{i\uparrow} n_{i\downarrow}
        - t \sum_{\langle i,j\rangle,\sigma} a^\dagger_{i\sigma} a_{j\sigma}
        \left[1 + (e^{J} - 1) n_{j\bar\sigma} + (e^{-J} - 1) n_{i\bar\sigma}
        - 2 (\cosh J - 1) n_{i\bar\sigma} n_{j\bar\sigma}\right].

    The transcorrelated Hamiltonian is not Hermitian but has the same spectrum as :math:`H`. Its
    right ground state :math:`e^{-\hat{g}}|\Psi_0\rangle` is more compact than :math:`|\Psi_0\rangle`
    for suitable (negative) :math:`J`, in particular in the momentum-space representation [1, 2].

    The Hamiltonian can be represented in the site basis (``basis="real"``) or, for periodic
    boundary conditions, in the basis of the momentum eigenstates
    :math:`c_k = N^{-1/2} \sum_r e^{-ikr} a_r` (``basis="momentum"``), in which it contains
    three-body terms. In the latter case, the momentum orbitals are ordered by their band energy
    :math:`\epsilon_k = -2t\cos k` (ascending, see :attr:`momenta`), such that the
    :class:`~qiskit_nature.second_q.circuit.library.HartreeFock` state is the Fermi sea.

    The spin orbitals are block-ordered: the first half of the register holds the spin-up
    orbitals, the second half the spin-down orbitals.

    For a periodic chain of two sites, the bond between the sites is counted twice, which doubles
    the hopping as in [1].

    The model can be solved as a :class:`.LatticeModelProblem`, e.g. with the
    :class:`.FermionicVarQITESolver` or the :class:`.VarQITEMinimumEigensolver`:

    .. code-block:: python

        from qiskit_nature.second_q.hamiltonians import TranscorrelatedFermiHubbardModel
        from qiskit_nature.second_q.problems import LatticeModelProblem

        model = TranscorrelatedFermiHubbardModel(4, correlation=-0.73, basis="momentum")
        problem = LatticeModelProblem(model)

    **References:**

    [1] I. O. Sokolov, W. Dobrautz, H. Luo, A. Alavi and I. Tavernelli, Orders of magnitude
    increased accuracy for quantum many-body problems on quantum computers via an exact
    transcorrelated method, `Phys. Rev. Research 5, 023174 (2023)
    <https://doi.org/10.1103/PhysRevResearch.5.023174>`_.

    [2] W. Dobrautz, H. Luo and A. Alavi, Compact numerical solutions to the two-dimensional
    repulsive Hubbard model obtained via nonunitary similarity transformations,
    `Phys. Rev. B 99, 075119 (2019) <https://doi.org/10.1103/PhysRevB.99.075119>`_.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        num_sites: int,
        hopping: float = 1.0,
        onsite_interaction: float = 4.0,
        correlation: float = 0.0,
        *,
        boundary: Literal["periodic", "open"] = "periodic",
        basis: Literal["real", "momentum"] = "real",
    ) -> None:
        """
        Args:
            num_sites: The number of sites of the chain.
            hopping: The hopping parameter :math:`t`.
            onsite_interaction: The on-site interaction :math:`U`.
            correlation: The correlation parameter :math:`J` of the Gutzwiller correlator. For
                ``correlation=0``, the (Hermitian) Hubbard Hamiltonian is obtained.
            boundary: The boundary conditions of the chain.
            basis: The single-particle basis, either the site basis (``"real"``) or the momentum
                basis (``"momentum"``, requires periodic boundary conditions).

        Raises:
            ValueError: If the momentum basis is requested for open boundary conditions or if
                the arguments are invalid.
        """
        if num_sites < 2:
            raise ValueError("The chain must have at least two sites.")
        if boundary not in ("periodic", "open"):
            raise ValueError(f"Invalid boundary conditions '{boundary}'.")
        if basis not in ("real", "momentum"):
            raise ValueError(f"Invalid basis '{basis}'.")
        if basis == "momentum" and boundary != "periodic":
            raise ValueError("The momentum basis requires periodic boundary conditions.")
        super().__init__(
            LineLattice(
                num_sites,
                edge_parameter=-hopping,
                boundary_condition=(
                    BoundaryCondition.PERIODIC if boundary == "periodic" else BoundaryCondition.OPEN
                ),
            )
        )
        self.num_sites = num_sites
        self.hopping = hopping
        self.onsite_interaction = onsite_interaction
        self.correlation = correlation
        self.boundary = boundary
        self.basis = basis

    @property
    def register_length(self) -> int:
        return 2 * self.num_sites

    @property
    def momenta(self) -> np.ndarray:
        """The momenta :math:`k` of the orbitals in the momentum basis, ordered by their band
        energy."""
        momenta = 2 * np.pi * np.arange(self.num_sites) / self.num_sites
        energies = np.round(-2 * self.hopping * np.cos(momenta), 12)
        return momenta[np.argsort(energies, kind="stable")]

    def second_q_op(self) -> FermionicOp:
        """Returns the second-quantized operator of the transcorrelated Hubbard model.

        Returns:
            The :class:`.FermionicOp` of the Hamiltonian.
        """
        terms = self._terms()
        if self.basis == "real":
            data = self._real_space(terms)
        else:
            data = self._momentum_space(terms)
        return FermionicOp(data, num_spin_orbitals=self.register_length).simplify()

    def _terms(self) -> list[_Term]:
        """The translation-invariant terms of the Hamiltonian."""
        hopping, interaction, j = self.hopping, self.onsite_interaction, self.correlation
        prefactors = (1.0, np.exp(-j) - 1, np.exp(j) - 1, -2 * (np.cosh(j) - 1))
        terms: list[_Term] = []
        for spin in (0, 1):
            other = 1 - spin
            # hopping from site j = i + offset to site i, for both orientations of the bond
            for offset_i, offset_j in ((0, 1), (1, 0)):
                hop = ((True, spin, offset_i), (False, spin, offset_j))
                n_i = ((True, other, offset_i), (False, other, offset_i))
                n_j = ((True, other, offset_j), (False, other, offset_j))
                for prefactor, densities in zip(prefactors, ((), n_i, n_j, n_i + n_j)):
                    if prefactor != 0.0:
                        terms.append((-hopping * prefactor, hop + densities))
        terms.append((interaction, ((True, 0, 0), (False, 0, 0), (True, 1, 0), (False, 1, 0))))
        return terms

    def _real_space(self, terms: list[_Term]) -> dict[str, complex]:
        num_sites = self.num_sites
        num_bonds = num_sites if self.boundary == "periodic" else num_sites - 1
        data: dict[str, complex] = {}
        for coeff, operators in terms:
            max_offset = max(offset for _, _, offset in operators)
            for site in range(num_bonds if max_offset > 0 else num_sites):
                label = " ".join(
                    f"{'+' if creator else '-'}_{spin * num_sites + (site + offset) % num_sites}"
                    for creator, spin, offset in operators
                )
                data[label] = data.get(label, 0.0) + coeff
        return data

    def _momentum_space(self, terms: list[_Term]) -> dict[str, complex]:
        num_sites = self.num_sites
        momenta = self.momenta
        # the orbital index of each momentum index n (k = 2 pi n / N)
        orbital = np.empty(num_sites, dtype=int)
        orbital[np.rint(momenta * num_sites / (2 * np.pi)).astype(int) % num_sites] = np.arange(
            num_sites
        )
        data: dict[str, complex] = {}
        for coeff, operators in terms:
            num_ops = len(operators)
            # sum_r prod_m o_m(r + d_m) = N^(1 - M/2) sum_{k: sum_m s_m k_m = 0} prod_m e^(i s_m k_m d_m)
            # with s_m = -1 for creation and +1 for annihilation operators
            signs = [-1 if creator else 1 for creator, _, _ in operators]
            prefactor = coeff * num_sites ** (1 - num_ops / 2)
            for free in itertools.product(range(num_sites), repeat=num_ops - 1):
                last = (-signs[-1] * sum(s * n for s, n in zip(signs, free))) % num_sites
                indices = free + (last,)
                phase = sum(
                    s * n * offset for s, n, (_, _, offset) in zip(signs, indices, operators)
                )
                value = prefactor * np.exp(2j * np.pi * phase / num_sites)
                label = " ".join(
                    f"{'+' if creator else '-'}_{spin * num_sites + orbital[n]}"
                    for n, (creator, spin, _) in zip(indices, operators)
                )
                data[label] = data.get(label, 0.0) + value
        return data
