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

"""Test the FCIDump and TCDump formats with transcorrelated (non-Hermitian) integrals.

The data files stem from https://github.com/dobrautz/tc-varqite-hamiltonians (MIT license), which
holds the transcorrelated Hamiltonians of W. Dobrautz, et al., J. Chem. Theory Comput. 20, 4146
(2024), computed with the TCHInt library.
"""

import tempfile
import unittest
from pathlib import Path
from test import QiskitNatureTestCase

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.formats import fcidump_to_problem
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.tcdump import TCDump
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals, S8Integrals

DATA = Path(__file__).parent


def sector_eigenvalues(qubit_op: SparsePauliOp, norb: int, nalpha: int, nbeta: int) -> np.ndarray:
    """The eigenvalues of a block-spin-ordered Jordan-Wigner operator in an (Na, Nb) sector,
    sorted by their real parts."""
    indices = np.arange(2 ** (2 * norb))
    bits = (indices[:, None] >> np.arange(2 * norb)[None, :]) & 1
    sector = np.flatnonzero((bits[:, :norb].sum(1) == nalpha) & (bits[:, norb:].sum(1) == nbeta))
    matrix = qubit_op.to_matrix(sparse=True).tocsr()[sector][:, sector].toarray()
    eigenvalues = np.linalg.eigvals(matrix)
    return eigenvalues[np.argsort(eigenvalues.real)]


def read_qubit_hamiltonian(path: Path) -> SparsePauliOp:
    """Reads a qubit Hamiltonian in the plain text format of the tc-varqite-hamiltonians data."""
    terms = []
    with path.open(encoding="utf8") as file:
        for line in file:
            tokens = line.split()
            if len(tokens) == 2:
                terms.append((tokens[0], complex(tokens[1].strip("()"))))
    return SparsePauliOp.from_list(terms)


def random_tc_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random 1-electron integrals and 2-electron integrals (chemist' order) with only the
    electron exchange symmetry, like transcorrelated integrals."""
    rng = np.random.default_rng(seed)
    hij = rng.normal(size=(norb, norb))
    hijkl = rng.normal(size=(norb,) * 4)
    hijkl = 0.5 * (hijkl + hijkl.transpose(2, 3, 0, 1))
    return hij, hijkl


class TestTranscorrelatedFCIDump(QiskitNatureTestCase):
    """Test the FCIDump format with non-Hermitian integrals."""

    def test_detect_non_hermitian(self):
        """Test that the non-Hermitian 2-electron integrals of H2 are preserved."""
        fcidump = FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump")
        self.assertFalse(fcidump.is_hermitian)
        self.assertIsInstance(fcidump.hijkl, S1Integrals)
        eri = np.asarray(fcidump.hijkl.array)
        # values listed in the file
        self.assertAlmostEqual(eri[1, 0, 1, 0], 0.28488894580452484)
        self.assertAlmostEqual(eri[0, 1, 0, 1], 0.030501663284391495)
        self.assertAlmostEqual(eri[0, 1, 1, 0], 0.15769530454445815)
        # the electron exchange symmetry holds, the other symmetries do not
        np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1))
        self.assertFalse(np.allclose(eri, eri.transpose(1, 0, 3, 2)))

    def test_exchange_partners_are_filled(self):
        """Test that only listing one of each pair of electron-exchanged integrals suffices."""
        hij, hijkl = random_tc_integrals(3)
        hij = 0.5 * (hij + hij.T)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            lines = ["&FCI NORB=3,NELEC=2,MS2=0,", " ST=1,", "/"]
            written = set()
            for index in np.ndindex(hijkl.shape):
                partner = index[2:] + index[:2]
                if partner not in written:
                    written.add(index)
                    lines.append(f"{hijkl[index]:.16e} " + " ".join(str(i + 1) for i in index))
            for i, j in zip(*np.triu_indices(3)):
                lines.append(f"{hij[i, j]:.16e} {i + 1} {j + 1} 0 0")
            path.write_text("\n".join(lines) + "\n", encoding="utf8")
            fcidump = FCIDump.from_file(path)
        np.testing.assert_allclose(np.asarray(fcidump.hijkl.array), hijkl, atol=1e-14)
        np.testing.assert_allclose(fcidump.hij, hij, atol=1e-14)

    def test_round_trip(self):
        """Test writing and reading non-Hermitian integrals, including non-symmetric 1-electron
        integrals."""
        hij, hijkl = random_tc_integrals(3)
        fcidump = FCIDump(num_electrons=2, hij=hij, hijkl=S1Integrals(hijkl), constant_energy=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            fcidump.to_file(path)
            self.assertIn("ST=1", path.read_text(encoding="utf8"))
            parsed = FCIDump.from_file(path)
        np.testing.assert_allclose(parsed.hij, hij, atol=1e-14)
        np.testing.assert_allclose(np.asarray(parsed.hijkl.array), hijkl, atol=1e-14)
        self.assertAlmostEqual(parsed.constant_energy, 0.5)

    def test_unrestricted_round_trip(self):
        """Test writing and reading non-Hermitian unrestricted-spin integrals."""
        hij, hijkl = random_tc_integrals(2, seed=1)
        hij_b, hijkl_bb = random_tc_integrals(2, seed=2)
        hijkl_ba = np.random.default_rng(3).normal(size=(2,) * 4)
        fcidump = FCIDump(
            num_electrons=2,
            hij=hij,
            hijkl=S1Integrals(hijkl),
            hij_b=hij_b,
            hijkl_bb=S1Integrals(hijkl_bb),
            hijkl_ba=S1Integrals(hijkl_ba),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            fcidump.to_file(path)
            parsed = FCIDump.from_file(path)
        np.testing.assert_allclose(parsed.hij, hij, atol=1e-14)
        np.testing.assert_allclose(parsed.hij_b, hij_b, atol=1e-14)
        np.testing.assert_allclose(np.asarray(parsed.hijkl.array), hijkl, atol=1e-14)
        np.testing.assert_allclose(np.asarray(parsed.hijkl_bb.array), hijkl_bb, atol=1e-14)
        np.testing.assert_allclose(np.asarray(parsed.hijkl_ba.array), hijkl_ba, atol=1e-14)

    def test_unrestricted_mixed_spin_order(self):
        """Test that mixed-spin entries of unrestricted files are read in either electron order."""
        hij, hijkl = random_tc_integrals(2, seed=4)
        hijkl_ba = np.random.default_rng(5).normal(size=(2,) * 4)
        fcidump = FCIDump(
            num_electrons=2,
            hij=hij,
            hijkl=S1Integrals(hijkl),
            hij_b=hij,
            hijkl_bb=S1Integrals(hijkl),
            hijkl_ba=S1Integrals(hijkl_ba),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            fcidump.to_file(path)
            lines = path.read_text(encoding="utf8").splitlines()
            # the dumper lists the mixed-spin entries alpha first, (ia|jb) with alpha i, a and
            # beta j, b; list them as the electron-exchanged (jb|ia) instead
            swapped = []
            for line in lines:
                tokens = line.split()
                numeric = len(tokens) == 5 and all(t.isdigit() for t in tokens[1:])
                indices = [int(t) for t in tokens[1:]] if numeric else []
                if indices and min(indices) > 0 and max(indices[:2]) <= 2 < min(indices[2:]):
                    line = f"{tokens[0]} {tokens[3]} {tokens[4]} {tokens[1]} {tokens[2]}"
                swapped.append(line)
            self.assertNotEqual(swapped, lines)
            path.write_text("\n".join(swapped) + "\n", encoding="utf8")
            parsed = FCIDump.from_file(path)
        np.testing.assert_allclose(np.asarray(parsed.hijkl_ba.array), hijkl_ba, atol=1e-14)

    def test_exchange_symmetry_violation(self):
        """Test that 2-electron entries which violate the electron exchange symmetry are
        symmetrized with a warning, which does not change the 2-electron operator."""
        lines = [
            "&FCI NORB=2, NELEC=2, MS2=0, ST=1,",
            "&END",
            "0.50 1 1 1 1",
            "0.40 2 2 2 2",
            "0.30 1 2 2 1",
            "0.10 2 1 1 2",
            "-1.0 1 1 0 0",
            "-0.5 2 2 0 0",
            "0.05 1 2 0 0",
            "0.02 2 1 0 0",
            "0.7 0 0 0 0",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            path.write_text("\n".join(lines) + "\n", encoding="utf8")
            with self.assertLogs("qiskit_nature.second_q.formats.fcidump.parser", "WARNING"):
                parsed = FCIDump.from_file(path)
        eri = np.asarray(parsed.hijkl.array)
        self.assertAlmostEqual(eri[0, 1, 1, 0], 0.2)
        self.assertAlmostEqual(eri[1, 0, 0, 1], 0.2)
        listed = np.zeros((2,) * 4)
        listed[0, 0, 0, 0], listed[1, 1, 1, 1], listed[0, 1, 1, 0], listed[1, 0, 0, 1] = (
            0.5,
            0.4,
            0.3,
            0.1,
        )
        unsymmetrized = ElectronicEnergy.from_raw_integrals(parsed.hij, S1Integrals(listed))
        symmetrized = ElectronicEnergy.from_raw_integrals(parsed.hij, parsed.hijkl)
        difference = (unsymmetrized.second_q_op() - symmetrized.second_q_op()).normal_order()
        self.assertEqual(len(difference.simplify()), 0)

    def test_invalid_files(self):
        """Test that malformed files raise errors."""
        header = "&FCI NORB=2, NELEC=2, MS2=0,\n&END\n"
        cases = {
            "missing header end": "&FCI NORB=2, NELEC=2, MS2=0,\n0.5 1 1 1 1\n",
            "four tokens": header + "0.5 1 1 1\n",
            "1-electron index": header + "0.5 1 3 0 0\n",
            "2-electron index": header + "0.5 1 1 3 1\n",
            "mixed-spin index": "&FCI NORB=2, NELEC=2, MS2=0, IUHF=1,\n&END\n0.5 3 1 1 1\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in cases.items():
                with self.subTest(name):
                    path = Path(tmp) / "FCIDUMP"
                    path.write_text(text, encoding="utf8")
                    with self.assertRaises(QiskitNatureError):
                        FCIDump.from_file(path)

    def test_adjoint_flag(self):
        """Test that the ADJOINT=1 header entry is kept and reported."""
        text = "&FCI NORB=1, NELEC=2, MS2=0, ST=1, ADJOINT=1,\n&END\n0.5 1 1 1 1\n-1.0 1 1 0 0\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            path.write_text(text, encoding="utf8")
            with self.assertLogs("qiskit_nature.second_q.formats.fcidump.parser", "INFO"):
                parsed = FCIDump.from_file(path)
        self.assertEqual(parsed.metadata.get("ADJOINT"), 1)

    def test_hermitian_header_contradiction(self):
        """Test that ST=0 raises for integrals without the 8-fold symmetry."""
        content = (DATA / "h2_sto6g_tc.fcidump").read_text(encoding="utf8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            path.write_text(content.replace("ST=1", "ST=0"), encoding="utf8")
            with self.assertRaises(QiskitNatureError):
                FCIDump.from_file(path)
            # without the ST entry, the symmetry is detected from the data
            path.write_text(content.replace("ST=1,", ""), encoding="utf8")
            self.assertIsInstance(FCIDump.from_file(path).hijkl, S1Integrals)

    def test_hermitian_files_are_unchanged(self):
        """Test that Hermitian integrals are still stored with their 8-fold symmetry."""
        fcidump = FCIDump.from_file(DATA.parent / "fcidump" / "test_fcidump_h2.fcidump")
        self.assertTrue(fcidump.is_hermitian)
        self.assertIsInstance(fcidump.hijkl, S8Integrals)

    def test_format_variants(self):
        """Test the &End terminator, Fortran exponents and spaces in ORBSYM."""
        content = (
            " &FCI NORB=  2,NELEC=  2,MS2= 0,\n    ORBSYM=1, 1,\n    ISYM=1,\n &End\n"
            "  0.5D+00  1  1  1  1\n  0.25D+00  2  2  1  1\n  0.4D+00  2  2  2  2\n"
            " -1.0D+00  1  1  0  0\n -0.5D+00  2  2  0  0\n  0.7D+00  0  0  0  0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            path.write_text(content, encoding="utf8")
            fcidump = FCIDump.from_file(path)
        self.assertAlmostEqual(fcidump.hijkl[0, 0, 1, 1], 0.25)
        self.assertAlmostEqual(fcidump.constant_energy, 0.7)
        self.assertEqual(list(fcidump.orbsym), [1, 1])

    def test_xtc_header(self):
        """Test an xTC file (TCHInt format) whose three-body terms are folded into lower-order
        terms, such that no TCDUMP is required."""
        hij, hijkl = random_tc_integrals(2, seed=4)
        hij = 0.5 * (hij + hij.T)
        lines = [
            "&FCI",
            " NORB=2,",
            " NELEC=2,",
            " MS2=0,",
            " ISYM=1,",
            " ORBSYM=1,1,",
            " ST=1,",
            " III=0,",
            " OCC=1,",
            " CLOSED=1,",
            " IUHF=0,",
            " NPY1=int1.npy,",
            " NPY2=int2.npy,",
            " ENUC=    0.75     ,",
            " /",
        ]
        for index in np.ndindex(hijkl.shape):
            lines.append(f"{hijkl[index]:.16e} " + " ".join(str(i + 1) for i in index))
        for i, j in zip(*np.triu_indices(2)):
            lines.append(f"{hij[i, j]:.16e} {i + 1} {j + 1} 0 0")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            path.write_text("\n".join(lines) + "\n", encoding="utf8")
            fcidump = FCIDump.from_file(path)
            with self.assertNoLogs("qiskit_nature", level="WARNING"):
                problem = fcidump_to_problem(fcidump)
            # the integrals of the npy-only variant are not supported
            path.write_text("\n".join(lines[:15]) + "\n", encoding="utf8")
            with self.assertRaises(QiskitNatureError):
                FCIDump.from_file(path)
        self.assertEqual(fcidump.metadata, {"ST": 1, "III": 0, "IUHF": 0})
        self.assertAlmostEqual(fcidump.constant_energy, 0.75)
        np.testing.assert_allclose(np.asarray(fcidump.hijkl.array), hijkl, atol=1e-14)
        self.assertIsNone(problem.hamiltonian.three_body)
        self.assertFalse(fcidump.is_hermitian)

    def test_missing_tcdump_warns(self):
        """Test that omitting the TCDUMP of a full transcorrelated Hamiltonian warns if the
        three-body terms contribute, i.e. for at least three electrons."""
        content = (DATA / "be_sto6g_tc.fcidump").read_text(encoding="utf8")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            path.write_text(content.replace("ISYM=1,", "ISYM=1, III=1,"), encoding="utf8")
            fcidump = FCIDump.from_file(path)
        self.assertEqual(fcidump.metadata["III"], 1)
        with self.assertLogs("qiskit_nature", level="WARNING"):
            fcidump_to_problem(fcidump)
        with self.assertNoLogs("qiskit_nature", level="WARNING"):
            fcidump_to_problem(FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"))

    def test_six_index_entries_raise(self):
        """Test that 3-electron entries in an FCIDump file raise an error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FCIDUMP"
            path.write_text("&FCI NORB=1,NELEC=2,\n&END\n0.1 1 1 1 1 1 1\n", encoding="utf8")
            with self.assertRaises(QiskitNatureError):
                FCIDump.from_file(path)


class TestTCDump(QiskitNatureTestCase):
    """Test the TCDump format."""

    def test_symmetry_reduced_and_full_files(self):
        """Test that the symmetry-reduced Be and the full H2 TCDUMP files are parsed."""
        be_tcdump = TCDump.from_file(DATA / "be_sto6g_tc.tcdump")
        self.assertEqual(be_tcdump.num_orbitals, 5)
        self.assertEqual(be_tcdump.three_body.symmetry, "s48")
        self.assertEqual(len(be_tcdump.three_body), 141)
        # the full listing of these integrals holds 2227 non-zero entries
        self.assertEqual(sum(len(v) for _, v in be_tcdump.three_body.iter_expanded()), 2227)

        h2_tcdump = TCDump.from_file(DATA / "h2_sto6g_tc.tcdump", 2)
        self.assertEqual(h2_tcdump.three_body.symmetry, "s48")
        self.assertEqual(sum(len(v) for _, v in h2_tcdump.three_body.iter_expanded()), 32)

    def test_round_trip(self):
        """Test writing symmetry-reduced and full TCDUMP files."""
        tcdump = TCDump.from_file(DATA / "be_sto6g_tc.tcdump")
        dense = tcdump.three_body.to_dense()
        with tempfile.TemporaryDirectory() as tmp:
            for symmetry_reduced in (True, False):
                with self.subTest(symmetry_reduced=symmetry_reduced):
                    path = Path(tmp) / "TCDUMP"
                    tcdump.to_file(path, symmetry_reduced=symmetry_reduced)
                    num_lines = len(path.read_text(encoding="utf8").splitlines())
                    self.assertEqual(num_lines, 1 + (141 if symmetry_reduced else 2227))
                    parsed = TCDump.from_file(path)
                    np.testing.assert_array_equal(parsed.three_body.to_dense(), dense)

    def test_file_without_header(self):
        """Test a TCDUMP file without the header line, with blank lines, and a missing file."""
        text = "\n 5.0E-01  1  1  1  1  1  1\n\n 2.5E-01  1  2  1  2  1  1\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "TCDUMP"
            path.write_text(text, encoding="utf8")
            tcdump = TCDump.from_file(path)
            with self.assertRaises(QiskitNatureError):
                TCDump.from_file(Path(tmp) / "missing")
        self.assertEqual(tcdump.num_orbitals, 2)
        self.assertAlmostEqual(tcdump.three_body[0, 1, 0, 1, 0, 0], 0.25)

    def test_inconsistent_number_of_orbitals(self):
        """Test that a mismatch with the header raises."""
        with self.assertRaises(QiskitNatureError):
            TCDump.from_file(DATA / "be_sto6g_tc.tcdump", 4)
        with self.assertRaises(QiskitNatureError):
            fcidump_to_problem(
                FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"),
                TCDump.from_file(DATA / "be_sto6g_tc.tcdump"),
            )


class TestTranscorrelatedHamiltonians(QiskitNatureTestCase):
    """Test transcorrelated Hamiltonians built from FCIDump and TCDump files."""

    def test_h2(self):
        """Test the H2 Hamiltonian against the published qubit Hamiltonian."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "h2_sto6g_tc.fcidump"),
            TCDump.from_file(DATA / "h2_sto6g_tc.tcdump", 2),
        )
        qubit_op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
        self.assertTrue(np.iscomplexobj(np.real_if_close(qubit_op.simplify().coeffs)))
        energy = sector_eigenvalues(qubit_op, 2, 1, 1)[0]
        energy += problem.hamiltonian.nuclear_repulsion_energy
        # lowest eigenvalue of the published qubit Hamiltonian plus core energy
        self.assertAlmostEqual(energy.real, -1.1717073880, places=9)
        self.assertAlmostEqual(energy.imag, 0.0, places=12)

    def test_lih_published_qubit_hamiltonian(self):
        """Test the parity-mapped LiH Hamiltonian (with two-qubit reduction) term by term against
        the published one, which omits terms with coefficients below 1e-8."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "lih_4nos_tc.fcidump"),
            TCDump.from_file(DATA / "lih_4nos_tc.tcdump"),
        )
        mapper = ParityMapper(num_particles=problem.num_particles)
        qubit_op = mapper.map(problem.hamiltonian.second_q_op())
        reference = read_qubit_hamiltonian(DATA / "lih_4nos_tc_parity_qubit_hamiltonian.txt")
        difference = (qubit_op - reference).simplify(atol=1e-12)
        self.assertLess(np.max(np.abs(difference.coeffs)), 2e-8)

    def test_be_three_body(self):
        """Test the Be Hamiltonian with explicit three-body terms."""
        problem = fcidump_to_problem(
            FCIDump.from_file(DATA / "be_sto6g_tc.fcidump"),
            TCDump.from_file(DATA / "be_sto6g_tc.tcdump"),
        )
        qubit_op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
        energy = sector_eigenvalues(qubit_op, 5, 2, 2)[0]
        # obtained by an independent determinant-based TC-FCI code from the same integrals
        self.assertAlmostEqual(energy.real, -14.66465358, places=7)
        self.assertAlmostEqual(energy.imag, 0.0, places=12)

        problem.hamiltonian.three_body = None
        qubit_op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
        energy_without_three_body = sector_eigenvalues(qubit_op, 5, 2, 2)[0]
        self.assertGreater(abs(energy_without_three_body.real - energy.real), 1e-4)


if __name__ == "__main__":
    unittest.main()
