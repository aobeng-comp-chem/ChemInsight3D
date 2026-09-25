"""
Regression tests on a real calculation: Fe(CO)5, RHF, 204 basis functions.

The same wavefunction is read from three formats:

    fchk   : test_files/closed-shell/gaussian/feco5-hf.fchk
    molden : test_files/closed-shell/cfour/MOLDEN_new.molden  (Molden2AIM)
    nbo    : test_files/closed-shell/cfour/MOLDEN.47 + MOLDEN.40

The tests check three things without needing any cube files:

1. Self-consistency of each source (overlap, orthonormality, electron count).
2. Agreement between the sources (orbital energies, electron density).
3. Pipek-Mezey localization invariants on the real occupied space.

The Gaussian fchk uses a different molecular orientation from the Molden/NBO
files, so fchk is compared to them only through rotation-invariant
quantities. Tests are skipped when the fixture files are not present.
"""

import os
import unittest
from itertools import combinations

import numpy as np

import density_analysis
import localization_io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAUSSIAN_DIR = os.path.join(ROOT, "test_files", "closed-shell", "gaussian")
CFOUR_DIR = os.path.join(ROOT, "test_files", "closed-shell", "cfour")

SOURCES = {
    "fchk": (os.path.join(GAUSSIAN_DIR, "feco5-hf.fchk"), None),
    "molden": (os.path.join(CFOUR_DIR, "MOLDEN_new.molden"), None),
    "nbo": (os.path.join(CFOUR_DIR, "MOLDEN.47"), os.path.join(CFOUR_DIR, "MOLDEN.40")),
}

N_ATOMS = 11
N_BASIS = 204
N_ELECTRONS = 96
N_OCC = 48
ATOMIC_NUMBERS = [26] + [6, 8] * 5

# Values printed in feco5-hf.log (5 decimals).
LOG_CORE_ENERGY = -261.36514
LOG_HOMO_ENERGY = -0.34958
LOG_LUMO_ENERGY = 0.09802

_MISSING = [p for path, key in SOURCES.values() for p in (path, key) if p and not os.path.exists(p)]
_SKIP_REASON = "Fe(CO)5 fixture files not found: " + ", ".join(_MISSING)

_LOADED = {}


def _load(name):
    """Load and cache everything the tests need from one source."""
    if name in _LOADED:
        return _LOADED[name]

    path, key = SOURCES[name]
    cmo, overlap, basis = localization_io.get_localization_inputs(path, key_path=key)
    fock = localization_io.get_fock_matrix(path, key_path=key, cmo=cmo, overlap=overlap)

    centers = {}
    for bf in basis:
        centers[bf["CENTER"]] = (bf["xcenter"], bf["ycenter"], bf["zcenter"])
    coords_bohr = np.array([centers[c] for c in sorted(centers)])

    data = {
        "cmo": cmo,
        "overlap": overlap,
        "fock": fock,
        "basis": basis,
        "coords_bohr": coords_bohr,
        "n_occ": localization_io.get_num_occupied_orbitals(path, key_path=key),
        "n_electrons": localization_io.get_electron_count(path, key_path=key),
    }
    _LOADED[name] = data
    return data


def _orbital_energies(data):
    c, f = data["cmo"], data["fock"]
    return np.diag(c.T @ f @ c)


def _total_density(data, points):
    """Closed-shell density rho = 2 * sum_occ |phi_i|^2 at points (bohr)."""
    c_occ = data["cmo"][:, :data["n_occ"]]
    alpha, _ = density_analysis._occupied_densities_at_points(
        data["basis"], data["coords_bohr"], points, c_occ, c_occ)
    return 2.0 * np.asarray(alpha)


def _invariant_points(coords):
    """Nuclei plus every pairwise midpoint: the same set in any orientation."""
    midpoints = [(coords[i] + coords[j]) / 2 for i, j in combinations(range(len(coords)), 2)]
    return np.vstack([coords, np.array(midpoints)])


def _pm_objective(cmo, overlap, center_ranges):
    return float((localization_io.atomic_populations(cmo, overlap, center_ranges) ** 2).sum())


# ── 1. Self-consistency of each source ─────────────────────────────────────


@unittest.skipIf(_MISSING, _SKIP_REASON)
class SourceConsistencyTests(unittest.TestCase):
    # NBO .47/.40 files store coefficients with fewer digits.
    ORTHO_TOL = {"fchk": 1e-6, "molden": 1e-6, "nbo": 1e-5}

    def test_sizes(self):
        for name in SOURCES:
            with self.subTest(source=name):
                data = _load(name)
                self.assertEqual(len(data["basis"]), N_BASIS)
                self.assertEqual(data["cmo"].shape, (N_BASIS, N_BASIS))
                self.assertEqual(len(data["coords_bohr"]), N_ATOMS)
                self.assertEqual(data["n_occ"], N_OCC)

    def test_atomic_numbers(self):
        import fchk_read
        import read_molden

        _, _, fchk_atoms = fchk_read.load_basis_from_fchk(SOURCES["fchk"][0])
        _, _, molden_atoms = read_molden.load_basis_from_molden(SOURCES["molden"][0])
        self.assertEqual([int(a[0]) for a in fchk_atoms], ATOMIC_NUMBERS)
        self.assertEqual([int(a[0]) for a in molden_atoms], ATOMIC_NUMBERS)

    def test_overlap_is_symmetric_normalized_positive_definite(self):
        for name in SOURCES:
            with self.subTest(source=name):
                s = _load(name)["overlap"]
                np.testing.assert_allclose(s, s.T, atol=1e-12)
                np.testing.assert_allclose(np.diag(s), 1.0, atol=1e-10)
                self.assertGreater(np.linalg.eigvalsh(s).min(), 0.0)

    def test_mos_are_orthonormal(self):
        for name in SOURCES:
            with self.subTest(source=name):
                data = _load(name)
                c, s = data["cmo"], data["overlap"]
                np.testing.assert_allclose(
                    c.T @ s @ c, np.eye(N_BASIS), atol=self.ORTHO_TOL[name])

    def test_electron_count(self):
        for name in SOURCES:
            with self.subTest(source=name):
                self.assertAlmostEqual(_load(name)["n_electrons"], N_ELECTRONS, delta=1e-4)

    def test_orbital_energies_match_gaussian_log(self):
        # C^T F C can swap degenerate pairs by rounding noise, larger for NBO.
        order_tol = {"fchk": 1e-6, "molden": 1e-6, "nbo": 1e-4}
        for name in SOURCES:
            with self.subTest(source=name):
                eps = _orbital_energies(_load(name))
                self.assertGreater(np.diff(eps).min(), -order_tol[name], "energies not ascending")
                self.assertAlmostEqual(eps[0], LOG_CORE_ENERGY, delta=1e-4)
                self.assertAlmostEqual(eps[N_OCC - 1], LOG_HOMO_ENERGY, delta=1e-5)
                self.assertAlmostEqual(eps[N_OCC], LOG_LUMO_ENERGY, delta=1e-5)


# ── 2. Agreement between formats ───────────────────────────────────────────


@unittest.skipIf(_MISSING, _SKIP_REASON)
class CrossFormatTests(unittest.TestCase):
    def test_orbital_energies_agree(self):
        reference = _orbital_energies(_load("fchk"))
        for name, tol in (("molden", 1e-6), ("nbo", 1e-4)):
            with self.subTest(source=name):
                np.testing.assert_allclose(_orbital_energies(_load(name)), reference, atol=tol)

    def test_molden_and_nbo_share_geometry(self):
        np.testing.assert_allclose(
            _load("molden")["coords_bohr"], _load("nbo")["coords_bohr"], atol=1e-5)

    def test_molden_and_nbo_density_agree_pointwise(self):
        # Same orientation, so compare directly at arbitrary points.
        points = np.random.default_rng(0).normal(scale=2.5, size=(200, 3))
        rho_molden = _total_density(_load("molden"), points)
        rho_nbo = _total_density(_load("nbo"), points)
        np.testing.assert_allclose(rho_nbo, rho_molden, rtol=1e-5, atol=1e-6)

    def test_density_at_invariant_points_agrees_with_fchk(self):
        # fchk is in a different orientation; the sorted density at nuclei
        # and bond midpoints does not depend on orientation or atom order.
        def invariant_density(name):
            data = _load(name)
            return np.sort(_total_density(data, _invariant_points(data["coords_bohr"])))

        reference = invariant_density("fchk")
        for name in ("molden", "nbo"):
            with self.subTest(source=name):
                np.testing.assert_allclose(invariant_density(name), reference, rtol=1e-5)


# ── 3. Pipek-Mezey localization ────────────────────────────────────────────


@unittest.skipIf(_MISSING, _SKIP_REASON)
class LocalizationTests(unittest.TestCase):
    ORTHO_TOL = {"fchk": 1e-6, "molden": 1e-6, "nbo": 1e-5}
    # Converged PM objective for the 48 occupied orbitals, same for all sources.
    EXPECTED_PM_OBJECTIVE = 39.6018

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for name in SOURCES:
            data = _load(name)
            basis, cmo, overlap, fock, _ = localization_io._reorder_for_contiguous_centers(
                data["basis"], data["cmo"], data["overlap"], data["fock"])
            ranges = localization_io.get_center_ranges(basis)
            for impl in (localization_io.localize_orbitals_cpp, localization_io.localize_orbitals):
                loc_c, loc_e = impl(cmo, overlap, fock, ranges,
                                    space="occupied", n_occ=N_OCC, seed=1)
                cls.results[name, impl.__name__] = (cmo, overlap, fock, ranges, loc_c, loc_e)

    def _each(self):
        for (name, impl), result in self.results.items():
            with self.subTest(source=name, impl=impl):
                yield name, result

    def test_localized_orbitals_are_orthonormal(self):
        for name, (_, s, _, _, loc_c, _) in self._each():
            self.assertEqual(loc_c.shape, (N_BASIS, N_OCC))
            np.testing.assert_allclose(
                loc_c.T @ s @ loc_c, np.eye(N_OCC), atol=self.ORTHO_TOL[name])

    def test_occupied_space_is_unchanged(self):
        # A unitary rotation within the occupied space leaves C_occ C_occ^T,
        # and therefore the density, exactly the same.
        for _, (c, _, _, _, loc_c, _) in self._each():
            c_occ = c[:, :N_OCC]
            np.testing.assert_allclose(loc_c @ loc_c.T, c_occ @ c_occ.T, atol=1e-10)

    def test_sum_of_orbital_energies_is_preserved(self):
        for _, (c, _, f, _, _, loc_e) in self._each():
            c_occ = c[:, :N_OCC]
            self.assertAlmostEqual(loc_e.sum(), np.trace(c_occ.T @ f @ c_occ), places=6)
            self.assertTrue(np.all(np.diff(loc_e) >= 0), "energies not sorted")

    def test_pm_objective_increases_to_expected_value(self):
        for _, (c, s, _, ranges, loc_c, _) in self._each():
            before = _pm_objective(c[:, :N_OCC], s, ranges)
            after = _pm_objective(loc_c, s, ranges)
            self.assertGreater(after, before)
            self.assertAlmostEqual(after, self.EXPECTED_PM_OBJECTIVE, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
