import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import density_analysis
import fchk_read
import nbo_read
import overlap_matrix
import read_molden
from source_cache import ComputationCache, file_cache_key


class ComputationCacheTests(unittest.TestCase):
    def test_file_signature_invalidates_cached_value(self):
        cache = ComputationCache()
        calls = []

        with tempfile.NamedTemporaryFile("w", delete=False) as handle:
            handle.write("first")
            path = handle.name

        try:
            def load():
                calls.append(None)
                return len(calls)

            key = ("value", file_cache_key(path))
            self.assertEqual(cache.get(key, load), 1)
            self.assertEqual(cache.get(key, load), 1)

            with open(path, "a") as handle:
                handle.write("-changed")

            changed_key = ("value", file_cache_key(path))
            self.assertNotEqual(key, changed_key)
            self.assertEqual(cache.get(changed_key, load), 2)
            self.assertEqual(len(calls), 2)
        finally:
            os.unlink(path)


class FchkCacheTests(unittest.TestCase):
    def setUp(self):
        fchk_read.clear_source_cache()
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "sample.fchk")

    def tearDown(self):
        fchk_read.clear_source_cache()
        self.tempdir.cleanup()

    def test_basis_normalization_runs_once_and_invalidates_on_change(self):
        with open(self.path, "w") as handle:
            handle.write("basis-v1")

        normalized = [{"CENTER": 1, "orb_val": "s"}]
        with (
            mock.patch.object(fchk_read, "_extract_basis_set", return_value=[{}]) as extract,
            mock.patch.object(fchk_read, "_extract_atoms", return_value=([(0, 0, 0)], [(1, 0, 0, 0)])),
            mock.patch.object(fchk_read, "_normalise_basis", return_value=normalized) as normalize,
        ):
            first = fchk_read.load_basis_from_fchk(self.path)
            second = fchk_read.load_basis_from_fchk(self.path)
            self.assertIs(first, second)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(normalize.call_count, 1)

            with open(self.path, "a") as handle:
                handle.write("-changed")

            third = fchk_read.load_basis_from_fchk(self.path)
            self.assertIsNot(third, first)
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(normalize.call_count, 2)

    def test_cmo_matrix_is_parsed_once_for_multiple_subsets(self):
        with open(self.path, "w") as handle:
            handle.write(
                "Number of basis functions                 I              2\n"
                "Alpha MO coefficients                     R   N=          4\n"
                " 1.0 0.0 0.0 1.0\n"
            )

        with mock.patch.object(fchk_read, "_parse_array", wraps=fchk_read._parse_array) as parse:
            first = fchk_read.load_cmos_from_fchk(self.path, [1], "alpha")
            second = fchk_read.load_cmos_from_fchk(self.path, [2], "alpha")

        np.testing.assert_array_equal(first[0], [1.0, 0.0])
        np.testing.assert_array_equal(second[0], [0.0, 1.0])
        self.assertEqual(parse.call_count, 1)


    def test_occupation_derivation_reuses_cached_cmo_matrix(self):
        with open(self.path, "w") as handle:
            handle.write(
                "Number of basis functions                 I              2\n"
                "Alpha Orbital Energies                    R   N=          2\n"
                " -0.5 0.2\n"
                "Alpha MO coefficients                     R   N=          4\n"
                " 1.0 0.0 0.0 1.0\n"
                "Total SCF Density                         R   N=          3\n"
                " 1.0 0.0 1.0\n"
            )

        with (
            mock.patch.object(fchk_read, "_parse_array", wraps=fchk_read._parse_array) as parse,
            mock.patch.object(fchk_read, "get_ao_overlap_matrix", return_value=np.eye(2)),
        ):
            fchk_read.load_cmos_from_fchk(self.path, [1], "alpha")
            fchk_read.get_orbital_energies_and_occupations_fchk(self.path)

        mo_reads = [
            call for call in parse.call_args_list
            if len(call.args) > 1 and call.args[1] == "Alpha MO coefficients"
        ]
        self.assertEqual(len(mo_reads), 1)


    def test_final_overlap_is_computed_once(self):
        with open(self.path, "w") as handle:
            handle.write("overlap")

        expected = np.eye(1)
        with (
            mock.patch.object(fchk_read, "load_basis_from_fchk", return_value=([{}], [], [])),
            mock.patch.object(overlap_matrix, "get_overlap_matrix", return_value=expected) as get_overlap,
        ):
            first = fchk_read.get_ao_overlap_matrix(self.path)
            second = fchk_read.get_ao_overlap_matrix(self.path)

        self.assertIs(first, second)
        self.assertEqual(get_overlap.call_count, 1)


class MoldenCacheTests(unittest.TestCase):
    def setUp(self):
        read_molden.clear_source_cache()
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "sample.molden")
        with open(self.path, "w") as handle:
            handle.write("molden-v1")

    def tearDown(self):
        read_molden.clear_source_cache()
        self.tempdir.cleanup()

    def test_full_parse_runs_once_and_invalidates_on_change(self):
        parsed = {"nbas": 1}
        with mock.patch.object(read_molden, "_parse_molden_uncached", return_value=parsed) as parse:
            self.assertIs(read_molden._parse_molden(self.path), parsed)
            self.assertIs(read_molden._parse_molden(self.path), parsed)
            self.assertEqual(parse.call_count, 1)

            with open(self.path, "a") as handle:
                handle.write("-changed")

            self.assertIs(read_molden._parse_molden(self.path), parsed)
            self.assertEqual(parse.call_count, 2)

    def test_basis_normalization_and_final_overlap_are_each_cached(self):
        parsed = {
            "raw_basis": [{}],
            "coordinates_ang": [(0, 0, 0)],
            "atom_info": [(1, 0, 0, 0)],
        }
        normalized = [{"CENTER": 1, "orb_val": "s"}]
        expected_overlap = np.eye(1)
        with (
            mock.patch.object(read_molden, "_parse_molden", return_value=parsed),
            mock.patch.object(read_molden, "_normalise_basis", return_value=normalized) as normalize,
            mock.patch.object(overlap_matrix, "get_overlap_matrix", return_value=expected_overlap) as get_overlap,
        ):
            first_basis = read_molden.load_basis_from_molden(self.path)
            second_basis = read_molden.load_basis_from_molden(self.path)
            first_overlap = read_molden.get_ao_overlap_matrix(self.path)
            second_overlap = read_molden.get_ao_overlap_matrix(self.path)

        self.assertIs(first_basis, second_basis)
        self.assertIs(first_overlap, second_overlap)
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual(get_overlap.call_count, 1)


class NboCacheTests(unittest.TestCase):
    def setUp(self):
        nbo_read.clear_source_cache()
        self.tempdir = tempfile.TemporaryDirectory()
        self.key_path = os.path.join(self.tempdir.name, "sample.40")
        with open(self.key_path, "w") as handle:
            handle.write("HEADER\nCMO\nCOMMENT\n1.0 0.0\n0.0 1.0\n")

    def tearDown(self):
        nbo_read.clear_source_cache()
        self.tempdir.cleanup()

    def test_full_cmo_matrix_is_loaded_once_for_multiple_subsets(self):
        with (
            mock.patch.object(nbo_read, "get_orbital_count", return_value=("CMO", 2, False)),
            mock.patch.object(nbo_read, "_detect_open_shell_key", return_value=False),
            mock.patch.object(nbo_read, "_single_block_open_shell_key", return_value=False),
            mock.patch.object(nbo_read, "_read_key_lines", wraps=nbo_read._read_key_lines) as read_lines,
        ):
            first = nbo_read.load_cmos_headless(self.key_path, [1], "alpha")
            second = nbo_read.load_cmos_headless(self.key_path, [2], "alpha")

        np.testing.assert_array_equal(first[0], [1.0, 0.0])
        np.testing.assert_array_equal(second[0], [0.0, 1.0])
        self.assertEqual(read_lines.call_count, 1)


class SpinDensityReuseTests(unittest.TestCase):
    def test_closed_shell_loads_only_alpha_and_aliases_beta_occ(self):
        basis = [{"CENTER": 1, "orb_val": "s", "coeffs": [1.0], "exps": [1.0]}]
        alpha_cmo = np.ones((1, 1))
        spins_loaded = []
        captured = {}

        def localization_inputs(path, key_path=None, spin="alpha"):
            spins_loaded.append(spin)
            return alpha_cmo, np.eye(1), basis

        def occupied_densities(basis_set, coordinates, points, alpha_occ, beta_occ):
            captured["alpha_occ"] = alpha_occ
            captured["beta_occ"] = beta_occ
            n = len(points)
            return np.ones(n), np.ones(n)

        with (
            mock.patch.object(read_molden, "load_basis_from_molden", return_value=(basis, [(0, 0, 0)], [(1, 0, 0, 0)])),
            mock.patch.object(density_analysis, "_get_occupation_arrays", return_value=("molden", np.array([2.0]), None)),
            mock.patch.object(density_analysis, "get_localization_inputs", side_effect=localization_inputs),
            mock.patch.object(density_analysis, "_build_uniform_grid", return_value=(np.zeros((1, 3)), (1, 1, 1), np.ones(3), np.zeros(3), np.zeros((1, 3)))),
            mock.patch.object(density_analysis, "_occupied_densities_at_points", side_effect=occupied_densities),
        ):
            result = density_analysis.compute_spin_density_cube_data("sample.molden")

        self.assertEqual(spins_loaded, ["alpha"])
        np.testing.assert_array_equal(captured["alpha_occ"], captured["beta_occ"])
        # alpha_density == beta_density here, so the default "spin" cube is zero.
        np.testing.assert_array_equal(result["cubes"]["spin"]["grid"], np.zeros((1, 1, 1)))

    def test_closed_shell_total_density_is_alpha_plus_beta(self):
        basis = [{"CENTER": 1, "orb_val": "s", "coeffs": [1.0], "exps": [1.0]}]
        alpha_cmo = np.ones((1, 1))

        def localization_inputs(path, key_path=None, spin="alpha"):
            return alpha_cmo, np.eye(1), basis

        def occupied_densities(basis_set, coordinates, points, alpha_occ, beta_occ):
            n = len(points)
            return np.full(n, 3.0), np.full(n, 3.0)

        with (
            mock.patch.object(read_molden, "load_basis_from_molden", return_value=(basis, [(0, 0, 0)], [(1, 0, 0, 0)])),
            mock.patch.object(density_analysis, "_get_occupation_arrays", return_value=("molden", np.array([2.0]), None)),
            mock.patch.object(density_analysis, "get_localization_inputs", side_effect=localization_inputs),
            mock.patch.object(density_analysis, "_build_uniform_grid", return_value=(np.zeros((1, 3)), (1, 1, 1), np.ones(3), np.zeros(3), np.zeros((1, 3)))),
            mock.patch.object(density_analysis, "_occupied_densities_at_points", side_effect=occupied_densities),
        ):
            result = density_analysis.compute_spin_density_cube_data(
                "sample.molden", density_type="total")

        self.assertEqual(set(result["cubes"]), {"total"})
        np.testing.assert_array_equal(result["cubes"]["total"]["grid"], np.full((1, 1, 1), 6.0))
        self.assertTrue(result["cubes"]["total"]["label"].endswith("_TOTAL_DENSITY"))

    def test_open_shell_passes_distinct_alpha_and_beta_occupied_cmo(self):
        basis = [{"CENTER": 1, "orb_val": "s", "coeffs": [1.0], "exps": [1.0]}]
        alpha_cmo = np.array([[2.0]])
        beta_cmo = np.array([[3.0]])
        captured = {}

        def localization_inputs(path, key_path=None, spin="alpha"):
            return (alpha_cmo if spin == "alpha" else beta_cmo), np.eye(1), basis

        def occupied_densities(basis_set, coordinates, points, alpha_occ, beta_occ):
            captured["alpha_occ"] = alpha_occ
            captured["beta_occ"] = beta_occ
            n = len(points)
            return np.ones(n), 2.0 * np.ones(n)

        with (
            mock.patch.object(read_molden, "load_basis_from_molden", return_value=(basis, [(0, 0, 0)], [(1, 0, 0, 0)])),
            mock.patch.object(density_analysis, "_get_occupation_arrays", return_value=("molden", np.array([2.0]), np.array([2.0]))),
            mock.patch.object(density_analysis, "get_localization_inputs", side_effect=localization_inputs),
            mock.patch.object(density_analysis, "_build_uniform_grid", return_value=(np.zeros((1, 3)), (1, 1, 1), np.ones(3), np.zeros(3), np.zeros((1, 3)))),
            mock.patch.object(density_analysis, "_occupied_densities_at_points", side_effect=occupied_densities),
        ):
            result = density_analysis.compute_spin_density_cube_data(
                "sample.molden", density_type=("alpha", "beta", "spin"))

        np.testing.assert_array_equal(captured["alpha_occ"], alpha_cmo)
        np.testing.assert_array_equal(captured["beta_occ"], beta_cmo)
        np.testing.assert_array_equal(result["cubes"]["alpha"]["grid"], np.ones((1, 1, 1)))
        np.testing.assert_array_equal(result["cubes"]["beta"]["grid"], np.full((1, 1, 1), 2.0))
        np.testing.assert_array_equal(result["cubes"]["spin"]["grid"], np.full((1, 1, 1), -1.0))

    def test_python_fallback_evaluates_ao_basis_only_once_for_multiple_types(self):
        # Force the pure-Python path by making the compiled extension look
        # unavailable, and confirm the (expensive) AO evaluation still
        # happens exactly once no matter how many density types are asked
        # for -- alpha/beta are evaluated once and every type is a cheap
        # elementwise combination of those two grids.
        basis = [{"CENTER": 1, "orb_val": "s", "coeffs": [1.0], "exps": [1.0]}]
        alpha_cmo = np.array([[2.0]])
        beta_cmo = np.array([[3.0]])

        def localization_inputs(path, key_path=None, spin="alpha"):
            return (alpha_cmo if spin == "alpha" else beta_cmo), np.eye(1), basis

        with (
            mock.patch.dict(sys.modules, {"electron_density_opt_omp": None}),
            mock.patch.object(read_molden, "load_basis_from_molden", return_value=(basis, [(0, 0, 0)], [(1, 0, 0, 0)])),
            mock.patch.object(density_analysis, "_get_occupation_arrays", return_value=("molden", np.array([2.0]), np.array([2.0]))),
            mock.patch.object(density_analysis, "get_localization_inputs", side_effect=localization_inputs),
            mock.patch.object(density_analysis, "_build_uniform_grid", return_value=(np.zeros((1, 3)), (1, 1, 1), np.ones(3), np.zeros(3), np.zeros((1, 3)))),
            mock.patch.object(density_analysis, "basis_values_at_points", wraps=density_analysis.basis_values_at_points) as ao_eval,
        ):
            result = density_analysis.compute_spin_density_cube_data(
                "sample.molden", density_type=("alpha", "beta", "total", "spin"))

        self.assertEqual(ao_eval.call_count, 1)
        self.assertEqual(set(result["cubes"]), {"alpha", "beta", "total", "spin"})

    def test_unknown_density_type_raises(self):
        with self.assertRaises(ValueError):
            density_analysis.compute_spin_density_cube_data(
                "sample.molden", density_type="bogus")

    def test_empty_density_type_raises(self):
        with self.assertRaises(ValueError):
            density_analysis.compute_spin_density_cube_data(
                "sample.molden", density_type=())


class NativeOccupiedDensityExtensionTests(unittest.TestCase):
    """
    Correctness check for electron_density_opt_omp.occupied_density(),
    the C++ kernel _occupied_densities_at_points() prefers: it must agree
    with the trusted NumPy formula (|ao_values @ occ|**2).sum(axis=1) --
    see density_analysis._occupied_densities_at_points -- across multiple
    angular-momentum shells, centers, and occupied orbitals.

    Skips (rather than fails) when the extension hasn't been compiled,
    same as the try/except ImportError fallback used at runtime.
    """

    def setUp(self):
        try:
            import electron_density_opt_omp as _cpp
        except ImportError:
            self.skipTest("electron_density_opt_omp extension is not built")
        if not hasattr(_cpp, "occupied_density"):
            self.skipTest("electron_density_opt_omp.occupied_density is not built")
        self.cpp = _cpp

    def test_occupied_density_matches_numpy_reference(self):
        rng = np.random.default_rng(0)
        basis = [
            {"CENTER": 1, "orb_val": "s",   "coeffs": [0.6, 0.4], "exps": [1.2, 0.4]},
            {"CENTER": 1, "orb_val": "px",  "coeffs": [1.0],      "exps": [0.8]},
            {"CENTER": 1, "orb_val": "dc2", "coeffs": [1.0],      "exps": [0.6]},
            {"CENTER": 2, "orb_val": "s",   "coeffs": [1.0],      "exps": [0.9]},
            {"CENTER": 2, "orb_val": "fc1", "coeffs": [1.0],      "exps": [0.5]},
        ]
        coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        points = rng.normal(scale=1.5, size=(200, 3))

        n_basis = len(basis)
        alpha_occ = rng.normal(size=(n_basis, 3))
        beta_occ = rng.normal(size=(n_basis, 2))

        alpha_cpp, beta_cpp = self.cpp.occupied_density(
            basis, coordinates, points, alpha_occ, beta_occ)

        ao_values = density_analysis.basis_values_at_points(basis, coordinates, points)
        alpha_ref = (np.abs(ao_values @ alpha_occ) ** 2).sum(axis=1)
        beta_ref = (np.abs(ao_values @ beta_occ) ** 2).sum(axis=1)

        np.testing.assert_allclose(alpha_cpp, alpha_ref, atol=1e-10)
        np.testing.assert_allclose(beta_cpp, beta_ref, atol=1e-10)

    def test_occupied_density_handles_zero_occupied_orbitals(self):
        basis = [{"CENTER": 1, "orb_val": "s", "coeffs": [1.0], "exps": [1.0]}]
        coordinates = np.zeros((1, 3))
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        alpha_density, beta_density = self.cpp.occupied_density(
            basis, coordinates, points,
            np.zeros((1, 0)), np.zeros((1, 0)),
        )

        np.testing.assert_array_equal(alpha_density, np.zeros(2))
        np.testing.assert_array_equal(beta_density, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
