"""
Localize first, then compute grids only for the localized orbitals the user
picks (localization_io.localize_source / compute_localized_cubes and the
localized-orbital picker dialog), on the tracked Fe(CO)5 fchk fixture.
"""
import os
import time
import unittest

import numpy as np

import localization_io as loc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FECO5_FCHK = os.path.join(REPO, "test_files", "closed-shell", "gaussian", "feco5-hf.fchk")
GRID = 30          # coarse grids keep the tests quick


@unittest.skipUnless(os.path.isfile(FECO5_FCHK), "Fe(CO)5 fixture missing")
class LocalizeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loc._LOCALIZATION_CACHE.clear()
        cls.localized = loc.localize_source(FECO5_FCHK, space="occupied_valence")

    def test_localizes_without_computing_any_grid(self):
        r = self.localized
        self.assertNotIn("cubes", r)
        n = r["localized_cmo"].shape[1]
        self.assertGreater(n, 10)
        self.assertEqual(r["orbital_indices"], list(range(1, n + 1)))
        self.assertEqual(len(r["energies"]), n)
        self.assertEqual(r["label_base"], "feco5-hf_LOC_occupied_valence")

    def test_centres_name_real_atoms_largest_first(self):
        n_atoms = len(self.localized["atom_info"])
        for per_orbital in self.localized["centres"]:
            self.assertGreaterEqual(len(per_orbital), 1)
            fractions = [f for _a, f in per_orbital]
            self.assertEqual(fractions, sorted(fractions, reverse=True))
            self.assertTrue(all(0 <= a < n_atoms for a, _f in per_orbital))
        # Pipek-Mezey orbitals of Fe(CO)5 sit on Fe, C or O: every element appears.
        top_elements = {int(self.localized["atom_info"][c[0][0]][0])
                        for c in self.localized["centres"]}
        self.assertEqual(top_elements, {26, 6, 8})

    def test_same_options_reuse_the_cached_localization(self):
        start = time.perf_counter()
        again = loc.localize_source(FECO5_FCHK, space="occupied_valence")
        self.assertIs(again, self.localized)
        self.assertLess(time.perf_counter() - start, 0.5)
        other = loc.localize_source(FECO5_FCHK, space="occupied")
        self.assertIsNot(other, self.localized)

    def test_grids_only_for_the_chosen_orbitals_in_the_order_given(self):
        cubes = loc.compute_localized_cubes(self.localized, [5, 2], grid_quality=GRID)
        self.assertEqual([c["label"] for c in cubes],
                         ["feco5-hf_LOC_occupied_valence-5", "feco5-hf_LOC_occupied_valence-2"])
        # Same grid as computing every localized orbital at once.
        everything = loc.compute_localized_cube_data(
            FECO5_FCHK, space="occupied_valence", grid_quality=GRID)
        np.testing.assert_allclose(cubes[0]["grid"], everything["cubes"][4]["grid"])
        self.assertEqual(len(everything["cubes"]), self.localized["localized_cmo"].shape[1])

    def test_out_of_range_numbers_are_refused(self):
        n = self.localized["localized_cmo"].shape[1]
        with self.assertRaises(ValueError):
            loc.compute_localized_cubes(self.localized, [0], grid_quality=GRID)
        with self.assertRaises(ValueError):
            loc.compute_localized_cubes(self.localized, [n + 1], grid_quality=GRID)

    def test_orbital_centres_thresholds(self):
        pops = np.array([[0.70, 0.05], [0.25, 0.06], [0.05, 0.89]])
        self.assertEqual(loc.orbital_centres(pops),
                         [[(0, 0.70), (1, 0.25)], [(2, 0.89)]])
        self.assertEqual(loc.orbital_centres(np.array([[0.04], [0.03]])), [[(0, 0.04)]])


try:
    from viewer_harness import SKIP_REASON
except Exception as exc:   # pragma: no cover
    SKIP_REASON = str(exc)


@unittest.skipIf(SKIP_REASON is not None or not os.path.isfile(FECO5_FCHK),
                 SKIP_REASON or "Fe(CO)5 fixture missing")
class PickerDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.localized = loc.localize_source(FECO5_FCHK, space="occupied_valence")

    def dialog(self):
        import ChemInsight3D as ci
        return ci._LocalizedOrbitalPickerDialog(self.localized, GRID, 4.0)

    def test_lists_every_orbital_with_nothing_checked(self):
        dlg = self.dialog()
        n = self.localized["localized_cmo"].shape[1]
        self.assertEqual(len(dlg._checkboxes), n)
        self.assertEqual(dlg.selected_numbers(), [])
        self.assertFalse(dlg.compute_btn.isEnabled())
        self.assertIn("Ha", dlg._energy_labels[0].text())
        dlg.unit_combo.setCurrentText("eV")
        self.assertIn("eV", dlg._energy_labels[0].text())

    def test_selection_tools(self):
        dlg = self.dialog()
        dlg.range_edit.setText("2-4, 7, x")
        dlg._set_checked(__import__("ChemInsight3D")._parse_index_ranges(dlg.range_edit.text()))
        self.assertEqual(dlg.selected_numbers(), [2, 3, 4, 7])
        self.assertEqual(dlg.count_label.text(), f"4 of {len(dlg._checkboxes)} selected")
        self.assertTrue(dlg.compute_btn.isEnabled())
        dlg.atom_combo.setCurrentIndex(dlg.atom_combo.findData(0))          # Fe1
        self.assertEqual(dlg.atom_combo.currentText(), "Fe1")
        dlg._select_by_atom()
        on_fe = [i + 1 for i, c in enumerate(self.localized["centres"]) if c[0][0] == 0]
        self.assertTrue(on_fe)
        self.assertEqual(dlg.selected_numbers(), on_fe)

    def test_computes_only_the_checked_orbitals(self):
        from PyQt5.QtWidgets import QApplication
        dlg = self.dialog()
        dlg._set_checked([3, 1])
        dlg._start_compute()
        self.assertFalse(dlg.cancel_btn.isEnabled())
        dlg._thread.wait(120000)
        for _ in range(50):
            QApplication.processEvents()
            if dlg.results:
                break
            time.sleep(0.05)
        labels = [c["label"] for c in dlg.results]
        self.assertEqual(labels, ["feco5-hf_LOC_occupied_valence-1",
                                  "feco5-hf_LOC_occupied_valence-3"])
        details = dlg.results[0]["source_details"]
        self.assertEqual([o["label"] for o in details["orbitals"]], labels)   # population lookup
        self.assertEqual(details["localization"]["space"], "occupied_valence")


if __name__ == "__main__":
    unittest.main()
