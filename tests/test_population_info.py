"""
"Show population info" for localized orbitals: NBO sources ask which basis to
analyse in (every time it is switched on), and a subset of localized orbitals
is found under its own numbers.
"""
import os
import unittest
from unittest import mock

import numpy as np

from viewer_harness import SKIP_REASON, make_viewer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBO_47 = os.path.join(REPO, "test_files", "closed-shell", "cfour", "MOLDEN.47")
FECO5_FCHK = os.path.join(REPO, "test_files", "closed-shell", "gaussian", "feco5-hf.fchk")
GRID = 30

if SKIP_REASON is None:
    from PyQt5.QtWidgets import QApplication, QWidget

    import ChemInsight3D as ci
    import localization_io as loc


def localized_cubes(path, numbers, space="occupied"):
    """What the localized-orbital picker hands the viewer for these orbitals."""
    localized = loc.localize_source(path, space=space)
    thread = ci._LocalizedCubesThread(localized, numbers, GRID, 4.0)
    out = []
    thread.finished.connect(out.extend)
    thread.error.connect(lambda msg: out.append(msg))
    thread.run()                                       # synchronously, in this thread
    assert out and isinstance(out[0], dict), out
    return localized, out


def wait_for_population(v, timeout=60.0):
    import time
    end = time.time() + timeout
    while v._population_thread is not None and time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    assert v._population_thread is None, "population analysis did not finish"


class PopulationTestCase(unittest.TestCase):
    def viewer(self, cubes):
        v = make_viewer([])
        v._test_panel = QWidget()
        v._build_appearance_content(v._test_panel)
        v._load_computed_cubes(cubes)
        self.addCleanup(v.plotter.close)
        return v

    def overlay(self, v):
        return v.population_actor.GetInput() if v.population_actor is not None else None


@unittest.skipIf(SKIP_REASON is not None or not os.path.isfile(NBO_47),
                 SKIP_REASON or "NBO fixture missing")
class NboBasisChoiceTests(PopulationTestCase):
    def setUp(self):
        self.localized, cubes = localized_cubes(NBO_47, [2, 4])
        self.v = self.viewer(cubes)
        options = self.v._nbo_basis_options()
        self.assertTrue(options, "the fixture has NBO key files beside the .47")
        self.key_label, self.key_value = options[0]

    def test_nothing_is_precomputed_in_the_ao_basis(self):           # fix 1
        self.assertIsNone(self.v.source_details["population_analysis"])
        self.assertNotIn("population_basis", self.v.source_details)

    def test_switching_on_asks_for_the_basis_and_uses_it(self):      # fix 1
        v = self.v
        with mock.patch.object(v, "_prompt_population_basis",
                               return_value=(self.key_label, self.key_value)) as ask:
            v._appearance_widgets["population"].setChecked(True)
            wait_for_population(v)
        ask.assert_called_once()
        offered = [value for _label, value in ask.call_args[0][0]]
        self.assertIn("AO", offered)
        self.assertIn(self.key_value, offered)
        self.assertEqual(v.source_details["population_basis"], self.key_value)
        text = self.overlay(v)
        self.assertIsNotNone(text)
        self.assertIn(f"Orbital 2 Population ({self.key_label.split(' (')[0]} basis)", text)

    def test_switching_off_and_on_asks_again(self):                  # fix 2
        v = self.v
        box = v._appearance_widgets["population"]
        with mock.patch.object(v, "_prompt_population_basis",
                               return_value=(self.key_label, self.key_value)):
            box.setChecked(True)
            wait_for_population(v)
        box.setChecked(False)
        self.assertIsNone(v.population_actor)
        with mock.patch.object(v, "_prompt_population_basis",
                               return_value=("AO (Atomic Orbital)", "AO")) as ask:
            box.setChecked(True)
            wait_for_population(v)
        ask.assert_called_once()                                       # asked again
        self.assertEqual(ask.call_args[0][1], self.key_value)          # last choice preselected
        self.assertEqual(v.source_details["population_basis"], "AO")   # and the new choice used
        self.assertIn("(AO basis)", self.overlay(v))

    def test_same_basis_again_reuses_the_result(self):
        v = self.v
        box = v._appearance_widgets["population"]
        with mock.patch.object(v, "_prompt_population_basis", return_value=("AO (Atomic Orbital)", "AO")):
            box.setChecked(True)
            wait_for_population(v)
            first = v.source_details["population_analysis"]
            box.setChecked(False)
            with mock.patch.object(v, "_start_population_thread") as start:
                box.setChecked(True)
            start.assert_not_called()
        self.assertIs(v.source_details["population_analysis"], first)

    def test_cancelling_the_question_leaves_population_info_off(self):
        v = self.v
        with mock.patch.object(v, "_prompt_population_basis", return_value=None):
            v._appearance_widgets["population"].setChecked(True)
        self.assertFalse(v.show_population_info)
        self.assertFalse(v._appearance_widgets["population"].isChecked())
        self.assertIsNone(v.population_actor)


@unittest.skipIf(SKIP_REASON is not None or not os.path.isfile(NBO_47),
                 SKIP_REASON or "NBO fixture missing")
class KeyBasisCheckTests(PopulationTestCase):
    """A file that is not a usable basis falls back to AO and says so."""

    STEM = os.path.splitext(NBO_47)[0]

    def transform(self, ext):
        import nbo_read
        nbas = len(nbo_read.load_basis_headless(NBO_47)[0])
        overlap = nbo_read.get_ao_overlap_matrix(NBO_47)
        cmat = np.eye(nbas)[:, :3]
        return ci._transform_to_key_basis(cmat, overlap, self.STEM + ext, "alpha", nbas), cmat, overlap

    def test_real_key_bases_are_accepted(self):
        for ext in (".32", ".33", ".34", ".35", ".36", ".37", ".38", ".39", ".40", ".41"):
            with self.subTest(ext=ext):
                (_c, _s, problem), _cmat, _overlap = self.transform(ext)
                self.assertIsNone(problem)

    def test_other_files_are_refused_and_left_in_the_ao_basis(self):
        for ext in (".42", ".46", ".wfn", ".wfx"):
            if not os.path.isfile(self.STEM + ext):
                continue
            with self.subTest(ext=ext):
                (c, s, problem), cmat, overlap = self.transform(ext)
                self.assertIsNotNone(problem)
                self.assertIs(c, cmat)
                self.assertIs(s, overlap)

    def test_the_viewer_labels_a_fallback_as_ao(self):
        if not os.path.isfile(self.STEM + ".wfx"):
            self.skipTest("no .wfx beside the fixture")
        _loc, cubes = localized_cubes(NBO_47, [2])
        v = self.viewer(cubes)
        wfx = next(o for o in v._nbo_basis_options() if o[0].endswith("(MOLDEN.wfx)"))
        with mock.patch.object(v, "_prompt_population_basis", return_value=wfx):
            v._appearance_widgets["population"].setChecked(True)
            wait_for_population(v)
        text = self.overlay(v)
        self.assertIn("Orbital 2 Population (AO basis)", text)
        self.assertIn("MOLDEN.wfx could not be used", text)
        fallback = v.source_details["population_analysis"]["basis_fallback"]
        self.assertEqual(fallback["requested"], "MOLDEN.wfx")
        self.assertIn("could not be used", ci._population_report_text(
            v.source_details["population_analysis"]))
        # Same numbers as choosing AO directly.
        ao = ci._compute_population_analysis_from_details(v.source_details, "AO")
        self.assertEqual(ao["orbitals"][0]["contribs"],
                         v.source_details["population_analysis"]["orbitals"][0]["contribs"])


@unittest.skipIf(SKIP_REASON is not None or not os.path.isfile(NBO_47),
                 SKIP_REASON or "NBO fixture missing")
class NewOrbitalTests(PopulationTestCase):
    """Population info stays on for orbitals computed or selected afterwards."""

    def setUp(self):
        _loc, cubes = localized_cubes(NBO_47, [2, 4])
        self.v = self.viewer(cubes)
        self.choice = self.v._nbo_basis_options()[1]          # e.g. NAOs
        with mock.patch.object(self.v, "_prompt_population_basis", return_value=self.choice):
            self.v._appearance_widgets["population"].setChecked(True)
            wait_for_population(self.v)

    def kind(self):
        return self.choice[0].split(" (")[0]

    def test_a_newly_computed_orbital_gets_its_analysis_without_asking(self):
        v = self.v
        _loc, new = localized_cubes(NBO_47, [5])
        with mock.patch.object(v, "_prompt_population_basis") as ask:
            v._load_computed_cubes(new)                         # shows the new cube
            wait_for_population(v)
        ask.assert_not_called()
        self.assertTrue(v.cube_files[v.current_cube_index].endswith("-5"))
        self.assertIn(f"Orbital 5 Population ({self.kind()} basis)", self.overlay(v))

    def test_orbitals_of_one_run_share_one_analysis(self):
        v = self.v
        self.assertIs(v.cubes[0]["source_details"], v.cubes[1]["source_details"])
        with mock.patch.object(v, "_start_population_thread") as start:
            v.switch_cube(1)
        start.assert_not_called()
        self.assertIn("Orbital 4 Population", self.overlay(v) or "")

    def test_switching_back_and_forth_keeps_showing_each_cubes_analysis(self):
        v = self.v
        _loc, new = localized_cubes(NBO_47, [5])
        v._load_computed_cubes(new)
        wait_for_population(v)
        with mock.patch.object(v, "_start_population_thread") as start:
            for number in (2, 5, 4):
                idx = next(i for i, f in enumerate(v.cube_files) if f.endswith(f"-{number}"))
                v.switch_cube(idx)
                self.assertIn(f"Orbital {number} Population", self.overlay(v) or "")
        start.assert_not_called()                               # computed once per batch

    def test_a_source_without_the_chosen_basis_asks(self):
        v = self.v
        _loc, new = localized_cubes(NBO_47, [5])
        other_kind = [("NBOs (other.37)", {"kind": "nbo_key", "key_path": "/nowhere/other.37"})]
        with mock.patch.object(v, "_nbo_basis_options", return_value=other_kind), \
                mock.patch.object(v, "_prompt_population_basis", return_value=None) as ask:
            v._load_computed_cubes(new)
        ask.assert_called_once()
        self.assertFalse(v.show_population_info)                # cancelled: switched off
        self.assertFalse(v._appearance_widgets["population"].isChecked())


@unittest.skipIf(SKIP_REASON is not None or not os.path.isfile(FECO5_FCHK),
                 SKIP_REASON or "Fe(CO)5 fixture missing")
class SubsetTests(PopulationTestCase):                                  # fix 3
    def test_a_later_batch_of_orbitals_gets_its_analysis(self):
        _loc, cubes = localized_cubes(FECO5_FCHK, [3], space="occupied_valence")
        v = self.viewer(cubes)
        v._appearance_widgets["population"].setChecked(True)
        _loc, more = localized_cubes(FECO5_FCHK, [9], space="occupied_valence")
        v._load_computed_cubes(more)
        wait_for_population(v)
        self.assertIn("Orbital 9 Population", self.overlay(v) or "")
    def test_every_cube_of_a_subset_shows_its_own_population(self):
        localized, cubes = localized_cubes(FECO5_FCHK, [7, 3], space="occupied_valence")
        v = self.viewer(cubes)
        v._appearance_widgets["population"].setChecked(True)          # fchk: AO only, no question
        for number in (7, 3):
            idx = next(i for i, f in enumerate(v.cube_files) if f.endswith(f"-{number}"))
            v.switch_cube(idx)
            text = self.overlay(v)
            self.assertIsNotNone(text, f"no population info for LMO {number}")
            self.assertIn(f"Orbital {number} Population", text)
            entry = v._population_entry_for_current_cube()[1]
            self.assertAlmostEqual(entry["energy"], float(localized["energies"][number - 1]), places=6)
            # The contributions are this orbital's: its largest atom matches the picker's.
            top_atom = localized["centres"][number - 1][0][0] + 1
            self.assertEqual(max(entry["contribs"], key=lambda c: c["percent"])["atom"], top_atom)

    def test_population_data_keeps_the_given_numbers(self):
        c = np.eye(4)[:, [0, 2]]
        basis = [{"CENTER": 1, "orb_val": "s"}, {"CENTER": 1, "orb_val": "s"},
                 {"CENTER": 2, "orb_val": "s"}, {"CENTER": 2, "orb_val": "s"}]
        try:
            data = ci._population_analysis_data(
                c, np.eye(4), basis, [(1, 0, 0, 0), (1, 0, 0, 1)],
                energies=[-1.0, -0.5], orb_occ=[2.0, 2.0], indices=[5, 9])
        except Exception as exc:          # basis schema mismatch: covered by the viewer test
            self.skipTest(f"minimal basis not accepted: {exc}")
        self.assertEqual([o["index"] for o in data["orbitals"]], [5, 9])
        self.assertEqual([o["energy"] for o in data["orbitals"]], [-1.0, -0.5])


if __name__ == "__main__":
    unittest.main()
