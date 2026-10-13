"""The viewer on top of the disk-backed cube store: records, saving, sessions."""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import viewer_harness as h

if h.SKIP_REASON is None:
    import cube_store
    from PyQt5.QtWidgets import QApplication, QListWidget

    ci = h.ci


def result_dict(label, seed=0, n=24, details=None):
    """A compute-thread result like the fchk/NBO/Molden paths produce."""
    rng = np.random.default_rng(seed)
    ax = np.linspace(-6, 6, n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    grid = (x + 0.3 * seed) * np.exp(-(x ** 2 + y ** 2 + z ** 2) / 4) + 1e-3 * rng.standard_normal(x.shape)
    return dict(label=label, atom_info=[(z_, a * h.BOHR, b * h.BOHR, c * h.BOHR) for z_, a, b, c in h.WATER],
                bohr_const=h.BOHR, origin=np.array([-6.0] * 3), spacing=np.array([ax[1] - ax[0]] * 3),
                nx=n, ny=n, nz=n, grid=grid,
                source_details=details if details is not None else {"file": f"{label}.fchk",
                                                                    "coefficients": np.arange(3.0)})


@unittest.skipIf(h.SKIP_REASON, h.SKIP_REASON or "")
class ViewerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ci3d-storage-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def viewer(self, with_list=True):
        v = h.make_viewer()
        if with_list:
            v.cube_list = QListWidget()
            v.cube_list.currentRowChanged.connect(v.switch_cube)
            self.addCleanup(v.cube_list.deleteLater)
        self.addCleanup(v.plotter.close)
        self.addCleanup(v.cube_store.close)
        return v

    def cube_file(self, name="a.cube", seed=0):
        """A .cube file on disk written by the viewer's own writer."""
        path = os.path.join(self.tmp, name)
        writer = ci.MultiCubeVisualizer()
        writer._write_cube(writer._result_to_cube(result_dict(name, seed)), path)
        return path


class CubeWriterTests(ViewerCase):
    def test_saved_cube_reopens_with_the_same_grid(self):
        v = self.viewer(with_list=False)
        cube = v._result_to_cube(result_dict("x", n=13))
        cube["data"] = cube["data"][:, :11, :]        # unequal sides catch transposes
        cube["dimensions"] = cube["data"].shape
        path = os.path.join(self.tmp, "x.cube")
        v._write_cube(cube, path)
        back = v.read_cube(path)
        self.assertEqual(back["dimensions"], (13, 11, 13))
        np.testing.assert_allclose(back["data"], cube["data"], rtol=1e-5, atol=1e-10)

    def test_writer_matches_the_nbo_writer_layout(self):
        import nbo_read
        r = result_dict("x", n=9)
        r.update(nx=9, ny=9, nz=9)
        v = self.viewer(with_list=False)
        a, b = os.path.join(self.tmp, "viewer.cube"), os.path.join(self.tmp, "nbo.cube")
        v._write_cube(v._result_to_cube(r), a)
        nbo_read.write_cube_from_result(r, b)
        np.testing.assert_allclose(v.read_cube(a)["data"], v.read_cube(b)["data"], rtol=1e-4, atol=1e-9)


class RecordTests(ViewerCase):
    def test_opened_files_become_records_with_paths(self):
        path = self.cube_file()
        v = self.viewer()
        v._replace_cubes_with([path])
        rec = v.cubes[0]
        self.assertIsInstance(rec, cube_store.CubeRecord)
        self.assertEqual(rec["source_path"], path)
        self.assertEqual(rec["label"], "a.cube")
        self.assertFalse(rec["unsaved"])
        self.assertTrue(os.path.isfile(v.cube_store.path(rec["grid_id"])))

    def test_vtk_grid_shares_the_stored_array(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("m")])
        stored = v.cube_store.get(v.cubes[0]["grid_id"])
        values = v._create_grid().point_data["values"]
        self.assertTrue(np.shares_memory(np.asarray(values), stored))

    def test_vtk_grid_shares_an_evicted_memory_map(self):
        v = self.viewer()
        v.cube_store.budget = 1
        v._load_computed_cubes([result_dict("a"), result_dict("b", 1)])
        v.cube_store._resident.clear()
        v.cube_store._resident_bytes = 0
        v.current_cube_index = 0
        mapped = v.cubes[0]["data"]
        self.assertIsInstance(mapped, np.memmap)
        values = np.asarray(v._create_grid().point_data["values"])
        self.assertTrue(np.shares_memory(values, mapped))

    def test_isovalue_range_uses_cached_maxima(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("a"), result_dict("b", 3)])
        with mock.patch.object(v.cube_store, "get", side_effect=AssertionError("paged a grid in")):
            got = v._cubes_max()
        expected = max(float(result_dict("a")["grid"].max()), float(result_dict("b", 3)["grid"].max()))
        self.assertAlmostEqual(got, expected)

    def test_switching_shows_each_cubes_own_source_details(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("a", details={"file": "a"}),
                                result_dict("b", 1, details={"file": "b"})])
        v.cube_list.setCurrentRow(1)
        self.assertEqual(v.source_details, {"file": "b"})
        v.cube_list.setCurrentRow(0)
        self.assertEqual(v.source_details, {"file": "a"})

    def test_replacing_the_list_frees_cache_files(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("a"), result_dict("b", 1)])
        paths = [v.cube_store.path(c["grid_id"]) for c in v.cubes]
        v._replace_cubes_with([self.cube_file()])
        self.assertTrue(all(not os.path.exists(p) for p in paths))
        self.assertEqual(len(v.cubes), 1)

    def test_background_loader_builds_records_off_the_gui_thread(self):
        paths = [self.cube_file("a.cube"), self.cube_file("b.cube", 1)]
        v = self.viewer()
        v._load_cube_files_in_background(paths)
        v._cube_load_thread.wait(30000)
        for _ in range(20):
            QApplication.processEvents()
        self.assertEqual([c["label"] for c in v.cubes], ["a.cube", "b.cube"])
        self.assertEqual(v.cube_files, paths)

    def test_background_loader_reports_unreadable_files_and_keeps_the_rest(self):
        bad = os.path.join(self.tmp, "bad.cube")
        open(bad, "w").write("not a cube\n")
        good = self.cube_file()
        v = self.viewer()
        with mock.patch.object(ci.QMessageBox, "warning") as warn:
            v._load_cube_files_in_background([bad, good])
            v._cube_load_thread.wait(30000)
            for _ in range(20):
                QApplication.processEvents()
        self.assertEqual(v.cube_files, [good])
        self.assertTrue(warn.called)


class UnsavedCubeTests(ViewerCase):
    def test_cache_files_do_not_count_as_saved(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("m")])
        self.assertTrue(os.path.isfile(v.cube_store.path(v.cubes[0]["grid_id"])))
        self.assertEqual(list(v._unsaved_cubes), [0])
        self.assertEqual(len(v._unprotected_computed_cubes()), 1)

    def test_new_results_keep_earlier_unsaved_cubes_by_default(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("first")])
        v._load_computed_cubes([result_dict("second", 1)])
        self.assertEqual([c["label"] for c in v.cubes], ["first", "second"])
        self.assertEqual(sorted(v._unsaved_cubes), [0, 1])

    def test_replace_choice_removes_them_and_their_files(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("first")])
        old = v.cube_store.path(v.cubes[0]["grid_id"])
        with mock.patch.object(v, "_keep_previous_results", return_value=False):
            v._load_computed_cubes([result_dict("second", 1)])
        self.assertEqual([c["label"] for c in v.cubes], ["second"])
        self.assertFalse(os.path.exists(old))

    def test_save_cubes_writes_from_the_cache_and_marks_saved(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("orb-7")])
        expected = np.array(v.cubes[0]["data"])
        err = v._save_one_cube(0, self.tmp, v._unsaved_cubes, QListWidget())
        self.assertIsNone(err)
        out = os.path.join(self.tmp, "orb-7.cube")
        np.testing.assert_allclose(v.read_cube(out)["data"], expected, rtol=1e-5, atol=1e-10)
        rec = v.cubes[0]
        self.assertFalse(rec["unsaved"])
        self.assertEqual(rec["source_path"], os.path.abspath(out))
        self.assertEqual(v._unsaved_cubes, {})
        self.assertEqual(v.cube_files[0], out)
        self.assertNotIn("●", v.cube_list.item(0).text())

    def test_quit_prompt_only_for_unprotected_cubes(self):
        v = self.viewer()
        v._load_computed_cubes([result_dict("m")])
        v.main_window = mock.Mock()                 # pretend there is a window
        self.addCleanup(setattr, v, "main_window", None)
        with mock.patch.object(ci.QMessageBox, "question", return_value=ci.QMessageBox.No) as ask:
            self.assertFalse(v._confirm_quit())
            ask.assert_called_once()
        v._write_session(os.path.join(self.tmp, "s.json"))
        with mock.patch.object(ci.QMessageBox, "question") as ask:
            self.assertTrue(v._confirm_quit())
            ask.assert_not_called()


class SessionTests(ViewerCase):
    def build(self):
        v = self.viewer()
        v._replace_cubes_with([self.cube_file("file.cube")])
        v._load_computed_cubes([result_dict("orb-1", 1), result_dict("orb-2", 2)])
        v.cube_list.setCurrentRow(2)
        return v

    def save(self, v, name="s.json"):
        path = os.path.join(self.tmp, name)
        v._write_session(path)
        with open(path) as f:
            return path, json.load(f)

    def load(self, path):
        v = self.viewer()
        with open(path) as f:
            v._read_session(path, json.load(f))
        return v

    def test_computed_cubes_round_trip_with_order_labels_flags_and_selection(self):
        v = self.build()
        grids = [np.array(c["data"]) for c in v.cubes]
        path, state = self.save(v)
        self.assertEqual(state["session_version"], 2)
        self.assertEqual([e["kind"] for e in state["cubes"]], ["file", "grid", "grid"])
        self.assertTrue(all(not os.path.isabs(e["grid"]) for e in state["cubes"][1:]))
        self.assertEqual(sorted(os.listdir(os.path.join(self.tmp, "s_assets"))),
                         sorted(f"{c['id']}.npy" for c in v.cubes[1:]))

        w = self.load(path)
        self.assertEqual([c["label"] for c in w.cubes], ["file.cube", "orb-1", "orb-2"])
        self.assertEqual([c["unsaved"] for c in w.cubes], [False, True, True])
        self.assertEqual([c["id"] for c in w.cubes[1:]], [c["id"] for c in v.cubes[1:]])
        self.assertEqual(w.current_cube_index, 2)
        for g, c in zip(grids, w.cubes):
            np.testing.assert_allclose(c["data"], g, rtol=1e-5, atol=1e-10)
        np.testing.assert_array_equal(w.cubes[1]["data"], grids[1])   # assets are exact
        self.assertEqual(w.cubes[2]["source_details"]["coefficients"], [0.0, 1.0, 2.0])
        self.assertEqual(w.source_details["file"], "orb-2.fchk")
        self.assertIn("●", w.cube_list.item(1).text())
        self.assertEqual(w.cubes[1]["bonds"], v.cubes[1]["bonds"])

    def test_session_assets_are_never_deleted_by_the_viewer(self):
        v = self.build()
        path, _ = self.save(v)
        w = self.load(path)
        assets = os.path.join(self.tmp, "s_assets")
        before = sorted(os.listdir(assets))
        w._release_records(w.cubes)
        w.cube_store.close()
        self.assertEqual(sorted(os.listdir(assets)), before)

    def test_resaving_a_loaded_session_keeps_its_assets(self):
        path, _ = self.save(self.build())
        w = self.load(path)
        _, state = self.save(w)                     # same path: assets are their own source
        x = self.load(path)
        self.assertEqual([c["label"] for c in x.cubes], ["file.cube", "orb-1", "orb-2"])
        self.assertEqual(len(state["cubes"]), 3)

    def test_unreferenced_assets_are_removed_on_resave(self):
        v = self.build()
        path, _ = self.save(v)
        gone = f"{v.cubes[1]['id']}.npy"
        v._save_one_cube(1, self.tmp, v._unsaved_cubes, QListWidget())    # now a real .cube file
        self.save(v)
        names = os.listdir(os.path.join(self.tmp, "s_assets"))
        self.assertNotIn(gone, names)
        self.assertEqual(len(names), 1)

    def test_missing_asset_is_reported_and_others_restore(self):
        v = self.build()
        path, state = self.save(v)
        os.remove(os.path.join(self.tmp, state["cubes"][1]["grid"]))
        with mock.patch("builtins.print") as printed:
            w = self.load(path)
        self.assertEqual([c["label"] for c in w.cubes], ["file.cube", "orb-2"])
        self.assertIn("orb-1", " ".join(str(a) for call in printed.call_args_list for a in call.args))

    def test_session_folder_can_move(self):
        v = self.build()
        path, _ = self.save(v)
        moved = os.path.join(self.tmp, "moved")
        os.makedirs(moved)
        shutil.move(path, moved)
        shutil.move(os.path.join(self.tmp, "s_assets"), moved)
        w = self.load(os.path.join(moved, "s.json"))
        self.assertEqual(len(w.cubes), 3)

    def test_legacy_session_with_file_paths_still_loads(self):
        a, b = self.cube_file("a.cube"), self.cube_file("b.cube", 1)
        path = os.path.join(self.tmp, "old.json")
        with open(path, "w") as f:
            json.dump({"cube_files": [a, b, "feco5-hf-48"], "current_index": 1,
                       "isovalue": 0.05}, f)
        with mock.patch("builtins.print"):
            w = self.load(path)
        self.assertEqual([c["label"] for c in w.cubes], ["a.cube", "b.cube"])
        self.assertEqual(w.current_cube_index, 1)

    def test_v1_readers_still_find_file_paths(self):
        _, state = self.save(self.build())
        self.assertEqual(len(state["cube_files"]), 1)
        self.assertTrue(state["cube_files"][0].endswith("file.cube"))

    def test_saving_a_session_protects_cubes_from_the_quit_prompt(self):
        v = self.build()
        self.assertEqual(len(v._unprotected_computed_cubes()), 2)
        self.save(v)
        self.assertEqual(v._unprotected_computed_cubes(), [])
        self.assertEqual(len(v._unsaved_cubes), 2)  # still not exported as .cube files


if __name__ == "__main__":
    unittest.main()
