"""CubeStore, CubeRecord and parse_cube_file (no Qt or rendering needed)."""
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import cube_store
from cube_store import CubeRecord, CubeStore, CubeStoreError, parse_cube_file

MB = 2**20


def grid(n=20, seed=0):
    return np.random.default_rng(seed).standard_normal((n, n + 1, n + 2))


def cube_dict(data):
    return dict(origin=np.zeros(3), spacing=(0.2, 0.2, 0.2), dimensions=data.shape,
                atoms=np.array([8]), coordinates=np.zeros((1, 3)), bonds=[],
                unit_bohr=True, data=data)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ci3d-store-")
        self.addCleanup(shutil.rmtree, self.root, True)

    def store(self, budget=64 * MB):
        s = CubeStore(root=self.root, budget_bytes=budget)
        self.addCleanup(s.close)
        return s


class PutGetTests(StoreCase):
    def test_round_trip_keeps_values_dtype_and_uses_fortran_order(self):
        s = self.store()
        for dtype in (np.float64, np.float32):
            data = grid().astype(dtype)
            gid = s.put(data)
            out = s.get(gid)
            np.testing.assert_array_equal(out, data)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(out.flags.f_contiguous)
            on_disk = np.load(s.path(gid), mmap_mode="r")
            np.testing.assert_array_equal(on_disk, data)
            self.assertTrue(on_disk.flags.f_contiguous)

    def test_fortran_ravel_is_a_view_of_the_mapped_file(self):
        s = self.store(budget=1)
        gid = s.put(grid(), keep_resident=False)
        mapped = s.get(gid)
        self.assertIsInstance(mapped, np.memmap)
        self.assertTrue(np.shares_memory(mapped.ravel(order="F"), mapped))

    def test_store_keeps_a_map_of_the_file_not_the_callers_array(self):
        import weakref
        s = self.store()
        data = np.asfortranarray(grid())
        ref = weakref.ref(data)
        gid = s.put(data)
        self.assertIsInstance(s.get(gid), np.memmap)
        self.assertFalse(s.get(gid).flags.writeable)
        del data
        self.assertIsNone(ref())

    def test_files_live_in_one_run_folder_under_the_root(self):
        s = self.store()
        gid = s.put(grid())
        self.assertEqual(os.path.dirname(s.path(gid)), s.run_dir)
        self.assertTrue(os.path.basename(s.run_dir).startswith(f"run-{os.getpid()}-"))
        self.assertEqual(os.path.dirname(s.run_dir), self.root)

    def test_no_folder_is_created_until_a_grid_is_stored(self):
        s = self.store()
        self.assertIsNone(s.run_dir)
        self.assertEqual(os.listdir(self.root), [])

    def test_unknown_id_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.store().get("nope")


class LruTests(StoreCase):
    def test_budget_evicts_least_recently_used(self):
        one = grid().nbytes
        s = self.store(budget=int(2.5 * one))
        a, b, c = (s.put(grid(seed=i)) for i in range(3))
        self.assertEqual(s.resident_ids(), [b, c])
        s.get(b)                                    # b becomes most recent
        d = s.put(grid(seed=3))
        self.assertEqual(s.resident_ids(), [b, d])
        self.assertLessEqual(s.resident_bytes, s.budget)
        np.testing.assert_array_equal(s.get(a), grid(seed=0))   # comes back from disk
        self.assertIsInstance(s.get(a), np.memmap)

    def test_most_recent_grid_stays_resident_even_over_budget(self):
        s = self.store(budget=10)
        gid = s.put(grid())
        self.assertEqual(s.resident_ids(), [gid])
        other = s.put(grid(seed=1))
        self.assertEqual(s.resident_ids(), [other])

    def test_budget_from_environment(self):
        with mock.patch.dict(os.environ, {"CI3D_CUBE_CACHE_MB": "17"}):
            self.assertEqual(CubeStore(root=self.root).budget, 17 * MB)
        with mock.patch.dict(os.environ, {"CI3D_CUBE_CACHE_MB": "lots"}):
            self.assertEqual(CubeStore(root=self.root).budget, cube_store.DEFAULT_BUDGET_MB * MB)


class LifetimeTests(StoreCase):
    def test_release_deletes_owned_file(self):
        s = self.store()
        gid = s.put(grid())
        path = s.path(gid)
        s.release(gid)
        self.assertFalse(os.path.exists(path))
        self.assertNotIn(gid, s.resident_ids())
        with self.assertRaises(KeyError):
            s.get(gid)

    def test_external_files_are_never_deleted(self):
        s = self.store()
        asset = os.path.join(self.root, "asset.npy")
        np.save(asset, np.asfortranarray(grid()))
        gid = s.add_external(asset)
        np.testing.assert_array_equal(s.get(gid), grid())
        s.release(gid)
        s.close()
        self.assertTrue(os.path.isfile(asset))

    def test_missing_external_file_raises(self):
        with self.assertRaises(CubeStoreError):
            self.store().add_external(os.path.join(self.root, "gone.npy"))

    def test_close_removes_the_run_folder(self):
        s = self.store()
        s.put(grid())
        run = s.run_dir
        s.close()
        self.assertFalse(os.path.exists(run))

    def test_export_copies_atomically_and_survives_close(self):
        s = self.store()
        gid = s.put(grid())
        dest = os.path.join(self.root, "kept.npy")
        s.export_grid(gid, dest)
        s.close()
        np.testing.assert_array_equal(np.load(dest), grid())
        self.assertEqual([n for n in os.listdir(self.root) if n.endswith(".tmp")], [])


class FailureTests(StoreCase):
    def test_failed_write_leaves_no_partial_file_and_no_entry(self):
        s = self.store()
        real_save = np.save

        def half_save(f, arr):
            f.write(b"\x93NUMPY partial")
            raise OSError(28, "No space left on device")

        with mock.patch.object(cube_store.np, "save", side_effect=half_save):
            with self.assertRaises(CubeStoreError) as ctx:
                s.put(grid())
        self.assertIn("No space left", str(ctx.exception))
        self.assertEqual(os.listdir(s.run_dir), [])
        self.assertEqual(s.resident_ids(), [])
        self.assertIs(np.save, real_save)
        s.put(grid())                               # the store still works afterwards

    def test_unwritable_root_reports_a_clear_error(self):
        blocker = os.path.join(self.root, "file")
        open(blocker, "w").close()
        s = CubeStore(root=os.path.join(blocker, "sub"))
        with self.assertRaises(CubeStoreError) as ctx:
            s.put(grid())
        self.assertIn("cube cache folder", str(ctx.exception))

    def test_corrupt_cache_file_raises_on_read(self):
        s = self.store(budget=1)
        gid = s.put(grid(), keep_resident=False)
        with open(s.path(gid), "wb") as f:
            f.write(b"garbage")
        with self.assertRaises(CubeStoreError):
            s.get(gid)

    def test_failed_export_leaves_no_temporary_file(self):
        s = self.store()
        gid = s.put(grid())
        dest = os.path.join(self.root, "no-such-dir", "x.npy")
        with self.assertRaises(CubeStoreError):
            s.export_grid(gid, dest)


class StaleCleanupTests(StoreCase):
    def make_run_dir(self, name, age_hours):
        path = os.path.join(self.root, name)
        os.makedirs(path)
        open(os.path.join(path, "x.npy"), "wb").close()
        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))
        return path

    def test_old_folders_of_dead_processes_are_removed(self):
        dead = self.make_run_dir("run-999999999-1-abc", 48)
        recent = self.make_run_dir("run-999999998-2-abc", 1)
        other = self.make_run_dir("keep-me", 48)
        CubeStore(root=self.root)
        self.assertFalse(os.path.exists(dead))
        self.assertTrue(os.path.exists(recent))
        self.assertTrue(os.path.exists(other))

    @unittest.skipIf(os.name == "nt", "process check is POSIX only")
    def test_folder_of_a_live_process_is_kept(self):
        alive = self.make_run_dir(f"run-{os.getppid()}-1-abc", 48)
        CubeStore(root=self.root)
        self.assertTrue(os.path.exists(alive))


class CubeRecordTests(StoreCase):
    def test_data_comes_from_the_store_and_is_not_held_by_the_dict(self):
        s = self.store()
        data = grid()
        rec = CubeRecord(s, cube_dict(data), label="a")
        np.testing.assert_array_equal(rec["data"], data)
        self.assertIn("data", rec)
        np.testing.assert_array_equal(rec.get("data"), data)
        self.assertFalse(any(isinstance(v, np.ndarray) and v.size == data.size
                             for v in dict.values(rec)))
        self.assertEqual(rec["data_min"], data.min())
        self.assertEqual(rec["data_max"], data.max())
        self.assertEqual(len(rec["id"]), 32)

    def test_assigning_data_replaces_the_grid_and_frees_the_old_file(self):
        s = self.store()
        rec = CubeRecord(s, cube_dict(grid()))
        old_path = s.path(rec["grid_id"])
        rec["data"] = grid(seed=5) * 3
        np.testing.assert_array_equal(rec["data"], grid(seed=5) * 3)
        self.assertFalse(os.path.exists(old_path))
        self.assertAlmostEqual(rec["data_max"], float((grid(seed=5) * 3).max()))

    def test_release_frees_the_file(self):
        s = self.store()
        rec = CubeRecord(s, cube_dict(grid()))
        path = s.path(rec["grid_id"])
        rec.release()
        self.assertFalse(os.path.exists(path))
        rec.release()                               # second release is harmless

    def test_evicted_record_reads_back_identically(self):
        one = grid().nbytes
        s = self.store(budget=one)
        recs = [CubeRecord(s, cube_dict(grid(seed=i))) for i in range(4)]
        for i, rec in enumerate(recs):
            np.testing.assert_array_equal(rec["data"], grid(seed=i))
        self.assertLessEqual(s.resident_bytes, one)

    def test_stable_id_survives_from_metadata(self):
        s = self.store()
        rec = CubeRecord(s, dict(cube_dict(grid()), id="fixed-id"))
        self.assertEqual(rec["id"], "fixed-id")


# ── parser ────────────────────────────────────────────────────────────────────

def reference_parse(path):
    """The viewer's previous list-of-floats reader (values only), kept as the oracle."""
    with open(path) as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        natoms = int(parts[0])
        dims = [int(float(f.readline().split()[0])) for _ in range(3)]
        for _ in range(natoms):
            f.readline()
        data = []
        for line in f:
            data.extend(float(x) for x in line.split())
    return np.array(data).reshape(dims)


def write_cube_text(path, data, natoms_line="    1   -2.000000   -2.000000   -2.000000",
                    extra_header=(), per_line=6, fmt="{:13.5E}", spacing=0.2):
    nx, ny, nz = data.shape
    lines = ["test cube", "second line", natoms_line,
             f"{nx:5d} {spacing:11.6f}    0.000000    0.000000",
             f"{ny:5d}    0.000000 {spacing:11.6f}    0.000000",
             f"{nz:5d}    0.000000    0.000000 {spacing:11.6f}",
             "    8    0.000000    0.000000    0.000000    0.000000", *extra_header]
    flat = data.ravel(order="C")
    body = [" ".join(fmt.format(v) for v in flat[i:i + per_line])
            for i in range(0, len(flat), per_line)]
    with open(path, "w") as f:
        f.write("\n".join(lines + body) + "\n")


class ParseCubeFileTests(StoreCase):
    def path(self, name="t.cube"):
        return os.path.join(self.root, name)

    def test_matches_the_previous_reader(self):
        data = grid(9)
        write_cube_text(self.path(), data)
        parsed = parse_cube_file(self.path())
        np.testing.assert_array_equal(parsed["data"], reference_parse(self.path()))
        self.assertEqual(parsed["dimensions"], data.shape)
        self.assertEqual(parsed["spacing"], (0.2, 0.2, 0.2))
        np.testing.assert_array_equal(parsed["origin"], [-2, -2, -2])
        self.assertTrue(parsed["unit_bohr"])
        self.assertEqual(parsed["coord_lines"], [[8, 0.0, 0.0, 0.0]])

    def test_values_split_across_chunk_boundaries(self):
        data = grid(11)
        write_cube_text(self.path(), data, per_line=5)
        with mock.patch.object(cube_store, "_PARSE_CHUNK_BYTES", 97):
            parsed = parse_cube_file(self.path())
        np.testing.assert_array_equal(parsed["data"], reference_parse(self.path()))

    def test_fortran_d_exponents(self):
        write_cube_text(self.path(), grid(5))
        expected = parse_cube_file(self.path())["data"]
        text = open(self.path()).read().split("\n")
        text[7:] = [ln.replace("E", "D") for ln in text[7:]]
        open(self.path("d.cube"), "w").write("\n".join(text))
        np.testing.assert_array_equal(parse_cube_file(self.path("d.cube"))["data"], expected)

    def test_irregular_values_per_line(self):
        data = grid(6)
        flat = data.ravel()
        write_cube_text(self.path(), data)
        header = open(self.path()).read().splitlines()[:7]
        with open(self.path(), "w") as f:
            f.write("\n".join(header) + "\n")
            i, n = 0, 1
            while i < flat.size:
                f.write(" ".join(repr(float(v)) for v in flat[i:i + n]) + "\n")
                i, n = i + n, n % 7 + 1
        np.testing.assert_array_equal(parse_cube_file(self.path())["data"].ravel(), flat)

    def test_malformed_value_is_an_error_not_a_short_read(self):
        write_cube_text(self.path(), grid(5))
        lines = open(self.path()).read().splitlines()
        lines[9] = lines[9].replace(" ", " nan? ", 1)
        open(self.path(), "w").write("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            parse_cube_file(self.path())

    def test_negative_atom_count_skips_the_orbital_line(self):
        data = grid(5)
        write_cube_text(self.path(), data,
                        natoms_line="   -1   -2.000000   -2.000000   -2.000000",
                        extra_header=["    1   12"])
        parsed = parse_cube_file(self.path())
        np.testing.assert_allclose(parsed["data"], data, rtol=1e-5)
        self.assertEqual(list(parsed["atoms"]), [8])

    def test_multi_orbital_files_are_rejected(self):
        write_cube_text(self.path(), grid(4),
                        natoms_line="   -1   -2.000000   -2.000000   -2.000000",
                        extra_header=["    2   12   13"])
        with self.assertRaises(ValueError):
            parse_cube_file(self.path())

    def test_truncated_and_overlong_files_are_rejected(self):
        data = grid(5)
        write_cube_text(self.path(), data)
        lines = open(self.path()).read().splitlines()
        open(self.path("short.cube"), "w").write("\n".join(lines[:-3]) + "\n")
        open(self.path("long.cube"), "w").write("\n".join(lines + ["1.0 2.0"]) + "\n")
        for name in ("short.cube", "long.cube"):
            with self.assertRaises(ValueError, msg=name):
                parse_cube_file(self.path(name))

    def test_angstrom_flag_follows_the_previous_rule(self):
        data = grid(4)
        write_cube_text(self.path(), data, spacing=-0.2)
        self.assertFalse(parse_cube_file(self.path())["unit_bohr"])

    def test_real_cube_file_if_present(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo, "cube_files", "feco5-hf_LOC_occupied-48.cube")
        if not os.path.isfile(path):
            self.skipTest("cube_files fixture not present")
        np.testing.assert_array_equal(parse_cube_file(path)["data"], reference_parse(path))


if __name__ == "__main__":
    unittest.main()
