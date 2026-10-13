"""
Camera regression tests on the real MultiCubeVisualizer with an off-screen
PyVista plotter. Skipped when off-screen rendering is unavailable.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from viewer_harness import SKIP_REASON, make_cube, make_viewer

if SKIP_REASON is None:
    import pyvista as pv

    import camera_utils as cu
    import ChemInsight3D as ci


def cam(v):
    return v.plotter.renderer.GetActiveCamera()


def state(v):
    return cu.capture_camera_state(cam(v))


def view_coords(v, points):
    """World -> view coordinates ([-1, 1] inside the frame) through VTK itself."""
    ren = v.plotter.renderer
    out = []
    for p in np.asarray(points, dtype=float):
        ren.SetWorldPoint(*p, 1.0)
        ren.WorldToView()
        out.append(ren.GetViewPoint()[:2])
    return np.abs(np.array(out))


def lobe_points(v):
    return np.vstack([m.points for m in (v.pos_mesh, v.neg_mesh) if m.n_points])[::7]


def corners(bounds):
    return np.array([(x, y, z) for x in bounds[0:2] for y in bounds[2:4] for z in bounds[4:6]])


def scene_in_front_of_camera(v):
    c = cam(v)
    pos, focal = np.array(c.GetPosition()), np.array(c.GetFocalPoint())
    d = (pos - focal) / np.linalg.norm(pos - focal)
    along = (corners(v.plotter.renderer.ComputeVisiblePropBounds()) - focal) @ d
    return bool(np.all(along < np.linalg.norm(pos - focal)))


def view_direction(v):
    return np.array(cam(v).GetDirectionOfProjection())


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class CameraViewerTestCase(unittest.TestCase):
    def tearDown(self):
        for v in getattr(self, "_viewers", []):
            v.plotter.close()

    def viewer(self, *args, **kwargs):
        v = make_viewer(*args, **kwargs)
        self._viewers = getattr(self, "_viewers", []) + [v]
        return v

    def assert_same_view(self, a, b, check_position=True):
        for key in ("focal", "viewup") + (("position",) if check_position else ()):
            np.testing.assert_allclose(a[key], b[key], atol=1e-6, err_msg=key)
        self.assertAlmostEqual(a["parallel_scale"], b["parallel_scale"], places=6)
        self.assertEqual(a["parallel_projection"], b["parallel_projection"])

    def user_view(self, v):
        """Give the viewer a non-default orientation, zoom and pan."""
        c = cam(v)
        c.Azimuth(40); c.Elevation(15); c.OrthogonalizeViewUp(); c.Zoom(1.7)
        f = np.array(c.GetFocalPoint()) + [0.3, -0.2, 0.1]
        p = np.array(c.GetPosition()) + [0.3, -0.2, 0.1]
        c.SetFocalPoint(*f); c.SetPosition(*p)
        v.plotter.renderer.ResetCameraClippingRange()
        return state(v)


class InitialFramingTests(CameraViewerTestCase):
    def test_startup_with_cubes_frames_the_atoms(self):
        v = self.viewer([make_cube()])
        self.assertTrue(v._camera_framed)
        ndc = view_coords(v, v.cubes[0]["coordinates"])
        self.assertLessEqual(ndc.max(), 1.0)
        self.assertGreater(ndc.max(), 0.3)    # framed, not a speck

    def test_first_gui_loaded_cube_is_framed_like_startup(self):
        reference = self.viewer([make_cube()])
        v = self.viewer()
        self.assertFalse(v._camera_framed)
        v._select_cube(v._append_cube_to_list("a.cube", make_cube()))
        self.assertTrue(v._camera_framed)
        self.assertAlmostEqual(cam(v).GetParallelScale(), cam(reference).GetParallelScale(), places=6)
        np.testing.assert_allclose(cam(v).GetFocalPoint(), cam(reference).GetFocalPoint(), atol=1e-6)

    def test_later_loads_do_not_refit(self):
        v = self.viewer()
        v._select_cube(v._append_cube_to_list("a.cube", make_cube()))
        before = self.user_view(v)
        v._select_cube(v._append_cube_to_list("b.cube", make_cube(width=60.0)))
        self.assert_same_view(before, state(v), check_position=False)


class CubeSwitchTests(CameraViewerTestCase):
    def test_switch_preserves_live_view(self):
        v = self.viewer([make_cube(), make_cube(width=60.0), make_cube(axis=2)])
        before = self.user_view(v)
        direction = view_direction(v)
        for i in (1, 2, 0):
            v.switch_cube(i)
            self.assert_same_view(before, state(v), check_position=False)
            np.testing.assert_allclose(view_direction(v), direction, atol=1e-9)

    def test_switch_to_larger_orbital_keeps_scene_in_front(self):
        v = self.viewer([make_cube(), make_cube(width=60.0)])
        scale = cam(v).GetParallelScale()
        v.switch_cube(1)
        self.assertTrue(scene_in_front_of_camera(v))
        self.assertAlmostEqual(cam(v).GetParallelScale(), scale)

    def test_near_plane_safeguard_keeps_geometry_visible(self):
        """Rendered check: an object behind the camera's position is clipped
        until _ensure_camera_clears_scene moves the camera back."""
        v = self.viewer([make_cube()])
        v.plotter.add_mesh(pv.Sphere(radius=1.0, center=(3, 0, 10)), color="magenta")
        c = cam(v)
        c.SetFocalPoint(0, 0, 0); c.SetPosition(0, 0, 5); c.SetViewUp(0, 1, 0); c.SetParallelScale(8)

        def magenta_pixels():
            v.plotter.renderer.ResetCameraClippingRange()
            v.plotter.render()
            img = v.plotter.screenshot(return_img=True).astype(int)
            return int(((img[..., 0] - img[..., 1] > 60) & (img[..., 2] - img[..., 1] > 60)).sum())

        self.assertLess(magenta_pixels(), 20)     # tolerance for anti-aliased lobe edges
        v._ensure_camera_clears_scene()
        self.assertGreater(magenta_pixels(), 500)
        self.assertEqual(c.GetParallelScale(), 8)

    def test_safeguard_handles_focal_point_outside_scene(self):
        v = self.viewer([make_cube(width=60.0)])
        c = cam(v)
        c.SetFocalPoint(0, 0, -100); c.SetPosition(0, 0, -100 + 50); c.SetViewUp(0, 1, 0)
        v._ensure_camera_clears_scene()
        self.assertTrue(scene_in_front_of_camera(v))
        np.testing.assert_allclose(view_direction(v), [0, 0, -1], atol=1e-9)


class StandardViewTests(CameraViewerTestCase):
    def test_presets_change_orientation_only(self):
        v = self.viewer([make_cube(width=60.0)])
        cam(v).Zoom(1.4)
        scale = cam(v).GetParallelScale()
        for name, spec in ci.MultiCubeVisualizer.CAMERA_PRESETS.items():
            with self.subTest(preset=name):
                v.set_camera_preset(name)
                expected = -np.array(spec["direction"], float) / np.linalg.norm(spec["direction"])
                np.testing.assert_allclose(view_direction(v), expected, atol=1e-9)
                np.testing.assert_allclose(cam(v).GetFocalPoint(), v._molecule_center(), atol=1e-9)
                self.assertAlmostEqual(cam(v).GetParallelScale(), scale)
                self.assertTrue(cam(v).GetParallelProjection())
                self.assertTrue(scene_in_front_of_camera(v))

    def test_oblique_replaces_misnamed_perspective(self):
        self.assertIn("Oblique", ci.MultiCubeVisualizer.CAMERA_PRESETS)
        self.assertNotIn("Perspective", ci.MultiCubeVisualizer.CAMERA_PRESETS)

    def test_presets_and_export_share_one_centre(self):
        asym = [(6, 0, 0, 0), (6, 2.9, 0, 0), (6, 5.8, 0, 0), (8, 8.7, 2.5, 0)]
        v = self.viewer([make_cube(atoms=asym)])
        v.set_camera_preset("Front")
        preset_focal = np.array(cam(v).GetFocalPoint())
        v._fit_camera_to_atoms()
        np.testing.assert_allclose(cam(v).GetFocalPoint(), preset_focal, atol=1e-6)

    def test_fit_view_frames_whole_isosurface_tightly(self):
        v = self.viewer([make_cube(width=60.0)])
        v.set_camera_preset("Oblique")
        v.fit_view()
        ndc = view_coords(v, lobe_points(v))
        self.assertLessEqual(ndc.max(), 0.97 + 1e-6)
        self.assertGreater(ndc.max(), 0.9)


class SessionTests(CameraViewerTestCase):
    def test_session_round_trip_restores_zoom(self):
        v = self.viewer([make_cube()])
        saved = self.user_view(v)
        block = json.loads(json.dumps(cu.camera_state_to_session(saved)))
        v.set_camera_preset("Top"); cam(v).Zoom(0.3)
        v._apply_session_camera(block)
        self.assert_same_view(saved, state(v))

    def test_legacy_session_keeps_orientation_and_refits_zoom(self):
        v = self.viewer([make_cube()])
        fitted = self.viewer([make_cube()])
        legacy = {"position": [0, -10, 0], "focal": [0, 0, 0], "viewup": [0, 0, 1], "distance": 10.0}
        cam(v).Zoom(5.0)
        v._apply_session_camera(legacy)
        np.testing.assert_allclose(view_direction(v), [0, 1, 0], atol=1e-9)
        np.testing.assert_allclose(cam(v).GetViewUp(), [0, 0, 1], atol=1e-9)
        cam(fitted).SetPosition(0, -10, 0); cam(fitted).SetFocalPoint(0, 0, 0)
        cam(fitted).SetViewUp(0, 0, 1)
        fitted._fit_camera_to_atoms()
        self.assertAlmostEqual(cam(v).GetParallelScale(), cam(fitted).GetParallelScale(), places=6)

    def test_missing_or_malformed_camera_keeps_current_view(self):
        v = self.viewer([make_cube()])
        before = self.user_view(v)
        for block in ({}, None, {"position": "bad"}, {"position": [0, 0, 1], "focal": [0, 0, 1],
                                                      "viewup": [0, 1, 0]}):
            with self.subTest(block=block):
                v._apply_session_camera(block)
                self.assert_same_view(before, state(v))


class ImageExportTests(CameraViewerTestCase):
    def overlay_state(self, v):
        a2d = v.plotter.renderer.GetActors2D()
        a2d.InitTraversal()
        vis, a = [], a2d.GetNextActor2D()
        while a:
            vis.append((a, a.GetVisibility()))
            a = a2d.GetNextActor2D()
        return vis

    def test_single_export_restores_camera_and_overlays(self):
        v = self.viewer([make_cube(width=60.0)])
        before = self.user_view(v)
        hidden_note = v.plotter.add_text("hidden note")
        hidden_note.VisibilityOff()
        # The title is recreated by _update_title, so compare every other overlay.
        overlays_before = [(a, vis) for a, vis in self.overlay_state(v) if a is not v.title_actor]
        img = v._render_export_image(1, False)
        self.assertEqual(img.shape[:2], (600, 800))
        self.assert_same_view(before, state(v))
        self.assertIsNotNone(v.title_actor)
        self.assertTrue(v.title_actor.GetVisibility())
        for actor, vis in overlays_before:
            self.assertEqual(actor.GetVisibility(), vis)
        self.assertFalse(hidden_note.GetVisibility())

    def test_single_export_restores_state_when_capture_fails(self):
        v = self.viewer([make_cube()])
        before = self.user_view(v)
        visible_before = [vis for _, vis in self.overlay_state(v)]
        # Full-size off-screen render unavailable, then the tiled fallback fails too.
        with mock.patch.object(ci.offscreen_export, "render_full_size", return_value=None), \
                mock.patch.object(v.plotter, "screenshot", side_effect=RuntimeError("GPU lost")):
            with self.assertRaises(RuntimeError):
                v._render_export_image(2, False)
        self.assert_same_view(before, state(v))
        self.assertEqual(sum(vis for _, vis in self.overlay_state(v)), sum(visible_before))

    def test_sequence_shares_one_camera_and_shows_every_orbital(self):
        cubes = [make_cube(), make_cube(width=60.0), make_cube(axis=1, centre=(3, 3, 0), width=8)]
        v = self.viewer(cubes)
        v.switch_cube(2)
        before = self.user_view(v)
        frames, real_capture = [], v._capture_frame

        def record(scale, transparent):
            frames.append((state(v), view_direction(v), view_coords(v, lobe_points(v)).max()))
            return real_capture(scale, transparent)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(v, "_capture_frame", side_effect=record):
            saved = v._write_image_sequence(
                {"folder": tmp, "prefix": "f", "scale": 1, "transparent": False})
            self.assertEqual([os.path.basename(p) for p in saved],
                             ["f_0001.png", "f_0002.png", "f_0003.png"])
            self.assertTrue(all(os.path.getsize(p) > 0 for p in saved))
        self.assertEqual(len(frames), 3)
        for st, direction, max_ndc in frames:
            self.assertAlmostEqual(st["parallel_scale"], frames[0][0]["parallel_scale"])
            np.testing.assert_allclose(st["focal"], frames[0][0]["focal"], atol=1e-9)
            np.testing.assert_allclose(direction, frames[0][1], atol=1e-9)
            self.assertLessEqual(max_ndc, 0.97 + 1e-6)          # nothing clipped at the edges
        self.assertGreater(max(f[2] for f in frames), 0.9)       # largest orbital sets the scale
        self.assertEqual(v.current_cube_index, 2)
        self.assert_same_view(before, state(v))

    def test_sequence_failure_restores_cube_and_camera(self):
        v = self.viewer([make_cube(), make_cube(width=60.0), make_cube(axis=2)])
        v.switch_cube(1)
        before = self.user_view(v)
        calls = []

        def fail_second(scale, transparent):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")
            return np.zeros((4, 4, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(v, "_capture_frame", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "after 1 of 3"):
                v._write_image_sequence(
                    {"folder": tmp, "prefix": "f", "scale": 1, "transparent": False})
        self.assertEqual(v.current_cube_index, 1)
        self.assert_same_view(before, state(v))


class GeometryAndViewportTests(CameraViewerTestCase):
    GEOMETRIES = {
        "single atom": [(26, 0.0, 0.0, 0.0)],
        "linear": [(8, -2.2, 0, 0), (6, 0, 0, 0), (8, 2.2, 0, 0)],
        "asymmetric": [(6, 0, 0, 0), (6, 2.9, 0, 0), (6, 5.8, 0, 0), (6, 8.7, 0, 0), (1, -1.0, 9.0, 0)],
        "coincident atoms": [(1, 1.0, 1.0, 1.0), (1, 1.0, 1.0, 1.0)],
    }

    def test_framing_handles_small_and_degenerate_geometries(self):
        for name, atoms in self.GEOMETRIES.items():
            with self.subTest(geometry=name):
                v = self.viewer([make_cube(atoms=atoms)])
                scale = cam(v).GetParallelScale()
                self.assertTrue(np.isfinite(scale) and scale > 0)
                self.assertTrue(np.all(np.isfinite(v._molecule_center())))
                self.assertLessEqual(view_coords(v, v.cubes[0]["coordinates"]).max(), 1.0)

    def test_non_square_viewports_keep_isosurface_in_frame(self):
        for size in ((300, 800), (1200, 300)):
            with self.subTest(size=size):
                v = self.viewer([make_cube(width=60.0)], size=size)
                v.set_camera_preset("Oblique")
                v.fit_view()
                self.assertLessEqual(view_coords(v, lobe_points(v)).max(), 0.97 + 1e-6)

    def test_resize_changes_no_camera_state_and_refit_uses_new_aspect(self):
        v = self.viewer([make_cube(width=60.0)])
        v.fit_view()
        before = state(v)
        v.plotter.window_size = (400, 900)
        v.plotter.render()
        self.assert_same_view(before, state(v))
        v.fit_view()
        self.assertLessEqual(view_coords(v, lobe_points(v)).max(), 0.97 + 1e-6)

    def test_cube_without_atoms_and_empty_scene(self):
        v = self.viewer([make_cube(atoms=[])])
        self.assertIsNone(v._atom_bounds())
        self.assertIsNone(v._molecule_center())
        before = state(v)
        v.set_camera_preset("Top")                     # no reference centre: no-op
        self.assert_same_view(before, state(v))
        v._fit_camera_to_atoms()                       # falls back to visible props
        self.assertTrue(np.isfinite(cam(v).GetParallelScale()))

        empty = self.viewer()
        before = state(empty)
        self.assertEqual(empty._scene_overflow_zoom(), 1.0)
        empty._ensure_camera_clears_scene()
        self.assert_same_view(before, state(empty))


if __name__ == "__main__":
    unittest.main()
