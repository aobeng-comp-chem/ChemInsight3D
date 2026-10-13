"""Off-screen tests of the app-owned lights, the Lighting panel and session handling."""
import json
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from viewer_harness import SKIP_REASON, make_cube, make_viewer, render

if SKIP_REASON is None:
    import pyvista as pv
    from PyQt5.QtWidgets import QWidget

    import rendering_style as rs

DEFAULT, GLOSSY = "Default", "Glossy"


def key_only(az, el):
    return rs.LightingSettings(key_azimuth=az, key_elevation=el, key_intensity=1.0,
                               fill_ratio=0.0, headlight=0.0, rim_lights=False,
                               key_color=rs.WHITE, fill_color=rs.WHITE)


def brightest_offset(az, el, lock_frame=None):
    """Lit sphere seen from +z; returns the brightest pixel's (dx, dy) from the centre, y up."""
    p = pv.Plotter(off_screen=True, window_size=(200, 200), lighting="none")
    p.set_background("black")
    try:
        p.add_mesh(pv.Sphere(radius=1.0, theta_resolution=120, phi_resolution=120),
                   color="white", ambient=0.0, diffuse=1.0, specular=0.0, smooth_shading=True)
        p.camera.position = (0, 0, 10); p.camera.focal_point = (0, 0, 0); p.camera.up = (0, 1, 0)
        p.enable_parallel_projection(); p.camera.parallel_scale = 1.2
        settings = key_only(az, el)
        if lock_frame is not None:
            settings = replace(settings, lock_to_camera=False, scene_frame=lock_frame)
        rs.apply_lighting(p.renderer, settings)
        p.render()
        img = p.screenshot(return_img=True).astype(float).sum(axis=2)
        row, col = np.unravel_index(np.argmax(img), img.shape)
        return col - img.shape[1] / 2, img.shape[0] / 2 - row
    finally:
        p.close()


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class LightDirectionTests(unittest.TestCase):
    def test_key_light_lands_where_its_angles_say(self):
        cases = {(0, 0): "centre", (90, 0): "right", (-90, 0): "left",
                 (0, 60): "top", (0, -60): "bottom"}
        for (az, el), where in cases.items():
            with self.subTest(az=az, el=el):
                dx, dy = brightest_offset(az, el)
                if where == "centre":
                    self.assertLess(max(abs(dx), abs(dy)), 10)
                elif where in ("right", "left"):
                    self.assertGreater(dx if where == "right" else -dx, 40)
                    self.assertLess(abs(dy), 10)
                else:
                    self.assertGreater(dy if where == "top" else -dy, 40)
                    self.assertLess(abs(dx), 10)

    def test_scene_locked_light_uses_frozen_frame(self):
        # Frame frozen from a camera that looked down -x: az 90 (its "right") is world -y,
        # which from the +z test camera appears at the bottom.
        frame = [[0, -1, 0], [0, 0, 1], [1, 0, 0]]
        dx, dy = brightest_offset(90, 0, lock_frame=frame)
        self.assertLess(dy, -40)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class LightingPanelTests(unittest.TestCase):
    def setUp(self):
        self.v = make_viewer([make_cube(width=4.0)])
        self.panel = QWidget()
        self.v._build_lighting_content(self.panel)
        self.addCleanup(self.v.plotter.close)

    def lights(self):
        return self.v.plotter.renderer._ci3d_lights

    def test_panel_offers_the_two_presets_and_shows_the_default(self):
        from PyQt5.QtWidgets import QComboBox, QRadioButton
        v = self.v
        buttons = self.panel.findChildren(QRadioButton)
        self.assertEqual([b.text() for b in buttons], [DEFAULT, GLOSSY])
        self.assertEqual(self.panel.findChildren(QComboBox), [])
        self.assertTrue(v._light_preset_buttons[DEFAULT].isChecked())       # the v1.0.0 look
        self.assertTrue(v._light_custom_label.isHidden())
        self.assertEqual(v._light_widgets["key_azimuth"][1].value(), 10)
        self.assertEqual(v._light_labels["fill_ratio"].text(), "Fill")
        self.assertTrue(v.rim_check.isEnabled() and v.rim_check.isChecked())
        self.assertTrue(v.lock_check.isChecked())
        v.set_lighting_preset(GLOSSY)
        self.assertEqual(v._light_widgets["key_azimuth"][1].value(), 45)
        self.assertEqual(v._light_labels["fill_ratio"].text(), "Lower pair")
        self.assertFalse(v.rim_check.isEnabled() or v.rim_check.isChecked())

    def test_editing_a_control_switches_to_custom_and_moves_the_light_in_place(self):
        v = self.v
        before = list(self.lights())
        v._light_widgets["key_azimuth"][0].setValue(90)     # drag the slider
        self.assertEqual(v._light_widgets["key_azimuth"][1].value(), 90)
        self.assertEqual(v.render_style.preset, rs.CUSTOM)
        self.assertFalse(any(b.isChecked() for b in v._light_preset_buttons.values()))
        self.assertFalse(v._light_custom_label.isHidden())
        self.assertIn(DEFAULT, v._light_custom_label.text())
        v._apply_lighting_now()                              # what the debounce timer runs
        self.assertEqual(self.lights(), before)              # same light objects
        self.assertEqual(v.plotter.renderer.GetLights().GetNumberOfItems(), 5)
        np.testing.assert_allclose(self.lights()[1].GetPosition(),
                                   rs.light_direction(90, 50), atol=1e-9)

    def test_reset_restores_the_preset_the_edit_started_from(self):
        v = self.v
        v.set_lighting_preset(GLOSSY)
        v._light_widgets["headlight"][1].setValue(0.45)
        self.assertEqual(v.render_style.preset, rs.CUSTOM)
        self.assertEqual(v.render_style.materials, "Glossy")         # edits keep the material
        v._reset_lighting()
        self.assertEqual(v.render_style, rs.RenderStyle.from_preset(GLOSSY))
        self.assertTrue(v._light_preset_buttons[GLOSSY].isChecked())
        self.assertTrue(v._light_custom_label.isHidden())

    def test_choosing_a_preset_after_an_edit_discards_it(self):
        v = self.v
        v._light_widgets["key_elevation"][1].setValue(-20)
        v._light_preset_buttons[DEFAULT].setChecked(True)
        self.assertEqual(v.render_style, rs.RenderStyle())

    def test_unlocking_freezes_lights_without_a_jump(self):
        image_before = render(self.v)
        self.v.lock_check.setChecked(False)
        self.v._apply_lighting_now()
        np.testing.assert_allclose(render(self.v), image_before, atol=2)
        world = [l.GetPosition() for l in self.lights()]
        self.v.plotter.renderer.GetActiveCamera().Azimuth(70)
        self.v._apply_lighting_now()
        self.assertEqual(world, [l.GetPosition() for l in self.lights()])
        self.assertEqual(self.lights()[1].GetLightType(), 3)    # scene light

    def test_choosing_glossy_restyles_in_place_and_back(self):
        v = self.v
        before = list(self.lights())
        actor = v.pos_actor
        v._light_preset_buttons[GLOSSY].setChecked(True)
        self.assertEqual(v.render_style.preset, GLOSSY)
        self.assertIs(v.pos_actor, actor)                             # restyled, not rebuilt
        self.assertAlmostEqual(actor.GetProperty().GetSpecular(), 0.80)
        self.assertEqual(self.lights(), before)                       # same light objects
        self.assertEqual(v.plotter.renderer.GetLights().GetNumberOfItems(), 5)
        on = [l for l in self.lights() if l.GetSwitch() and l.GetIntensity() > 0]
        self.assertEqual(len(on), 4)
        np.testing.assert_allclose(self.lights()[1].GetPosition(), rs.light_direction(45, 35),
                                   atol=1e-9)
        v._light_preset_buttons[DEFAULT].setChecked(True)
        self.assertAlmostEqual(actor.GetProperty().GetSpecular(), 0.40)
        self.assertTrue(self.lights()[3].GetSwitch())                 # rim lights back

    def test_background_never_changes_the_lighting(self):
        for preset in (DEFAULT, GLOSSY):
            self.v.set_lighting_preset(preset)
            for color in ("white", "black"):
                self.v.set_background(color)
                self.assertEqual(self.v.render_style.preset, preset)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class RenderStyleSessionTests(unittest.TestCase):
    def viewer_with_panel(self):
        v = make_viewer([make_cube(width=4.0)])
        v._test_panel = QWidget()            # keep the panel alive; its widgets die with it
        v._build_lighting_content(v._test_panel)
        self.addCleanup(v.plotter.close)
        return v

    def test_preset_survives_save_and_load(self):
        v = self.viewer_with_panel()
        v.set_lighting_preset(GLOSSY)
        saved = json.loads(json.dumps({"render_style": v.render_style.to_dict()}))
        w = self.viewer_with_panel()
        w._set_render_style(w._session_render_style(saved))
        self.assertEqual(w.render_style.preset, GLOSSY)
        self.assertTrue(w._light_preset_buttons[GLOSSY].isChecked())
        self.assertEqual(w.render_style.to_dict(), saved["render_style"])

    def test_custom_lighting_round_trips(self):
        v = self.viewer_with_panel()
        v._light_widgets["key_elevation"][1].setValue(-20)
        v.lock_check.setChecked(False)
        first = json.loads(json.dumps({"render_style": v.render_style.to_dict()}))
        w = self.viewer_with_panel()
        w._set_render_style(w._session_render_style(first))
        self.assertEqual(w.render_style.to_dict(), first["render_style"])
        self.assertEqual(w._light_widgets["key_elevation"][1].value(), -20)
        self.assertFalse(w.lock_check.isChecked())
        self.assertFalse(w._light_custom_label.isHidden())

    def test_v1_session_loads_the_default_and_ignores_its_ssao_flag(self):
        w = self.viewer_with_panel()
        w.set_lighting_preset(GLOSSY)
        w._set_render_style(w._session_render_style({"ssao": True}))
        self.assertEqual(w.render_style.preset, DEFAULT)
        w.plotter.ren_win.Render()
        self.assertFalse(w.plotter.renderer.GetUseSSAO())


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class DepthPeelingTests(unittest.TestCase):
    VIEWS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    def test_translucent_layers_composite_front_to_back_from_six_viewpoints(self):
        """A red 50 % plate nearer the camera than a blue 50 % plate must give
        (127, 0, 64) on black; VTK's default OIT gives about (96, 0, 96)."""
        style = rs.RenderStyle()
        for d in self.VIEWS:
            with self.subTest(view=d):
                d = np.array(d, float)
                p = pv.Plotter(off_screen=True, window_size=(120, 120), lighting="none")
                try:
                    p.set_background("black")
                    for offset, colour in ((1.0, "red"), (0.0, "blue")):
                        p.add_mesh(pv.Plane(center=d * offset, direction=d, i_size=2, j_size=2),
                                   color=colour, opacity=0.5, lighting=False)
                    up = (0, 0, 1) if abs(d[2]) < 0.5 else (0, 1, 0)
                    p.camera.position = d * 10; p.camera.focal_point = (0, 0, 0); p.camera.up = up
                    p.enable_parallel_projection(); p.camera.parallel_scale = 0.5
                    rs.install_auto_render_passes(p, lambda: style)
                    p.ren_win.Render()
                    centre = p.screenshot(return_img=True)[60, 60].astype(int)
                    np.testing.assert_allclose(centre, [127, 0, 64], atol=3)
                finally:
                    p.close()

    def test_peeling_follows_every_source_of_translucency(self):
        v = make_viewer([make_cube(width=4.0)])
        self.addCleanup(v.plotter.close)
        ren, rw = v.plotter.renderer, v.plotter.ren_win

        def peeling():
            rw.Render()
            return bool(ren.GetUseDepthPeeling())

        self.assertTrue(peeling())                       # default: the 75 % mesh lines
        v.toggle_wireframe(False)
        self.assertFalse(peeling())                      # everything opaque
        v.surface_opacity = 0.5; v.update_visualization()
        self.assertTrue(peeling())
        v.surface_opacity = 1.0; v.update_visualization()
        v.selected_atoms = [0]; v._update_selected_highlights()
        self.assertTrue(peeling())                       # 0.55 highlight
        v.render_style = replace(v.render_style, depth_peeling=False)
        self.assertFalse(peeling())                      # the style can turn it off


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class MaterialAndOverlayTests(unittest.TestCase):
    def test_vertex_colours_drive_the_ambient_term(self):
        """Review item #7: the light-blue AmbientColor PyVista leaves on scalar
        meshes is not used; VTK 9.6 takes ambient from the vertex colour."""
        p = pv.Plotter(off_screen=True, window_size=(100, 100), lighting="none")
        self.addCleanup(p.close)
        p.set_background("black")
        rs.apply_lighting(p.renderer, rs.LightingSettings())
        sphere = pv.Sphere(theta_resolution=60, phi_resolution=60)
        sphere.point_data["colors"] = np.tile(np.array([0, 255, 0], np.uint8), (sphere.n_points, 1))
        actor = p.add_mesh(sphere, scalars="colors", rgb=True, ambient=1.0, diffuse=0.0, specular=0.0)
        actor.GetProperty().SetAmbientColor(1, 0, 0)
        self.assertEqual(p.screenshot(return_img=True)[50, 50].tolist(), [0, 255, 0])

    def test_grid_overlay_is_unlit_and_drawn_in_front_of_its_surface(self):
        import vtk
        v = make_viewer([make_cube(width=6.0)])
        self.addCleanup(v.plotter.close)
        for wire in (v.pos_wire_actor, v.neg_wire_actor):
            self.assertFalse(wire.GetProperty().GetLighting())
            factor, units = vtk.reference(0.0), vtk.reference(0.0)
            wire.GetMapper().GetRelativeCoincidentTopologyLineOffsetParameters(factor, units)
            self.assertEqual((float(factor), float(units)), (0.0, -4.0))
        self.assertEqual(vtk.vtkMapper.GetResolveCoincidentTopology(), 1)   # polygon offset

        def visible_line_pixels():
            """Pixels the overlay visibly changes (near-black lines on the lobes)."""
            with_lines = render(v).astype(int)
            for wire in (v.pos_wire_actor, v.neg_wire_actor):
                wire.SetVisibility(False)
            without = render(v).astype(int)
            for wire in (v.pos_wire_actor, v.neg_wire_actor):
                wire.SetVisibility(True)
            return int((np.abs(with_lines - without).sum(axis=2) > 30).sum())

        from viewer_harness import set_fixed_camera
        set_fixed_camera(v, scale_factor=0.9)
        offset = visible_line_pixels()
        vtk.vtkMapper.SetResolveCoincidentTopologyToOff()
        try:
            no_offset = visible_line_pixels()
        finally:
            vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.assertGreater(offset, no_offset * 1.2)        # z-fighting hid part of the grid


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class LobeOrientationTests(unittest.TestCase):
    def test_both_lobes_face_outward(self):
        v = make_viewer([make_cube(width=4.0)])
        self.addCleanup(v.plotter.close)
        for mesh in (v.pos_mesh, v.neg_mesh):
            normals = mesh.point_data["Normals"]
            outward = np.sum(normals * (mesh.points - np.asarray(mesh.center)), axis=1) > 0
            self.assertGreater(outward.mean(), 0.99)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class FullSizeExportTests(unittest.TestCase):
    def setUp(self):
        import offscreen_export
        self.oe = offscreen_export
        self.v = make_viewer([make_cube(width=4.0)])
        self.addCleanup(self.v.plotter.close)
        from viewer_harness import set_fixed_camera
        set_fixed_camera(self.v)

    def tiled(self, scale):
        with self.v._export_overlays_hidden():
            self.v.plotter.ren_win.Render()
            return self.v.plotter.screenshot(return_img=True, scale=scale)

    def test_full_size_matches_the_live_scene_at_1x(self):
        full = self.oe.render_full_size(self.v.plotter, self.v.render_style, 1, False)
        self.assertLess(np.abs(full.astype(float) - self.tiled(1).astype(float)).mean(), 0.5)

    def test_labels_keep_their_proportions_at_3x(self):
        v = self.v
        v.show_atom_labels = True
        v.update_visualization()
        for a in (v.pos_actor, v.neg_actor, v.pos_wire_actor, v.neg_wire_actor):
            a.SetVisibility(False)

        def label_height(img):
            rows = np.where((img.min(axis=2) > 180).any(axis=1))[0]
            return rows.max() - rows.min() + 1

        one = label_height(self.tiled(1))
        full = label_height(self.oe.render_full_size(v.plotter, v.render_style, 3, False))
        self.assertAlmostEqual(full / one, 3.0, delta=0.15)

    def test_falls_back_when_too_large_or_failing(self):
        with mock.patch.object(self.oe, "MAX_DIMENSION", 100):
            self.assertIsNone(self.oe.render_full_size(self.v.plotter, self.v.render_style, 2, False))
        with mock.patch.object(pv, "Plotter", side_effect=RuntimeError("no context")):
            self.assertIsNone(self.oe.render_full_size(self.v.plotter, self.v.render_style, 2, False))
            img = self.v._capture_frame(2, False)          # the viewer then uses tiling
        self.assertEqual(img.shape[:2], (1200, 1600))


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class OpenLobeTests(unittest.TestCase):
    def viewer(self, width):
        v = make_viewer([make_cube(width=width)])
        self.addCleanup(v.plotter.close)
        return v

    def test_closed_lobes_get_no_backface_property(self):
        v = self.viewer(4.0)
        self.assertFalse(v.open_lobes)
        self.assertIsNone(v.pos_actor.GetBackfaceProperty())

    def test_lobe_cut_by_the_grid_edge_is_flagged_and_shaded_inside(self):
        import ChemInsight3D as ci
        v = self.viewer(60.0)                       # reaches the grid boundary
        self.assertTrue(v.open_lobes)
        back = v.pos_actor.GetBackfaceProperty()
        self.assertIsNotNone(back)
        np.testing.assert_allclose(back.GetColor(),
                                   0.6 * np.array(pv.Color(v.lobe_pos_color).float_rgb), atol=1e-6)
        v.toggle_close_open_lobes(True)
        self.assertFalse(v.open_lobes)
        self.assertFalse(ci.is_open_surface(v.pos_mesh) or ci.is_open_surface(v.neg_mesh))
        self.assertIsNone(v.pos_actor.GetBackfaceProperty())

    def test_padding_keeps_values_in_place(self):
        import ChemInsight3D as ci
        v = self.viewer(4.0)
        grid = v._create_grid()
        padded = ci.pad_grid_with_zeros(grid)
        self.assertEqual(padded.dimensions, tuple(d + 2 for d in grid.dimensions))
        points = pv.PolyData(np.array([[0.7, 0.3, -1.1], [-5.0, 2.0, 4.0]]))
        np.testing.assert_allclose(points.sample(padded)["values"], points.sample(grid)["values"])


if __name__ == "__main__":
    unittest.main()
