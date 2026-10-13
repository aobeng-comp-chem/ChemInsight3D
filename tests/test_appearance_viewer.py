"""Off-screen tests of the Appearance panel against the real viewer."""
import json
import unittest
from unittest import mock

import numpy as np

from viewer_harness import SKIP_REASON, WATER, make_cube, make_viewer, render

if SKIP_REASON is None:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

    import appearance as ap
    import camera_utils as cu
    import rendering_style as rs


def session_dict(v):
    """What save_session writes for appearance and lighting."""
    s = v._appearance_settings()
    return json.loads(json.dumps({**ap.legacy_session_keys(s),
                                  "appearance": ap.appearance_to_session(s),
                                  "render_style": v.render_style.to_dict()}))


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class AppearanceTestCase(unittest.TestCase):
    def viewer(self, cubes=None):
        v = make_viewer(cubes if cubes is not None else [make_cube(width=4.0)])
        v._test_panel = QWidget()                     # keep widgets alive
        v._build_appearance_content(v._test_panel)
        self.addCleanup(v.plotter.close)
        return v

    def w(self, v):
        return v._appearance_widgets


class WidgetSyncTests(AppearanceTestCase):
    def test_session_load_updates_every_widget(self):                         # T01 / F01
        v = self.viewer()
        v.update_atom_scale(250)
        state = {"mol_theme": "Sticks", "atom_scale": 0.5, "opacity": 0.4, "show_wireframe": False}
        v._apply_session_settings(state)
        w = self.w(v)
        self.assertTrue(w["representation"]["Sticks"].isChecked())
        self.assertFalse(w["representation"]["Ball and Stick"].isChecked())
        self.assertEqual(w["atom_slider"].value(), 50)
        self.assertEqual(w["atom_value"].text(), "0.50×")
        self.assertEqual(w["opacity_slider"].value(), 40)
        self.assertEqual(w["opacity_value"].text(), "40 %")
        self.assertFalse(w["mesh"].isChecked())

    def test_touching_an_unrelated_control_keeps_loaded_values(self):
        v = self.viewer()
        v._apply_session_settings({"mol_theme": "Sticks", "atom_scale": 0.5, "opacity": 0.4})
        self.w(v)["labels"].setChecked(True)
        self.assertEqual((v.current_mol_theme, v.atom_scale, v.surface_opacity), ("Sticks", 0.5, 0.4))

    def test_sync_never_changes_the_scene(self):
        v = self.viewer()
        with mock.patch.object(v, "_contour_lobes") as contour, \
                mock.patch.object(v, "_rebuild_molecule") as rebuild:
            v._sync_appearance_widgets()
        contour.assert_not_called(); rebuild.assert_not_called()

    def test_controls_without_effect_are_disabled(self):                       # T13
        v = self.viewer()
        v.set_molecule_theme("Space Filling")
        self.assertFalse(self.w(v)["bond_slider"].isEnabled())
        self.assertTrue(self.w(v)["atom_slider"].isEnabled())
        self.assertEqual(v.bond_actors, [])
        for theme in ("Sticks", "Tubes"):
            v.set_molecule_theme(theme)
            self.assertFalse(self.w(v)["atom_slider"].isEnabled())
            self.assertTrue(self.w(v)["bond_slider"].isEnabled())


class LobeColourTests(AppearanceTestCase):
    GREEN, MAGENTA = (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)

    def test_custom_colours_survive_save_and_load(self):                       # T02 / F02
        v = self.viewer()
        with mock.patch.object(v, "_contour_lobes") as contour:
            v._set_lobe_colors(self.GREEN, self.MAGENTA)
        contour.assert_not_called()                                            # recolour only
        self.assertEqual(self.w(v)["palette"].currentData(), ap.CUSTOM_PALETTE)
        saved = session_dict(v)
        u = self.viewer()
        u._apply_session_settings(saved)
        u.update_visualization()
        self.assertEqual((u.lobe_pos_color, u.lobe_neg_color), (self.GREEN, self.MAGENTA))
        self.assertEqual(self.w(u)["palette"].currentData(), ap.CUSTOM_PALETTE)
        np.testing.assert_allclose(u.pos_actor.GetProperty().GetColor(), self.GREEN)
        self.assertEqual(session_dict(u)["appearance"], saved["appearance"])     # T12

    def test_old_named_scheme_restores_its_colours(self):                      # T03
        u = self.viewer()
        u._apply_session_settings({"color_scheme": "Crimson / Steel-blue (default)"})
        self.assertEqual(u.lobe_pos_color, (0.60, 0.15, 0.18))
        self.assertEqual(self.w(u)["palette"].currentData(), "crimson_steel_blue")

    def test_palette_and_swap_recolour_in_place(self):
        v = self.viewer()
        actor = v.pos_actor
        v.set_lobe_palette("orange_blue")
        self.assertIs(v.pos_actor, actor)                                      # same actor
        np.testing.assert_allclose(actor.GetProperty().GetColor(),
                                   ap.PALETTES_BY_ID["orange_blue"].positive)
        v.swap_lobe_colors()
        self.assertEqual(v.lobe_pos_color, ap.PALETTES_BY_ID["orange_blue"].negative)
        self.assertEqual(self.w(v)["palette"].currentData(), ap.CUSTOM_PALETTE)


class ThemeAndBackgroundTests(AppearanceTestCase):
    def test_ui_theme_leaves_the_scene_background_alone(self):                 # T04 / F03
        v = self.viewer()
        v.main_window = QMainWindow()
        v.set_background("white")
        for _ in range(2):
            v.toggle_ui_theme()
            self.assertEqual(v.background_color, "white")
            np.testing.assert_allclose(v.plotter.renderer.GetBackground(), (1, 1, 1))
            self.assertTrue(self.w(v)["background"]["white"].isChecked())
        v.main_window = None

    def test_background_keeps_text_readable(self):                            # T05
        v = self.viewer()
        v.toggle_atom_labels(True)
        for color, expected in (("white", (0.05, 0.05, 0.05)), ("black", (1.0, 1.0, 1.0))):
            v.set_background(color)
            np.testing.assert_allclose(v.atom_label_actors[0].GetTextProperty().GetColor(), expected)
            self.assertEqual(v._title_color(), "black" if color == "white" else "white")

    def test_a_chosen_lighting_preset_survives_a_background_change(self):
        v = self.viewer()
        v._test_lighting_panel = QWidget()
        v._build_lighting_content(v._test_lighting_panel)
        v.set_lighting_preset("Glossy")
        v.set_background("white")
        self.assertEqual(v.render_style.preset, "Glossy")


class AtomLabelTests(AppearanceTestCase):
    def offsets(self, v):
        v.plotter.ren_win.Render()                     # the per-frame hook re-places labels
        return np.array([a.GetDisplayOffset()[1] for a in v.atom_label_actors])

    def sphere_px(self, v):
        """On-screen radius of each drawn atom sphere, from the camera alone."""
        cam = v.plotter.renderer.GetActiveCamera()
        px_per_unit = v.plotter.renderer.GetSize()[1] / (2 * cam.GetParallelScale())
        cube = v.cubes[0]
        return np.array([v._atom_display_radius(an) * px_per_unit for an in cube["atoms"]])

    def test_labels_clear_the_spheres_at_any_size_and_zoom(self):              # T06 / F04 F05
        v = self.viewer()
        v.toggle_atom_labels(True)
        for scale in (50, 100, 250):
            for zoom in (1.0, 2.0, 0.5):
                with self.subTest(scale=scale, zoom=zoom):
                    v.update_atom_scale(scale)
                    v.plotter.renderer.GetActiveCamera().Zoom(zoom)
                    offsets = self.offsets(v)
                    self.assertTrue(np.all(offsets > self.sphere_px(v)))
                    self.assertTrue(np.all(offsets <= self.sphere_px(v) + ap_margin() + 2))

    def test_no_duplicate_labels(self):
        v = self.viewer()
        for _ in range(3):
            v.toggle_atom_labels(True)
            v.set_molecule_theme("Space Filling")
            v.set_molecule_theme("Ball and Stick")
        self.assertEqual(len(v.atom_label_actors), len(WATER))
        labels = [p for p in self._view_props(v) if p.GetClassName().endswith("BillboardTextActor3D")]
        self.assertEqual(len(labels), len(WATER))

    @staticmethod
    def _view_props(v):
        props = v.plotter.renderer.GetViewProps()
        props.InitTraversal()
        out, p = [], props.GetNextProp()
        while p is not None:
            out.append(p); p = props.GetNextProp()
        return out


def ap_margin():
    import ChemInsight3D as ci
    return ci.LABEL_MARGIN_PX


class RebuildCostTests(AppearanceTestCase):
    def test_representation_changes_never_recontour(self):                     # T07 / F06
        v = self.viewer()
        cam_before = cu.capture_camera_state(v.plotter.renderer.GetActiveCamera())
        lobes = (v.pos_actor, v.neg_actor)
        with mock.patch.object(v, "_contour_lobes", wraps=v._contour_lobes) as contour:
            for theme in ("Sticks", "Tubes", "Space Filling", "Ball and Stick"):
                v.set_molecule_theme(theme)
        self.assertEqual(contour.call_count, 0)
        self.assertEqual((v.pos_actor, v.neg_actor), lobes)
        self.assertEqual(cu.capture_camera_state(v.plotter.renderer.GetActiveCamera()), cam_before)

    def test_slider_drag_is_coalesced_and_lands_on_the_final_value(self):    # T08
        v = self.viewer([make_cube(atoms=[(8, 0.0, 0.0, 0.0)], width=4.0)])
        with mock.patch.object(v, "_add_atoms", wraps=v._add_atoms) as add_atoms:
            for value in range(100, 176, 5):
                self.w(v)["atom_slider"].setValue(value)
            self.assertEqual(add_atoms.call_count, 0)              # nothing rebuilt mid-drag
            self.assertEqual(self.w(v)["atom_value"].text(), "1.75×")
            v._appearance_timer.timeout.emit()                     # the debounce fires
            self.assertEqual(add_atoms.call_count, 1)
        b = v.atom_actors[0].GetMapper().GetInput().bounds
        expected = v._atom_display_radius(8)
        self.assertAlmostEqual((b[1] - b[0]) / 2, expected, places=2)

    def test_session_load_cancels_a_pending_slider_rebuild(self):              # T14
        v = self.viewer()
        self.w(v)["atom_slider"].setValue(200)
        v._apply_session_settings({"atom_scale": 0.5})
        self.assertFalse(v._appearance_timer.isActive())
        self.assertEqual(v.atom_scale, 0.5)

    def test_mesh_toggle_does_not_recontour_but_closing_lobes_does(self):     # T09 / T10
        v = self.viewer()
        with mock.patch.object(v, "_contour_lobes", wraps=v._contour_lobes) as contour:
            v.toggle_wireframe(False); v.toggle_wireframe(True)
            self.assertEqual(contour.call_count, 0)
            v.toggle_close_open_lobes(True)
            self.assertEqual(contour.call_count, 1)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class UvGridTests(unittest.TestCase):
    """The v1.0.0 mesh overlay (build_uv_grid_lines)."""

    def test_lines_are_v1_meridians_about_y_and_parallels_along_y(self):
        import pyvista as pv
        import ChemInsight3D as ci
        lobe = pv.Sphere(radius=2.0, center=(1.0, -0.5, 0.3),
                         theta_resolution=80, phi_resolution=80).triangulate()
        pts = np.asarray(ci.build_uv_grid_lines(lobe).points)
        self.assertGreater(len(pts), 0)
        rel = pts - np.asarray(lobe.center)
        b = lobe.bounds
        lo, hi = b[2] + 0.05 * (b[3] - b[2]), b[3] - 0.05 * (b[3] - b[2])
        levels = lo + (hi - lo) * np.arange(1, ci.GRID_LINES_V) / ci.GRID_LINES_V
        angles = np.pi * np.arange(ci.GRID_LINES_U) / ci.GRID_LINES_U
        normals = np.stack([np.cos(angles), np.zeros_like(angles), np.sin(angles)], axis=1)
        on_meridian = np.abs(rel @ normals.T).min(axis=1) < 1e-6
        on_parallel = np.abs(pts[:, 1:2] - levels[None, :]).min(axis=1) < 1e-6
        self.assertTrue(np.all(on_meridian | on_parallel))
        self.assertEqual(len(np.unique(np.round(pts[on_parallel & ~on_meridian, 1], 6))),
                         ci.GRID_LINES_V - 1)

    def test_lines_do_not_erase_translucent_lobes(self):
        """With depth peeling, the lines must not hide parts of translucent lobes."""
        from viewer_harness import set_fixed_camera
        v = make_viewer([make_cube(width=6.0)])
        self.addCleanup(v.plotter.close)

        def lobe_pixels(mesh_on):
            v.toggle_wireframe(mesh_on)
            set_fixed_camera(v)
            img = render(v).astype(int)
            return int((img.sum(axis=2) > 60).sum())

        v.update_opacity(80)
        self.assertGreater(lobe_pixels(True), 0.97 * lobe_pixels(False))


class MeshOverlayTests(AppearanceTestCase):
    def test_on_by_default_in_the_v1_style(self):
        import ChemInsight3D as ci
        v = self.viewer()
        self.assertTrue(v.show_wireframe)
        self.assertTrue(self.w(v)["mesh"].isChecked())
        for wire in (v.pos_wire_actor, v.neg_wire_actor):
            prop = wire.GetProperty()
            np.testing.assert_allclose(prop.GetColor(), ci.MESH_LINE_COLOR, atol=1 / 255)
            self.assertAlmostEqual(prop.GetLineWidth(), 0.8, places=5)
            self.assertAlmostEqual(prop.GetOpacity(), 0.75)

    def test_switch_and_recolour_never_recontour(self):
        import ChemInsight3D as ci
        v = self.viewer()
        with mock.patch.object(v, "_contour_lobes") as contour:
            self.w(v)["mesh"].setChecked(False)
            self.assertIsNone(v.pos_wire_actor)
            self.w(v)["mesh"].setChecked(True)
            self.assertIsNotNone(v.pos_wire_actor)
            v.set_lobe_palette("orange_blue")                  # lines stay near-black
            np.testing.assert_allclose(v.pos_wire_actor.GetProperty().GetColor(),
                                       ci.MESH_LINE_COLOR, atol=1 / 255)
        contour.assert_not_called()

    def test_switch_round_trips_through_a_session(self):
        v = self.viewer()
        v.toggle_wireframe(False)
        u = self.viewer()
        u._apply_session_settings(session_dict(v))
        self.assertFalse(u.show_wireframe)
        self.assertFalse(self.w(u)["mesh"].isChecked())


class V1SizeTests(AppearanceTestCase):
    def test_ball_and_stick_atoms_and_bonds_have_the_v1_sizes(self):
        import ChemInsight3D as ci
        v = self.viewer()
        cube = v.cubes[0]
        for an in set(cube["atoms"]):
            bs_r = v.ELEMENT_DATA[an][3]
            self.assertAlmostEqual(v._atom_display_radius(an), bs_r / 0.529177)
        themes = ci.MultiCubeVisualizer.MOLECULE_THEMES
        self.assertEqual(themes["Ball and Stick"]["bond_radius_A"], 0.10)
        self.assertEqual(themes["Sticks"]["bond_radius_A"], 0.10)


class TubesTests(AppearanceTestCase):
    def tube_mesh(self, v):
        self.assertEqual(v.atom_actors, [])                  # the ends belong to the bond mesh
        self.assertEqual(len(v.bond_actors), 1)
        return v.bond_actors[0].GetMapper().GetInput()

    def test_every_atom_ends_in_a_sphere_of_the_tube_radius(self):
        import pyvista as pv
        import ChemInsight3D as ci
        v = self.viewer()
        v.set_molecule_theme("Tubes")
        cube = v.cubes[0]
        r = ci.MultiCubeVisualizer.MOLECULE_THEMES["Tubes"]["bond_radius_A"] / 0.529177
        mesh = pv.wrap(self.tube_mesh(v))
        pts = np.asarray(mesh.points)
        for pos in np.asarray(cube["coordinates"], dtype=float):
            dist = np.linalg.norm(pts - pos, axis=1)
            # the farthest point of the mesh near this atom is its rounded end, at the tube radius
            cap = dist[dist < 1.5 * r]
            self.assertAlmostEqual(cap.max(), r, delta=0.01 * r)
        v.update_bond_scale(200)                             # bond thickness sizes the ends
        pts = np.asarray(pv.wrap(self.tube_mesh(v)).points)
        o = np.asarray(cube["coordinates"][0], dtype=float)
        self.assertAlmostEqual(np.linalg.norm(pts - o, axis=1)[
            np.linalg.norm(pts - o, axis=1) < 3 * r].max(), 2 * r, delta=0.02 * r)

    def test_bonds_change_colour_at_the_midpoint(self):
        """Ball and stick splits an O-H bond at the covalent-radius ratio (~0.68 from O);
        Tubes splits it at 0.5, as Mathematica does."""
        import pyvista as pv
        v = self.viewer()
        v.set_molecule_theme("Tubes")
        mesh = pv.wrap(self.tube_mesh(v))
        bond = v.cubes[0]["bonds"][0]
        p1, p2 = np.asarray(bond["c1"], float), np.asarray(bond["c2"], float)
        length = np.linalg.norm(p2 - p1)
        axis = (p2 - p1) / length
        rel = np.asarray(mesh.points) - p1
        t = rel @ axis / length
        radial = np.linalg.norm(rel - np.outer(t * length, axis), axis=1)
        # Tube segments only have vertex rings at their two ends, so the only
        # vertices well inside the bond are where the two halves meet.
        inside = (radial < 0.5) & (t > 0.25) & (t < 0.75)
        np.testing.assert_allclose(t[inside], 0.5, atol=1e-6)
        colours = {tuple(c) for c in mesh.point_data["colors"][inside]}
        self.assertEqual(len(colours), 2)                     # O red meets H white

    def test_labels_clear_the_tube_ends(self):
        v = self.viewer()
        v.set_molecule_theme("Tubes")
        v.toggle_atom_labels(True)
        v.plotter.ren_win.Render()
        cam = v.plotter.renderer.GetActiveCamera()
        px_per_unit = v.plotter.renderer.GetSize()[1] / (2 * cam.GetParallelScale())
        cube = v.cubes[0]
        end_px = v._rendered_atom_radius(cube["atoms"][0], 0.529177) * px_per_unit
        self.assertGreater(end_px, 1)
        offsets = [a.GetDisplayOffset()[1] for a in v.atom_label_actors]
        self.assertTrue(all(o > end_px for o in offsets))


class PanelLayoutTests(AppearanceTestCase):
    def test_panel_scrolls_instead_of_running_off_a_short_window(self):
        from PyQt5.QtWidgets import QScrollArea
        v = self.viewer()
        window = QMainWindow(); window.resize(500, 300); window.show()
        panel = v._test_panel
        panel.setParent(window); panel.move(0, 50); panel.show()
        v.main_window = window
        try:
            v._fit_dropdown_height(panel)
            QApplication.processEvents()
            self.assertLessEqual(panel.y() + panel.height(), window.height())
            self.assertGreater(panel.findChild(QScrollArea).verticalScrollBar().maximum(), 0)
        finally:
            v.main_window = None


class ResetTests(AppearanceTestCase):
    def test_reset_restores_defaults_only(self):
        v = self.viewer()
        v.set_molecule_theme("Sticks"); v.update_atom_scale(200); v.set_lobe_palette("red_blue")
        v.update_opacity(40); v.set_background("white")
        cam = cu.capture_camera_state(v.plotter.renderer.GetActiveCamera())
        style = v.render_style
        iso = v.current_isovalue
        v.reset_appearance_defaults()
        self.assertEqual(v._appearance_settings(), ap.AppearanceSettings())
        self.assertEqual(cu.capture_camera_state(v.plotter.renderer.GetActiveCamera()), cam)
        self.assertEqual(v.render_style, style)
        self.assertEqual(v.current_isovalue, iso)
        self.assertTrue(self.w(v)["representation"]["Ball and Stick"].isChecked())


if __name__ == "__main__":
    unittest.main()
