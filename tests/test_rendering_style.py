"""Pure tests of the render-style data (no Qt, no rendering)."""
import json
import math
import unittest
from dataclasses import replace

import numpy as np

import rendering_style as rs


class DefaultPresetTests(unittest.TestCase):
    def test_default_rig_reproduces_pyvista_light_kit(self):
        """Same intensities, directions and colours as the lights PyVista's
        default lighting='light kit' adds (what v1.0.0 rendered with)."""
        import pyvista as pv
        plotter = pv.Plotter(off_screen=True)
        kit_lights = plotter.renderer.GetLights()
        kit_lights.InitTraversal()
        expected = [kit_lights.GetNextItem() for _ in range(kit_lights.GetNumberOfItems())]
        plotter.close()
        self.assertEqual(len(expected), 5)

        specs = rs.light_specs(rs.RenderStyle().lighting)
        # Match each kit light to a rig light by intensity and direction.
        for light in expected:
            match = [s for s in specs
                     if math.isclose(s[2], light.GetIntensity(), rel_tol=1e-9)
                     and np.allclose(s[1], light.GetPosition(), atol=1e-6)]
            self.assertEqual(len(match), 1, f"no rig light matches {light.GetPosition()}")
            np.testing.assert_allclose(match[0][3], light.GetDiffuseColor(), atol=1e-6)

    def test_default_materials_are_the_v1_values(self):
        style = rs.RenderStyle()
        self.assertEqual(style.preset, "Default")
        self.assertEqual(style.material("lobe"), rs.Material(0.15, 0.85, 0.40, 20))
        self.assertEqual(style.material("atom"), rs.Material(0.08, 0.85, 0.50, 40))
        self.assertEqual(style.material("bond"), rs.Material(0.08, 0.90, 0.30, 20))

    def test_exactly_two_presets_each_defining_every_actor_class(self):
        self.assertEqual(list(rs.STYLE_PRESETS), ["Default", "Glossy"])
        for name, (_lights, materials) in rs.STYLE_PRESETS.items():
            with self.subTest(preset=name):
                self.assertEqual(set(rs.MATERIAL_PRESETS[materials]), set(rs.ACTOR_CLASSES))


class GlossyPresetTests(unittest.TestCase):
    def test_glossy_puts_four_white_lights_on_the_diagonals(self):
        style = rs.RenderStyle().with_preset("Glossy")
        self.assertEqual(style.material("lobe"), rs.Material(0.30, 0.75, 0.80, 50))
        specs = {name: (d, i, c) for name, d, i, c, _on in rs.light_specs(style.lighting)}
        self.assertEqual(len(specs), 5)                          # same five-light rig
        for name, az, el in (("upper_right", 45, 35), ("upper_left", -45, 35),
                             ("lower_right", 45, -35), ("lower_left", -45, -35)):
            with self.subTest(light=name):
                d, i, c = specs[name]
                np.testing.assert_allclose(d, rs.light_direction(az, el), atol=1e-12)
                self.assertAlmostEqual(i, 0.45)
                self.assertEqual(c, rs.WHITE)
        self.assertEqual(specs["head"][1], 0.0)                 # no headlight

    def test_quad_lower_pair_follows_fill_ratio(self):
        lights = replace(rs.STYLE_PRESETS["Glossy"][0], fill_ratio=0.5)
        specs = {name: i for name, _d, i, _c, _on in rs.light_specs(lights)}
        self.assertAlmostEqual(specs["lower_left"], 0.225)


class LightGeometryTests(unittest.TestCase):
    def test_direction_convention(self):
        np.testing.assert_allclose(rs.light_direction(0, 0), (0, 0, 1), atol=1e-12)
        np.testing.assert_allclose(rs.light_direction(90, 0), (1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(rs.light_direction(0, 90), (0, 1, 0), atol=1e-12)

    def test_fill_mirrors_key(self):
        s = rs.LightingSettings(key_azimuth=40, key_elevation=30, key_intensity=1.0,
                                fill_ratio=0.5, fill_elevation_factor=-0.5)
        fill = dict((name, (d, i)) for name, d, i, _c, _on in rs.light_specs(s))["fill"]
        np.testing.assert_allclose(fill[0], rs.light_direction(-40, -15), atol=1e-12)
        self.assertAlmostEqual(fill[1], 0.5)

    def test_camera_frame_is_orthonormal_and_right_handed(self):
        import vtk
        cam = vtk.vtkCamera()
        cam.SetPosition(3, -4, 5); cam.SetFocalPoint(1, 1, 1); cam.SetViewUp(0.2, 0.1, 1)
        right, up, back = (np.array(a) for a in rs.camera_frame(cam))
        for a in (right, up, back):
            self.assertAlmostEqual(np.linalg.norm(a), 1.0)
        self.assertAlmostEqual(float(right @ up), 0.0)
        self.assertAlmostEqual(float(up @ back), 0.0)
        np.testing.assert_allclose(np.cross(right, up), back, atol=1e-12)


class SerializationTests(unittest.TestCase):
    def custom(self):
        style = rs.RenderStyle.from_preset("Glossy", depth_peeling=False)
        return replace(style, preset=rs.CUSTOM, lighting=replace(
            style.lighting, key_azimuth=-75.0, lock_to_camera=False,
            scene_frame=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    def test_round_trip_through_json(self):
        for style in (rs.RenderStyle(), rs.RenderStyle.from_preset("Glossy"), self.custom()):
            with self.subTest(preset=style.preset):
                back = rs.RenderStyle.from_dict(json.loads(json.dumps(style.to_dict())))
                self.assertEqual(back, style)
                self.assertEqual(back.to_dict(), style.to_dict())

    def test_older_preset_names_still_load(self):
        cases = {"Default (v1.0.0)": "Default", "Legacy light kit": "Default",
                 "Neutral (glossy)": "Glossy", "Publication": "Default",
                 "Print (white bg)": "Default", "Screen (black bg)": "Default"}
        for old, new in cases.items():
            with self.subTest(old=old):
                block = {"preset": old, "materials": "Publication", "ssao": True,
                         "lighting": {"key_azimuth": -75.0}}
                style = rs.RenderStyle.from_dict(block)
                self.assertEqual(style.preset, new)
                self.assertEqual(style.materials, rs.STYLE_PRESETS[new][1])

    def test_a_block_without_lighting_takes_the_presets(self):
        # Blocks saved by the two-preset build held only the preset name.
        style = rs.RenderStyle.from_dict({"preset": "Glossy", "depth_peeling": True, "version": 2})
        self.assertEqual(style, rs.RenderStyle.from_preset("Glossy"))

    def test_malformed_values_fall_back_to_defaults(self):
        base = rs.RenderStyle()
        bad = {"preset": rs.CUSTOM, "materials": 7, "depth_peeling": "yes",
               "lighting": {"key_azimuth": "left", "key_intensity": 99, "key_color": [2, -1],
                            "lock_to_camera": False, "scene_frame": "frame", "layout": "ring"}}
        style = rs.RenderStyle.from_dict(bad)
        self.assertEqual(style.preset, rs.CUSTOM)
        self.assertEqual(style.materials, "Legacy")
        self.assertEqual(style.depth_peeling, base.depth_peeling)
        self.assertEqual(style.lighting.key_azimuth, base.lighting.key_azimuth)
        self.assertEqual(style.lighting.layout, "kit")
        self.assertEqual(style.lighting.key_intensity, 1.5)            # clamped
        self.assertEqual(style.lighting.key_color, base.lighting.key_color)
        self.assertTrue(style.lighting.lock_to_camera)                 # unusable frame
        self.assertEqual(rs.RenderStyle.from_dict({"preset": 7}), base)
        for junk in (None, [], "style"):
            self.assertEqual(rs.RenderStyle.from_dict(junk), base)


if __name__ == "__main__":
    unittest.main()
