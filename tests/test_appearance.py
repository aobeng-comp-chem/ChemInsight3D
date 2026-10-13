"""Pure tests of appearance settings: palettes and session (de)serialisation."""
import json
import unittest
from dataclasses import replace

import appearance as ap


class PaletteTests(unittest.TestCase):
    def test_ids_are_unique_and_default_exists(self):
        ids = [p.id for p in ap.PALETTES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(ap.DEFAULT_PALETTE, ap.PALETTES_BY_ID)
        self.assertNotIn(ap.CUSTOM_PALETTE, ap.PALETTES_BY_ID)

    def test_every_old_scheme_name_still_maps_to_its_colours(self):
        old = {
            "Crimson / Steel-blue (default)": ((0.60, 0.15, 0.18), (0.55, 0.72, 0.82)),
            "Mathematica (default)":          ((0.60, 0.15, 0.18), (0.55, 0.72, 0.82)),
            "Red / Blue":                     ((0.85, 0.10, 0.10), (0.10, 0.30, 0.85)),
            "Red / Green":                    ((0.85, 0.12, 0.12), (0.12, 0.70, 0.25)),
            "Orange / Teal":                  ((0.92, 0.50, 0.05), (0.05, 0.60, 0.65)),
            "Purple / Gold":                  ((0.55, 0.10, 0.75), (0.85, 0.72, 0.05)),
            "White / Grey":                   ((0.95, 0.95, 0.95), (0.45, 0.45, 0.45)),
        }
        for name, (pos, neg) in old.items():
            with self.subTest(name=name):
                p = ap.PALETTES_BY_ID[ap.LEGACY_PALETTE_NAMES[name]]
                self.assertEqual((p.positive, p.negative), (pos, neg))

    def test_red_green_is_flagged(self):
        self.assertFalse(ap.PALETTES_BY_ID["red_green"].colorblind_safe)
        self.assertTrue(all(p.colorblind_safe for p in ap.PALETTES if p.id != "red_green"))

    def test_palette_for_colours(self):
        p = ap.PALETTES_BY_ID["orange_blue"]
        self.assertEqual(ap.palette_for_colors(p.positive, p.negative), "orange_blue")
        self.assertEqual(ap.palette_for_colors(p.negative, p.positive), ap.CUSTOM_PALETTE)


class SessionTests(unittest.TestCase):
    CUSTOM = ap.AppearanceSettings(
        representation="Sticks", atom_scale=0.5, bond_scale=1.75, palette=ap.CUSTOM_PALETTE,
        positive_rgb=(0.2, 0.8, 0.3), negative_rgb=(0.8, 0.2, 0.7), surface_opacity=0.4,
        background="white", close_open_lobes=True, show_mesh=False, show_atom_labels=True)

    def session(self, settings):
        return json.loads(json.dumps({**ap.legacy_session_keys(settings),
                                      "appearance": ap.appearance_to_session(settings)}))

    def test_round_trip_keeps_every_value(self):
        self.assertEqual(ap.appearance_from_session(self.session(self.CUSTOM)), self.CUSTOM)
        named = replace(ap.AppearanceSettings(), palette="purple_teal",
                        positive_rgb=ap.PALETTES_BY_ID["purple_teal"].positive,
                        negative_rgb=ap.PALETTES_BY_ID["purple_teal"].negative)
        self.assertEqual(ap.appearance_from_session(self.session(named)), named)

    def test_save_load_save_is_stable(self):
        first = self.session(self.CUSTOM)
        second = self.session(ap.appearance_from_session(first))
        self.assertEqual(first, second)

    def test_explicit_rgb_wins_over_a_conflicting_scheme_name(self):
        state = self.session(self.CUSTOM)
        state["appearance"]["scheme_id"] = "crimson_steel_blue"
        loaded = ap.appearance_from_session(state)
        self.assertEqual(loaded.positive_rgb, (0.2, 0.8, 0.3))
        self.assertEqual(loaded.palette, ap.CUSTOM_PALETTE)

    def test_legacy_sessions_read_the_flat_keys(self):
        old = {"mol_theme": "Space Filling", "atom_scale": 2.0, "opacity": 0.6,
               "background": "white", "show_wireframe": False, "show_labels": True,
               "color_scheme": "Mathematica (default)"}
        s = ap.appearance_from_session(old)
        self.assertEqual((s.representation, s.atom_scale, s.surface_opacity, s.background),
                         ("Space Filling", 2.0, 0.6, "white"))
        self.assertEqual((s.show_mesh, s.show_atom_labels), (False, True))
        self.assertEqual(s.palette, "crimson_steel_blue")
        self.assertEqual(ap.appearance_from_session({}), ap.AppearanceSettings())

    def test_v1_reader_keys_are_written(self):
        keys = ap.legacy_session_keys(self.CUSTOM)
        self.assertEqual(keys["mol_theme"], "Sticks")
        self.assertEqual(keys["color_scheme"], ap.CUSTOM_LABEL)
        named = ap.legacy_session_keys(ap.AppearanceSettings())
        self.assertEqual(named["color_scheme"], "Crimson / Steel-blue (default)")

    def test_tubes_round_trip_and_older_readers_get_sticks(self):
        tubes = replace(ap.AppearanceSettings(), representation="Tubes")
        state = self.session(tubes)
        self.assertEqual(state["appearance"]["representation"], "tubes")
        self.assertEqual(state["mol_theme"], "Sticks")       # v1.0.0 has no Tubes theme
        self.assertEqual(ap.appearance_from_session(state).representation, "Tubes")

    def test_mesh_switch_from_every_file_generation(self):
        self.assertTrue(ap.AppearanceSettings().show_mesh)                      # v1.0.0 default
        self.assertFalse(ap.appearance_from_session({"show_wireframe": False}).show_mesh)
        self.assertTrue(ap.appearance_from_session({"show_wireframe": True}).show_mesh)
        block = {"appearance": {"show_mesh_overlay": False}}
        self.assertFalse(ap.appearance_from_session(block).show_mesh)
        self.assertIs(ap.legacy_session_keys(self.CUSTOM)["show_wireframe"], False)
        # Development builds saved a style name: anything but "off" means on.
        for style, shown in (("fine", True), ("globe", True), ("off", False)):
            block = {"appearance": {"mesh_style": style}}
            self.assertIs(ap.appearance_from_session(block).show_mesh, shown)

    def test_malformed_values_fall_back_or_clamp(self):
        state = {"appearance": {"representation": "cartoon", "atom_scale": 99, "bond_scale": "x",
                                "positive_rgb": [2, -1, 0.5], "negative_rgb": "red",
                                "surface_opacity": 0.0, "background": "purple",
                                "show_mesh_overlay": "yes", "scheme_id": "nope"}}
        s = ap.appearance_from_session(state)
        base = ap.AppearanceSettings()
        self.assertEqual(s.representation, base.representation)
        self.assertEqual(s.atom_scale, 3.0)                    # clamped
        self.assertEqual(s.bond_scale, base.bond_scale)
        self.assertEqual(s.surface_opacity, 0.10)              # clamped
        self.assertEqual(s.background, base.background)
        self.assertEqual(s.show_mesh, base.show_mesh)
        self.assertEqual(s.palette, ap.DEFAULT_PALETTE)        # one RGB unusable -> palette
        for junk in (None, [], "session", {"appearance": "bad"}):
            self.assertEqual(ap.appearance_from_session(junk), base)


if __name__ == "__main__":
    unittest.main()
