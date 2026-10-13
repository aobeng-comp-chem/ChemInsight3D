"""
Image-regression tests for lighting and materials.

Renders fixed scenes off-screen (800x600, fixed camera) and compares them with
reference PNGs by mean absolute difference: fail above 1/255. References are
only comparable on the machine setup that made them (see
viewer_harness.environment_fingerprint); elsewhere the comparison is skipped.

Regenerate the references only for an intended change of look:
    CI3D_UPDATE_REFERENCES=1 python -m pytest tests/test_render_baseline.py
"""
import json
import os
import unittest

import numpy as np

from viewer_harness import (SKIP_REASON, environment_fingerprint, feco5_homo_cube,
                            make_cube, make_viewer, render, set_fixed_camera)

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_images")
META = os.path.join(REF_DIR, "references.json")
UPDATE = os.environ.get("CI3D_UPDATE_REFERENCES") == "1"
MAX_MEAN_DIFF = 1.0   # in 0-255 units, i.e. 1/255


def fixtures():
    """name -> cube factory. Water is synthetic; Fe(CO)5 is the HOMO from the fchk fixture."""
    return {
        "water": lambda: make_cube(width=4.0),
        "feco5": feco5_homo_cube,
    }


def render_fixture(name, opacity):
    cube = fixtures()[name]()
    if cube is None:
        return None
    v = make_viewer([cube])
    try:
        v.surface_opacity = opacity
        v.update_visualization()
        set_fixed_camera(v)
        return render(v)
    finally:
        v.plotter.close()


def load_meta():
    if not os.path.isfile(META):
        return {}
    with open(META) as f:
        return json.load(f)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class RenderBaselineTests(unittest.TestCase):
    CASES = [("water", 1.0), ("water", 0.5), ("feco5", 1.0), ("feco5", 0.5)]

    def test_renders_match_references(self):
        from PIL import Image

        meta = load_meta()
        fingerprint = environment_fingerprint()
        if UPDATE:
            os.makedirs(REF_DIR, exist_ok=True)
            meta = {"fingerprint": fingerprint, "images": {}}
        elif meta.get("fingerprint") != fingerprint:
            self.skipTest(f"references were made on '{meta.get('fingerprint')}', "
                          f"this run is '{fingerprint}'")

        for name, opacity in self.CASES:
            key = f"{name}_opacity{int(opacity * 100):03d}"
            with self.subTest(case=key):
                img = render_fixture(name, opacity)
                if img is None:
                    self.skipTest(f"fixture {name} unavailable")
                path = os.path.join(REF_DIR, key + ".png")
                if UPDATE:
                    Image.fromarray(img).save(path)
                    meta["images"][key] = list(img.shape)
                    continue
                ref = np.asarray(Image.open(path).convert(Image.open(path).mode))
                self.assertEqual(img.shape, ref.shape)
                diff = float(np.abs(img.astype(float) - ref.astype(float)).mean())
                self.assertLessEqual(diff, MAX_MEAN_DIFF, f"{key}: mean |diff| {diff:.3f}/255")

        if UPDATE:
            with open(META, "w") as f:
                json.dump(meta, f, indent=2)


if __name__ == "__main__":
    unittest.main()
