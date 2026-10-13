"""Pure camera-state and projection tests (no Qt, no rendering)."""
import json
import unittest

import numpy as np
import vtk

import camera_utils as cu


def make_camera(position=(1.0, 2.0, 30.0), focal=(0.5, -1.0, 0.0),
                viewup=(0.0, 1.0, 0.0), scale=4.2, parallel=True):
    cam = vtk.vtkCamera()
    cam.SetFocalPoint(*focal)
    cam.SetPosition(*position)
    cam.SetViewUp(*viewup)
    cam.SetParallelProjection(parallel)
    cam.SetParallelScale(scale)
    return cam


class CameraStateRoundTripTests(unittest.TestCase):
    def assert_cameras_match(self, a, b):
        np.testing.assert_allclose(a.GetPosition(), b.GetPosition(), atol=1e-9)
        np.testing.assert_allclose(a.GetFocalPoint(), b.GetFocalPoint(), atol=1e-9)
        np.testing.assert_allclose(a.GetViewUp(), b.GetViewUp(), atol=1e-9)
        self.assertAlmostEqual(a.GetParallelScale(), b.GetParallelScale())
        self.assertEqual(a.GetParallelProjection(), b.GetParallelProjection())
        self.assertAlmostEqual(a.GetDistance(), b.GetDistance())

    def test_capture_apply_round_trip_keeps_zoom_and_projection(self):
        src = make_camera()
        dst = make_camera(position=(9, 9, 9), focal=(0, 0, 0), scale=1.0, parallel=False)
        cu.apply_camera_state(dst, cu.capture_camera_state(src))
        self.assert_cameras_match(src, dst)

    def test_state_has_no_independent_distance(self):
        self.assertNotIn("distance", cu.capture_camera_state(make_camera()))

    def test_session_round_trip_through_json(self):
        src = make_camera()
        block = json.loads(json.dumps(cu.camera_state_to_session(cu.capture_camera_state(src))))
        dst = make_camera(position=(5, 0, 0), focal=(0, 0, 0), viewup=(0, 0, 1), scale=9.0)
        cu.apply_camera_state(dst, cu.camera_state_from_session(block))
        self.assert_cameras_match(src, dst)

    def test_session_block_keeps_distance_consistent_for_v1_0_0(self):
        cam = make_camera()
        block = cu.camera_state_to_session(cu.capture_camera_state(cam))
        self.assertAlmostEqual(block["distance"], cam.GetDistance())


class SessionCompatibilityTests(unittest.TestCase):
    LEGACY = {"position": [0, -10, 0], "focal": [0, 0, 0], "viewup": [0, 0, 1], "distance": 10.0}

    def test_legacy_block_has_no_zoom_and_defaults_to_parallel(self):
        state = cu.camera_state_from_session(dict(self.LEGACY))
        self.assertIsNone(state["parallel_scale"])
        self.assertTrue(state["parallel_projection"])

    def test_legacy_state_leaves_current_zoom_alone(self):
        cam = make_camera(scale=3.3)
        cu.apply_camera_state(cam, cu.camera_state_from_session(dict(self.LEGACY)))
        self.assertAlmostEqual(cam.GetParallelScale(), 3.3)
        np.testing.assert_allclose(cam.GetPosition(), [0, -10, 0])

    def test_malformed_blocks_are_rejected(self):
        bad = [
            None, [], "camera",
            {**self.LEGACY, "position": [0, 0]},                 # wrong length
            {**self.LEGACY, "focal": ["a", 0, 0]},               # not numeric
            {**self.LEGACY, "viewup": [0, float("nan"), 1]},     # non-finite
            {**self.LEGACY, "position": [0, 0, 0]},              # position == focal
            {**self.LEGACY, "viewup": [0, 0, 0]},                # zero view-up
            {**self.LEGACY, "viewup": [0, 1, 0]},                # view-up along view direction
            {k: v for k, v in self.LEGACY.items() if k != "focal"},
        ]
        for block in bad:
            with self.subTest(block=block):
                self.assertIsNone(cu.camera_state_from_session(block))

    def test_bad_optional_fields_fall_back_to_defaults(self):
        for scale in (0, -1, float("inf"), "big", True):
            with self.subTest(scale=scale):
                state = cu.camera_state_from_session({**self.LEGACY, "parallel_scale": scale,
                                                      "parallel_projection": "yes"})
                self.assertIsNone(state["parallel_scale"])
                self.assertTrue(state["parallel_projection"])


class OverflowZoomTests(unittest.TestCase):
    @staticmethod
    def matrix(cam, aspect):
        m = cam.GetCompositeProjectionTransformMatrix(aspect, *cam.GetClippingRange())
        return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])

    def setUp(self):
        self.cam = make_camera(position=(0, 0, 50), focal=(0, 0, 0), viewup=(0, 1, 0), scale=5.0)
        self.cam.SetClippingRange(1, 100)

    def test_inside_frame_needs_no_zoom(self):
        self.assertEqual(cu.overflow_zoom([[1, 1, 0], [-2, 3, 5]], self.matrix(self.cam, 1.0)), 1.0)

    def test_overflow_scales_to_margin(self):
        # Point at y = 10 with half-height 5 -> |ndc| = 2 -> zoom 0.97 / 2.
        self.assertAlmostEqual(cu.overflow_zoom([[0, 10, 0]], self.matrix(self.cam, 1.0)), 0.485)

    def test_non_square_aspect_uses_the_narrow_axis(self):
        # Wide window: x spans 2x the height, so x = 8 fits; tall window it does not.
        self.assertEqual(cu.overflow_zoom([[8, 0, 0]], self.matrix(self.cam, 2.0)), 1.0)
        self.assertAlmostEqual(cu.overflow_zoom([[8, 0, 0]], self.matrix(self.cam, 0.5)),
                               0.97 / (8 / 2.5))

    def test_empty_and_degenerate_input(self):
        M = self.matrix(self.cam, 1.0)
        self.assertEqual(cu.overflow_zoom(np.empty((0, 3)), M), 1.0)
        self.assertEqual(cu.overflow_zoom([[np.nan, 0, 0]], M), 1.0)
        self.assertEqual(cu.overflow_zoom([[1, 1, 1]], np.zeros((4, 4))), 1.0)   # w = 0


if __name__ == "__main__":
    unittest.main()
