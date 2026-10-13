import unittest
from unittest import mock

import numpy as np

import grid_utils
from grid_utils import estimate_remaining_seconds


class EstimateRemainingSecondsTests(unittest.TestCase):
    def test_no_estimate_before_first_cube(self):
        self.assertIsNone(estimate_remaining_seconds(0, 30, 0.0, None, 5.0))

    def test_first_cube_sets_initial_estimate(self):
        # 30 cubes, first took 10 s -> ~300 s total, 290 s left right after it.
        self.assertAlmostEqual(estimate_remaining_seconds(1, 30, 0.0, 10.0, 10.0), 290.0)

    def test_counts_down_between_completions(self):
        self.assertAlmostEqual(estimate_remaining_seconds(1, 30, 0.0, 10.0, 14.0), 286.0)

    def test_average_uses_actual_elapsed_time(self):
        # 12 cubes done after 180 s -> 15 s/cube, 18 left -> 270 s; the first
        # cube's 10 s estimate is not reused.
        self.assertAlmostEqual(estimate_remaining_seconds(12, 30, 0.0, 180.0, 180.0), 270.0)

    def test_never_negative_when_a_cube_runs_long(self):
        self.assertEqual(estimate_remaining_seconds(1, 3, 0.0, 10.0, 100.0), 0.0)

    def test_zero_when_all_done(self):
        self.assertEqual(estimate_remaining_seconds(30, 30, 0.0, 300.0, 300.0), 0.0)


class EvaluateOrbitalGridsProgressTests(unittest.TestCase):
    def _run(self, n_orbitals, **kwargs):
        fake_evaluator = (lambda cmo: np.zeros((2, 2, 2)), "fake")
        with mock.patch.object(grid_utils, "_orbital_evaluator", return_value=fake_evaluator):
            return grid_utils.evaluate_orbital_grids(
                basis=[], coordinates_ang=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                atom_info=[], cmos=[np.zeros(1)] * n_orbitals,
                orbital_indices=list(range(1, n_orbitals + 1)), label_base="mol",
                grid_quality=5, ext_dist=1.0, bohr_const=0.529177, **kwargs)

    def test_reports_start_then_each_finished_cube(self):
        calls = []
        results = self._run(3, progress_callback=lambda done, total: calls.append((done, total)))
        self.assertEqual(len(results), 3)
        self.assertEqual(calls, [(0, 3), (1, 3), (2, 3), (3, 3)])

    def test_callback_is_optional(self):
        self.assertEqual(len(self._run(2)), 2)

    def test_failure_stops_reporting_before_completion(self):
        calls = []

        def evaluate(cmo):
            if len(calls) == 2:   # (0, 3) and (1, 3) seen -> second cube fails
                raise RuntimeError("boom")
            return np.zeros((2, 2, 2))

        with mock.patch.object(grid_utils, "_orbital_evaluator", return_value=(evaluate, "fake")):
            with self.assertRaises(RuntimeError):
                grid_utils.evaluate_orbital_grids(
                    [], [(0.0, 0.0, 0.0)], [], [np.zeros(1)] * 3, [1, 2, 3], "mol",
                    5, 1.0, 0.529177,
                    progress_callback=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls, [(0, 3), (1, 3)])


if __name__ == "__main__":
    unittest.main()
