import unittest

import numpy as np

from visualisation.error_by_energy_visualisation.histogram import (
    build_histogram,
    choose_edges,
    histogram_heights,
)
from visualisation.error_by_energy_visualisation.models import Metrics, StepObservation


class HistogramTests(unittest.TestCase):
    def test_only_decision_frames_are_included(self):
        metrics = Metrics(0.0, 0.0, 0, 0)
        observation = StepObservation(
            energy=np.asarray([[1.0, -1.0], [100.0, 200.0]]),
            threshold=np.asarray([0.0, 0.0]),
            margin=np.asarray([[1.0, -1.0], [100.0, 200.0]]),
            should_flip=np.asarray([[False, True], [False, False]]),
            flip_mask=np.asarray([[False, True], [False, False]]),
            correct_action=np.asarray([[True, True], [True, True]]),
            decision_frames=np.asarray([True, False]),
            before=metrics,
            after=metrics,
        )
        result = build_histogram(observation, bins=4)
        np.testing.assert_array_equal(np.sort(result.correct_values), [-1.0, 1.0])
        self.assertEqual(result.correct_counts.sum(), 2)

    def test_density_integrates_to_one_per_nonempty_group(self):
        edges, _ = choose_edges(np.arange(20), bins=5)
        from visualisation.error_by_energy_visualisation.models import HistogramResult
        result = HistogramResult(
            edges=edges,
            correct_counts=np.asarray([1, 2, 3, 4, 5]),
            incorrect_counts=np.asarray([5, 4, 3, 2, 1]),
            correct_values=np.arange(15),
            incorrect_values=np.arange(15),
            method="manual",
        )
        correct, incorrect = histogram_heights(result, "density")
        self.assertAlmostEqual(float(np.sum(correct * np.diff(edges))), 1.0)
        self.assertAlmostEqual(float(np.sum(incorrect * np.diff(edges))), 1.0)

    def test_constant_values_have_valid_edges(self):
        edges, method = choose_edges(np.ones(30), bins="adaptive")
        self.assertEqual(len(edges), 11)
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertIn("constant", method)


if __name__ == "__main__":
    unittest.main()
