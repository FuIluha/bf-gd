import unittest

import numpy as np

from visualisation.error_by_energy_visualisation.histogram import build_histogram, choose_edges, histogram_heights
from visualisation.error_by_energy_visualisation.models import CategorySamples, HistogramResult, Metrics, StepObservation


class HistogramTests(unittest.TestCase):
    def test_decoder_categories_can_overlap_and_toggle(self):
        metrics = Metrics(0, 0, 0, 0)
        sample = CategorySamples(np.asarray([0, 0]), np.asarray([0, 1]), {"score": np.asarray([-1., 2.])})
        observation = StepObservation({"first": sample, "second": sample}, np.asarray([True, False]), metrics, metrics)
        both = build_histogram(observation, "score", ["first", "second"], bins=4)
        self.assertEqual(sum(counts.sum() for counts in both.counts.values()), 4)
        one = build_histogram(observation, "score", ["second"], bins=4)
        self.assertEqual(one.counts["second"].sum(), 2)

    def test_density_integrates_for_each_nonempty_category(self):
        edges, _ = choose_edges(np.arange(20), bins=5)
        counts = {"one": np.asarray([1, 2, 3, 4, 5]), "two": np.asarray([5, 4, 3, 2, 1])}
        result = HistogramResult(edges, counts, {}, "manual")
        for heights in histogram_heights(result, "density").values():
            self.assertAlmostEqual(float(np.sum(heights * np.diff(edges))), 1.0)

    def test_absent_category_is_empty_not_an_error(self):
        metrics = Metrics(0, 0, 0, 0)
        observation = StepObservation({}, np.asarray([True]), metrics, metrics)
        result = build_histogram(observation, "score", ["no_samples"])
        self.assertEqual(result.counts["no_samples"].sum(), 0)

    def test_constant_values_have_valid_edges(self):
        edges, method = choose_edges(np.ones(30), bins="adaptive")
        self.assertEqual(len(edges), 11)
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertIn("constant", method)


if __name__ == "__main__":
    unittest.main()
