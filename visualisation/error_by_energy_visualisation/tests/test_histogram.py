import unittest

import numpy as np

from visualisation.error_by_energy_visualisation.histogram import build_histogram, choose_edges, histogram_heights
from visualisation.error_by_energy_visualisation.models import CategorySamples, HistogramResult, Metrics, StepObservation


class HistogramTests(unittest.TestCase):
    def test_group_density_uses_one_denominator_and_survives_toggle(self):
        metrics = Metrics(0, 0, 0, 0)
        first = CategorySamples(
            np.asarray([0, 0]), np.asarray([0, 1]), {"score": np.asarray([-1.0, 2.0])},
        )
        second = CategorySamples(
            np.asarray([0]), np.asarray([2]), {"score": np.asarray([0.5])},
        )
        observation = StepObservation(
            {"first": first, "second": second}, np.asarray([True]), metrics, metrics,
        )
        both = build_histogram(
            observation, "score", ["first", "second"], bins=4,
            normalization_keys=["first", "second"],
        )
        heights = histogram_heights(both, "density")
        widths = np.diff(both.edges)
        self.assertAlmostEqual(float(np.sum((heights["first"] + heights["second"]) * widths)), 1.0)
        self.assertAlmostEqual(float(np.sum(heights["first"] * widths)), 2.0 / 3.0)

        one = build_histogram(
            observation, "score", ["second"], bins=4,
            normalization_keys=["first", "second"],
        )
        one_height = histogram_heights(one, "density")["second"]
        self.assertEqual(one.normalization_count, 3)
        self.assertAlmostEqual(float(np.sum(one_height * np.diff(one.edges))), 1.0 / 3.0)
        np.testing.assert_array_equal(one.edges, both.edges)

    def test_density_components_integrate_to_one_together(self):
        edges, _ = choose_edges(np.arange(20), bins=5)
        counts = {"one": np.asarray([1, 2, 3, 4, 5]), "two": np.asarray([5, 4, 3, 2, 1])}
        result = HistogramResult(edges, counts, {}, "manual")
        heights = histogram_heights(result, "density")
        total = sum(heights.values())
        self.assertAlmostEqual(float(np.sum(total * np.diff(edges))), 1.0)

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
