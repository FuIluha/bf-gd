import unittest
from types import SimpleNamespace

import numpy as np

from visualisation.error_by_energy_visualisation.figures import histogram_figure, performance_figure
from visualisation.error_by_energy_visualisation.app import _layout
from visualisation.error_by_energy_visualisation.models import (
    CategoryGroupSpec, CategorySpec, DecoderSpec, HistogramResult, Metrics, ObservableSpec,
)


class PerformanceFigureTests(unittest.TestCase):
    def test_distribution_components_are_overlaid_without_total_line(self):
        histogram = HistogramResult(
            np.asarray([0.0, 1.0, 2.0]),
            {"first": np.asarray([2, 0]), "second": np.asarray([1, 1])},
            {"first": np.asarray([0.2, 0.4]), "second": np.asarray([0.8, 1.2])},
            "manual", normalization_count=4,
        )
        spec = DecoderSpec(
            "test", "Test", (), (ObservableSpec("value", "Value"),),
            (CategorySpec("first", "First", "#111111"),
             CategorySpec("second", "Second", "#222222")),
            (CategoryGroupSpec("all", "All", ("first", "second")),),
        )

        figure = histogram_figure(
            histogram, "density", "linear", spec.observables[0], spec, 0,
        )

        self.assertEqual(figure.layout.barmode, "overlay")
        self.assertEqual(len(figure.data), 2)
        self.assertTrue(all(trace.type == "bar" for trace in figure.data))

    def test_ber_fer_are_always_logarithmic(self):
        view = SimpleNamespace(
            snapshots=[
                SimpleNamespace(metrics=Metrics(0.1, 0.5, 1, 1)),
                SimpleNamespace(metrics=Metrics(0.0, 0.0, 0, 0)),
            ],
            cursor=1,
            comparison=None,
        )

        figure = performance_figure(view)

        self.assertEqual(figure.layout.yaxis.type, "log")
        self.assertAlmostEqual(figure.data[0].y[0], 0.1)
        self.assertAlmostEqual(figure.data[1].y[0], 0.5)
        self.assertTrue(np.isnan(figure.data[0].y[1]))
        self.assertTrue(np.isnan(figure.data[1].y[1]))

    def test_distribution_is_pinned_above_scrollable_controls(self):
        page = _layout({"seed": 1, "workers": 1, "frames": 1})
        pinned = next(child for child in page.children
                      if getattr(child, "className", None) == "pinned-distribution")
        scrolling = next(child for child in page.children
                         if getattr(child, "className", None) == "scroll-region")

        def descendants(component):
            yield component
            children = getattr(component, "children", None)
            if children is None:
                children = []
            elif not isinstance(children, (list, tuple)):
                children = [children]
            for child in children:
                if not isinstance(child, str):
                    yield from descendants(child)

        pinned_ids = [getattr(item, "id", None) for item in descendants(pinned)]
        scrolling_ids = [getattr(item, "id", None) for item in descendants(scrolling)]
        self.assertIn("histogram", pinned_ids)
        self.assertIn("forward", pinned_ids)
        self.assertIn("backward", pinned_ids)
        self.assertIn("algorithm", scrolling_ids)
        self.assertNotIn("histogram", scrolling_ids)
        learning_rate = next(item for item in descendants(scrolling)
                             if getattr(item, "id", None) == {
                                 "type": "parameter-input", "algorithm": "gdms",
                                 "name": "learning_rate",
                             })
        self.assertEqual(learning_rate.step, "any")
        slider = next(item for item in descendants(scrolling)
                      if getattr(item, "id", None) == {
                          "type": "parameter-slider", "algorithm": "gdms",
                          "name": "learning_rate",
                      })
        self.assertAlmostEqual(slider.min, 0.01)


if __name__ == "__main__":
    unittest.main()
