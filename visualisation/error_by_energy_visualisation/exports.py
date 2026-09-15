"""Text exports for the current interactive view."""

import csv
from io import StringIO
import json

import numpy as np

from .histogram import histogram_heights


def histogram_csv(histogram, mode):
    correct, incorrect = histogram_heights(histogram, mode)
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "bin_left",
        "bin_right",
        "bin_center",
        f"correct_{mode}",
        f"incorrect_{mode}",
        "correct_count",
        "incorrect_count",
    ])
    for index in range(len(histogram.edges) - 1):
        left = histogram.edges[index]
        right = histogram.edges[index + 1]
        writer.writerow([
            left,
            right,
            (left + right) / 2.0,
            correct[index],
            incorrect[index],
            histogram.correct_counts[index],
            histogram.incorrect_counts[index],
        ])
    return output.getvalue()


def summary_json(view, observation, histogram, display_settings):
    payload = {
        "schema": "bf-gd-energy-view/v1",
        "algorithm": view.algorithm.value,
        "current_iteration": view.cursor,
        "dataset": view.metadata,
        "current_metrics": view.snapshots[view.cursor].metrics.to_dict(),
        "preview_metrics": observation.after.to_dict(),
        "active_frames": int(np.count_nonzero(observation.decision_frames)),
        "histogram": {
            "method": histogram.method,
            "settings": dict(display_settings),
            "edges": histogram.edges.tolist(),
            "correct_counts": histogram.correct_counts.tolist(),
            "incorrect_counts": histogram.incorrect_counts.tolist(),
        },
        "history": [transition.to_dict() for transition in view.transitions],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
