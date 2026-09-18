"""Text exports for the current interactive view."""

import csv
from io import StringIO
import json

import numpy as np

from .histogram import histogram_heights


def histogram_csv(histogram, mode):
    heights = histogram_heights(histogram, mode)
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "bin_left",
        "bin_right",
        "bin_center",
        *[f"{key}_{mode}" for key in histogram.counts],
        *[f"{key}_count" for key in histogram.counts],
    ])
    for index in range(len(histogram.edges) - 1):
        left = histogram.edges[index]
        right = histogram.edges[index + 1]
        writer.writerow([
            left,
            right,
            (left + right) / 2.0,
            *[heights[key][index] for key in histogram.counts],
            *[histogram.counts[key][index] for key in histogram.counts],
        ])
    return output.getvalue()


def summary_json(view, observation, histogram, display_settings, decoder_spec):
    payload = {
        "schema": "bf-gd-energy-view/v2",
        "algorithm": view.algorithm,
        "decoder_title": decoder_spec.title,
        "observables": {item.key: item.label for item in decoder_spec.observables},
        "categories": {item.key: item.label for item in decoder_spec.categories},
        "current_iteration": view.cursor,
        "dataset": view.metadata,
        "current_metrics": view.snapshots[view.cursor].metrics.to_dict(),
        "preview_metrics": observation.after.to_dict(),
        "active_frames": int(np.count_nonzero(observation.decision_frames)),
        "histogram": {
            "method": histogram.method,
            "settings": dict(display_settings),
            "edges": histogram.edges.tolist(),
            "category_counts": {key: value.tolist() for key, value in histogram.counts.items()},
        },
        "history": [transition.to_dict() for transition in view.transitions],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
