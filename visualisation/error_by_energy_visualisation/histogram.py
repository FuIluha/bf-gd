"""Histogram preparation independent of the web framework."""

import numpy as np

from .models import HistogramResult


MIN_ADAPTIVE_BINS = 10
MAX_ADAPTIVE_BINS = 250


def build_histogram(observation, value_kind="energy", bins="adaptive"):
    """Split values by correctness and place both groups on common bin edges."""
    if value_kind == "energy":
        values = observation.energy
    elif value_kind == "margin":
        values = observation.margin
    else:
        raise ValueError("value_kind must be 'energy' or 'margin'")

    decision_mask = np.broadcast_to(
        observation.decision_frames[:, None],
        values.shape,
    )
    finite = np.isfinite(values) & decision_mask
    correct_values = values[finite & observation.correct_action]
    incorrect_values = values[finite & ~observation.correct_action]
    combined = np.concatenate((correct_values, incorrect_values))
    edges, method = choose_edges(combined, bins)
    return HistogramResult(
        edges=edges,
        correct_counts=np.histogram(correct_values, bins=edges)[0],
        incorrect_counts=np.histogram(incorrect_values, bins=edges)[0],
        correct_values=correct_values,
        incorrect_values=incorrect_values,
        method=method,
    )


def choose_edges(values, bins="adaptive"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(-1.0, 1.0, 11), "empty fallback (10 bins)"

    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        padding = max(abs(low) * 0.05, 0.5)
        return np.linspace(low - padding, high + padding, 11), "constant fallback (10 bins)"

    if bins != "adaptive":
        count = int(bins)
        if count < 1 or count > 2_000:
            raise ValueError("manual bin count must be between 1 and 2000")
        return np.linspace(low, high, count + 1), f"manual ({count} bins)"

    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = float(q75 - q25)
    width = 2.0 * iqr / np.cbrt(values.size)
    if width > 0 and np.isfinite(width):
        count = int(np.ceil((high - low) / width))
        method = "Freedman–Diaconis"
    else:
        count = int(np.ceil(np.log2(values.size) + 1.0))
        method = "Sturges fallback"
    count = min(MAX_ADAPTIVE_BINS, max(MIN_ADAPTIVE_BINS, count))
    return np.linspace(low, high, count + 1), f"{method} ({count} bins)"


def histogram_heights(histogram, mode):
    if mode == "count":
        return (
            histogram.correct_counts.astype(np.float64),
            histogram.incorrect_counts.astype(np.float64),
        )
    if mode != "density":
        raise ValueError("histogram mode must be 'count' or 'density'")
    widths = np.diff(histogram.edges)
    return (
        _density(histogram.correct_counts, widths),
        _density(histogram.incorrect_counts, widths),
    )


def _density(counts, widths):
    total = counts.sum()
    if total == 0:
        return np.zeros_like(widths, dtype=np.float64)
    return counts / (float(total) * widths)
