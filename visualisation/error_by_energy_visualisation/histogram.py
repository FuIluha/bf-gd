"""Histogram preparation independent of the web framework."""

import numpy as np

from .models import HistogramResult


MIN_ADAPTIVE_BINS = 10
MAX_ADAPTIVE_BINS = 250


def build_histogram(observation, observable_key, category_keys, bins="adaptive"):
    """Place selected decoder-defined categories on shared bin edges."""
    values = {}
    for key in category_keys:
        category = observation.categories.get(key)
        sample = (
            np.asarray(category.values[observable_key], dtype=np.float64)
            if category is not None else np.empty(0, dtype=np.float64)
        )
        values[key] = sample[np.isfinite(sample)]
    combined = np.concatenate(list(values.values())) if values else np.empty(0)
    edges, method = choose_edges(combined, bins)
    return HistogramResult(
        edges=edges,
        counts={key: np.histogram(sample, bins=edges)[0] for key, sample in values.items()},
        values=values,
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
        return {key: counts.astype(np.float64) for key, counts in histogram.counts.items()}
    if mode != "density":
        raise ValueError("histogram mode must be 'count' or 'density'")
    widths = np.diff(histogram.edges)
    return {key: _density(counts, widths) for key, counts in histogram.counts.items()}


def _density(counts, widths):
    total = counts.sum()
    if total == 0:
        return np.zeros_like(widths, dtype=np.float64)
    return counts / (float(total) * widths)
