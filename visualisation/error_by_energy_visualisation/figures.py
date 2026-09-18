"""Plotly figure builders for the energy explorer."""

import numpy as np
import plotly.graph_objects as go

from .histogram import histogram_heights


COLORS = {
    "correct": "#2e7d32",
    "incorrect": "#c62828",
    "ber": "#1565c0",
    "fer": "#ef6c00",
    "comparison_ber": "#6a1b9a",
    "comparison_fer": "#ad1457",
    "grid": "#dddddd",
    "text": "#222222",
    "paper": "#ffffff",
}

FIGURE_HEIGHT = 560


def histogram_figure(
    histogram,
    mode,
    y_scale,
    observable,
    decoder_spec,
    iteration,
):
    heights = histogram_heights(histogram, mode)
    edges = histogram.edges
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    y_title = "Плотность вероятности" if mode == "density" else "Количество наблюдений"
    x_title = observable.label

    figure = go.Figure()
    for category in decoder_spec.categories:
        if category.key not in heights:
            continue
        figure.add_bar(
            x=centers, y=heights[category.key], width=widths,
            name=f"{category.label} ({histogram.values[category.key].size:,})",
            marker_color=category.color, opacity=0.68,
            hovertemplate=f"{x_title}: %{{x:.5g}}<br>{y_title}: %{{y:.5g}}<extra>{category.label}</extra>",
        )
    if observable.zero_line:
        figure.add_vline(
            x=0.0,
            line_dash="dash",
            line_color="#64748b",
            opacity=0.8,
            annotation_text="0",
            annotation_position="top right",
        )
    figure.update_layout(
        height=FIGURE_HEIGHT,
        autosize=True,
        title={
            "text": (
                f"{decoder_spec.title}: {x_title} · решение {iteration} → {iteration + 1}"
                f"<br><sup>{histogram.method}</sup>"
            ),
            "x": 0.01,
        },
        barmode="overlay",
        bargap=0.0,
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis_type=y_scale,
        legend={"orientation": "v", "y": 1, "x": 1, "xanchor": "right"},
        hovermode="closest",
        margin={"l": 64, "r": 24, "t": 82, "b": 60},
    )
    return _theme(figure)


def performance_figure(view):
    iterations = np.arange(len(view.snapshots))
    ber = np.asarray([item.metrics.ber for item in view.snapshots])
    fer = np.asarray([item.metrics.fer for item in view.snapshots])
    figure = go.Figure()
    _add_metric_trace(figure, iterations, ber, "BER", COLORS["ber"])
    _add_metric_trace(figure, iterations, fer, "FER", COLORS["fer"])

    if view.comparison is not None:
        label = f"{view.comparison.algorithm} · {view.comparison.name}"
        _add_metric_trace(
            figure,
            view.comparison.iterations,
            view.comparison.ber,
            f"BER · {label}",
            COLORS["comparison_ber"],
            dash="dot",
        )
        _add_metric_trace(
            figure,
            view.comparison.iterations,
            view.comparison.fer,
            f"FER · {label}",
            COLORS["comparison_fer"],
            dash="dot",
        )
    figure.add_vline(
        x=view.cursor,
        line_color="#e2e8f0",
        line_dash="dash",
        opacity=0.7,
        annotation_text="текущий снимок",
    )
    figure.update_layout(
        height=FIGURE_HEIGHT,
        autosize=True,
        title={"text": "BER и FER по истории", "x": 0.01},
        xaxis_title="Номер итерации",
        yaxis_title="Доля ошибок",
        yaxis_type="log",
        legend={"orientation": "v", "y": 1, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        margin={"l": 64, "r": 24, "t": 60, "b": 60},
    )
    return _theme(figure)


def empty_figure(message):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": COLORS["text"]},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(height=FIGURE_HEIGHT, autosize=True)
    return _theme(figure)


def _add_metric_trace(figure, x, y, name, color, dash="solid"):
    shown = np.asarray(y, dtype=np.float64)
    shown = np.where(shown > 0, shown, np.nan)
    figure.add_scatter(
        x=x,
        y=shown,
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 2, "dash": dash},
        marker={"size": 6},
        customdata=np.asarray(y),
        hovertemplate=(
            "Итерация: %{x}<br>Значение: %{customdata:.6g}<extra>"
            + name
            + "</extra>"
        ),
    )


def _theme(figure):
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor="#ffffff",
        font={"family": "Inter, system-ui, sans-serif", "color": COLORS["text"]},
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    figure.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return figure
