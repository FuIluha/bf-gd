"""Plotly figure builders for the energy explorer."""

import numpy as np
import plotly.graph_objects as go

from .histogram import histogram_heights
from .models import Algorithm


COLORS = {
    "correct": "#34d399",
    "incorrect": "#fb7185",
    "ber": "#60a5fa",
    "fer": "#f59e0b",
    "comparison_ber": "#a78bfa",
    "comparison_fer": "#f472b6",
    "grid": "rgba(148, 163, 184, 0.16)",
    "text": "#dbeafe",
    "paper": "rgba(0,0,0,0)",
}


def histogram_figure(
    histogram,
    mode,
    y_scale,
    value_kind,
    algorithm,
    iteration,
):
    correct, incorrect = histogram_heights(histogram, mode)
    edges = histogram.edges
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    y_title = "Плотность вероятности" if mode == "density" else "Количество битов"
    x_title = "E" if value_kind == "energy" else "E − E_threshold"
    title_value = "энергии" if value_kind == "energy" else "запаса до порога"

    figure = go.Figure()
    figure.add_bar(
        x=centers,
        y=correct,
        width=widths,
        name=f"Верное действие ({histogram.correct_values.size:,})",
        marker_color=COLORS["correct"],
        opacity=0.68,
        hovertemplate=(
            f"{x_title}: %{{x:.5g}}<br>{y_title}: %{{y:.5g}}"
            "<extra>Верное действие</extra>"
        ),
    )
    figure.add_bar(
        x=centers,
        y=incorrect,
        width=widths,
        name=f"Ошибочное действие ({histogram.incorrect_values.size:,})",
        marker_color=COLORS["incorrect"],
        opacity=0.68,
        hovertemplate=(
            f"{x_title}: %{{x:.5g}}<br>{y_title}: %{{y:.5g}}"
            "<extra>Ошибочное действие</extra>"
        ),
    )
    if value_kind == "margin" or Algorithm(algorithm) is Algorithm.FTGDBF:
        figure.add_vline(
            x=0.0,
            line_dash="dash",
            line_color="#f8fafc",
            opacity=0.8,
            annotation_text="порог",
            annotation_position="top right",
        )
    figure.update_layout(
        title={
            "text": (
                f"Распределение {title_value}: решение {iteration} → {iteration + 1}"
                f"<br><sup>{histogram.method}</sup>"
            ),
            "x": 0.01,
        },
        barmode="overlay",
        bargap=0.0,
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis_type=y_scale,
        legend={"orientation": "h", "y": 1.11, "x": 0},
        hovermode="x unified",
        margin={"l": 64, "r": 24, "t": 96, "b": 60},
    )
    return _theme(figure)


def performance_figure(view, y_scale="log"):
    iterations = np.arange(len(view.snapshots))
    ber = np.asarray([item.metrics.ber for item in view.snapshots])
    fer = np.asarray([item.metrics.fer for item in view.snapshots])
    figure = go.Figure()
    _add_metric_trace(figure, iterations, ber, "BER", COLORS["ber"], y_scale)
    _add_metric_trace(figure, iterations, fer, "FER", COLORS["fer"], y_scale)

    if view.comparison is not None:
        label = f"{view.comparison.algorithm} · {view.comparison.name}"
        _add_metric_trace(
            figure,
            view.comparison.iterations,
            view.comparison.ber,
            f"BER · {label}",
            COLORS["comparison_ber"],
            y_scale,
            dash="dot",
        )
        _add_metric_trace(
            figure,
            view.comparison.iterations,
            view.comparison.fer,
            f"FER · {label}",
            COLORS["comparison_fer"],
            y_scale,
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
        title={"text": "BER и FER по истории", "x": 0.01},
        xaxis_title="Номер итерации",
        yaxis_title="Доля ошибок",
        yaxis_type=y_scale,
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
        margin={"l": 64, "r": 24, "t": 84, "b": 60},
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
    return _theme(figure)


def _add_metric_trace(figure, x, y, name, color, y_scale, dash="solid"):
    shown = np.asarray(y, dtype=np.float64)
    if y_scale == "log":
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
        template="plotly_dark",
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor="rgba(15, 23, 42, 0.52)",
        font={"family": "Inter, system-ui, sans-serif", "color": COLORS["text"]},
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    figure.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return figure
