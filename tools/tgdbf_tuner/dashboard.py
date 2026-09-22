"""Read-only live dashboard for TGDBF threshold tuning and observation."""

from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go


def _metric_figure(records, key, color, title):
    iterations = [record["iteration"] for record in records]
    values = np.asarray([record["eval"][key] for record in records], dtype=float)
    figure = go.Figure()
    figure.add_scatter(
        x=iterations, y=np.where(values > 0, values, np.nan),
        mode="lines+markers", name=key.upper(), line_color=color,
        customdata=values,
        hovertemplate="Итерация %{x}<br>Значение %{customdata:.6g}<extra></extra>",
    )
    figure.update_layout(
        title=title, height=390, xaxis_title="Номер итерации",
        yaxis_title=key.upper(), yaxis_type="log",
        margin={"l": 70, "r": 25, "t": 55, "b": 55},
    )
    return figure


def _ratio_figure(records):
    figure = go.Figure()
    for cohort, color, label in (("train", "#2e7d32", "Подбор"),
                                 ("eval", "#7b1fa2", "Проверка")):
        points = [(record["iteration"], record.get(f"flips_{cohort}"))
                  for record in records]
        finite = [(iteration, flips["ratio"] if flips is not None
                   and flips["ratio_kind"] == "finite" else None)
                  for iteration, flips in points]
        figure.add_scatter(
            x=[point[0] for point in finite], y=[point[1] for point in finite],
            mode="lines+markers", name=label, line_color=color,
            hovertemplate="Итерация %{x}<br>N=%{y:.6g}<extra></extra>",
        )
        infinite = [iteration for iteration, flips in points
                    if flips is not None and flips["ratio_kind"] == "infinite"]
        if infinite:
            top = max((value for _, value in finite if value is not None), default=1.0)
            figure.add_scatter(
                x=infinite, y=[top * 1.1] * len(infinite), mode="markers",
                name=f"{label}: N=∞", marker={"color": color, "symbol": "diamond"},
                hovertemplate="Итерация %{x}<br>N=∞<extra></extra>",
            )
    figure.update_layout(
        title="N = правильные / неправильные фактические flip",
        height=390, xaxis_title="Номер итерации", yaxis_title="N",
        margin={"l": 70, "r": 25, "t": 55, "b": 55},
    )
    return figure


def create_app(state):
    app = Dash(__name__, title="TGDBF — анализ порога")
    metadata = state.snapshot()["metadata"]
    fixed = metadata["mode"] == "fixed"
    mode_details = (f"фиксированная delta=[0, {metadata['fixed_delta']:g}]"
                    if fixed else
                    f"подбор: N={metadata['ratio']:g}, бин={metadata['bin_width']:g}")
    details = (f"SNR={metadata['snr_db']:g} дБ · {mode_details} · "
               f"обучение={metadata['train_frames']} слов · "
               f"проверка={metadata['eval_frames']} слов · "
               f"процессов={metadata['workers']}")
    app.layout = html.Main(style={"maxWidth": "1100px", "margin": "24px auto",
                                  "fontFamily": "sans-serif", "padding": "0 16px"}, children=[
        html.H1("TGDBF: порог и статистика по итерациям"),
        html.P(details),
        html.P(id="status"),
        html.P(id="latest"),
        dcc.Graph(id="ber", style={"height": "390px"}),
        dcc.Graph(id="fer", style={"height": "390px"}),
        dcc.Graph(id="ratio", style={"height": "390px"} if fixed else {"display": "none"}),
        html.H2("Последние параметры"),
        html.Div(id="history"),
        dcc.Interval(id="refresh", interval=2_000, n_intervals=0),
    ])

    @app.callback(
        Output("ber", "figure"), Output("fer", "figure"), Output("ratio", "figure"),
        Output("status", "children"), Output("latest", "children"),
        Output("history", "children"), Input("refresh", "n_intervals"),
    )
    def refresh(_):
        snapshot = state.snapshot()
        records = snapshot["records"]
        status = f"{snapshot['status']}: {snapshot['message']}"
        if snapshot.get("output"):
            status += f" · лог: {snapshot['output']}"
        latest = records[-1] if records else None
        summary = (
            f"Итерация {latest['iteration']}: "
            f"BER={latest['eval']['ber']:.6g}, FER={latest['eval']['fer']:.6g}"
            if latest else "Ожидание начальной статистики"
        )
        if fixed and latest and latest.get("flips_train"):
            flips = latest["flips_train"]
            ratio = ("∞" if flips["ratio_kind"] == "infinite" else
                     "—" if flips["ratio_kind"] == "undefined" else
                     f"{flips['ratio']:.6g}")
            summary += f", N={ratio}"
        rows = [
            html.Tr([html.Th("Итерация"), html.Th("delta"),
                     html.Th("Правильные flip"), html.Th("Неправильные flip"),
                     html.Th("N"), html.Th("BER"), html.Th("FER")]),
        ]
        for record in reversed(records[-12:]):
            bounds = record.get("delta")
            flips = record.get("flips_train")
            ratio = ("—" if not flips or flips["ratio_kind"] == "undefined" else
                     "∞" if flips["ratio_kind"] == "infinite" else
                     f"{flips['ratio']:.6g}")
            rows.append(html.Tr([
                html.Td(record["iteration"]),
                html.Td("—" if bounds is None else f"[0, {bounds[1]:.6g}]"),
                html.Td(record.get("correct_flips_train", "—")),
                html.Td(record.get("incorrect_flips_train", "—")),
                html.Td(ratio),
                html.Td(f"{record['eval']['ber']:.6g}"),
                html.Td(f"{record['eval']['fer']:.6g}"),
            ]))
        table = html.Table(rows, style={"width": "100%", "textAlign": "left"})
        return (
            _metric_figure(records, "ber", "#1565c0", "BER проверочной выборки"),
            _metric_figure(records, "fer", "#ef6c00", "FER проверочной выборки"),
            _ratio_figure(records) if fixed else go.Figure(),
            status, summary, table,
        )

    return app
