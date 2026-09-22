"""Read-only live dashboard for a running TGDBF tuning job."""

from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go


def _metric_figure(records, key, color, title):
    iterations = [record["iteration"] for record in records]
    figure = go.Figure()
    for cohort, name, dash in (("train", "Подбор", "dash"),
                               ("eval", "Проверка", "solid")):
        values = np.asarray([record[cohort][key] for record in records], dtype=float)
        figure.add_scatter(
            x=iterations, y=np.where(values > 0, values, np.nan),
            mode="lines+markers", name=name,
            line={"color": color, "dash": dash}, customdata=values,
            hovertemplate=f"{name} · итерация %{{x}}<br>Значение %{{customdata:.6g}}<extra></extra>",
        )
    figure.update_layout(
        title=title, height=390, xaxis_title="Номер итерации",
        yaxis_title=key.upper(), yaxis_type="log",
        margin={"l": 70, "r": 25, "t": 55, "b": 55},
    )
    return figure


def create_app(state):
    app = Dash(__name__, title="TGDBF — подбор delta")
    metadata = state.snapshot()["metadata"]
    details = (
        f"SNR={metadata['snr_db']:g} дБ · шаг порога={metadata['bin_width']:g} · "
        f"подбор={metadata['train_frames']} слов · "
        f"оценка={metadata['eval_frames']} слов · "
        f"процессов={metadata['workers']}"
    )
    app.layout = html.Main(style={"maxWidth": "1100px", "margin": "24px auto",
                                  "fontFamily": "sans-serif", "padding": "0 16px"}, children=[
        html.H1("TGDBF: подбор delta по итерациям"),
        html.P(details),
        html.P(id="status"),
        html.P(id="latest"),
        dcc.Graph(id="ber", style={"height": "390px"}),
        dcc.Graph(id="fer", style={"height": "390px"}),
        html.H2("Последние параметры"),
        html.Div(id="history"),
        dcc.Interval(id="refresh", interval=2_000, n_intervals=0),
    ])

    @app.callback(
        Output("ber", "figure"), Output("fer", "figure"),
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
        rows = [
            html.Tr([html.Th("Итерация"), html.Th("delta"),
                     html.Th("Порогов"), html.Th("FER подбор"),
                     html.Th("BER подбор"), html.Th("FER проверка"),
                     html.Th("BER проверка")]),
        ]
        for record in reversed(records[-12:]):
            bounds = record.get("delta")
            rows.append(html.Tr([
                html.Td(record["iteration"]),
                html.Td("—" if bounds is None else f"[0, {bounds[1]:.6g}]"),
                html.Td(record.get("candidate_count", "—")),
                html.Td(f"{record['train']['fer']:.6g}"),
                html.Td(f"{record['train']['ber']:.6g}"),
                html.Td(f"{record['eval']['fer']:.6g}"),
                html.Td(f"{record['eval']['ber']:.6g}"),
            ]))
        table = html.Table(rows, style={"width": "100%", "textAlign": "left"})
        return (
            _metric_figure(records, "ber", "#1565c0", "BER проверочной выборки"),
            _metric_figure(records, "fer", "#ef6c00", "FER проверочной выборки"),
            status, summary, table,
        )

    return app
