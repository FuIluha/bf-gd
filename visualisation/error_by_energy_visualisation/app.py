"""Dash web application for interactive FTGDBF/PMGDBF inspection."""

import base64
from datetime import datetime
from pathlib import Path
import uuid

from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from .exports import histogram_csv, summary_json
from .figures import empty_figure, histogram_figure, performance_figure
from .histogram import build_histogram
from .models import (
    Algorithm,
    FtgdbfParameters,
    PmgdbfParameters,
    parse_rho,
)
from .registry import SessionRegistry
from .session import ExplorerSession, comparison_from_bytes, compatible_datasets


DEFAULTS = {
    "alpha": 1.8,
    "delta": 1.0,
    "p": 0.9,
    "L": 7,
    "rho": "2, 2, 2, 2, 2, 1, 1",
}

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "bf-gd-energy-distribution",
        "scale": 2,
    },
}

GRAPH_STYLE = {
    "height": "560px",
    "minHeight": "560px",
    "width": "100%",
}


def create_app(dataset, seed=None, workers=None, max_sessions=16):
    """Build an isolated, server-side session app over one fixed frame batch."""
    effective_seed = dataset.metadata["seed"] if seed is None else seed
    effective_workers = dataset.metadata["workers"] if workers is None else workers
    registry = SessionRegistry(
        dataset,
        seed=effective_seed,
        workers=effective_workers,
        max_sessions=max_sessions,
    )
    assets = Path(__file__).resolve().parent / "assets"
    app = Dash(
        __name__,
        title="LDPC Energy Explorer",
        assets_folder=str(assets),
        suppress_callback_exceptions=False,
    )
    app.layout = lambda: _layout(dataset.metadata)
    _register_callbacks(app, registry)
    return app


def _layout(metadata):
    session_id = str(uuid.uuid4())
    return html.Div(className="app-shell", children=[
        dcc.Store(id="browser-session-id", data=session_id, storage_type="session"),
        dcc.Store(id="revision", data=0),
        dcc.Download(id="download-session"),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-json"),
        html.Header(className="hero", children=[
            html.Div([
                html.Div("BF–GD LAB", className="eyebrow"),
                html.H1("Исследователь решений по энергии"),
                html.P(
                    "Пошаговый FTGDBF и PMGDBF на фиксированных реализациях канала. "
                    "Каждый снимок состояния неизменяем.",
                    className="hero-copy",
                ),
            ]),
            html.Div(_metadata_text(metadata), id="dataset-metadata", className="dataset-pill"),
        ]),
        html.Main(className="workspace", children=[
            html.Aside(className="control-column", children=[
                _section("Алгоритм и история", [
                    _label("Декодер"),
                    dcc.Dropdown(
                        id="algorithm",
                        options=[
                            {"label": "FTGDBF", "value": Algorithm.FTGDBF.value},
                            {"label": "PMGDBF", "value": Algorithm.PMGDBF.value},
                        ],
                        value=Algorithm.FTGDBF.value,
                        clearable=False,
                        searchable=False,
                    ),
                    html.Div(className="step-row", children=[
                        html.Button("← Назад", id="backward", n_clicks=0, className="button secondary"),
                        html.Button("Вперёд →", id="forward", n_clicks=0, className="button primary"),
                    ]),
                    html.Button("Сбросить к итерации 0", id="reset", n_clicks=0, className="button ghost full"),
                    html.Div("Снимок 0 · решение 0 → 1", id="iteration-label", className="iteration-badge"),
                ]),
                _section("Параметры декодера", [
                    _numeric_control("alpha", "alpha", 0.0, 4.0, 0.01, DEFAULTS["alpha"]),
                    html.Div(id="pm-controls", children=[
                        _numeric_control("delta", "delta", 0.0, 4.0, 0.01, DEFAULTS["delta"]),
                        _numeric_control("p", "p", 0.0, 1.0, 0.01, DEFAULTS["p"]),
                        _numeric_control("momentum-length", "L", 1, 32, 1, DEFAULTS["L"]),
                        _label("rho (через запятую)"),
                        dcc.Input(
                            id="rho",
                            type="text",
                            value=DEFAULTS["rho"],
                            debounce=True,
                            className="text-input",
                        ),
                    ]),
                ]),
                _section("Отображение", [
                    _label("Величина по оси X"),
                    dcc.RadioItems(
                        id="value-kind",
                        options=[
                            {"label": "Энергия E", "value": "energy"},
                            {"label": "Запас E − E_threshold", "value": "margin"},
                        ],
                        value="energy",
                        className="radio-stack",
                    ),
                    _label("Ось Y"),
                    dcc.RadioItems(
                        id="histogram-mode",
                        options=[
                            {"label": "Плотность", "value": "density"},
                            {"label": "Количество", "value": "count"},
                        ],
                        value="density",
                        inline=True,
                        className="radio-row",
                    ),
                    dcc.RadioItems(
                        id="y-scale",
                        options=[
                            {"label": "Линейная", "value": "linear"},
                            {"label": "Логарифмическая", "value": "log"},
                        ],
                        value="linear",
                        inline=True,
                        className="radio-row",
                    ),
                    _label("Ширина бинов"),
                    dcc.RadioItems(
                        id="bin-mode",
                        options=[
                            {"label": "Адаптивно", "value": "adaptive"},
                            {"label": "Вручную", "value": "manual"},
                        ],
                        value="adaptive",
                        inline=True,
                        className="radio-row",
                    ),
                    dcc.Input(
                        id="manual-bins",
                        type="number",
                        min=1,
                        max=2000,
                        step=1,
                        value=80,
                        disabled=True,
                        className="number-input full",
                    ),
                ]),
                _section("Сессия и экспорт", [
                    html.Div(className="button-grid", children=[
                        html.Button("Сохранить сессию", id="save-session", n_clicks=0, className="button secondary"),
                        dcc.Upload(
                            id="session-upload",
                            accept=".npz",
                            children=html.Button("Загрузить сессию", className="button secondary"),
                        ),
                        html.Button("Экспорт CSV", id="export-csv", n_clicks=0, className="button secondary"),
                        html.Button("Экспорт JSON", id="export-json", n_clicks=0, className="button secondary"),
                    ]),
                    html.P("PNG сохраняется кнопкой камеры на графике.", className="hint"),
                    _label("Сравнить с другой сессией"),
                    dcc.Upload(
                        id="comparison-upload",
                        accept=".npz",
                        children=html.Button("Добавить сравнение", className="button ghost full"),
                    ),
                    html.Button("Убрать сравнение", id="clear-comparison", n_clicks=0, className="button link full"),
                ]),
                html.Div("Готово", id="action-status", className="status-line"),
            ]),
            html.Section(className="content-column", children=[
                html.Div(className="metric-grid", children=[
                    _metric("Текущий BER", "current-ber"),
                    _metric("Текущий FER", "current-fer"),
                    _metric("BER после шага", "preview-ber"),
                    _metric("FER после шага", "preview-fer"),
                    _metric("Активные кадры", "active-frames"),
                ]),
                html.Div(id="plot-status", className="plot-status"),
                html.Div(className="chart-card", children=[
                    dcc.Loading(
                        type="circle",
                        children=dcc.Graph(
                            id="histogram",
                            config=GRAPH_CONFIG,
                            responsive=True,
                            style=GRAPH_STYLE,
                        ),
                    ),
                    html.Div(id="decision-summary", className="decision-summary"),
                ]),
                html.Div(className="chart-card", children=[
                    dcc.Graph(
                        id="performance",
                        config=GRAPH_CONFIG,
                        responsive=True,
                        style=GRAPH_STYLE,
                    ),
                ]),
                html.Div(className="table-card", children=[
                    html.Div(className="card-heading", children=[
                        html.H2("История параметров"),
                        html.P("Шаг из прошлого удаляет прежнее будущее и создаёт новую линейную историю."),
                    ]),
                    dash_table.DataTable(
                        id="parameter-history",
                        columns=[
                            {"name": "Шаг", "id": "step"},
                            {"name": "Параметры", "id": "parameters"},
                            {"name": "BER", "id": "ber", "type": "numeric", "format": {"specifier": ".6e"}},
                            {"name": "FER", "id": "fer", "type": "numeric", "format": {"specifier": ".6e"}},
                            {"name": "Положение", "id": "position"},
                        ],
                        data=[],
                        page_size=12,
                        sort_action="native",
                        style_as_list_view=True,
                        style_cell={
                            "backgroundColor": "#ffffff",
                            "color": "#222222",
                            "border": "none",
                            "borderBottom": "1px solid #dddddd",
                            "padding": "10px 12px",
                            "textAlign": "left",
                            "fontFamily": "Arial, Helvetica, sans-serif",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "minWidth": "100px",
                            "maxWidth": "420px",
                        },
                        style_header={
                            "backgroundColor": "#f2f2f2",
                            "fontWeight": 700,
                            "borderBottom": "1px solid #aaaaaa",
                        },
                        style_data_conditional=[{
                            "if": {"filter_query": "{position} = 'будущее'"},
                            "opacity": 0.45,
                        }],
                    ),
                ]),
            ]),
        ]),
    ])


def _register_callbacks(app, registry):
    _register_sync_callbacks(app)

    @app.callback(Output("manual-bins", "disabled"), Input("bin-mode", "value"))
    def toggle_manual_bins(mode):
        return mode != "manual"

    @app.callback(
        Output("revision", "data"),
        Output("action-status", "children"),
        Output("algorithm", "value"),
        Input("forward", "n_clicks"),
        Input("backward", "n_clicks"),
        Input("reset", "n_clicks"),
        Input("algorithm", "value"),
        Input("session-upload", "contents"),
        Input("comparison-upload", "contents"),
        Input("clear-comparison", "n_clicks"),
        State("browser-session-id", "data"),
        State("revision", "data"),
        State("alpha-input", "value"),
        State("delta-input", "value"),
        State("p-input", "value"),
        State("momentum-length-input", "value"),
        State("rho", "value"),
        State("session-upload", "filename"),
        State("comparison-upload", "filename"),
        prevent_initial_call=True,
    )
    def mutate(
        _forward,
        _backward,
        _reset,
        algorithm_value,
        session_contents,
        comparison_contents,
        _clear_comparison,
        session_id,
        revision,
        alpha,
        delta,
        probability,
        momentum_length,
        rho,
        session_filename,
        comparison_filename,
    ):
        revision = int(revision or 0)
        try:
            session = registry.get(session_id)
            trigger = ctx.triggered_id
            if trigger == "forward":
                parameters = _parameters(
                    session.algorithm,
                    alpha,
                    delta,
                    probability,
                    momentum_length,
                    rho,
                )
                session.step_forward(parameters)
                return revision + 1, f"Сохранён снимок {session.cursor}", no_update
            if trigger == "backward":
                before = session.cursor
                session.step_back()
                message = "Уже выбрана итерация 0" if before == 0 else f"Возврат к снимку {session.cursor}"
                return revision + 1, message, no_update
            if trigger == "reset":
                session.reset(session.algorithm, int(momentum_length))
                return revision + 1, "История сброшена к итерации 0", no_update
            if trigger == "algorithm":
                requested = Algorithm(algorithm_value)
                if requested is session.algorithm:
                    return revision, no_update, no_update
                session.reset(requested, int(momentum_length))
                return revision + 1, f"Выбран {requested.title}; история начата заново", no_update
            if trigger == "session-upload":
                payload = _upload_bytes(session_contents)
                loaded = ExplorerSession.from_bytes(payload, workers=registry.workers)
                registry.replace(session_id, loaded)
                return (
                    revision + 1,
                    f"Загружена сессия {session_filename or ''}",
                    loaded.algorithm.value,
                )
            if trigger == "comparison-upload":
                payload = _upload_bytes(comparison_contents)
                label = Path(comparison_filename or "сравнение").stem
                comparison = comparison_from_bytes(payload, name=label)
                with session.lock:
                    if not compatible_datasets(session.metadata, comparison.metadata):
                        raise ValueError(
                            "сравниваемая история рассчитана на другом наборе кадров"
                        )
                    session.comparison = comparison
                return revision + 1, f"Добавлено сравнение: {label}", no_update
            if trigger == "clear-comparison":
                with session.lock:
                    session.comparison = None
                return revision + 1, "Сравнение убрано", no_update
            return revision, no_update, no_update
        except Exception as exc:  # Dash must report validation errors in the UI.
            return revision, f"Ошибка: {exc}", no_update

    @app.callback(
        Output("iteration-label", "children"),
        Output("current-ber", "children"),
        Output("current-fer", "children"),
        Output("preview-ber", "children"),
        Output("preview-fer", "children"),
        Output("active-frames", "children"),
        Output("decision-summary", "children"),
        Output("histogram", "figure"),
        Output("performance", "figure"),
        Output("parameter-history", "data"),
        Output("pm-controls", "style"),
        Output("plot-status", "children"),
        Output("backward", "disabled"),
        Output("forward", "disabled"),
        Output("dataset-metadata", "children"),
        Input("revision", "data"),
        Input("algorithm", "value"),
        Input("alpha-input", "value"),
        Input("delta-input", "value"),
        Input("p-input", "value"),
        Input("momentum-length-input", "value"),
        Input("rho", "value"),
        Input("value-kind", "value"),
        Input("histogram-mode", "value"),
        Input("y-scale", "value"),
        Input("bin-mode", "value"),
        Input("manual-bins", "value"),
        State("browser-session-id", "data"),
    )
    def render(
        _revision,
        _algorithm_value,
        alpha,
        delta,
        probability,
        momentum_length,
        rho,
        value_kind,
        histogram_mode,
        y_scale,
        bin_mode,
        manual_bins,
        session_id,
    ):
        session = registry.get(session_id)
        try:
            with session.lock:
                view = session.view()
                parameters = _parameters(
                    view.algorithm,
                    alpha,
                    delta,
                    probability,
                    momentum_length,
                    rho,
                )
                _, observation = session.preview(parameters)
            pm_style = {} if view.algorithm is Algorithm.PMGDBF else {"display": "none"}
            history = _history_rows(view)
            current = view.snapshots[view.cursor].metrics
            bins = "adaptive" if bin_mode == "adaptive" else int(manual_bins)
            histogram = build_histogram(observation, value_kind=value_kind, bins=bins)
            histogram_plot = histogram_figure(
                histogram,
                mode=histogram_mode,
                y_scale=y_scale,
                value_kind=value_kind,
                algorithm=view.algorithm,
                iteration=view.cursor,
            )
            active = int(observation.decision_frames.sum())
            summary = _decision_summary(observation)
            status = (
                f"Предпросмотр с текущими параметрами · {histogram.method} · "
                f"{histogram.correct_values.size + histogram.incorrect_values.size:,} решений"
            )
            if view.algorithm is Algorithm.PMGDBF and value_kind == "energy":
                status += " · порог PMGDBF различается между кадрами"
            forward_disabled = active == 0
            preview = observation.after
        except Exception as exc:
            view = session.view()
            pm_style = {} if view.algorithm is Algorithm.PMGDBF else {"display": "none"}
            history = _history_rows(view)
            current = view.snapshots[view.cursor].metrics
            histogram_plot = empty_figure(f"Исправьте параметры: {exc}")
            active = int(view.snapshots[view.cursor].state.active.sum())
            summary = "Предпросмотр недоступен"
            status = f"Ошибка параметров: {exc}"
            forward_disabled = True
            preview = current
        return (
            f"Снимок {view.cursor} · решение {view.cursor} → {view.cursor + 1}",
            _rate(current.ber, current.bit_errors),
            _rate(current.fer, current.frame_errors),
            _rate(preview.ber, preview.bit_errors),
            _rate(preview.fer, preview.frame_errors),
            f"{active:,} / {session.batch.frames:,}",
            summary,
            histogram_plot,
            performance_figure(view, y_scale=y_scale),
            history,
            pm_style,
            status,
            view.cursor == 0,
            forward_disabled,
            _metadata_text(view.metadata),
        )

    @app.callback(
        Output("download-session", "data"),
        Input("save-session", "n_clicks"),
        State("browser-session-id", "data"),
        prevent_initial_call=True,
    )
    def save_session(_clicks, session_id):
        payload = registry.get(session_id).to_bytes()
        filename = f"energy-session-{_timestamp()}.npz"
        return dcc.send_bytes(lambda buffer: buffer.write(payload), filename)

    @app.callback(
        Output("download-csv", "data"),
        Output("download-json", "data"),
        Input("export-csv", "n_clicks"),
        Input("export-json", "n_clicks"),
        State("browser-session-id", "data"),
        State("alpha-input", "value"),
        State("delta-input", "value"),
        State("p-input", "value"),
        State("momentum-length-input", "value"),
        State("rho", "value"),
        State("value-kind", "value"),
        State("histogram-mode", "value"),
        State("y-scale", "value"),
        State("bin-mode", "value"),
        State("manual-bins", "value"),
        prevent_initial_call=True,
    )
    def export_view(
        _csv_clicks,
        _json_clicks,
        session_id,
        alpha,
        delta,
        probability,
        momentum_length,
        rho,
        value_kind,
        histogram_mode,
        y_scale,
        bin_mode,
        manual_bins,
    ):
        session = registry.get(session_id)
        with session.lock:
            view = session.view()
            parameters = _parameters(
                view.algorithm,
                alpha,
                delta,
                probability,
                momentum_length,
                rho,
            )
            _, observation = session.preview(parameters)
        bins = "adaptive" if bin_mode == "adaptive" else int(manual_bins)
        histogram = build_histogram(observation, value_kind=value_kind, bins=bins)
        filename = f"energy-iteration-{view.cursor}-{_timestamp()}"
        if ctx.triggered_id == "export-csv":
            return dcc.send_string(
                histogram_csv(histogram, histogram_mode),
                filename + ".csv",
            ), no_update
        settings = {
            "value_kind": value_kind,
            "histogram_mode": histogram_mode,
            "y_scale": y_scale,
            "bins": bins,
            "preview_parameters": parameters.to_dict(),
        }
        return no_update, dcc.send_string(
            summary_json(view, observation, histogram, settings),
            filename + ".json",
        )


def _register_sync_callbacks(app):
    specs = (
        ("alpha-slider", "alpha-input"),
        ("delta-slider", "delta-input"),
        ("p-slider", "p-input"),
        ("momentum-length-slider", "momentum-length-input"),
    )
    for slider_id, input_id in specs:
        def sync(slider_value, input_value, slider_id=slider_id):
            value = slider_value if ctx.triggered_id == slider_id else input_value
            if value is None:
                return no_update, no_update
            return value, value

        app.callback(
            Output(slider_id, "value"),
            Output(input_id, "value"),
            Input(slider_id, "value"),
            Input(input_id, "value"),
            prevent_initial_call=True,
        )(sync)


def _parameters(algorithm, alpha, delta, probability, momentum_length, rho):
    if alpha is None:
        raise ValueError("alpha не задан")
    if Algorithm(algorithm) is Algorithm.FTGDBF:
        result = FtgdbfParameters(alpha=float(alpha))
    else:
        if delta is None or probability is None or momentum_length is None:
            raise ValueError("заданы не все параметры PMGDBF")
        result = PmgdbfParameters(
            delta=float(delta),
            alpha=float(alpha),
            p=float(probability),
            rho=parse_rho(rho),
            L=int(momentum_length),
        )
    result.validate()
    return result


def _history_rows(view):
    rows = []
    for transition in view.transitions:
        parameters = transition.parameters.to_dict()
        values = ", ".join(f"{key}={value}" for key, value in parameters.items())
        rows.append({
            "step": f"{transition.from_iteration} → {transition.to_iteration}",
            "parameters": values,
            "ber": transition.after.ber,
            "fer": transition.after.fer,
            "position": "пройдено" if transition.to_iteration <= view.cursor else "будущее",
        })
    return rows


def _decision_summary(observation):
    active = observation.decision_frames[:, None]
    should = observation.should_flip & active
    should_not = ~observation.should_flip & active
    did = observation.flip_mask & active
    did_not = ~observation.flip_mask & active
    return " · ".join([
        f"верный flip: {int((should & did).sum()):,}",
        f"верный no-flip: {int((should_not & did_not).sum()):,}",
        f"ошибочный flip: {int((should_not & did).sum()):,}",
        f"пропущенный flip: {int((should & did_not).sum()):,}",
    ])


def _upload_bytes(contents):
    if not contents or "," not in contents:
        raise ValueError("файл не передан")
    _header, encoded = contents.split(",", 1)
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("повреждённые данные загрузки") from exc


def _numeric_control(identifier, label, minimum, maximum, step, value):
    return html.Div(className="numeric-control", children=[
        html.Div(className="control-label-row", children=[
            _label(label),
            dcc.Input(
                id=f"{identifier}-input",
                type="number",
                min=minimum,
                max=maximum,
                step=step,
                value=value,
                debounce=True,
                className="number-input",
            ),
        ]),
        dcc.Slider(
            id=f"{identifier}-slider",
            min=minimum,
            max=maximum,
            step=step,
            value=value,
            updatemode="mouseup",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ])


def _section(title, children):
    return html.Section(className="control-card", children=[html.H2(title), *children])


def _label(text):
    return html.Div(text, className="field-label")


def _metric(label, identifier):
    return html.Div(className="metric-card", children=[
        html.Div(label, className="metric-label"),
        html.Div("—", id=identifier, className="metric-value"),
    ])


def _metadata_text(metadata):
    code = metadata.get("code", "LDPC code")
    frames = int(metadata.get("frames", 0))
    snr_db = float(metadata.get("snr_db", 0.0))
    seed = metadata.get("seed", "—")
    workers = metadata.get("workers", "—")
    return (
        f"{code} · {frames:,} кадров · SNR {snr_db:g} dB · "
        f"seed {seed} · {workers} поток(а/ов)"
    )


def _rate(value, errors):
    return f"{value:.6e} · {errors:,}"


def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")
