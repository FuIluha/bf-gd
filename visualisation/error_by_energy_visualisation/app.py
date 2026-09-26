"""Decoder-agnostic Dash explorer for LDPC step diagnostics."""

import base64
from datetime import datetime
import math
from pathlib import Path
import uuid

from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

from .data import DatasetConfig, load_dataset
from .exports import histogram_csv, summary_json
from .figures import empty_figure, histogram_figure, performance_figure
from .histogram import build_histogram
from .registry import SessionRegistry
from .session import ExplorerSession, comparison_from_bytes, compatible_datasets
from .supported import decoder_catalog


GRAPH_CONFIG = {"displaylogo": False, "responsive": True, "toImageButtonOptions": {
    "format": "png", "filename": "bf-gd-energy-distribution", "scale": 2,
}}
GRAPH_STYLE = {"height": "560px", "minHeight": "560px", "width": "100%"}


def create_app(dataset, seed=None, workers=None, max_sessions=16):
    effective_seed = dataset.metadata["seed"] if seed is None else seed
    effective_workers = dataset.metadata["workers"] if workers is None else workers
    registry = SessionRegistry(dataset, effective_seed, effective_workers, max_sessions)
    app = Dash(
        __name__, title="LDPC Energy Explorer",
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
    )
    app.layout = lambda: _layout(dataset.metadata)
    _register_callbacks(app, registry)
    return app


def _layout(metadata):
    catalog = decoder_catalog()
    specs = [item.describe() for item in catalog.values()]
    first = specs[0]
    first_group = first.groups()[0]
    first_categories = [item for item in first.categories if item.key in first_group.categories]
    return html.Div(className="app-shell", children=[
        dcc.Store(id="browser-session-id", data=str(uuid.uuid4()), storage_type="session"),
        dcc.Store(id="revision", data=0),
        dcc.Download(id="download-session"), dcc.Download(id="download-csv"),
        dcc.Download(id="download-json"),
        html.Div(className="pinned-distribution", children=[
            html.Nav(className="step-toolbar", children=[
                html.Div(className="step-toolbar-inner", children=[
                    html.Button("← Назад", id="backward", n_clicks=0, className="button secondary"),
                    html.Div("Снимок 0 · решение 0 → 1", id="iteration-label", className="iteration-badge"),
                    html.Button("Вперёд →", id="forward", n_clicks=0, className="button primary"),
                ]),
            ]),
            html.Div(className="pinned-chart", children=[
                dcc.Loading(type="circle", children=dcc.Graph(
                    id="histogram", config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE,
                )),
            ]),
        ]),
        html.Div(className="scroll-region", children=[
        html.Header(className="hero", children=[
            html.Div([
                html.Div("BF–GD LAB", className="eyebrow"),
                html.H1("Исследователь решений декодера"),
                html.P("Пошаговое сравнение на фиксированных реализациях канала. Каждый снимок состояния неизменяем.", className="hero-copy"),
            ]),
            html.Div(_metadata_text(metadata), id="dataset-metadata", className="dataset-pill"),
        ]),
        html.Main(className="workspace", children=[
            html.Aside(className="control-column", children=[
                _section("Набор кадров", [
                    html.Div(className="dataset-fields", children=[
                        html.Div([
                            _label("Количество слов"),
                            dcc.Input(id="dataset-frames", type="number", min=1, step=1,
                                      value=int(metadata.get("frames", 1000)), debounce=True,
                                      className="number-input full"),
                        ]),
                        html.Div([
                            _label("SNR, дБ"),
                            dcc.Input(id="dataset-snr", type="number", step="any",
                                      value=float(metadata.get("snr_db", 1.0)), debounce=True,
                                      className="number-input full"),
                        ]),
                    ]),
                    html.Button("Применить набор", id="apply-dataset", n_clicks=0,
                                className="button secondary full"),
                    html.P("Новый набор начинает историю с шага 0; seed сохраняется.", className="hint"),
                ]),
                html.Div("Готово", id="action-status", className="status-line"),
                _section("Алгоритм и история", [
                    _label("Декодер"),
                    dcc.Dropdown(
                        id="algorithm",
                        options=[{"label": spec.title, "value": spec.key} for spec in specs],
                        value=first.key, clearable=False, searchable=False,
                    ),
                    html.Button("Сбросить к итерации 0", id="reset", n_clicks=0, className="button ghost full"),
                ]),
                _section("Параметры декодера", [
                    html.Div(
                        id={"type": "decoder-panel", "algorithm": spec.key},
                        style={} if spec.key == first.key else {"display": "none"},
                        children=[_parameter_control(spec.key, item) for item in spec.parameters],
                    ) for spec in specs
                ]),
                _section("Отображение", [
                    _label("Величина по оси X"),
                    dcc.Dropdown(
                        id="observable", value=first.observables[0].key,
                        options=[{"label": item.label, "value": item.key} for item in first.observables],
                        clearable=False,
                    ),
                    _label("Разложение"),
                    dcc.Dropdown(
                        id="category-group", value=first_group.key,
                        options=[{"label": item.label, "value": item.key} for item in first.groups()],
                        clearable=False, searchable=False,
                    ),
                    _label("Категории"),
                    dcc.Checklist(
                        id="visible-categories", value=[item.key for item in first_categories],
                        options=[{"label": item.label, "value": item.key} for item in first_categories],
                        className="radio-stack",
                    ),
                    _label("Ось Y гистограммы"),
                    dcc.RadioItems(id="histogram-mode", options=[
                        {"label": "Совместная плотность", "value": "density"},
                        {"label": "Количество", "value": "count"},
                    ], value="density", inline=True, className="radio-row"),
                    dcc.RadioItems(id="y-scale", options=[
                        {"label": "Линейная", "value": "linear"},
                        {"label": "Логарифмическая", "value": "log"},
                    ], value="linear", inline=True, className="radio-row"),
                    _label("Ширина бинов"),
                    dcc.RadioItems(id="bin-mode", options=[
                        {"label": "Адаптивно", "value": "adaptive"},
                        {"label": "Вручную", "value": "manual"},
                    ], value="adaptive", inline=True, className="radio-row"),
                    dcc.Input(id="manual-bins", type="number", min=1, max=2000, step=1,
                              value=80, disabled=True, className="number-input full"),
                ]),
                _section("Сессия и экспорт", [
                    html.Div(className="button-grid", children=[
                        html.Button("Сохранить сессию", id="save-session", n_clicks=0, className="button secondary"),
                        dcc.Upload(id="session-upload", accept=".npz", children=html.Button("Загрузить сессию", className="button secondary")),
                        html.Button("Экспорт CSV", id="export-csv", n_clicks=0, className="button secondary"),
                        html.Button("Экспорт JSON", id="export-json", n_clicks=0, className="button secondary"),
                    ]),
                    html.P("PNG сохраняется кнопкой камеры на графике.", className="hint"),
                    _label("Сравнить с другой сессией"),
                    dcc.Upload(id="comparison-upload", accept=".npz", children=html.Button("Добавить сравнение", className="button ghost full")),
                    html.Button("Убрать сравнение", id="clear-comparison", n_clicks=0, className="button link full"),
                ]),
            ]),
            html.Section(className="content-column", children=[
                html.Div(className="metric-grid", children=[
                    _metric("Текущий BER", "current-ber"), _metric("Текущий FER", "current-fer"),
                    _metric("BER после шага", "preview-ber"), _metric("FER после шага", "preview-fer"),
                    _metric("Активные кадры", "active-frames"),
                ]),
                html.Div(id="plot-status", className="plot-status"),
                html.Div(id="decision-summary", className="decision-summary"),
                html.Div(className="chart-card", children=[
                    dcc.Graph(id="performance", config=GRAPH_CONFIG, responsive=True, style=GRAPH_STYLE),
                ]),
                html.Div(className="table-card", children=[
                    html.Div(className="card-heading", children=[
                        html.H2("История параметров"),
                        html.P("Шаг из прошлого удаляет прежнее будущее и создаёт новую линейную историю."),
                    ]),
                    dash_table.DataTable(
                        id="parameter-history", columns=[
                            {"name": "Шаг", "id": "step"}, {"name": "Параметры", "id": "parameters"},
                            {"name": "BER", "id": "ber", "type": "numeric", "format": {"specifier": ".6e"}},
                            {"name": "FER", "id": "fer", "type": "numeric", "format": {"specifier": ".6e"}},
                            {"name": "Положение", "id": "position"},
                        ], data=[], page_size=12, sort_action="native", style_as_list_view=True,
                        style_cell={"backgroundColor": "#ffffff", "color": "#222222", "border": "none",
                                    "borderBottom": "1px solid #dddddd", "padding": "10px 12px", "textAlign": "left",
                                    "fontFamily": "Arial, Helvetica, sans-serif", "whiteSpace": "normal", "height": "auto",
                                    "minWidth": "100px", "maxWidth": "420px"},
                        style_header={"backgroundColor": "#f2f2f2", "fontWeight": 700, "borderBottom": "1px solid #aaaaaa"},
                        style_data_conditional=[{"if": {"filter_query": "{position} = 'будущее'"}, "opacity": 0.45}],
                    ),
                ]),
            ]),
        ]),
        ]),
    ])


def _parameter_control(algorithm, parameter):
    identifier = {"algorithm": algorithm, "name": parameter.key}
    if parameter.kind == "float_list":
        return html.Div([
            _label(parameter.label),
            dcc.Input(id={"type": "parameter-input", **identifier}, type="text",
                      value=", ".join(str(value) for value in parameter.default), debounce=True,
                      className="text-input"),
        ])
    input_control = dcc.Input(
        id={"type": "parameter-input", **identifier}, type="number", value=parameter.default,
        min=parameter.minimum, max=parameter.maximum,
        step="any" if parameter.kind == "float" else 1,
        debounce=True, className="number-input",
    )
    slider_minimum = parameter.minimum or 0
    if parameter.kind == "float" and parameter.step and slider_minimum > 0:
        aligned = math.ceil(slider_minimum / parameter.step) * parameter.step
        if aligned <= parameter.default:
            slider_minimum = aligned
    slider = dcc.Slider(
        id={"type": "parameter-slider", **identifier}, min=slider_minimum,
        max=parameter.maximum or 10, step=parameter.step or 0.01, value=parameter.default,
        updatemode="mouseup", tooltip={"placement": "bottom", "always_visible": False},
    )
    return html.Div(className="numeric-control", children=[
        html.Div(className="control-label-row", children=[_label(parameter.label), input_control]), slider,
    ])


def _register_callbacks(app, registry):
    @app.callback(
        Output({"type": "parameter-slider", "algorithm": MATCH, "name": MATCH}, "value"),
        Output({"type": "parameter-input", "algorithm": MATCH, "name": MATCH}, "value"),
        Input({"type": "parameter-slider", "algorithm": MATCH, "name": MATCH}, "value"),
        Input({"type": "parameter-input", "algorithm": MATCH, "name": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def sync_parameter(slider, manual):
        trigger = ctx.triggered_id
        value = slider if trigger and trigger.get("type") == "parameter-slider" else manual
        return (value, value) if value is not None else (no_update, no_update)

    @app.callback(Output("manual-bins", "disabled"), Input("bin-mode", "value"))
    def toggle_manual_bins(mode):
        return mode != "manual"

    @app.callback(
        Output({"type": "decoder-panel", "algorithm": ALL}, "style"),
        Output("observable", "options"), Output("observable", "value"),
        Output("category-group", "options"), Output("category-group", "value"),
        Input("algorithm", "value"),
        State({"type": "decoder-panel", "algorithm": ALL}, "id"),
    )
    def decoder_controls(algorithm, panel_ids):
        spec = decoder_catalog()[algorithm].describe()
        return (
            [{} if item["algorithm"] == algorithm else {"display": "none"} for item in panel_ids],
            [{"label": item.label, "value": item.key} for item in spec.observables],
            spec.observables[0].key,
            [{"label": item.label, "value": item.key} for item in spec.groups()],
            spec.groups()[0].key,
        )

    @app.callback(
        Output("visible-categories", "options"), Output("visible-categories", "value"),
        Input("algorithm", "value"), Input("category-group", "value"),
    )
    def category_controls(algorithm, group_key):
        spec = decoder_catalog()[algorithm].describe()
        groups = spec.groups()
        group = next((item for item in groups if item.key == group_key), groups[0])
        categories = [item for item in spec.categories if item.key in group.categories]
        return (
            [{"label": item.label, "value": item.key} for item in categories],
            [item.key for item in categories],
        )

    @app.callback(
        Output("revision", "data"), Output("action-status", "children"), Output("algorithm", "value"),
        Output("dataset-frames", "value"), Output("dataset-snr", "value"),
        Input("forward", "n_clicks"), Input("backward", "n_clicks"), Input("reset", "n_clicks"),
        Input("apply-dataset", "n_clicks"),
        Input("algorithm", "value"), Input("session-upload", "contents"),
        Input("comparison-upload", "contents"), Input("clear-comparison", "n_clicks"),
        State("browser-session-id", "data"), State("revision", "data"),
        State({"type": "parameter-input", "algorithm": ALL, "name": ALL}, "id"),
        State({"type": "parameter-input", "algorithm": ALL, "name": ALL}, "value"),
        State("session-upload", "filename"), State("comparison-upload", "filename"),
        State("dataset-frames", "value"), State("dataset-snr", "value"),
        prevent_initial_call=True,
    )
    def mutate(_forward, _backward, _reset, _apply_dataset, algorithm, upload, comparison_upload,
               _clear, session_id, revision, parameter_ids, parameter_values,
               session_filename, comparison_filename, frames, snr):
        revision = int(revision or 0)
        try:
            session = registry.get(session_id)
            trigger = ctx.triggered_id
            if trigger == "forward":
                session.step_forward(_parameters(session, parameter_ids, parameter_values))
                return revision + 1, f"Сохранён снимок {session.cursor}", no_update, no_update, no_update
            if trigger == "backward":
                before = session.cursor
                session.step_back()
                message = "Уже выбрана итерация 0" if before == 0 else f"Возврат к снимку {session.cursor}"
                return revision + 1, message, no_update, no_update, no_update
            if trigger == "reset":
                session.reset(session.algorithm, _parameters(session, parameter_ids, parameter_values))
                return revision + 1, "История сброшена к итерации 0", no_update, no_update, no_update
            if trigger == "apply-dataset":
                loaded = _rebuild_dataset_session(session, frames, snr)
                registry.replace(session_id, loaded)
                return (
                    revision + 1,
                    f"Новый набор: {loaded.batch.frames:,} слов, SNR {loaded.metadata['snr_db']:g} дБ; история начата заново",
                    no_update, loaded.batch.frames, loaded.metadata["snr_db"],
                )
            if trigger == "algorithm":
                if algorithm == session.algorithm:
                    return revision, no_update, no_update, no_update, no_update
                session.reset(algorithm)
                return revision + 1, f"Выбран {session.spec.title}; история начата заново", no_update, no_update, no_update
            if trigger == "session-upload":
                loaded = ExplorerSession.from_bytes(_upload_bytes(upload), workers=registry.workers)
                registry.replace(session_id, loaded)
                return (
                    revision + 1, f"Загружена сессия {session_filename or ''}", loaded.algorithm,
                    loaded.batch.frames, loaded.metadata.get("snr_db", 1.0),
                )
            if trigger == "comparison-upload":
                label = Path(comparison_filename or "сравнение").stem
                comparison = comparison_from_bytes(_upload_bytes(comparison_upload), name=label)
                with session.lock:
                    if not compatible_datasets(session.metadata, comparison.metadata):
                        raise ValueError("сравниваемая история рассчитана на другом наборе кадров")
                    session.comparison = comparison
                return revision + 1, f"Добавлено сравнение: {label}", no_update, no_update, no_update
            if trigger == "clear-comparison":
                with session.lock:
                    session.comparison = None
                return revision + 1, "Сравнение убрано", no_update, no_update, no_update
            return revision, no_update, no_update, no_update, no_update
        except Exception as exc:
            return revision, f"Ошибка: {exc}", no_update, no_update, no_update

    @app.callback(
        Output("iteration-label", "children"), Output("current-ber", "children"),
        Output("current-fer", "children"), Output("preview-ber", "children"),
        Output("preview-fer", "children"), Output("active-frames", "children"),
        Output("decision-summary", "children"), Output("histogram", "figure"),
        Output("performance", "figure"), Output("parameter-history", "data"),
        Output("plot-status", "children"), Output("backward", "disabled"),
        Output("forward", "disabled"), Output("dataset-metadata", "children"),
        Input("revision", "data"), Input("algorithm", "value"),
        Input({"type": "parameter-input", "algorithm": ALL, "name": ALL}, "value"),
        Input("observable", "value"), Input("category-group", "value"),
        Input("visible-categories", "value"),
        Input("histogram-mode", "value"), Input("y-scale", "value"),
        Input("bin-mode", "value"), Input("manual-bins", "value"),
        State("browser-session-id", "data"),
        State({"type": "parameter-input", "algorithm": ALL, "name": ALL}, "id"),
    )
    def render(_revision, _algorithm, values, observable_key, group_key, selected_categories,
               mode, scale, bin_mode, manual_bins, session_id, parameter_ids):
        session = registry.get(session_id)
        view = session.view()
        spec = session.spec
        current = view.snapshots[view.cursor].metrics
        history = _history_rows(view)
        try:
            parameters = _parameters(session, parameter_ids, values)
            _, observation = session.preview(parameters)
            observables = {item.key: item for item in spec.observables}
            if observable_key not in observables:
                observable_key = spec.observables[0].key
            groups = spec.groups()
            group = next((item for item in groups if item.key == group_key), groups[0])
            allowed = set(group.categories)
            selected = [key for key in (selected_categories or []) if key in allowed]
            bins = "adaptive" if bin_mode == "adaptive" else int(manual_bins)
            histogram = build_histogram(
                observation, observable_key, selected, bins,
                normalization_keys=group.categories,
            )
            plot = histogram_figure(histogram, mode, scale, observables[observable_key], spec, view.cursor)
            active = int(observation.decision_frames.sum())
            summary = " · ".join(
                f"{item.label}: {observation.categories[item.key].frame_indices.size if item.key in observation.categories else 0:,}"
                for item in spec.categories if item.key in selected
            ) or "Категории скрыты"
            status = (
                f"Предпросмотр с текущими параметрами · {histogram.method} · "
                f"нормировка по {histogram.normalization_count:,} наблюдениям"
            )
            preview = observation.after
            forward_disabled = active == 0
        except Exception as exc:
            plot = empty_figure(f"Исправьте параметры: {exc}")
            active = int(view.snapshots[view.cursor].state.active.sum())
            summary, status = "Предпросмотр недоступен", f"Ошибка параметров: {exc}"
            preview, forward_disabled = current, True
        return (
            f"Снимок {view.cursor} · решение {view.cursor} → {view.cursor + 1}",
            _rate(current.ber, current.bit_errors), _rate(current.fer, current.frame_errors),
            _rate(preview.ber, preview.bit_errors), _rate(preview.fer, preview.frame_errors),
            f"{active:,} / {session.batch.frames:,}", summary, plot,
            performance_figure(view), history, status, view.cursor == 0,
            forward_disabled, _metadata_text(view.metadata),
        )

    @app.callback(Output("download-session", "data"), Input("save-session", "n_clicks"),
                  State("browser-session-id", "data"), prevent_initial_call=True)
    def save_session(_clicks, session_id):
        payload = registry.get(session_id).to_bytes()
        return dcc.send_bytes(lambda buffer: buffer.write(payload), f"energy-session-{_timestamp()}.npz")

    @app.callback(
        Output("download-csv", "data"), Output("download-json", "data"),
        Input("export-csv", "n_clicks"), Input("export-json", "n_clicks"),
        State("browser-session-id", "data"),
        State({"type": "parameter-input", "algorithm": ALL, "name": ALL}, "id"),
        State({"type": "parameter-input", "algorithm": ALL, "name": ALL}, "value"),
        State("observable", "value"), State("category-group", "value"),
        State("visible-categories", "value"),
        State("histogram-mode", "value"), State("y-scale", "value"),
        State("bin-mode", "value"), State("manual-bins", "value"),
        prevent_initial_call=True,
    )
    def export_view(_csv, _json, session_id, ids, values, observable_key,
                    group_key, selected_categories, mode, scale, bin_mode, manual_bins):
        session = registry.get(session_id)
        parameters = _parameters(session, ids, values)
        view = session.view()
        _, observation = session.preview(parameters)
        bins = "adaptive" if bin_mode == "adaptive" else int(manual_bins)
        spec = session.spec
        observables = {item.key for item in spec.observables}
        if observable_key not in observables:
            observable_key = spec.observables[0].key
        groups = spec.groups()
        group = next((item for item in groups if item.key == group_key), groups[0])
        selected = [key for key in (selected_categories or []) if key in set(group.categories)]
        histogram = build_histogram(
            observation, observable_key, selected, bins,
            normalization_keys=group.categories,
        )
        filename = f"energy-iteration-{view.cursor}-{_timestamp()}"
        if ctx.triggered_id == "export-csv":
            return dcc.send_string(histogram_csv(histogram, mode), filename + ".csv"), no_update
        settings = {
            "observable": observable_key, "category_group": group.key,
            "visible_categories": selected,
            "histogram_mode": mode, "histogram_y_scale": scale,
            "performance_y_scale": "log", "bins": bins, "preview_parameters": parameters,
        }
        return no_update, dcc.send_string(
            summary_json(view, observation, histogram, settings, spec), filename + ".json",
        )


def _parameters(session, ids, values):
    selected = {
        identifier["name"]: value for identifier, value in zip(ids, values)
        if identifier["algorithm"] == session.algorithm
    }
    return session.decoder.validate_parameters(selected)


def _rebuild_dataset_session(session, frames, snr):
    if frames is None or not float(frames).is_integer():
        raise ValueError("Количество слов должно быть целым положительным числом")
    if snr is None:
        raise ValueError("SNR не задан")
    code_path = session.metadata.get("code_path")
    if not code_path:
        raise ValueError("В сессии нет пути к описанию кода; набор нельзя пересоздать")
    dataset = load_dataset(DatasetConfig(
        code_path=Path(code_path), frames=int(frames), snr_db=float(snr),
        seed=session.seed, workers=session.workers,
    ))
    return ExplorerSession(
        dataset.graph, dataset.batch, seed=session.seed, workers=session.workers,
        algorithm=session.algorithm, metadata=dataset.metadata,
    )


def _history_rows(view):
    return [{
        "step": f"{item.from_iteration} → {item.to_iteration}",
        "parameters": ", ".join(f"{key}={value}" for key, value in item.parameters.items()),
        "ber": item.after.ber, "fer": item.after.fer,
        "position": "пройдено" if item.to_iteration <= view.cursor else "будущее",
    } for item in view.transitions]


def _upload_bytes(contents):
    if not contents or "," not in contents:
        raise ValueError("файл не передан")
    try:
        return base64.b64decode(contents.split(",", 1)[1], validate=True)
    except ValueError as exc:
        raise ValueError("повреждённые данные загрузки") from exc


def _section(title, children):
    return html.Section(className="control-card", children=[html.H2(title), *children])


def _label(text):
    return html.Div(text, className="field-label")


def _metric(label, identifier):
    return html.Div(className="metric-card", children=[
        html.Div(label, className="metric-label"), html.Div("—", id=identifier, className="metric-value"),
    ])


def _metadata_text(metadata):
    return (
        f"{metadata.get('code', 'LDPC code')} · {int(metadata.get('frames', 0)):,} кадров · "
        f"SNR {float(metadata.get('snr_db', 0.0)):g} dB · "
        f"seed {metadata.get('seed', '—')} · {metadata.get('workers', '—')} поток(а/ов)"
    )


def _rate(value, errors):
    return f"{value:.6e} · {errors:,}"


def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")
