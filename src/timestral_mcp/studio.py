"""Forecast Studio e caricamento file: interfacce interattive (MCP App) per impostare tutto da un'unica schermata.

Lo Studio è una FastMCPApp: un tool d'ingresso visibile al modello (`forecast_studio`) disegna il modulo,
mentre i tool di backend (`studio_run`, `studio_export`) sono richiamati solo dall'interfaccia. I risultati
restano nella sessione (stessi archivi dei tool normali), quindi Claude può poi rispondere alle domande con
`get_forecast_details`.
"""

from __future__ import annotations

import base64
import binascii
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pandas as pd
from fastmcp import FastMCPApp
from fastmcp.exceptions import ToolError
from pandas.tseries.frequencies import to_offset
from prefab_ui.actions import SetState, ShowToast, UpdateContext
from prefab_ui.actions.mcp import CallTool, SendMessage
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Alert,
    AlertDescription,
    Badge,
    Button,
    Card,
    CardContent,
    CardHeader,
    Column,
    DataTable,
    DataTableColumn,
    DropZone,
    Heading,
    Input,
    Label,
    Loader,
    Metric,
    Row,
    Select,
    SelectOption,
    Separator,
    Switch,
    Tab,
    Tabs,
    Text,
)
from prefab_ui.components.charts import ChartSeries, LineChart
from prefab_ui.components.control_flow import ForEach, If
from prefab_ui.rx import ERROR, RESULT, STATE, Rx

from timestral_mcp import models
from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics.covariates import future_block
from timestral_mcp.analytics.forecast import run_forecast
from timestral_mcp.analytics.summary import (
    _date,
    _pct,
    backtest_summary,
    fmt,
    forecast_summary,
    known_covariate_lines,
)
from timestral_mcp.config import MODEL_SIZES, MODEL_WEIGHTS_MB
from timestral_mcp.data.ingest import Dataset, build_dataset, read_bytes
from timestral_mcp.session import backtests, datasets, forecasts, settings, uploads
from timestral_mcp.ui.charts import forecast_rows
from timestral_mcp.ui.theme import C_BOUND, C_HISTORY, C_MEDIAN, THEME

studio = FastMCPApp("Forecast Studio")
uploader = FastMCPApp("Dataset Upload")

MAX_COVARIATE_ROWS = 12
MAX_SERIES_OPTIONS = 200
INTERVALS = [("0.5", "50%"), ("0.8", "80%"), ("0.9", "90%"), ("0.98", "98%")]
COVARIATE_MODES = [
    ("none", "Non usare"),
    ("past", "Passata (nota solo fino a oggi)"),
    ("known", "Nota in anticipo"),
]
UPLOAD_SUFFIXES = ".csv,.tsv,.txt,.xlsx,.xlsm"


# ----------------------------------------------------------------------------- impostazioni di partenza


def default_horizon(freq: str) -> int:
    off = to_offset(freq)
    if isinstance(off, pd.offsets.Minute):
        return max(1440 // max(off.n, 1), 24)
    if isinstance(off, pd.offsets.Hour):
        return 48
    if isinstance(off, pd.offsets.Week):
        return 12
    if isinstance(off, pd.offsets.MonthBegin | pd.offsets.MonthEnd):
        return 12
    if isinstance(off, pd.offsets.QuarterBegin | pd.offsets.QuarterEnd):
        return 4
    if isinstance(off, pd.offsets.YearBegin | pd.offsets.YearEnd):
        return 3
    return 30


def _is_binary(ds: Dataset, col: str) -> bool:
    vals = np.unique(np.concatenate([df[col].dropna().to_numpy(dtype=float) for df in ds.series.values()]))
    return vals.size > 0 and set(vals) <= {0.0, 1.0}


def _default_target(ds: Dataset) -> str:
    return next((c for c in ds.value_columns if not _is_binary(ds, c)), ds.value_columns[0])


def _model_options() -> tuple[list[tuple[str, str]], str]:
    options = []
    cached = []
    for size in MODEL_SIZES:
        ok = models.is_cached(size)
        if ok:
            cached.append(size)
        status = "scaricato" if ok else "da scaricare"
        options.append((size, f"{size} · {MODEL_WEIGHTS_MB[size]} MB · {status}"))
    default = (
        settings.default_model
        if settings.default_model in cached
        else (cached[0] if cached else settings.default_model)
    )
    return options, default


def _safe_column(name: str) -> bool:
    return not any(ch in name for ch in ",'\"{}")


def _covariate_columns(ds: Dataset, target: str) -> tuple[list[str], list[str]]:
    usable, skipped = [], []
    for c in ds.value_columns:
        if c == target:
            continue
        (usable if _safe_column(c) else skipped).append(c)
    return usable[:MAX_COVARIATE_ROWS], skipped + usable[MAX_COVARIATE_ROWS:]


def _join_modes(columns: list[str], mode: str):
    """Espressione lato client: elenco 'col1,col2,' delle colonne scelte con quella modalità."""
    expr = None
    for i, col in enumerate(columns):
        term = (Rx(f"cov_{i}") == mode).then(f"{col},", "")
        expr = term if expr is None else expr + term
    return expr if expr is not None else ""


def _split(value: str | None) -> list[str]:
    if value and "{{" in value:
        raise ToolError(
            "L'interfaccia non ha valutato i campi del modulo (formula non risolta). "
            "Riapri il Forecast Studio oppure usa direttamente il tool forecast."
        )
    return [c.strip() for c in (value or "").split(",") if c.strip()]


# ----------------------------------------------------------------------------- backend dello Studio


@studio.tool()
async def studio_run(
    dataset_id: str,
    target: str,
    horizon: int,
    series: str | None = None,
    model: str | None = None,
    interval: float = 0.8,
    known: str = "",
    past: str = "",
    with_backtest: bool = False,
    allow_download: bool = False,
) -> dict[str, Any]:
    """Esegue previsione (e, se richiesto, backtest) con le scelte fatte nell'interfaccia."""
    try:
        ds = datasets.get(dataset_id)
        size = models.normalize_size(model)
        models.ensure_available(size, bool(allow_download))
        known_cols, past_cols = _split(known), _split(past)
        dropped = [c for c in known_cols + past_cols if c == target]
        known_cols = [c for c in known_cols if c != target]
        past_cols = [c for c in past_cols if c != target]
        key = series or None
        fc_id = forecasts.new_id()
        result = await anyio.to_thread.run_sync(
            partial(
                run_forecast,
                ds,
                forecast_id=fc_id,
                target=target,
                horizon=int(horizon),
                series=key,
                model=size,
                interval=float(interval),
                past_covariates=past_cols or None,
                known_covariates=known_cols or None,
            )
        )
        forecasts.put(fc_id, result)

        backtest_text, backtest_metrics = "", None
        if with_backtest:
            try:
                plan = bt.plan_backtest(
                    ds,
                    target=target,
                    horizon=int(horizon),
                    series=key,
                    model=size,
                    n_folds=3,
                    interval=float(interval),
                    past_covariates=past_cols or None,
                    known_covariates=known_cols or None,
                )
                folds = [await anyio.to_thread.run_sync(bt.run_fold, plan, o) for o in plan.origins]
                bt_id = backtests.new_id()
                back = bt.finish_backtest(plan, folds, bt_id)
                backtests.put(bt_id, back)
                backtest_text, backtest_metrics = backtest_summary(back), back.summary
            except ValueError as exc:
                backtest_text = f"Backtest non eseguito: {exc.args[0] if exc.args else exc}"
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise ToolError(str(exc.args[0]) if exc.args else str(exc)) from exc

    lo, hi = result.lower, result.upper
    med = result.median
    last_val = float(result.history.dropna().iloc[-1])
    peak = int(np.argmax(med))
    warnings = list(result.warnings)
    if dropped:
        warnings.append(f"'{target}' è il target: l'ho tolto dalle covariate.")
    cov_lines = known_covariate_lines(result)
    message = (
        f"Forecast Studio: ho calcolato la previsione {result.id} (serie '{result.series}', target '{target}', "
        f"{result.horizon} periodi, Timestral-{result.model}). Usa get_forecast_details con forecast_id='{result.id}' "
        f"per rispondere alle mie domande.\n\n{forecast_summary(result)}"
        + (f"\n\n{backtest_text}" if backtest_text else "")
    )
    return {
        "forecast_id": result.id,
        "metrics": {
            "last": fmt(last_val),
            "median_end": fmt(med[-1]),
            "change": _pct(med[-1], last_val),
            "interval_end": f"{fmt(lo[-1])} – {fmt(hi[-1])}",
            "interval_label": f"Intervallo {result.interval:.0%} a fine orizzonte",
            "peak": fmt(med[peak]),
            "peak_date": _date(result.future_index[peak], result.freq),
            "model": f"Timestral-{result.model}",
        },
        "chart": forecast_rows(result),
        "table": [
            {
                "data": _date(ts, result.freq),
                "mediana": round(float(med[i]), 2),
                "basso": round(float(lo[i]), 2),
                "alto": round(float(hi[i]), 2),
            }
            for i, ts in enumerate(result.future_index)
        ],
        "warnings_text": "\n".join(warnings),
        "covariate_text": "\n".join(cov_lines),
        "backtest_text": backtest_text,
        "backtest_metrics": backtest_metrics,
        "message": message,
    }


@studio.tool()
async def studio_export(forecast_id: str, format: str = "xlsx") -> dict[str, Any]:
    """Salva la previsione (mediana e 21 quantili) nella cartella di esportazione."""
    from timestral_mcp.session import resolve_forecast

    try:
        r = resolve_forecast(forecast_id)
    except KeyError as exc:
        raise ToolError(str(exc.args[0])) from exc
    if format not in {"csv", "xlsx"}:
        raise ToolError("format deve essere csv o xlsx.")
    out = settings.export_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{r.id.replace('::', '_')}.{format}"
    df = r.to_frame().reset_index()
    df.to_csv(path, index=False) if format == "csv" else df.to_excel(
        path, index=False, sheet_name="previsione"
    )
    return {"path": str(path), "rows": len(df)}


# ----------------------------------------------------------------------------- interfaccia dello Studio


@studio.ui(
    name="forecast_studio",
    description=(
        "Apre il Forecast Studio: un'unica schermata interattiva in cui l'utente imposta serie, colonna da prevedere, "
        "orizzonte, modello, intervallo e covariate, avvia la previsione (con backtest opzionale) e vede grafico e tabella. "
        "Richiede un dataset già caricato (load_dataset o upload_dataset); senza dataset_id usa l'ultimo caricato. "
        "Dopo averlo aperto, FERMATI e attendi: l'utente imposta e calcola dall'interfaccia, poi può inviarti i risultati."
    ),
)
def forecast_studio(dataset_id: str | None = None, series: str | None = None) -> PrefabApp:
    items = datasets.items()
    if dataset_id is None:
        if not items:
            raise ToolError(
                "Nessun dataset in sessione: caricane uno con load_dataset (percorso file o dati incollati) "
                "oppure con upload_dataset, poi riapri il Forecast Studio."
            )
        dataset_id = items[-1][0]
    try:
        ds = datasets.get(dataset_id)
        key, _ = ds.get_series(series) if series else (next(iter(ds.series)), None)
    except (ValueError, KeyError) as exc:
        raise ToolError(str(exc.args[0]) if exc.args else str(exc)) from exc

    target = _default_target(ds)
    cov_cols, skipped = _covariate_columns(ds, target)
    block = future_block(ds.series[key], ds.freq)
    future_cols = set(block["columns"]) if block else set()
    model_options, model_default = _model_options()
    series_keys = list(ds.series)[:MAX_SERIES_OPTIONS]
    horizon = default_horizon(ds.freq)
    df = ds.series[key]
    period = f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}"

    state: dict[str, Any] = {
        "dataset_id": dataset_id,
        "series": key,
        "target": target,
        "horizon": horizon,
        "model": model_default,
        "interval": "0.8",
        "with_backtest": False,
        "allow_download": False,
        "running": False,
        "ready": False,
        "error": "",
        "chart": [],
        "table": [],
        "metrics": {
            "last": "",
            "median_end": "",
            "change": "",
            "interval_end": "",
            "interval_label": "",
            "peak": "",
            "peak_date": "",
            "model": "",
        },
        "message": "",
        "forecast_id": "",
        "warnings_text": "",
        "covariate_text": "",
        "backtest_text": "",
        **{f"cov_{i}": ("known" if c in future_cols else "none") for i, c in enumerate(cov_cols)},
    }

    arguments = {
        "dataset_id": Rx("dataset_id"),
        "target": Rx("target"),
        "horizon": Rx("horizon"),
        "series": Rx("series"),
        "model": Rx("model"),
        "interval": Rx("interval"),
        "known": _join_modes(cov_cols, "known"),
        "past": _join_modes(cov_cols, "past"),
        "with_backtest": Rx("with_backtest"),
        "allow_download": Rx("allow_download"),
    }

    with Card(css_class="max-w-4xl mx-auto") as view:
        with CardHeader(), Column(gap=1):
            Heading("Forecast Studio")
            Text(
                f"Dataset '{ds.name}' · {len(ds.series)} serie · frequenza {ds.freq_label} · {period}",
                css_class="text-sm text-muted-foreground",
            )
        with CardContent(), Column(gap=4):
            for warning in ds.warnings:
                with Alert(variant="warning"):
                    AlertDescription(warning)
            with Row(gap=4, css_class="flex-wrap"):
                if len(series_keys) > 1:
                    with Column(gap=1):
                        Label("Serie")
                        with Select(name="series"):
                            for k in series_keys:
                                SelectOption(value=k, label=k)
                        if len(ds.series) > len(series_keys):
                            Text(
                                f"Mostrate le prime {len(series_keys)} serie su {len(ds.series)}.",
                                css_class="text-xs text-muted-foreground",
                            )
                with Column(gap=1):
                    Label("Colonna da prevedere")
                    with Select(name="target"):
                        for c in ds.value_columns:
                            SelectOption(value=c, label=c)
                with Column(gap=1):
                    Label(f"Orizzonte (periodi: {ds.freq_label})")
                    Input(name="horizon", input_type="number", min=1, max=3660, step=1)
            with Row(gap=4, css_class="flex-wrap"):
                with Column(gap=1):
                    Label("Modello")
                    with Select(name="model"):
                        for value, label in model_options:
                            SelectOption(value=value, label=label)
                with Column(gap=1):
                    Label("Intervallo di incertezza")
                    with Select(name="interval"):
                        for value, label in INTERVALS:
                            SelectOption(value=value, label=label)
            if cov_cols:
                Separator()
                Heading("Covariate", css_class="text-base")
                Text(
                    "Colonne che possono aiutare la previsione. «Nota in anticipo» richiede righe future nel dataset "
                    "(es. promozioni pianificate); sono preselezionate quelle che le hanno.",
                    css_class="text-xs text-muted-foreground",
                )
                for i, c in enumerate(cov_cols):
                    with Row(gap=3, align="center"):
                        Text(c, css_class="w-48 text-sm font-medium")
                        with Select(name=f"cov_{i}"):
                            for value, label in COVARIATE_MODES:
                                SelectOption(value=value, label=label)
                if skipped:
                    Text(f"Non mostrate: {', '.join(skipped)}.", css_class="text-xs text-muted-foreground")
            Separator()
            with Row(gap=6, css_class="flex-wrap"):
                Switch(name="with_backtest", label="Esegui anche il backtest (più lento)")
                Switch(name="allow_download", label="Consenti il download del modello se manca")
            with Row(gap=3, align="center"):
                Button(
                    "Calcola previsione",
                    disabled=Rx("running"),
                    on_click=[
                        SetState("running", True),
                        SetState("error", ""),
                        SetState("ready", False),
                        CallTool(
                            studio_run,
                            arguments=arguments,
                            on_success=[
                                SetState("chart", RESULT.chart),
                                SetState("table", RESULT.table),
                                SetState("metrics", RESULT.metrics),
                                SetState("message", RESULT.message),
                                SetState("forecast_id", RESULT.forecast_id),
                                SetState("warnings_text", RESULT.warnings_text),
                                SetState("covariate_text", RESULT.covariate_text),
                                SetState("backtest_text", RESULT.backtest_text),
                                SetState("ready", True),
                                SetState("running", False),
                            ],
                            on_error=[SetState("error", ERROR), SetState("running", False)],
                        ),
                    ],
                )
                with If(Rx("running")):
                    Loader(variant="spin")
                    Text("Calcolo in corso…", css_class="text-sm text-muted-foreground")
            with If(Rx("error")):
                with Alert(variant="destructive"):
                    AlertDescription(Rx("error"))

            with If(Rx("ready")), Column(gap=4):
                Separator()
                with If(Rx("warnings_text")):
                    with Alert(variant="warning"):
                        AlertDescription(Rx("warnings_text"))
                with If(Rx("covariate_text")):
                    Text(Rx("covariate_text"), css_class="text-sm text-muted-foreground whitespace-pre-wrap")
                with Row(gap=3, css_class="flex-wrap"):
                    Metric(label="Ultimo osservato", value=Rx("metrics.last"))
                    Metric(
                        label="Mediana a fine orizzonte",
                        value=Rx("metrics.median_end"),
                        delta=Rx("metrics.change"),
                    )
                    Metric(label=Rx("metrics.interval_label"), value=Rx("metrics.interval_end"))
                    Metric(
                        label="Picco previsto", value=Rx("metrics.peak"), description=Rx("metrics.peak_date")
                    )
                with Tabs(value="grafico", variant="line"):
                    with Tab("Grafico", value="grafico"):
                        LineChart(
                            data=Rx("chart"),
                            series=[
                                ChartSeries(data_key="storico", label="Storico", color=C_HISTORY),
                                ChartSeries(data_key="mediana", label="Previsione (mediana)", color=C_MEDIAN),
                                ChartSeries(data_key="basso", label="Limite basso", color=C_BOUND),
                                ChartSeries(data_key="alto", label="Limite alto", color=C_BOUND),
                            ],
                            x_axis="date",
                            height=380,
                            show_dots=False,
                            animate=False,
                        )
                    with Tab("Tabella", value="tabella"):
                        DataTable(
                            columns=[
                                DataTableColumn(key="data", header="Data", sortable=True),
                                DataTableColumn(key="mediana", header="Mediana", sortable=True),
                                DataTableColumn(key="basso", header="Limite basso", sortable=True),
                                DataTableColumn(key="alto", header="Limite alto", sortable=True),
                            ],
                            rows=Rx("table"),
                            search=True,
                            paginated=True,
                            page_size=12,
                        )
                    with Tab("Backtest", value="backtest"):
                        with If(Rx("backtest_text")):
                            Text(Rx("backtest_text"), css_class="text-sm whitespace-pre-wrap")
                        with If(~Rx("backtest_text")):
                            Text(
                                "Attiva «Esegui anche il backtest» e ricalcola per vedere quanto fidarsi.",
                                css_class="text-sm text-muted-foreground",
                            )
                with Row(gap=3, css_class="flex-wrap"):
                    Button(
                        "Invia i risultati a Claude", variant="default", on_click=SendMessage(Rx("message"))
                    )
                    Button(
                        "Salva in Excel",
                        variant="outline",
                        on_click=CallTool(
                            studio_export,
                            arguments={"forecast_id": Rx("forecast_id"), "format": "xlsx"},
                            on_success=ShowToast(f"Salvato: {RESULT.path}", variant="success"),
                            on_error=ShowToast(ERROR, variant="error"),
                        ),
                    )
    return PrefabApp(theme=THEME, title="Forecast Studio", view=view, state=state)


# ----------------------------------------------------------------------------- caricamento file dall'interfaccia


def _b64_size(b64: str) -> int:
    n = len(b64)
    return 0 if n == 0 else n * 3 // 4 - b64.count("=", max(0, n - 2))


@uploader.tool()
async def store_upload(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Riceve i file trascinati nell'interfaccia (base64), li legge e crea un dataset per ciascuno."""
    limit = settings.max_upload_mb * 1024 * 1024
    items: list[dict[str, Any]] = []
    for f in files:
        name = Path(str(f.get("name", "file"))).name
        item: dict[str, Any] = {"name": name, "ok": False, "dataset_id": "", "info": ""}
        try:
            if _b64_size(f.get("data", "")) > limit:
                raise ValueError(f"supera il limite di {settings.max_upload_mb} MB")
            try:
                raw = base64.b64decode(f.get("data", ""), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("contenuto non valido") from exc
            df = await anyio.to_thread.run_sync(partial(read_bytes, name, raw, settings.max_rows))
            uploads.put(name, df)  # sempre disponibile come source='upload:<nome>' per scelte manuali
            ds_id = datasets.new_id()
            try:
                ds = await anyio.to_thread.run_sync(
                    partial(
                        build_dataset, df, dataset_id=ds_id, name=Path(name).stem, source=f"upload {name}"
                    )
                )
            except ValueError as exc:
                item["info"] = (
                    f"letto ({len(df)} righe) ma non riconosciuto in automatico: {exc.args[0] if exc.args else exc} "
                    f"Usa load_dataset(source='upload:{name}', ...) con le colonne giuste."
                )
            else:
                datasets.put(ds_id, ds)
                item.update(
                    ok=True,
                    dataset_id=ds_id,
                    info=f"{len(ds.series)} serie · {ds.freq_label} · colonne: {', '.join(ds.value_columns[:6])}",
                )
        except ValueError as exc:
            item["info"] = str(exc.args[0]) if exc.args else str(exc)
        items.append(item)
    ok = [i for i in items if i["ok"]]
    parts = [f"{i['name']} → dataset {i['dataset_id']} ({i['info']})" for i in ok]
    parts += [f"{i['name']}: {i['info']}" for i in items if not i["ok"]]
    message = "L'utente ha caricato file dall'interfaccia. " + "; ".join(parts) + "."
    return {"items": items, "message": message, "loaded": len(ok)}


@uploader.ui(
    name="upload_dataset",
    description=(
        "Apre un pannello in cui l'utente trascina file CSV o Excel (fino a 25 MB) che diventano dataset senza passare "
        "dalla chat. Usalo quando l'utente non ha un percorso di file da indicare. Dopo averlo aperto, FERMATI e attendi: "
        "i dataset creati compaiono nel contesto e in list_session_items."
    ),
)
def upload_dataset() -> PrefabApp:
    with Card(css_class="max-w-2xl mx-auto") as view:
        with CardHeader():
            Heading("Carica i tuoi dati")
        with CardContent(), Column(gap=4):
            Text(
                "Trascina qui file CSV o Excel: diventano dataset pronti per la previsione, senza passare dalla chat.",
                css_class="text-sm text-muted-foreground",
            )
            DropZone(
                name="pending",
                icon="inbox",
                label="Trascina qui i file",
                description=f"CSV, TSV, TXT, Excel (.xlsx) · fino a {settings.max_upload_mb} MB",
                accept=UPLOAD_SUFFIXES,
                multiple=True,
                max_size=settings.max_upload_mb * 1024 * 1024,
            )
            with If(STATE.pending.length()), Column(gap=2):
                with ForEach("pending"), Row(gap=2, align="center"):
                    Text(Rx("$item.name"), css_class="text-sm")
                Button(
                    "Carica",
                    on_click=CallTool(
                        store_upload,
                        arguments={"files": Rx("pending")},
                        on_success=[
                            SetState("stored", RESULT.items),
                            SetState("pending", []),
                            UpdateContext(content=RESULT.message),
                            ShowToast("File elaborati.", variant="success"),
                        ],
                        on_error=ShowToast(ERROR, variant="error"),
                    ),
                )
            with If(STATE.stored.length()):
                Separator()
                with ForEach("stored") as f, Column(gap=1):
                    with Row(gap=2, align="center"):
                        Text(f.name, css_class="text-sm font-medium")
                        with If(f.ok):
                            Badge("pronto", variant="success")
                        with If(~f.ok):
                            Badge("da verificare", variant="warning")
                    Text(f.info, css_class="text-xs text-muted-foreground")
                    with If(f.ok):
                        Button(
                            "Apri nel Forecast Studio",
                            variant="outline",
                            size="sm",
                            on_click=SendMessage(f"Apri il Forecast Studio sul dataset {f.dataset_id}"),
                        )
    return PrefabApp(theme=THEME, title="Carica dati", view=view, state={"pending": [], "stored": []})
