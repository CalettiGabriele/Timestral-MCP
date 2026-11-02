"""Server MCP: previsioni con Timestral, analisi e grafici interattivi."""

from __future__ import annotations

import functools
import json
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import anyio
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from pydantic import Field

from timestral_mcp import models
from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics.covariates import future_block
from timestral_mcp.analytics.forecast import run_forecast
from timestral_mcp.analytics.profile import profile_dataset as build_profile
from timestral_mcp.analytics.summary import backtest_summary, forecast_details, forecast_summary
from timestral_mcp.config import MODEL_SIZES, MODEL_WEIGHTS_MB
from timestral_mcp.data.ingest import Dataset, build_dataset, read_source, trailing_future_rows
from timestral_mcp.session import (
    backtests,
    datasets,
    forecasts,
    settings,
)
from timestral_mcp.ui.charts import (
    backtest_app,
    forecast_app,
    forecast_png,
)

INSTRUCTIONS = """\
Server per previsioni probabilistiche di serie temporali con il modello Timestral di Mistral.

Flusso consigliato: 1) load_dataset (percorso di un file CSV/Excel oppure dati incollati); \
2) profile_dataset per controllare la qualità; 3) backtest per capire quanto fidarsi; \
4) forecast per la previsione con grafico interattivo; 5) get_forecast_details per rispondere a \
domande su periodi, quantili o totali mensili; 6) export_results per salvare i numeri.

Non differenziare né imputare i dati prima del modello: Timestral gestisce valori mancanti e scala da solo. \
Rispondi alle domande sui grafici usando i numeri restituiti dai tool, non indovinando dall'immagine. \
Consulta le skill del server (forecasting-workflow, timestral-specifics, forecast-interpretation, sales-forecasting).

Prima di commentare l'effetto di covariate note in anticipo (es. promozioni), leggi i loro valori futuri \
nel testo della previsione ("Valori futuri usati dal modello"): non assumerli. \
Per un totale mensile completo controlla `partial_periods` in get_forecast_details ed estendi l'orizzonte se serve. \
Se un modello non è ancora scaricato, forecast/backtest lo segnalano: chiedi conferma all'utente prima di ripetere \
con allow_download=true (il download può richiedere minuti).
"""


def _preload() -> None:
    try:
        if models.is_cached(settings.default_model):  # niente download silenziosi all'avvio
            models.get_forecaster(settings.default_model)
    except Exception:  # noqa: BLE001 - l'errore emergerà al primo uso con un messaggio chiaro
        pass


@asynccontextmanager
async def lifespan(_server: FastMCP):
    if settings.preload and models._factory is None:
        threading.Thread(target=_preload, name="timestral-preload", daemon=True).start()
    yield


mcp = FastMCP("Timestral", instructions=INSTRUCTIONS, lifespan=lifespan)

_skills_root = settings.skills_dir or Path(__file__).resolve().parents[2] / "skills"
_skill_roots = [p for p in (_skills_root / "core", _skills_root / "verticals" / "sales") if p.is_dir()]
if _skill_roots:
    mcp.add_provider(SkillsDirectoryProvider(roots=_skill_roots))


def user_errors(fn):
    """Trasforma gli errori d'uso in messaggi leggibili per l'utente/LLM."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise ToolError(str(exc.args[0]) if exc.args else str(exc)) from exc

    return wrapper


# --------------------------------------------------------------------------- dati


@mcp.tool(annotations={"readOnlyHint": False, "idempotentHint": False})
@user_errors
async def load_dataset(
    source: Annotated[
        str,
        Field(
            description="Percorso di un file CSV/Excel sul computer (es. ~/Documents/vendite.csv) oppure dati CSV incollati come testo."
        ),
    ],
    name: Annotated[str | None, Field(description="Nome descrittivo del dataset.")] = None,
    date_column: Annotated[
        str | None, Field(description="Colonna con le date; se omessa viene riconosciuta.")
    ] = None,
    series_column: Annotated[
        str | None,
        Field(
            description="Colonna che identifica la serie (es. prodotto o negozio) se il file ne contiene più di una in formato lungo."
        ),
    ] = None,
    value_columns: Annotated[
        list[str] | None, Field(description="Colonne numeriche da tenere; se omesse tutte quelle numeriche.")
    ] = None,
    sheet: Annotated[str | None, Field(description="Nome del foglio per i file Excel.")] = None,
    dayfirst: Annotated[
        bool | None, Field(description="True se le date sono giorno/mese/anno, False se mese/giorno/anno.")
    ] = None,
    aggregate_duplicates: Annotated[
        Literal["sum", "mean", "last"] | None,
        Field(description="Come combinare più righe con la stessa data."),
    ] = None,
    ctx: Context | None = None,
) -> dict:
    """Carica i dati e li prepara per la previsione. Restituisce un dataset_id da usare negli altri tool."""
    df, origin = await anyio.to_thread.run_sync(partial(read_source, source, sheet, settings.max_rows))
    ds_id = datasets.new_id()
    ds = await anyio.to_thread.run_sync(
        partial(
            build_dataset,
            df,
            dataset_id=ds_id,
            name=name or (Path(origin).stem if origin != "testo incollato" else "dati incollati"),
            source=origin,
            date_column=date_column,
            series_column=series_column,
            value_columns=value_columns,
            dayfirst=dayfirst,
            aggregate_duplicates=aggregate_duplicates,
        )
    )
    datasets.put(ds_id, ds)
    return _dataset_overview(ds)


def _dataset_overview(ds: Dataset) -> dict:
    series = []
    for key, df in list(ds.series.items())[:20]:
        series.append(
            {
                "series": key,
                "start": f"{df.index.min():%Y-%m-%d}",
                "end": f"{df.index.max():%Y-%m-%d}",
                "rows": int(len(df)),
                "rows_without_value_at_end": {c: trailing_future_rows(df, c) for c in ds.value_columns},
                "future_covariates": future_block(df, ds.freq),
            }
        )
    return {
        "dataset_id": ds.id,
        "name": ds.name,
        "source": ds.source,
        "frequency": ds.freq,
        "frequency_label": ds.freq_label,
        "date_column": ds.date_column,
        "series_column": ds.series_column,
        "numeric_columns": ds.value_columns,
        "n_series": len(ds.series),
        "series": series,
        "warnings": ds.warnings,
        "next_steps": "Usa profile_dataset per i controlli di qualità, poi backtest e forecast con questo dataset_id.",
    }


@mcp.tool(annotations={"readOnlyHint": True})
@user_errors
async def profile_dataset(
    dataset_id: str,
    series: Annotated[
        str | None, Field(description="Una sola serie da analizzare (default: tutte, max 20).")
    ] = None,
) -> dict:
    """Profilo di qualità dei dati: periodo, valori mancanti, zeri, anomalie, statistiche e avvisi."""
    ds = datasets.get(dataset_id)
    return await anyio.to_thread.run_sync(partial(build_profile, ds, series))


@mcp.tool(annotations={"readOnlyHint": True})
async def list_session_items() -> dict:
    """Elenca dataset, previsioni e backtest disponibili in questa sessione (utile se si perde un id)."""
    return {
        "datasets": [
            {"id": k, "name": v.name, "series": len(v.series), "columns": v.value_columns}
            for k, v in datasets.items()
        ],
        "forecasts": [
            {"id": k, "target": v.target, "series": v.series, "horizon": v.horizon, "model": v.model}
            for k, v in forecasts.items()
        ],
        "backtests": [
            {"id": k, "target": v.target, "series": v.series, "horizon": v.horizon}
            for k, v in backtests.items()
        ],
    }


# --------------------------------------------------------------------------- previsione


@mcp.tool(app=True, annotations={"readOnlyHint": False})
@user_errors
async def forecast(
    dataset_id: str,
    target: Annotated[str, Field(description="Colonna numerica da prevedere.")],
    horizon: Annotated[
        int, Field(ge=1, description="Quanti periodi futuri prevedere (nella frequenza del dataset).")
    ],
    series: Annotated[
        str | None, Field(description="Serie da prevedere se il dataset ne contiene più di una.")
    ] = None,
    model: Annotated[
        Literal["5M", "28M", "123M", "209M"] | None,
        Field(description="Taglia del modello (default: quella configurata, 209M = più accurato)."),
    ] = None,
    interval: Annotated[
        float, Field(gt=0, lt=1, description="Intervallo di incertezza centrale mostrato, es. 0.8 = 80%.")
    ] = 0.8,
    past_covariates: Annotated[
        list[str] | None, Field(description="Colonne note solo nel passato (es. meteo osservato).")
    ] = None,
    known_covariates: Annotated[
        list[str] | None,
        Field(
            description="Colonne note anche per il futuro (es. promozioni pianificate): servono righe future con target vuoto."
        ),
    ] = None,
    context_length: Annotated[
        int | None, Field(ge=8, description="Massimo storico da usare (default: tutto, fino a 32768).")
    ] = None,
    allow_download: Annotated[
        bool,
        Field(
            description="Consenti il download dei pesi del modello se non è ancora sul disco. Usare solo dopo la conferma dell'utente."
        ),
    ] = False,
) -> ToolResult:
    """Previsione probabilistica con Timestral. Mostra un grafico interattivo; il testo contiene i numeri per rispondere alle domande."""
    ds = datasets.get(dataset_id)
    size = models.normalize_size(model)
    models.ensure_available(size, allow_download)
    fc_id = forecasts.new_id()
    result = await anyio.to_thread.run_sync(
        partial(
            run_forecast,
            ds,
            forecast_id=fc_id,
            target=target,
            horizon=horizon,
            series=series,
            model=size,
            interval=interval,
            past_covariates=past_covariates,
            known_covariates=known_covariates,
            context_length=context_length,
        )
    )
    forecasts.put(fc_id, result)
    return ToolResult(content=forecast_summary(result), structured_content=forecast_app(result))


@mcp.tool(annotations={"readOnlyHint": True})
@user_errors
async def get_forecast_details(
    forecast_id: str,
    start: Annotated[str | None, Field(description="Data iniziale (YYYY-MM-DD).")] = None,
    end: Annotated[str | None, Field(description="Data finale (YYYY-MM-DD).")] = None,
    quantiles: Annotated[
        list[float] | None, Field(description="Quantili richiesti, tra 0.01 e 0.99 (es. [0.05, 0.5, 0.95]).")
    ] = None,
    aggregate: Annotated[
        Literal["week", "month", "quarter", "year"] | None,
        Field(description="Aggrega per settimana/mese/trimestre/anno."),
    ] = None,
    how: Annotated[Literal["sum", "mean"], Field(description="Somma o media nell'aggregazione.")] = "sum",
) -> dict:
    """Valori numerici di una previsione per un periodo, con quantili a scelta e totali aggregati (approssimati per gli intervalli)."""
    r = forecasts.get(forecast_id)
    return forecast_details(r, start=start, end=end, quantiles=quantiles, aggregate=aggregate, how=how)


@mcp.tool(annotations={"readOnlyHint": True})
@user_errors
async def forecast_chart_image(forecast_id: str) -> Image:
    """Grafico PNG della previsione con bande di incertezza (per client senza interfacce interattive)."""
    r = forecasts.get(forecast_id)
    data = await anyio.to_thread.run_sync(forecast_png, r)
    return Image(data=data, format="png")


@mcp.tool(app=True, annotations={"readOnlyHint": False})
@user_errors
async def backtest(
    dataset_id: str,
    target: str,
    horizon: Annotated[int, Field(ge=1, description="Periodi previsti in ogni finestra di test.")],
    series: str | None = None,
    model: Literal["5M", "28M", "123M", "209M"] | None = None,
    n_folds: Annotated[
        int, Field(ge=1, le=12, description="Numero di finestre di test consecutive (default 3).")
    ] = 3,
    interval: Annotated[float, Field(gt=0, lt=1)] = 0.8,
    past_covariates: list[str] | None = None,
    known_covariates: list[str] | None = None,
    allow_download: Annotated[
        bool,
        Field(
            description="Consenti il download dei pesi del modello se non è ancora sul disco. Usare solo dopo la conferma dell'utente."
        ),
    ] = False,
    ctx: Context | None = None,
) -> ToolResult:
    """Valida il modello sul passato: prevede le ultime finestre usando solo i dati precedenti e confronta con i valori reali e con una baseline stagionale."""
    ds = datasets.get(dataset_id)
    models.ensure_available(models.normalize_size(model), allow_download)
    plan = bt.plan_backtest(
        ds,
        target=target,
        horizon=horizon,
        series=series,
        model=model,
        n_folds=n_folds,
        interval=interval,
        past_covariates=past_covariates,
        known_covariates=known_covariates,
    )
    folds = []
    total = len(plan.origins)
    for i, origin in enumerate(plan.origins):
        if ctx:
            await ctx.report_progress(i, total, f"Finestra di test {i + 1}/{total}")
        folds.append(await anyio.to_thread.run_sync(bt.run_fold, plan, origin))
    if ctx:
        await ctx.report_progress(total, total, "Completato")
    bt_id = backtests.new_id()
    result = bt.finish_backtest(plan, folds, bt_id)
    backtests.put(bt_id, result)
    return ToolResult(content=backtest_summary(result), structured_content=backtest_app(result))


# --------------------------------------------------------------------------- esportazione


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
@user_errors
async def export_results(
    forecast_id: str,
    format: Annotated[Literal["csv", "xlsx"], Field(description="Formato del file.")] = "csv",
    filename: Annotated[
        str | None, Field(description="Nome del file (senza cartella); default: l'id della previsione.")
    ] = None,
) -> dict:
    """Salva mediana e tutti i 21 quantili della previsione in un file nella cartella di esportazione."""
    r = forecasts.get(forecast_id)
    out_dir = settings.export_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename or r.id).name
    path = out_dir / (base if base.lower().endswith(f".{format}") else f"{base}.{format}")
    df = r.to_frame().reset_index()
    if format == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False, sheet_name="previsione")
    return {"path": str(path), "rows": len(df), "columns": list(df.columns)}


# --------------------------------------------------------------------------- resource e prompt


@mcp.resource("timestral://models")
def models_info() -> str:
    """Taglie disponibili del modello, con dimensioni e limiti."""
    return json.dumps(
        {
            "models": {
                size: {
                    "weights_mb": MODEL_WEIGHTS_MB[size],
                    "downloaded": models.is_cached(size),
                    "note": note,
                }
                for size, note in {
                    "5M": "il più veloce, accuratezza inferiore",
                    "28M": "ottimo compromesso velocità/accuratezza",
                    "123M": "quasi come 209M",
                    "209M": "il più accurato",
                }.items()
            },
            "default": settings.default_model,
            "max_context": settings.max_context,
            "max_horizon_per_block": settings.model_horizon_block,
            "quantile_levels": "0.01, 0.05, 0.10, 0.15 … 0.90, 0.95, 0.99",
            "sizes": list(MODEL_SIZES),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.resource("dataset://{dataset_id}")
def dataset_resource(dataset_id: str) -> str:
    """Profilo di un dataset caricato."""
    return json.dumps(build_profile(datasets.get(dataset_id)), ensure_ascii=False, indent=2)


@mcp.resource("forecast://{forecast_id}")
def forecast_resource(forecast_id: str) -> str:
    """Riassunto testuale di una previsione."""
    return forecast_summary(forecasts.get(forecast_id))


@mcp.prompt
def previsione_guidata(percorso_file: str, obiettivo: str = "") -> str:
    """Guida passo passo da un file di dati a una previsione validata."""
    goal = f"\nObiettivo dell'utente: {obiettivo}" if obiettivo else ""
    return (
        f"Aiutami a prevedere i dati nel file {percorso_file}.{goal}\n\n"
        "Procedi così, riportandomi i risultati a ogni passo: 1) carica il file con load_dataset e dimmi "
        "cosa hai riconosciuto (date, frequenza, serie, colonne); se qualcosa è ambiguo chiedimelo; "
        "2) esegui profile_dataset e segnalami problemi di qualità; 3) chiedimi colonna da prevedere, "
        "orizzonte e covariate; 4) esegui un backtest e spiegami quanto fidarmi; 5) esegui la previsione "
        "e commenta trend, picchi e incertezza."
    )


def main() -> None:
    import sys

    if sys.argv[1:2] == ["download"]:  # uv run timestral-mcp download 28M 209M
        sizes = sys.argv[2:] or [settings.default_model]
        for size in sizes:
            print(f"Scarico Timestral-{models.normalize_size(size)}...", file=sys.stderr)
            print(models.download(size))
        return
    mcp.run()


if __name__ == "__main__":
    main()
