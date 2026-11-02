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
import pandas as pd
from fastmcp import Context, FastMCP
from fastmcp.dependencies import Progress
from fastmcp.exceptions import ToolError
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.tools import ToolResult
from fastmcp.utilities.types import Image
from fastmcp_tasks import TasksExtension
from pydantic import BaseModel, Field

from timestral_mcp import models
from timestral_mcp.analytics import anomalies as an
from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics import batch as bm
from timestral_mcp.analytics import importance as im
from timestral_mcp.analytics import scenarios as sc
from timestral_mcp.analytics.calendar import FEATURES as CALENDAR_FEATURES
from timestral_mcp.analytics.calendar import add_calendar_features as build_calendar
from timestral_mcp.analytics.covariates import future_block
from timestral_mcp.analytics.decision import quantity_for_service_level as service_quantity
from timestral_mcp.analytics.decision import threshold_probability as threshold_prob
from timestral_mcp.analytics.forecast import run_forecast
from timestral_mcp.analytics.profile import profile_dataset as build_profile
from timestral_mcp.analytics.summary import backtest_summary, forecast_details, forecast_summary
from timestral_mcp.config import MODEL_SIZES, MODEL_WEIGHTS_MB
from timestral_mcp.data.ingest import Dataset, build_dataset, read_source, trailing_future_rows
from timestral_mcp.progress import Reporter, effective_seconds
from timestral_mcp.session import (
    anomaly_runs,
    backtests,
    batches,
    datasets,
    forecasts,
    importance_runs,
    resolve_forecast,
    settings,
)
from timestral_mcp.ui.charts import (
    anomalies_app,
    backtest_app,
    batch_app,
    forecast_app,
    forecast_png,
    importance_app,
    scenario_app,
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

Altri strumenti: forecast_many per molti prodotti/negozi insieme (poi batch_series_forecast per una serie); add_calendar_features per festività e calendario commerciale come covariate note; scenario_forecast per what-if (es. promozioni); threshold_probability e quantity_for_service_level per soglie e scorte; detect_anomalies per picchi, crolli e possibili rotture di stock; covariate_importance per capire quali covariate aiutano davvero (ablation). backtest, forecast_many, detect_anomalies e covariate_importance possono durare: se il client supporta i Task MCP girano in background, altrimenti entro un tempo massimo e dichiarano cosa non hanno eseguito. Le probabilità e i totali su più periodi sono approssimati dai quantili: dillo.
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
mcp.add_extension(TasksExtension())  # i tool con task=True girano in background se il client lo supporta

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
        "batches": [
            {"id": k, "target": v.target, "series": len(v.results), "horizon": v.horizon}
            for k, v in batches.items()
        ],
        "importance_runs": [
            {"id": k, "target": v.target, "series": v.series, "covariates": len(v.rows)}
            for k, v in importance_runs.items()
        ],
        "anomaly_runs": [
            {"id": k, "target": v.target, "series": v.series, "anomalies": len(v.anomalies)}
            for k, v in anomaly_runs.items()
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
    r = resolve_forecast(forecast_id)
    return forecast_details(r, start=start, end=end, quantiles=quantiles, aggregate=aggregate, how=how)


@mcp.tool(annotations={"readOnlyHint": True})
@user_errors
async def forecast_chart_image(forecast_id: str) -> Image:
    """Grafico PNG della previsione con bande di incertezza (per client senza interfacce interattive)."""
    r = resolve_forecast(forecast_id)
    data = await anyio.to_thread.run_sync(forecast_png, r)
    return Image(data=data, format="png")


@mcp.tool(app=True, task=True, annotations={"readOnlyHint": False})
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
    progress: Progress = Progress(),
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
    reporter = Reporter(progress, ctx)
    for i, origin in enumerate(plan.origins):
        await reporter.update(i, total, f"Finestra di test {i + 1}/{total}")
        folds.append(await anyio.to_thread.run_sync(bt.run_fold, plan, origin))
    await reporter.update(total, total, "Completato")
    bt_id = backtests.new_id()
    result = bt.finish_backtest(plan, folds, bt_id)
    backtests.put(bt_id, result)
    return ToolResult(content=backtest_summary(result), structured_content=backtest_app(result))


# --------------------------------------------------------------------------- più serie


ModelSize = Literal["5M", "28M", "123M", "209M"]
DownloadFlag = Annotated[
    bool,
    Field(
        description="Consenti il download dei pesi del modello se non è ancora sul disco. Usare solo dopo la conferma dell'utente."
    ),
]


@mcp.tool(app=True, task=True, annotations={"readOnlyHint": False})
@user_errors
async def forecast_many(
    dataset_id: str,
    target: Annotated[str, Field(description="Colonna numerica da prevedere, per ogni serie.")],
    horizon: Annotated[int, Field(ge=1, description="Periodi futuri da prevedere.")],
    series: Annotated[
        list[str] | None, Field(description="Serie da prevedere (default: tutte, fino al limite).")
    ] = None,
    model: ModelSize | None = None,
    interval: Annotated[float, Field(gt=0, lt=1)] = 0.8,
    past_covariates: list[str] | None = None,
    known_covariates: Annotated[
        list[str] | None,
        Field(description="Covariate note in anticipo: ogni serie deve avere righe future con i valori."),
    ] = None,
    with_backtest: Annotated[
        bool,
        Field(
            description="Calcola anche l'affidabilità di ogni serie con un backtest (fino a 30 serie, più lento)."
        ),
    ] = False,
    sort_by: Annotated[
        Literal["volume", "change", "uncertainty"], Field(description="Ordine della tabella riassuntiva.")
    ] = "volume",
    max_seconds: Annotated[
        int | None,
        Field(
            ge=10,
            le=3600,
            description="Tempo massimo di calcolo; le serie non eseguite sono elencate. Default: 90 s, oppure 900 s se gira come Task in background.",
        ),
    ] = None,
    allow_download: DownloadFlag = False,
    ctx: Context | None = None,
    progress: Progress = Progress(),
) -> ToolResult:
    """Prevede molte serie insieme (prodotti, negozi...): tabella riassuntiva con variazione, incertezza e segnalazioni, classifiche di attenzione e totale."""
    ds = datasets.get(dataset_id)
    size = models.normalize_size(model)
    models.ensure_available(size, allow_download)
    batch_id = batches.new_id()
    result = await anyio.to_thread.run_sync(
        partial(
            bm.run_batch,
            ds,
            batch_id=batch_id,
            target=target,
            horizon=horizon,
            series=series,
            model=size,
            interval=interval,
            past_covariates=past_covariates,
            known_covariates=known_covariates,
            with_backtest=with_backtest,
            max_series=settings.max_batch_series,
            max_seconds=effective_seconds(max_seconds),
            progress=Reporter(progress, ctx).from_thread(),
        )
    )
    batches.put(batch_id, result)
    return ToolResult(
        content=bm.batch_summary(result, sort_by), structured_content=batch_app(result, sort_by)
    )


@mcp.tool(app=True, annotations={"readOnlyHint": True})
@user_errors
async def batch_series_forecast(
    batch_id: str, series: Annotated[str, Field(description="Nome della serie del batch.")]
) -> ToolResult:
    """Grafico interattivo e numeri di una singola serie di una previsione su più serie."""
    r = resolve_forecast(bm.forecast_key(batch_id, series))
    return ToolResult(content=forecast_summary(r), structured_content=forecast_app(r))


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
@user_errors
async def export_batch(
    batch_id: str,
    format: Annotated[
        Literal["csv", "xlsx"], Field(description="xlsx: un file con due fogli; csv: due file.")
    ] = "xlsx",
) -> dict:
    """Salva la tabella riassuntiva e le previsioni (mediana e 21 quantili) di tutte le serie del batch."""
    b = batches.get(batch_id)
    out_dir = settings.export_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, long = bm.batch_frames(b)
    if format == "xlsx":
        path = out_dir / f"{b.id}.xlsx"
        with pd.ExcelWriter(path) as xl:
            summary.to_excel(xl, sheet_name="riepilogo", index=False)
            long.to_excel(xl, sheet_name="previsioni", index=False)
        files = [str(path)]
    else:
        a, c = out_dir / f"{b.id}_riepilogo.csv", out_dir / f"{b.id}_previsioni.csv"
        summary.to_csv(a, index=False)
        long.to_csv(c, index=False)
        files = [str(a), str(c)]
    return {"files": files, "series": len(summary), "forecast_rows": len(long)}


# --------------------------------------------------------------------------- decisioni


@mcp.tool(annotations={"readOnlyHint": False})
@user_errors
async def add_calendar_features(
    dataset_id: str,
    country: Annotated[
        str, Field(description="Codice paese ISO per le festività (IT, DE, FR, ES, GB, US...).")
    ] = "IT",
    subdiv: Annotated[
        str | None, Field(description="Regione/provincia/stato per festività locali (es. 'MI' per Milano).")
    ] = None,
    features: Annotated[
        list[str] | None,
        Field(description=f"Quali colonne creare (default tutte): {', '.join(CALENDAR_FEATURES)}."),
    ] = None,
    extend_future_periods: Annotated[
        int,
        Field(
            ge=0,
            le=3660,
            description="Righe future da aggiungere dopo l'ultima riga, con le colonne di calendario (per usarle come covariate note).",
        ),
    ] = 0,
) -> dict:
    """Crea un nuovo dataset con covariate di calendario (festività, ponti, Black Friday, periodo natalizio, Pasqua), anche per il futuro. Solo dati giornalieri o più frequenti."""
    ds = datasets.get(dataset_id)
    new_id = datasets.new_id()
    new_ds, info = await anyio.to_thread.run_sync(
        partial(
            build_calendar,
            ds,
            new_id=new_id,
            country=country,
            subdiv=subdiv,
            features=features,
            extend_future_periods=extend_future_periods,
        )
    )
    datasets.put(new_id, new_ds)
    return info


class ScenarioChange(BaseModel):
    column: str = Field(description="Covariata nota da modificare (es. 'promo').")
    value: float = Field(description="Nuovo valore (es. 1 per attivare un indicatore 0/1).")
    start: str | None = Field(
        default=None, description="Prima data modificata (YYYY-MM-DD); default: inizio orizzonte."
    )
    end: str | None = Field(
        default=None, description="Ultima data modificata inclusa; default: fine orizzonte."
    )


@mcp.tool(app=True, annotations={"readOnlyHint": False})
@user_errors
async def scenario_forecast(
    dataset_id: str,
    target: str,
    horizon: Annotated[int, Field(ge=1)],
    known_covariates: Annotated[
        list[str], Field(description="Covariate note in anticipo (servono righe future nel dataset).")
    ],
    scenarios: Annotated[
        dict[str, list[ScenarioChange]],
        Field(
            description="Scenari what-if: nome → cambi alle covariate future. Es. {'promo_10_16_feb': [{'column':'promo','value':1,'start':'2026-02-10','end':'2026-02-16'}]}. Max 5."
        ),
    ],
    series: str | None = None,
    model: ModelSize | None = None,
    interval: Annotated[float, Field(gt=0, lt=1)] = 0.8,
    past_covariates: list[str] | None = None,
    allow_download: DownloadFlag = False,
) -> ToolResult:
    """What-if: confronta la previsione base con scenari in cui cambiano i valori futuri di covariate note (es. 'e se faccio una promozione?'). Mostra grafico e differenze."""
    ds = datasets.get(dataset_id)
    size = models.normalize_size(model)
    models.ensure_available(size, allow_download)
    result = await anyio.to_thread.run_sync(
        partial(
            sc.run_scenarios,
            ds,
            target=target,
            horizon=horizon,
            series=series,
            model=size,
            interval=interval,
            past_covariates=past_covariates,
            known_covariates=known_covariates,
            id_factory=forecasts.new_id,
            scenarios={n: [c.model_dump() for c in chs] for n, chs in scenarios.items()},
        )
    )
    for r in (result.base, *result.scenarios.values()):
        forecasts.put(r.id, r)
    return ToolResult(content=sc.scenario_summary(result), structured_content=scenario_app(result))


@mcp.tool(annotations={"readOnlyHint": True})
@user_errors
async def threshold_probability(
    forecast_id: str,
    threshold: Annotated[float, Field(description="Soglia da confrontare (es. 150).")],
    direction: Annotated[
        Literal["below", "above"],
        Field(description="below: probabilità di stare sotto la soglia; above: sopra."),
    ] = "below",
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Probabilità di essere sotto/sopra una soglia in ciascun periodo, interpolata tra i quantili (con limiti espliciti nelle code)."""
    return threshold_prob(resolve_forecast(forecast_id), threshold, direction, start, end)


@mcp.tool(annotations={"readOnlyHint": True})
@user_errors
async def quantity_for_service_level(
    forecast_id: str,
    service_level: Annotated[
        float, Field(ge=0.5, le=0.99, description="Quota di casi da coprire, es. 0.9 = 90%.")
    ] = 0.9,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Quantità necessaria per coprire una data quota di casi (es. scorte): per periodo e totale sulla finestra (lead time), con intervallo cautelativo/indipendente."""
    return service_quantity(resolve_forecast(forecast_id), service_level, start, end)


@mcp.tool(app=True, task=True, annotations={"readOnlyHint": False})
@user_errors
async def detect_anomalies(
    dataset_id: str,
    target: str,
    series: str | None = None,
    model: ModelSize | None = None,
    interval: Annotated[
        float, Field(gt=0, lt=1, description="Intervallo atteso: fuori da questo è anomalia (default 0.98).")
    ] = 0.98,
    lookback: Annotated[
        int | None, Field(ge=1, description="Quanti periodi recenti analizzare (default 365).")
    ] = None,
    block: Annotated[
        int | None,
        Field(ge=2, description="Periodi per finestra di previsione (default: ciclo stagionale, almeno 14)."),
    ] = None,
    past_covariates: list[str] | None = None,
    known_covariates: Annotated[
        list[str] | None,
        Field(description="Covariate che spiegano i picchi (es. promo): così non risultano anomalie."),
    ] = None,
    allow_download: DownloadFlag = False,
    ctx: Context | None = None,
    progress: Progress = Progress(),
) -> ToolResult:
    """Trova valori reali fuori dall'intervallo atteso (picchi, crolli, possibili rotture di stock), confrontandoli con previsioni fatte solo sui dati precedenti."""
    ds = datasets.get(dataset_id)
    size = models.normalize_size(model)
    models.ensure_available(size, allow_download)
    plan = an.plan_anomalies(
        ds,
        target=target,
        series=series,
        model=size,
        interval=interval,
        lookback=lookback,
        block=block,
        past_covariates=past_covariates,
        known_covariates=known_covariates,
    )
    parts = []
    reporter = Reporter(progress, ctx)
    for i, origin in enumerate(plan.origins):
        await reporter.update(i, len(plan.origins), f"Finestra {i + 1}/{len(plan.origins)}")
        parts.append(await anyio.to_thread.run_sync(an.run_window, plan, origin))
    await reporter.update(len(plan.origins), len(plan.origins), "Completato")
    run_id = anomaly_runs.new_id()
    result = an.finish_anomalies(plan, parts, run_id)
    anomaly_runs.put(run_id, result)
    return ToolResult(content=an.anomaly_summary(result), structured_content=anomalies_app(result))


@mcp.tool(app=True, task=True, annotations={"readOnlyHint": False})
@user_errors
async def covariate_importance(
    dataset_id: str,
    target: str,
    horizon: Annotated[int, Field(ge=1, description="Periodi previsti in ogni finestra di test.")],
    series: str | None = None,
    model: ModelSize | None = None,
    known_covariates: Annotated[
        list[str] | None,
        Field(
            description="Covariate note in anticipo da valutare. Se né queste né le passate sono indicate, le sceglie il server."
        ),
    ] = None,
    past_covariates: Annotated[
        list[str] | None, Field(description="Covariate note solo nel passato da valutare.")
    ] = None,
    n_folds: Annotated[
        int,
        Field(
            ge=2,
            le=8,
            description="Finestre di test (default 4): più finestre, risultato più affidabile ma più lento.",
        ),
    ] = 4,
    interval: Annotated[float, Field(gt=0, lt=1)] = 0.8,
    max_seconds: Annotated[
        int | None,
        Field(
            ge=10,
            le=3600,
            description="Tempo massimo. Default: 90 s, oppure 900 s se gira come Task in background.",
        ),
    ] = None,
    allow_download: DownloadFlag = False,
    ctx: Context | None = None,
    progress: Progress = Progress(),
) -> ToolResult:
    """Importanza delle covariate per ablation: quanto peggiora la previsione sul passato (backtest) se si toglie ciascuna covariata. Misura l'utilità predittiva, non la causalità."""
    ds = datasets.get(dataset_id)
    size = models.normalize_size(model)
    models.ensure_available(size, allow_download)
    plan = im.plan_importance(
        ds,
        target=target,
        horizon=horizon,
        series=series,
        model=size,
        n_folds=n_folds,
        interval=interval,
        known_covariates=known_covariates,
        past_covariates=past_covariates,
    )
    run_id = importance_runs.new_id()
    result = await anyio.to_thread.run_sync(
        partial(
            im.run_importance,
            plan,
            run_id,
            max_seconds=effective_seconds(max_seconds),
            progress=Reporter(progress, ctx).from_thread(),
        )
    )
    importance_runs.put(run_id, result)
    return ToolResult(content=im.importance_summary(result), structured_content=importance_app(result))


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
    r = resolve_forecast(forecast_id)
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
    return forecast_summary(resolve_forecast(forecast_id))


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
