"""Scenari what-if: cambiano i valori futuri delle covariate note e si confrontano le previsioni."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from timestral_mcp.analytics.forecast import ForecastResult, run_forecast
from timestral_mcp.analytics.summary import _date, fmt
from timestral_mcp.data.ingest import Dataset

BASE = "base"


@dataclass
class ScenarioResult:
    target: str
    series: str
    horizon: int
    base: ForecastResult
    scenarios: dict[str, ForecastResult]
    comparison: list[dict[str, Any]]
    changes: dict[str, list[dict[str, Any]]]
    warnings: list[str] = field(default_factory=list)


def _validate_change(ch: dict[str, Any], known: list[str], future: pd.DatetimeIndex) -> dict[str, Any]:
    col = ch.get("column")
    if col not in known:
        raise ValueError(
            f"La covariata {col!r} non è tra le known_covariates ({', '.join(known)}): "
            "gli scenari possono cambiare solo covariate note in anticipo."
        )
    try:
        value = float(ch["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cambio su {col!r}: serve un 'value' numerico.") from exc
    start = pd.Timestamp(ch["start"]) if ch.get("start") else future[0]
    end = pd.Timestamp(ch["end"]) if ch.get("end") else future[-1]
    if end < future[0] or start > future[-1]:
        raise ValueError(
            f"Cambio su {col!r}: {start:%Y-%m-%d} → {end:%Y-%m-%d} è fuori dall'orizzonte della previsione "
            f"({future[0]:%Y-%m-%d} → {future[-1]:%Y-%m-%d})."
        )
    if end < start:
        raise ValueError(f"Cambio su {col!r}: la data finale precede quella iniziale.")
    return {"column": col, "value": value, "start": max(start, future[0]), "end": min(end, future[-1])}


def apply_changes(ds: Dataset, key: str, changes: list[dict[str, Any]]) -> Dataset:
    df = ds.series[key].copy()
    for ch in changes:
        df.loc[(df.index >= ch["start"]) & (df.index <= ch["end"]), ch["column"]] = ch["value"]
    return dataclasses.replace(ds, series={**ds.series, key: df})


def _history_warnings(
    ds: Dataset, key: str, target: str, name: str, changes: list[dict[str, Any]]
) -> list[str]:
    df = ds.series[key]
    last = df[target].last_valid_index()
    hist = df.loc[:last]
    out = []
    for ch in changes:
        s = hist[ch["column"]].dropna()
        if s.empty:
            continue
        seen = int((s == ch["value"]).sum())
        if set(np.unique(s)) <= {0.0, 1.0}:
            if ch["value"] == 1.0 and seen < 5:
                out.append(
                    f"Scenario '{name}': '{ch['column']}' era attiva solo {seen} volte nello storico: "
                    "l'effetto stimato dal modello è poco affidabile."
                )
        elif not (s.min() <= ch["value"] <= s.max()):
            out.append(
                f"Scenario '{name}': {ch['column']}={ch['value']:g} è fuori dall'intervallo osservato nello storico "
                f"({s.min():g}–{s.max():g}): il modello estrapola."
            )
    return out


def run_scenarios(
    ds: Dataset,
    *,
    target: str,
    horizon: int,
    scenarios: dict[str, list[dict[str, Any]]],
    known_covariates: list[str],
    id_factory,
    series: str | None = None,
    model: str | None = None,
    interval: float = 0.8,
    past_covariates: list[str] | None = None,
    context_length: int | None = None,
    max_scenarios: int = 5,
) -> ScenarioResult:
    if not known_covariates:
        raise ValueError(
            "Servono known_covariates: gli scenari modificano i valori futuri di covariate note in anticipo "
            "(es. promozioni, prezzi). Indica quali colonne usare."
        )
    if not scenarios:
        raise ValueError(
            "Indica almeno uno scenario, es. {'con_promo': [{'column': 'promo', 'value': 1, 'start': '2026-02-10', 'end': '2026-02-16'}]}."
        )
    if len(scenarios) > max_scenarios:
        raise ValueError(f"Massimo {max_scenarios} scenari per confronto (ricevuti {len(scenarios)}).")
    if BASE in scenarios:
        raise ValueError(f"Il nome '{BASE}' è riservato allo scenario di partenza.")
    key, _ = ds.get_series(series)
    kw = dict(
        target=target,
        horizon=horizon,
        series=key,
        model=model,
        interval=interval,
        past_covariates=past_covariates,
        known_covariates=known_covariates,
        context_length=context_length,
    )
    base = run_forecast(ds, forecast_id=id_factory(), **kw)
    results: dict[str, ForecastResult] = {}
    parsed: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for name, changes in scenarios.items():
        if not changes:
            raise ValueError(f"Lo scenario {name!r} non contiene cambi.")
        ch = [_validate_change(c, known_covariates, base.future_index) for c in changes]
        parsed[name] = ch
        warnings += _history_warnings(ds, key, target, name, ch)
        results[name] = run_forecast(apply_changes(ds, key, ch), forecast_id=id_factory(), **kw)

    base_total = float(base.median.sum())
    rows = [_row(BASE, base, base, base_total, None)]
    rows += [_row(n, r, base, base_total, parsed[n]) for n, r in results.items()]
    return ScenarioResult(target, key, horizon, base, results, rows, parsed, warnings)


def _row(name: str, r: ForecastResult, base: ForecastResult, base_total: float, changes) -> dict[str, Any]:
    total = float(r.median.sum())
    row: dict[str, Any] = {
        "scenario": name,
        "forecast_id": r.id,
        "total_median": round(total, 2),
        "delta_vs_base": round(total - base_total, 2),
        "delta_vs_base_pct": None if base_total == 0 else round(100 * (total / base_total - 1), 2),
        "peak": round(float(r.median.max()), 2),
        "peak_date": _date(r.future_index[int(np.argmax(r.median))], r.freq),
    }
    if changes:
        a, b = min(c["start"] for c in changes), max(c["end"] for c in changes)
        mask = (r.future_index >= a) & (r.future_index <= b)
        base_w, scen_w = float(base.median[mask].sum()), float(r.median[mask].sum())
        row["changed_window"] = f"{_date(a, r.freq)} → {_date(b, r.freq)}"
        row["window_delta"] = round(scen_w - base_w, 2)
        row["window_delta_pct"] = None if base_w == 0 else round(100 * (scen_w / base_w - 1), 2)
        # effetto dopo la finestra (anticipo/rinvio degli acquisti): periodi successivi alla modifica
        after = r.future_index > b
        row["after_window_delta"] = (
            round(float(r.median[after].sum() - base.median[after].sum()), 2) if after.any() else 0.0
        )
    return row


def scenario_summary(sr: ScenarioResult) -> str:
    base = sr.base
    lines = [
        f"Confronto scenari · target '{sr.target}' · serie '{sr.series}' · {sr.horizon} periodi "
        f"({_date(base.future_index[0], base.freq)} → {_date(base.future_index[-1], base.freq)}) · Timestral-{base.model}",
        "Totali = somma delle mediane sull'intero orizzonte. Gli effetti sono quelli che il modello stima dalla relazione "
        "storica con la covariata: non sono una prova di causalità.",
        "",
        "scenario | totale | vs base | vs base % | finestra modificata | effetto nella finestra | dopo la finestra",
    ]
    for r in sr.comparison:
        if r["scenario"] == BASE:
            lines.append(f"base | {fmt(r['total_median'])} | - | - | - | - | -")
        else:
            wp = r["window_delta_pct"]
            lines.append(
                f"{r['scenario']} | {fmt(r['total_median'])} | {r['delta_vs_base']:+,.0f} | "
                f"{'n/d' if r['delta_vs_base_pct'] is None else f'{r["delta_vs_base_pct"]:+.1f}%'} | {r['changed_window']} | "
                f"{r['window_delta']:+,.0f} ({'n/d' if wp is None else f'{wp:+.1f}%'}) | {r['after_window_delta']:+,.0f}"
            )
    for name, chs in sr.changes.items():
        desc = "; ".join(
            f"{c['column']}={c['value']:g} dal {c['start']:%Y-%m-%d} al {c['end']:%Y-%m-%d}" for c in chs
        )
        lines.append(f"Scenario '{name}': {desc}.")
    ids = ", ".join(f"{r['scenario']}={r['forecast_id']}" for r in sr.comparison)
    lines.append(f"Previsioni complete (per get_forecast_details): {ids}.")
    for w in sr.warnings + base.warnings:
        lines.append(f"ATTENZIONE: {w}")
    return "\n".join(lines)
