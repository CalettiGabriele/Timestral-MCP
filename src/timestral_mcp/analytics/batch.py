"""Previsione su molte serie insieme (prodotti, negozi...) con tabella riassuntiva e classifiche."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics.covariates import fmt_date
from timestral_mcp.analytics.forecast import ForecastResult, run_forecast
from timestral_mcp.analytics.summary import fmt
from timestral_mcp.data.ingest import Dataset

MIN_NOTABLE_CHANGE_PCT = 5  # sotto questa variazione non è una segnalazione
SHORT_HISTORY_FLAG = "storico corto"
ZEROS_FLAG = "molti zeri"
UNCERTAIN_FLAG = "incertezza alta"
DROP_FLAG = "calo forte"
RISE_FLAG = "crescita forte"
SortBy = Literal["volume", "change", "uncertainty"]


@dataclass
class BatchResult:
    id: str
    dataset_id: str
    dataset_name: str
    target: str
    horizon: int
    model: str
    freq: str
    interval: float
    rows: list[dict[str, Any]]
    results: dict[str, ForecastResult]
    skipped: dict[str, str]
    not_run: list[str] = field(default_factory=list)
    known_covariates: list[str] = field(default_factory=list)
    past_covariates: list[str] = field(default_factory=list)
    with_backtest: bool = False
    warnings: list[str] = field(default_factory=list)


def select_series(
    ds: Dataset, target: str, horizon: int, series: list[str] | None, max_series: int
) -> tuple[list[str], list[str], list[str]]:
    """(da prevedere, escluse per limite, non trovate). Con troppe serie tiene le più grandi per volume recente."""
    if target not in ds.value_columns:
        raise ValueError(f"Target {target!r} non trovato. Colonne numeriche: {', '.join(ds.value_columns)}")
    if series:
        missing = [s for s in series if s not in ds.series]
        keys = [s for s in series if s in ds.series]
    else:
        missing, keys = [], list(ds.series)
    if not keys:
        raise ValueError(f"Nessuna serie valida. Disponibili (prime 20): {', '.join(list(ds.series)[:20])}")

    def volume(k: str) -> float:
        s = ds.series[k][target].dropna()
        return float(s.iloc[-horizon:].abs().sum()) if len(s) else 0.0

    if len(keys) > max_series:
        ranked = sorted(keys, key=volume, reverse=True)
        return ranked[:max_series], ranked[max_series:], missing
    return keys, [], missing


def _previous_total(r: ForecastResult, horizon: int) -> float | None:
    """Somma degli ultimi `horizon` valori osservati (None se troppo incompleti)."""
    tail = r.history.iloc[-horizon:]
    if len(tail) < horizon or tail.notna().mean() < 0.8:
        return None
    return float(tail.sum(skipna=True))


def series_row(ds: Dataset, r: ForecastResult, backtest: bt.BacktestResult | None = None) -> dict[str, Any]:
    med, lo, hi = r.median, r.lower, r.upper
    total = float(med.sum())
    prev = _previous_total(r, r.horizon)
    change = None if prev in (None, 0) else 100 * (total / prev - 1)
    # ampiezza totale dell'intervallo rispetto al volume totale: stabile anche con mediane vicine a zero
    rel_unc = float((hi - lo).sum() / max(float(np.abs(med).sum()), 1e-9)) * 100
    recent = r.history.iloc[-90:].dropna().to_numpy()
    zeros = float((recent == 0).mean()) * 100 if recent.size else 0.0
    flags = []
    if len(r.history.dropna()) < 2 * ds.season:
        flags.append(SHORT_HISTORY_FLAG)
    if zeros >= 30:
        flags.append(ZEROS_FLAG)
    if rel_unc >= 100:
        flags.append(UNCERTAIN_FLAG)
    if change is not None and change <= -30:
        flags.append(DROP_FLAG)
    if change is not None and change >= 30:
        flags.append(RISE_FLAG)
    row: dict[str, Any] = {
        "series": r.series,
        "status": "ok",
        "forecast_id": r.id,
        "history_points": int(len(r.history.dropna())),
        "last_value": round(float(r.history.dropna().iloc[-1]), 4),
        "previous_total": None if prev is None else round(prev, 2),
        "forecast_total": round(total, 2),
        "change_pct": None if change is None else round(change, 1),
        "change_abs": None if prev is None else round(total - prev, 2),
        "uncertainty_pct": round(rel_unc, 1),
        "peak": round(float(med.max()), 2),
        "peak_date": fmt_date(r.future_index[int(np.argmax(med))], r.freq),
        "flags": flags,
    }
    if backtest is not None:
        s = backtest.summary
        skill = s["skill_vs_seasonal_naive_pct"]
        row.update(
            mase=None if s["mase"] is None else round(s["mase"], 3),
            skill_pct=None if skill is None else round(skill, 1),
            coverage_pct=None if s["interval_coverage_pct"] is None else round(s["interval_coverage_pct"], 1),
            reliability=None
            if skill is None
            else ("alta" if skill >= 20 else "media" if skill >= 0 else "bassa"),
        )
    return row


def finish_batch(
    batch_id: str,
    ds: Dataset,
    *,
    target: str,
    horizon: int,
    model: str,
    interval: float,
    results: dict[str, ForecastResult],
    rows: list[dict[str, Any]],
    skipped: dict[str, str],
    not_run: list[str],
    known_covariates: list[str],
    past_covariates: list[str],
    with_backtest: bool,
    warnings: list[str],
) -> BatchResult:
    return BatchResult(
        id=batch_id,
        dataset_id=ds.id,
        dataset_name=ds.name,
        target=target,
        horizon=horizon,
        model=model,
        freq=ds.freq,
        interval=interval,
        rows=rows,
        results=results,
        skipped=skipped,
        not_run=not_run,
        known_covariates=known_covariates,
        past_covariates=past_covariates,
        with_backtest=with_backtest,
        warnings=warnings,
    )


def sorted_rows(rows: list[dict[str, Any]], sort_by: SortBy = "volume") -> list[dict[str, Any]]:
    keyf = {
        "volume": lambda r: -r["forecast_total"],
        "change": lambda r: (r["change_abs"] is None, r["change_abs"] if r["change_abs"] is not None else 0),
        "uncertainty": lambda r: -r["uncertainty_pct"],
    }[sort_by]
    return sorted(rows, key=keyf)


def totals(b: BatchResult) -> dict[str, Any]:
    ok = b.rows
    both = [r for r in ok if r["previous_total"] is not None]
    fc_both = sum(r["forecast_total"] for r in both)
    prev_both = sum(r["previous_total"] for r in both)
    return {
        "series_forecasted": len(ok),
        "forecast_total": round(sum(r["forecast_total"] for r in ok), 2),
        "comparable_series": len(both),
        "previous_total_comparable": round(prev_both, 2),
        "forecast_total_comparable": round(fc_both, 2),
        "change_pct_comparable": None if prev_both == 0 else round(100 * (fc_both / prev_both - 1), 1),
    }


def attention(b: BatchResult, top: int = 5) -> dict[str, list[dict[str, Any]]]:
    comparable = [r for r in b.rows if r["change_abs"] is not None]
    med_vol = float(np.median([r["forecast_total"] for r in b.rows])) if b.rows else 0.0

    def brief(r: dict[str, Any]) -> dict[str, Any]:
        return {
            k: r[k]
            for k in (
                "series",
                "previous_total",
                "forecast_total",
                "change_pct",
                "change_abs",
                "uncertainty_pct",
                "flags",
            )
        }

    return {
        "biggest_drops": [
            brief(r)
            for r in sorted(comparable, key=lambda r: r["change_abs"])[:top]
            if r["change_pct"] is not None and r["change_pct"] <= -MIN_NOTABLE_CHANGE_PCT
        ],
        "biggest_rises": [
            brief(r)
            for r in sorted(comparable, key=lambda r: -r["change_abs"])[:top]
            if r["change_pct"] is not None and r["change_pct"] >= MIN_NOTABLE_CHANGE_PCT
        ],
        "most_uncertain_large_series": [
            brief(r)
            for r in sorted(
                (r for r in b.rows if r["forecast_total"] >= med_vol), key=lambda r: -r["uncertainty_pct"]
            )[:top]
        ],
        "data_quality_flags": [
            brief(r) for r in b.rows if SHORT_HISTORY_FLAG in r["flags"] or ZEROS_FLAG in r["flags"]
        ][:top],
    }


def batch_summary(b: BatchResult, sort_by: SortBy = "volume", max_rows: int = 30) -> str:
    t = totals(b)
    first = next(iter(b.results.values()))
    lines = [
        f"Previsione su più serie {b.id} · Timestral-{b.model} · target '{b.target}' · {b.horizon} periodi "
        f"({fmt_date(first.future_index[0], b.freq)} → {fmt_date(first.future_index[-1], b.freq)}) · intervallo {b.interval:.0%}",
        f"Serie previste: {t['series_forecasted']}; saltate: {len(b.skipped)}; non eseguite per limiti: {len(b.not_run)}.",
        f"Totale previsto (somma delle mediane, tutte le serie): {fmt(t['forecast_total'])}.",
    ]
    if t["change_pct_comparable"] is not None:
        lines.append(
            f"Rispetto agli ultimi {b.horizon} periodi osservati (su {t['comparable_series']} serie confrontabili): "
            f"{fmt(t['previous_total_comparable'])} → {fmt(t['forecast_total_comparable'])} ({t['change_pct_comparable']:+.1f}%). "
            "Il confronto non tiene conto della stagionalità: un periodo diverso dall'anno prima può cambiare molto."
        )
    if b.known_covariates:
        lines.append(f"Covariate note usate: {', '.join(b.known_covariates)} (valori futuri dal dataset).")
    cols = "serie | prima | previsto | var% | incertezza% | segnalazioni"
    if b.with_backtest:
        cols += " | affidabilità"
    lines += ["", f"Prime {min(max_rows, len(b.rows))} serie per {_sort_label(sort_by)}:", cols]
    for r in sorted_rows(b.rows, sort_by)[:max_rows]:
        cp = "n/d" if r["change_pct"] is None else f"{r['change_pct']:+.0f}%"
        line = (
            f"{r['series']} | {fmt(r['previous_total']) if r['previous_total'] is not None else 'n/d'} | "
            f"{fmt(r['forecast_total'])} | {cp} | {r['uncertainty_pct']:.0f}% | {', '.join(r['flags']) or '-'}"
        )
        if b.with_backtest:
            line += f" | {r.get('reliability') or 'n/d'}"
        lines.append(line)
    if len(b.rows) > max_rows:
        lines.append(f"(altre {len(b.rows) - max_rows} serie: usa export_batch o batch_series_forecast.)")
    att = attention(b)
    for title, key in (
        ("Maggiori cali (assoluti)", "biggest_drops"),
        ("Maggiori crescite (assolute)", "biggest_rises"),
        ("Serie grandi più incerte", "most_uncertain_large_series"),
    ):
        if att[key]:
            items = ", ".join(
                f"{x['series']} ({'n/d' if x['change_pct'] is None else f'{x["change_pct"]:+.0f}%'})"
                for x in att[key]
            )
            lines.append(f"{title}: {items}.")
    if att["data_quality_flags"]:
        lines.append(
            "Dati da controllare (storico corto o molti zeri): "
            + ", ".join(x["series"] for x in att["data_quality_flags"])
            + "."
        )
    if b.skipped:
        lines.append(
            "Saltate: "
            + "; ".join(f"{k}: {v}" for k, v in list(b.skipped.items())[:10])
            + ("…" if len(b.skipped) > 10 else "")
        )
    if b.not_run:
        lines.append(
            f"Non eseguite per i limiti di lavoro ({len(b.not_run)}): "
            + ", ".join(b.not_run[:15])
            + ("…" if len(b.not_run) > 15 else "")
            + ". Ripeti con series=[...] per completarle."
        )
    for w in b.warnings:
        lines.append(f"ATTENZIONE: {w}")
    lines.append(f"Dettaglio e grafico di una serie: batch_series_forecast(batch_id='{b.id}', series=...).")
    return "\n".join(lines)


def _sort_label(sort_by: str) -> str:
    return {
        "volume": "volume previsto",
        "change": "variazione assoluta (dal calo maggiore)",
        "uncertainty": "incertezza relativa",
    }[sort_by]


def batch_frames(b: BatchResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tabella riassuntiva, previsioni in formato lungo: serie, data, mediana e quantili)."""
    summary = pd.DataFrame(b.rows)
    if not summary.empty:
        summary["flags"] = summary["flags"].map(", ".join)
    parts = []
    for key, r in b.results.items():
        f = r.to_frame().reset_index()
        f.insert(0, "series", key)
        parts.append(f)
    return summary, (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())


MAX_BACKTEST_SERIES = 30
SEP = "::"


def forecast_key(batch_id: str, series: str) -> str:
    return f"{batch_id}{SEP}{series}"


def run_batch(
    ds: Dataset,
    *,
    batch_id: str,
    target: str,
    horizon: int,
    series: list[str] | None = None,
    model: str | None = None,
    interval: float = 0.8,
    past_covariates: list[str] | None = None,
    known_covariates: list[str] | None = None,
    with_backtest: bool = False,
    backtest_folds: int = 2,
    max_series: int = 100,
    max_seconds: float | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> BatchResult:
    """Prevede ogni serie; errori per singola serie diventano 'saltate', il resto prosegue."""
    selected, over_limit, missing = select_series(ds, target, horizon, series, max_series)
    warnings: list[str] = []
    if missing:
        warnings.append(f"Serie non trovate: {', '.join(missing[:10])}.")
    if over_limit:
        warnings.append(
            f"Il dataset ha più serie del limite ({max_series}): ho previsto le {len(selected)} con più volume recente. "
            f"Escluse {len(over_limit)}; per altre indica series=[...]."
        )
    if with_backtest and len(selected) > MAX_BACKTEST_SERIES:
        warnings.append(
            f"Il backtest per serie è disponibile fino a {MAX_BACKTEST_SERIES} serie (richiesta: {len(selected)}): "
            "l'ho disattivato."
        )
        with_backtest = False

    results: dict[str, ForecastResult] = {}
    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    not_run: list[str] = []
    started = time.monotonic()
    total = len(selected)
    model_used = None
    for i, key in enumerate(selected):
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            not_run = selected[i:]
            warnings.append(
                f"Tempo massimo di {max_seconds:.0f}s raggiunto dopo {i} serie: le restanti {len(not_run)} non sono state eseguite."
            )
            break
        if progress:
            progress(i, total, f"Serie {i + 1}/{total}: {key}")
        try:
            r = run_forecast(
                ds,
                forecast_id=forecast_key(batch_id, key),
                target=target,
                horizon=horizon,
                series=key,
                model=model,
                interval=interval,
                past_covariates=past_covariates,
                known_covariates=known_covariates,
            )
            back = None
            if with_backtest:
                try:
                    plan = bt.plan_backtest(
                        ds,
                        target=target,
                        horizon=horizon,
                        series=key,
                        model=model,
                        n_folds=backtest_folds,
                        interval=interval,
                        past_covariates=past_covariates,
                        known_covariates=known_covariates,
                    )
                    back = bt.finish_backtest(
                        plan, [bt.run_fold(plan, o) for o in plan.origins], f"{batch_id}-bt"
                    )
                except ValueError:
                    back = None  # storico troppo corto: la previsione resta valida, l'affidabilità è n/d
            results[key] = r
            rows.append(series_row(ds, r, back))
            model_used = r.model
        except ValueError as exc:
            skipped[key] = str(exc.args[0])[:160] if exc.args else "errore"
    if progress:
        progress(total, total, "Completato")
    if not results:
        reasons = "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:3])
        raise ValueError(f"Nessuna serie prevista. {reasons}")
    return finish_batch(
        batch_id,
        ds,
        target=target,
        horizon=horizon,
        model=model_used or "",
        interval=interval,
        results=results,
        rows=rows,
        skipped=skipped,
        not_run=not_run + over_limit,
        known_covariates=list(known_covariates or []),
        past_covariates=list(past_covariates or []),
        with_backtest=with_backtest,
        warnings=warnings,
    )
