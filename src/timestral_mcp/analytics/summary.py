"""Riassunti testuali e dettagli interrogabili delle previsioni (per Claude e per l'utente)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from timestral_mcp.analytics.backtest import BacktestResult
from timestral_mcp.analytics.covariates import describe_future_values, format_future_values
from timestral_mcp.analytics.forecast import ForecastResult

MAX_ROWS_INLINE = 36


def fmt(v: float | None, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "n/d"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}".replace(",", ".")
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"


def _date(ts: pd.Timestamp, freq: str) -> str:
    return f"{ts:%Y-%m-%d %H:%M}" if freq.lower().endswith(("h", "min")) else f"{ts:%Y-%m-%d}"


def _pct(a: float, b: float) -> str:
    return "n/d" if b == 0 or not np.isfinite(a) else f"{100 * (a / b - 1):+.1f}%"


def forecast_summary(r: ForecastResult) -> str:
    lo_l, hi_l = r.bounds
    med, lo, hi = r.median, r.lower, r.upper
    last_val = float(r.history.dropna().iloc[-1])
    last_date = r.history.last_valid_index()
    peak, trough = int(np.argmax(med)), int(np.argmin(med))
    width0, width1 = hi[0] - lo[0], hi[-1] - lo[-1]
    lines = [
        f"Previsione {r.id} · modello Timestral-{r.model} · serie '{r.series}' · target '{r.target}'",
        f"Orizzonte: {r.horizon} periodi ({_date(r.future_index[0], r.freq)} → {_date(r.future_index[-1], r.freq)}), "
        f"frequenza {r.freq}. Contesto usato: {r.context_length} punti. "
        f"Intervallo mostrato: {r.interval:.0%} (quantili {lo_l:g}–{hi_l:g}).",
        f"Ultimo valore osservato: {fmt(last_val)} ({_date(last_date, r.freq)}).",
        f"Mediana prevista: primo periodo {fmt(med[0])}, ultimo {fmt(med[-1])} ({_pct(med[-1], last_val)} rispetto all'ultimo osservato). "
        f"Media sul periodo {fmt(med.mean())}.",
        f"Massimo atteso (mediana): {fmt(med[peak])} il {_date(r.future_index[peak], r.freq)}. "
        f"Minimo: {fmt(med[trough])} il {_date(r.future_index[trough], r.freq)}.",
        f"Ampiezza dell'intervallo: {fmt(width0)} al primo periodo, {fmt(width1)} all'ultimo "
        f"({'cresce' if width1 > width0 * 1.05 else 'resta simile'}).",
    ]
    if r.past_covariates or r.known_covariates:
        lines.append(
            "Covariate: passate "
            f"[{', '.join(r.past_covariates) or '-'}], note in anticipo [{', '.join(r.known_covariates) or '-'}]."
        )
    for text in known_covariate_lines(r):
        lines.append(f"Valori futuri usati dal modello — {text}")
    for w in r.warnings:
        lines.append(f"ATTENZIONE: {w}")
    lines.append("")
    lines.append(
        _table(r, np.arange(r.horizon) if r.horizon <= MAX_ROWS_INLINE else _sample_positions(r.horizon))
    )
    if r.horizon > MAX_ROWS_INLINE:
        lines.append(
            f"(Mostrati {len(_sample_positions(r.horizon))} periodi su {r.horizon}. "
            f"Per altri periodi o totali mensili/settimanali usa get_forecast_details con forecast_id='{r.id}'.)"
        )
    else:
        lines.append(
            f"Per quantili diversi o totali aggregati usa get_forecast_details con forecast_id='{r.id}'."
        )
    return "\n".join(lines)


def known_covariate_lines(r: ForecastResult) -> list[str]:
    """Cosa valgono, nel futuro, le covariate note in anticipo (es. giorni di promozione)."""
    if r.known_future is None or r.known_future.empty:
        return []
    return format_future_values(describe_future_values(r.known_future, r.freq))


def _sample_positions(h: int, n: int = 24) -> np.ndarray:
    return np.unique(np.linspace(0, h - 1, n).round().astype(int))


def _table(r: ForecastResult, pos: np.ndarray) -> str:
    lo_l, hi_l = r.bounds
    rows = ["data | mediana | " + f"q{lo_l:g} | q{hi_l:g}"]
    for i in pos:
        rows.append(
            f"{_date(r.future_index[i], r.freq)} | {fmt(r.median[i])} | {fmt(r.lower[i])} | {fmt(r.upper[i])}"
        )
    return "\n".join(rows)


def forecast_details(
    r: ForecastResult,
    *,
    start: str | None = None,
    end: str | None = None,
    quantiles: list[float] | None = None,
    aggregate: Literal["week", "month", "quarter", "year"] | None = None,
    how: Literal["sum", "mean"] = "sum",
    max_rows: int = 200,
) -> dict[str, Any]:
    levels = quantiles or [r.bounds[0], 0.5, r.bounds[1]]
    df = r.to_frame(sorted(set(levels) | {0.5}))
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError(
            f"Nessun periodo nell'intervallo richiesto. La previsione copre "
            f"{_date(r.future_index[0], r.freq)} → {_date(r.future_index[-1], r.freq)}."
        )
    df = df.drop(columns=["q50"], errors="ignore")  # uguale a "median"
    if r.known_future is not None:  # affianca le covariate note (es. promo) ai quantili
        df = df.join(r.known_future.reindex(df.index))
    notes: list[str] = []
    partial_periods: list[dict[str, Any]] = []
    if aggregate:
        rule = {"week": "W", "month": "MS", "quarter": "QS", "year": "YS"}[aggregate]
        grouped = df.resample(rule)
        counts = grouped.size()
        df = (grouped.sum() if how == "sum" else grouped.mean())[counts > 0]
        df.insert(0, "periods", counts[counts > 0])
        notes.append(
            "Aggregazione approssimata: i quantili sommati assumono dipendenza perfetta tra i periodi, "
            "quindi gli intervalli aggregati risultano più larghi del reale; la mediana aggregata è la somma delle mediane."
            if how == "sum"
            else "Media dei quantili per periodo (approssimazione)."
        )
        if r.known_future is not None and how == "sum":
            notes.append(
                "Le colonne delle covariate note sono sommate: per un indicatore 0/1 il valore è il numero di periodi attivi."
            )
        partial_periods, partial_notes = _partial_periods(r, df.index, aggregate)
        notes += partial_notes
    truncated = len(df) > max_rows
    df = df.head(max_rows)
    out = df.round(4).reset_index()
    out["date"] = out["date"].map(lambda t: _date(t, r.freq))
    return {
        "forecast_id": r.id,
        "target": r.target,
        "series": r.series,
        "frequency": r.freq,
        "aggregate": aggregate,
        "how": how if aggregate else None,
        "rows": out.to_dict(orient="records"),
        "partial_periods": partial_periods,
        "truncated": truncated,
        "notes": notes + r.warnings,
    }


def _period_bounds(label: pd.Timestamp, aggregate: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Primo e ultimo istante del mese/trimestre/anno/settimana identificato da ``label``."""
    one_ns = pd.Timedelta(nanoseconds=1)
    if aggregate == "week":  # resample("W"): l'etichetta è la domenica che chiude la settimana
        end = label.normalize() + pd.Timedelta(days=1) - one_ns
        return end - pd.Timedelta(days=7) + one_ns, end
    nxt = label + pd.tseries.frequencies.to_offset({"month": "MS", "quarter": "QS", "year": "YS"}[aggregate])
    return label, nxt - one_ns


def _partial_periods(
    r: ForecastResult, labels: pd.DatetimeIndex, aggregate: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Periodi aggregati che escono dall'orizzonte della previsione (inizio o fine incompleti)."""
    first, last = r.future_index[0], r.future_index[-1]
    unit = {"week": "settimana", "month": "mese", "quarter": "trimestre", "year": "anno"}[aggregate]
    partial: list[dict[str, Any]] = []
    notes: list[str] = []
    for label in labels:
        start, end = _period_bounds(label, aggregate)
        grid = pd.date_range(start, end, freq=r.freq)
        before, after = int((grid < first).sum()), int((grid > last).sum())
        if not (before or after):
            continue
        covered = len(grid) - before - after
        item = {
            "period": str(label.date()),
            "covered": covered,
            "expected": len(grid),
            "missing_before": before,
            "missing_after": after,
        }
        if after:
            item["horizon_for_complete_period"] = r.horizon + after
            notes.append(
                f"Il {unit} {label.date()} è incompleto: la previsione copre {covered} periodi su {len(grid)} "
                f"e termina il {_date(last, r.freq)}. Per il totale completo servono altri {after} periodi: "
                f"rifai la previsione con horizon={r.horizon + after}."
            )
        if before:
            notes.append(
                f"Il {unit} {label.date()} è coperto dalla previsione per {covered} periodi su {len(grid)}: "
                f"la previsione inizia il {_date(first, r.freq)} (gli altri {before} sono storico)."
            )
        partial.append(item)
    return partial, notes


def backtest_summary(b: BacktestResult) -> str:
    s = b.summary
    lines = [
        f"Backtest {b.id} · Timestral-{b.model} · serie '{b.series}' · target '{b.target}'",
        f"{s['folds']} finestre di test da {b.horizon} periodi, ognuna prevista usando solo i dati precedenti.",
        f"MASE medio: {fmt(s['mase'])} (1 = errore di una previsione ingenua; più basso è meglio). "
        f"Baseline stagionale ingenua: {fmt(s['naive_mase'])}.",
    ]
    if s["skill_vs_seasonal_naive_pct"] is not None:
        sk = s["skill_vs_seasonal_naive_pct"]
        lines.append(
            f"Rispetto alla baseline stagionale: {abs(sk):.0f}% {'meglio' if sk >= 0 else 'peggio'}."
        )
    lines.append(f"WQL (errore pesato sui quantili, più basso è meglio): {fmt(s['wql'], 3)}.")
    if s["interval_coverage_pct"] is not None:
        lines.append(
            f"Copertura intervallo: il {s['interval_coverage_pct']:.0f}% dei valori reali è caduto nell'intervallo "
            f"nominale del {s['interval_nominal_pct']:.0f}%."
        )
    lines.append("")
    lines.append("origine | MASE | baseline | copertura")
    for f in b.folds:
        lines.append(
            f"{f.origin:%Y-%m-%d} | {fmt(f.mase)} | {fmt(f.naive_mase)} | "
            f"{'n/d' if f.coverage is None else f'{f.coverage:.0%}'}"
        )
    for w in b.warnings:
        lines.append(f"ATTENZIONE: {w}")
    return "\n".join(lines)
