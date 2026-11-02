"""Interfacce Prefab (MCP App) e grafico PNG di riserva per previsioni e backtest."""

from __future__ import annotations

import io

import numpy as np
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Alert,
    AlertDescription,
    Column,
    DataTable,
    DataTableColumn,
    Heading,
    Metric,
    Row,
    Tab,
    Tabs,
    Text,
)
from prefab_ui.components.charts import BarChart, ChartSeries, LineChart

from timestral_mcp.analytics import batch as batchmod
from timestral_mcp.analytics.anomalies import AnomalyResult
from timestral_mcp.analytics.backtest import BacktestResult
from timestral_mcp.analytics.forecast import ForecastResult
from timestral_mcp.analytics.importance import ImportanceResult
from timestral_mcp.analytics.scenarios import BASE, ScenarioResult
from timestral_mcp.analytics.summary import _date, _pct, fmt, known_covariate_lines

HISTORY_POINTS = 120
C_HISTORY, C_MEDIAN, C_BOUND = "#2563eb", "#ea580c", "#94a3b8"


def _val(v: float | None, nd: int = 4) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _warnings(items: list[str]) -> None:
    for w in items:
        with Alert(variant="warning"):
            AlertDescription(w)


def forecast_rows(r: ForecastResult, history_points: int = HISTORY_POINTS) -> list[dict]:
    hist = r.history.iloc[-history_points:]
    rows: list[dict] = [
        {"date": _date(ts, r.freq), "storico": _val(v), "mediana": None, "basso": None, "alto": None}
        for ts, v in hist.items()
    ]
    anchor = rows[-1]["storico"]  # raccorda storico e previsione
    rows[-1].update(mediana=anchor, basso=anchor, alto=anchor)
    med, lo, hi = r.median, r.lower, r.upper
    rows += [
        {
            "date": _date(ts, r.freq),
            "storico": None,
            "mediana": _val(med[i]),
            "basso": _val(lo[i]),
            "alto": _val(hi[i]),
        }
        for i, ts in enumerate(r.future_index)
    ]
    return rows


def forecast_app(r: ForecastResult) -> PrefabApp:
    lo_l, hi_l = r.bounds
    med, lo, hi = r.median, r.lower, r.upper
    last_val = float(r.history.dropna().iloc[-1])
    peak = int(np.argmax(med))
    table = [
        {
            "data": _date(ts, r.freq),
            "mediana": _val(med[i], 2),
            "basso": _val(lo[i], 2),
            "alto": _val(hi[i], 2),
        }
        for i, ts in enumerate(r.future_index)
    ]
    with Column(gap=4, css_class="p-4") as view:
        Heading(f"Previsione di {r.target}")
        Text(
            f"Timestral-{r.model} · serie '{r.series}' · {r.horizon} periodi · "
            f"intervallo {r.interval:.0%} · contesto {r.context_length} punti",
            css_class="text-sm text-muted-foreground",
        )
        _warnings(r.warnings)
        for line in known_covariate_lines(r):
            Text(f"Covariata usata: {line}", css_class="text-sm text-muted-foreground")
        with Row(gap=3, css_class="flex-wrap"):
            Metric(label="Ultimo osservato", value=fmt(last_val))
            Metric(
                label="Mediana a fine orizzonte",
                value=fmt(med[-1]),
                delta=_pct(med[-1], last_val),
            )
            Metric(
                label=f"Intervallo {r.interval:.0%} a fine orizzonte", value=f"{fmt(lo[-1])} – {fmt(hi[-1])}"
            )
            Metric(
                label="Picco previsto", value=fmt(med[peak]), description=_date(r.future_index[peak], r.freq)
            )
        with Tabs(value="grafico", variant="line"):
            with Tab("Grafico", value="grafico"):
                LineChart(
                    data=forecast_rows(r),
                    series=[
                        ChartSeries(data_key="storico", label="Storico", color=C_HISTORY),
                        ChartSeries(data_key="mediana", label="Previsione (mediana)", color=C_MEDIAN),
                        ChartSeries(
                            data_key="basso", label=f"Limite basso (P{round(lo_l * 100)})", color=C_BOUND
                        ),
                        ChartSeries(
                            data_key="alto", label=f"Limite alto (P{round(hi_l * 100)})", color=C_BOUND
                        ),
                    ],
                    x_axis="date",
                    height=380,
                    show_dots=False,
                    show_legend=True,
                    animate=False,
                )
            with Tab("Tabella", value="tabella"):
                DataTable(
                    columns=[
                        DataTableColumn(key="data", header="Data", sortable=True),
                        DataTableColumn(key="mediana", header="Mediana", sortable=True),
                        DataTableColumn(key="basso", header=f"P{round(lo_l * 100)}", sortable=True),
                        DataTableColumn(key="alto", header=f"P{round(hi_l * 100)}", sortable=True),
                    ],
                    rows=table,
                    search=True,
                    paginated=True,
                    page_size=12,
                )
    return PrefabApp(title=f"Previsione {r.target}", view=view)


def backtest_app(b: BacktestResult) -> PrefabApp:
    s = b.summary
    rows: list[dict] = []
    for f in b.folds:
        for i, ts in enumerate(f.dates):
            rows.append(
                {
                    "date": f"{ts:%Y-%m-%d}",
                    "reale": _val(f.actual[i]),
                    "previsione": _val(f.median[i]),
                    "basso": _val(f.lower[i]),
                    "alto": _val(f.upper[i]),
                }
            )
    skill = s["skill_vs_seasonal_naive_pct"]
    with Column(gap=4, css_class="p-4") as view:
        Heading(f"Backtest di {b.target}")
        Text(
            f"Timestral-{b.model} · serie '{b.series}' · {s['folds']} finestre da {b.horizon} periodi",
            css_class="text-sm text-muted-foreground",
        )
        _warnings(b.warnings)
        with Row(gap=3, css_class="flex-wrap"):
            Metric(
                label="MASE (più basso = meglio)",
                value=fmt(s["mase"]),
                description=f"baseline {fmt(s['naive_mase'])}",
            )
            if skill is not None:
                Metric(
                    label="Rispetto alla baseline stagionale",
                    value=f"{skill:+.0f}%",
                    trend="up" if skill >= 0 else "down",
                    trend_sentiment="positive" if skill >= 0 else "negative",
                )
            Metric(label="WQL (più basso = meglio)", value=fmt(s["wql"], 3))
            if s["interval_coverage_pct"] is not None:
                Metric(
                    label=f"Copertura intervallo {s['interval_nominal_pct']:.0f}%",
                    value=f"{s['interval_coverage_pct']:.0f}%",
                    description="valori reali dentro l'intervallo",
                )
        LineChart(
            data=rows,
            series=[
                ChartSeries(data_key="reale", label="Reale", color=C_HISTORY),
                ChartSeries(data_key="previsione", label="Previsione (mediana)", color=C_MEDIAN),
                ChartSeries(data_key="basso", label="Limite basso", color=C_BOUND),
                ChartSeries(data_key="alto", label="Limite alto", color=C_BOUND),
            ],
            x_axis="date",
            height=340,
            show_dots=False,
            animate=False,
        )
    return PrefabApp(title=f"Backtest {b.target}", view=view)


SCENARIO_COLORS = ["#2563eb", "#16a34a", "#9333ea", "#dc2626", "#0891b2", "#ca8a04"]


def batch_app(b: batchmod.BatchResult, sort_by: batchmod.SortBy = "volume") -> PrefabApp:
    t = batchmod.totals(b)
    rows = batchmod.sorted_rows(b.rows, sort_by)
    table = [
        {
            "serie": r["series"],
            "prima": r["previous_total"],
            "previsto": r["forecast_total"],
            "var": r["change_pct"],
            "incertezza": r["uncertainty_pct"],
            "segnalazioni": ", ".join(r["flags"]) or "-",
            **({"affidabilita": r.get("reliability") or "n/d"} if b.with_backtest else {}),
        }
        for r in rows
    ]
    top = sorted(b.rows, key=lambda r: -r["forecast_total"])[:10]
    bars = [
        {"serie": r["series"], "prima": r["previous_total"] or 0, "previsto": r["forecast_total"]}
        for r in top
    ]
    att = batchmod.attention(b)
    cols = [
        DataTableColumn(key="serie", header="Serie", sortable=True),
        DataTableColumn(key="prima", header=f"Ultimi {b.horizon}", sortable=True),
        DataTableColumn(key="previsto", header="Previsto", sortable=True),
        DataTableColumn(key="var", header="Var. %", sortable=True),
        DataTableColumn(key="incertezza", header=f"Incertezza % ({b.interval:.0%})", sortable=True),
        DataTableColumn(key="segnalazioni", header="Segnalazioni"),
    ]
    if b.with_backtest:
        cols.append(DataTableColumn(key="affidabilita", header="Affidabilità", sortable=True))
    with Column(gap=4, css_class="p-4") as view:
        Heading(f"Previsione di {b.target} su {t['series_forecasted']} serie")
        Text(
            f"Timestral-{b.model} · {b.horizon} periodi · intervallo {b.interval:.0%} · id {b.id}",
            css_class="text-sm text-muted-foreground",
        )
        _warnings(b.warnings)
        with Row(gap=3, css_class="flex-wrap"):
            Metric(label="Totale previsto", value=fmt(t["forecast_total"]))
            if t["change_pct_comparable"] is not None:
                Metric(
                    label=f"Rispetto agli ultimi {b.horizon} periodi",
                    value=f"{t['change_pct_comparable']:+.1f}%",
                    description=f"su {t['comparable_series']} serie confrontabili",
                )
            Metric(
                label="Serie saltate",
                value=len(b.skipped),
                description=f"{len(b.not_run)} non eseguite per limiti",
            )
        with Tabs(value="tabella", variant="line"):
            with Tab("Tabella", value="tabella"):
                DataTable(columns=cols, rows=table, search=True, paginated=True, page_size=15)
            with Tab("Prime 10", value="prime"):
                BarChart(
                    data=bars,
                    series=[
                        ChartSeries(data_key="prima", label=f"Ultimi {b.horizon} periodi", color=C_BOUND),
                        ChartSeries(data_key="previsto", label="Previsto", color=C_MEDIAN),
                    ],
                    x_axis="serie",
                    height=340,
                    animate=False,
                )
            with Tab("Attenzione", value="attenzione"):
                for title, key in (
                    ("Maggiori cali", "biggest_drops"),
                    ("Maggiori crescite", "biggest_rises"),
                    ("Serie grandi più incerte", "most_uncertain_large_series"),
                    ("Dati da controllare", "data_quality_flags"),
                ):
                    if att[key]:
                        Heading(title, css_class="text-base")
                        for x in att[key]:
                            ch = "n/d" if x["change_pct"] is None else f"{x['change_pct']:+.0f}%"
                            Text(
                                f"{x['series']}: {fmt(x['previous_total'])} → {fmt(x['forecast_total'])} ({ch})"
                            )
                if b.skipped:
                    Heading("Saltate", css_class="text-base")
                    for k, v in list(b.skipped.items())[:15]:
                        Text(f"{k}: {v}", css_class="text-sm text-muted-foreground")
    return PrefabApp(title=f"Previsione {b.target}", view=view)


def scenario_app(sr: ScenarioResult) -> PrefabApp:
    base = sr.base
    names = [BASE, *sr.scenarios]
    results = {BASE: base, **sr.scenarios}
    keys = {n: f"s{i}" for i, n in enumerate(names)}
    hist = base.history.iloc[-60:]
    rows: list[dict] = [{"date": _date(ts, base.freq), "storico": _val(v)} for ts, v in hist.items()]
    anchor = rows[-1]["storico"]
    for n in names:
        rows[-1][keys[n]] = anchor
    for i, ts in enumerate(base.future_index):
        row = {"date": _date(ts, base.freq)}
        for n in names:
            row[keys[n]] = _val(results[n].median[i])
        rows.append(row)
    comp = [
        {
            "scenario": r["scenario"],
            "totale": r["total_median"],
            "vs_base": r["delta_vs_base"],
            "vs_base_pct": r["delta_vs_base_pct"],
            "finestra": r.get("changed_window", "-"),
            "effetto_finestra": r.get("window_delta"),
            "dopo_finestra": r.get("after_window_delta"),
        }
        for r in sr.comparison
    ]
    with Column(gap=4, css_class="p-4") as view:
        Heading(f"Scenari per {sr.target}")
        Text(
            f"Timestral-{base.model} · serie '{sr.series}' · {sr.horizon} periodi · mediane a confronto",
            css_class="text-sm text-muted-foreground",
        )
        _warnings(sr.warnings)
        with Row(gap=3, css_class="flex-wrap"):
            for r in sr.comparison[1:]:
                Metric(
                    label=f"{r['scenario']} vs base",
                    value=f"{r['delta_vs_base']:+,.0f}",
                    delta=None if r["delta_vs_base_pct"] is None else f"{r['delta_vs_base_pct']:+.1f}%",
                )
        LineChart(
            data=rows,
            series=[
                ChartSeries(data_key="storico", label="Storico", color="#64748b"),
                *[
                    ChartSeries(
                        data_key=keys[n],
                        label="Base" if n == BASE else n,
                        color=SCENARIO_COLORS[i % len(SCENARIO_COLORS)],
                    )
                    for i, n in enumerate(names)
                ],
            ],
            x_axis="date",
            height=380,
            show_dots=False,
            animate=False,
        )
        DataTable(
            columns=[
                DataTableColumn(key="scenario", header="Scenario"),
                DataTableColumn(key="totale", header="Totale", sortable=True),
                DataTableColumn(key="vs_base", header="vs base", sortable=True),
                DataTableColumn(key="vs_base_pct", header="vs base %", sortable=True),
                DataTableColumn(key="finestra", header="Finestra modificata"),
                DataTableColumn(key="effetto_finestra", header="Effetto nella finestra"),
                DataTableColumn(key="dopo_finestra", header="Dopo la finestra"),
            ],
            rows=comp,
        )
    return PrefabApp(title=f"Scenari {sr.target}", view=view)


def anomalies_app(a: AnomalyResult, max_points: int = 400) -> PrefabApp:
    pts = a.points.iloc[-max_points:]
    rows = [
        {
            "date": _date(ts, a.freq),
            "reale": _val(r["actual"]),
            "atteso": _val(r["median"]),
            "basso": _val(r["lower"]),
            "alto": _val(r["upper"]),
        }
        for ts, r in pts.iterrows()
    ]
    table = [
        {
            "data": r["date"],
            "reale": r["actual"],
            "atteso": r["expected_median"],
            "intervallo": f"{fmt(r['expected_range'][0])} – {fmt(r['expected_range'][1])}",
            "direzione": r["direction"],
            "gravita": r["score"],
            "note": "; ".join(r["notes"]) or "-",
        }
        for r in a.anomalies
    ]
    total = int(a.points["anomaly"].sum())
    with Column(gap=4, css_class="p-4") as view:
        Heading(f"Anomalie in {a.target}")
        Text(
            f"Timestral-{a.model} · serie '{a.series}' · intervallo {a.interval:.0%} · {len(a.points)} periodi analizzati",
            css_class="text-sm text-muted-foreground",
        )
        _warnings(a.warnings)
        with Row(gap=3, css_class="flex-wrap"):
            Metric(label="Anomalie trovate", value=total)
            Metric(label="Attese per puro caso", value=a.expected_by_chance)
        LineChart(
            data=rows,
            series=[
                ChartSeries(data_key="reale", label="Reale", color=C_HISTORY),
                ChartSeries(data_key="atteso", label="Atteso (mediana)", color=C_MEDIAN),
                ChartSeries(data_key="basso", label="Limite basso", color=C_BOUND),
                ChartSeries(data_key="alto", label="Limite alto", color=C_BOUND),
            ],
            x_axis="date",
            height=340,
            show_dots=False,
            animate=False,
        )
        if table:
            DataTable(
                columns=[
                    DataTableColumn(key="data", header="Data", sortable=True),
                    DataTableColumn(key="reale", header="Reale", sortable=True),
                    DataTableColumn(key="atteso", header="Atteso", sortable=True),
                    DataTableColumn(key="intervallo", header="Intervallo"),
                    DataTableColumn(key="direzione", header="Direzione"),
                    DataTableColumn(key="gravita", header="Gravità", sortable=True),
                    DataTableColumn(key="note", header="Note"),
                ],
                rows=table,
                search=True,
                paginated=True,
                page_size=10,
            )
    return PrefabApp(title=f"Anomalie {a.target}", view=view)


def importance_app(r: ImportanceResult) -> PrefabApp:
    g = r.gain_vs_none_pct["wql"]
    bars = [{"covariata": row["covariate"], "delta": row["delta_wql_pct"] or 0.0} for row in r.rows]
    table = [
        {
            "covariata": row["covariate"],
            "ruolo": row["role"],
            "delta_wql": row["delta_wql_pct"],
            "delta_mase": row["delta_mase_pct"],
            "finestre": row["folds_worse_without"],
            "giudizio": row["verdict"],
        }
        for row in r.rows
    ]
    with Column(gap=4, css_class="p-4") as view:
        Heading(f"Importanza delle covariate per {r.target}")
        Text(
            f"Timestral-{r.model} · serie '{r.series}' · {r.folds} finestre da {r.horizon} periodi · id {r.id}",
            css_class="text-sm text-muted-foreground",
        )
        _warnings(r.warnings)
        with Row(gap=3, css_class="flex-wrap"):
            Metric(label="Errore (WQL) con tutte le covariate", value=fmt(r.all_covariates["wql"], 3))
            Metric(label="Errore (WQL) senza covariate", value=fmt(r.no_covariates["wql"], 3))
            if g is not None:
                Metric(
                    label="Senza covariate l'errore",
                    value=f"{g:+.1f}%",
                    trend="up" if g >= 0 else "down",
                    trend_sentiment="negative" if g >= 0 else "positive",
                )
        Text(
            "Quanto peggiora la previsione sul passato se si toglie una covariata (positivo = covariata utile, "
            "negativo = rumore). Misura l'utilità per prevedere, non la causalità.",
            css_class="text-sm text-muted-foreground",
        )
        BarChart(
            data=bars,
            series=[ChartSeries(data_key="delta", label="Peggioramento % se tolta (WQL)", color=C_MEDIAN)],
            x_axis="covariata",
            height=320,
            animate=False,
        )
        DataTable(
            columns=[
                DataTableColumn(key="covariata", header="Covariata"),
                DataTableColumn(key="ruolo", header="Ruolo"),
                DataTableColumn(key="delta_wql", header="Δ WQL %", sortable=True),
                DataTableColumn(key="delta_mase", header="Δ MASE %", sortable=True),
                DataTableColumn(key="finestre", header="Finestre peggiori senza"),
                DataTableColumn(key="giudizio", header="Giudizio"),
            ],
            rows=table,
        )
    return PrefabApp(title=f"Importanza covariate {r.target}", view=view)


def forecast_png(r: ForecastResult, history_points: int = HISTORY_POINTS) -> bytes:
    """Grafico statico con bande piene (alternativa per i client senza MCP Apps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = r.history.iloc[-history_points:]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=130)
    ax.plot(hist.index, hist.values, color=C_HISTORY, lw=1.6, label="Storico")
    outer_lo, outer_hi = r.bounds
    ax.fill_between(
        r.future_index, r.lower, r.upper, color=C_MEDIAN, alpha=0.15, label=f"Intervallo {r.interval:.0%}"
    )
    ax.fill_between(r.future_index, r.q(0.25), r.q(0.75), color=C_MEDIAN, alpha=0.3, label="Intervallo 50%")
    ax.plot(r.future_index, r.median, color=C_MEDIAN, lw=2, label="Previsione (mediana)")
    ax.axvline(hist.index[-1], color="#64748b", lw=0.8, ls="--")
    ax.set_title(f"{r.target} · serie '{r.series}' · Timestral-{r.model}")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
