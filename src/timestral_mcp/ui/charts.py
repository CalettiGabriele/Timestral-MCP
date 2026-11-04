"""Interfacce Prefab (MCP App) e grafico PNG di riserva per previsioni e backtest.

Ogni schermata segue lo stesso schema, pensato per chi usa lo strumento la prima volta: un titolo in parole
semplici, i numeri chiave, il grafico con le bande di incertezza, una spiegazione su come leggerlo e i pulsanti
per chiedere a Claude il passo successivo.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Column,
    DataTable,
    DataTableColumn,
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
from timestral_mcp.analytics.covariates import describe_future_values
from timestral_mcp.analytics.forecast import ForecastResult
from timestral_mcp.analytics.importance import ImportanceResult
from timestral_mcp.analytics.scenarios import BASE, ScenarioResult
from timestral_mcp.ui import kit, theme
from timestral_mcp.ui.kit import BAND_INNER, BAND_OUTER, day, num, period_word, signed_pct
from timestral_mcp.ui.theme import C_MEDIAN, C_MUTED, SCENARIO_COLORS

HISTORY_POINTS = 120  # grafico PNG


def _series_note(series: str) -> str:
    return "" if series in ("serie", "") else f"Serie «{series}». "


def _subject(target: str) -> str:
    return target[:1].upper() + target[1:]


def _axis_year(dates: pd.DatetimeIndex) -> bool:
    return len(dates) > 1 and (dates[-1] - dates[0]).days > 300


def history_window(horizon: int) -> int:
    """Storico mostrato: abbastanza per vedere la stagionalità, non tanto da schiacciare la previsione."""
    return int(min(max(3 * horizon, 60), 180))


def forecast_rows(r: ForecastResult, history_points: int | None = None) -> list[dict]:
    """Righe del grafico: storico, mediana e bande (intervallo scelto e 50%)."""
    hist = r.history.iloc[-(history_points or history_window(r.horizon)) :]
    d = kit.digits_for(list(r.median) + list(hist.dropna().values))
    dates = hist.index.append(r.future_index)
    year = _axis_year(dates)
    rows: list[dict] = [
        {"date": day(ts, r.freq, year), "storico": kit.chart_value(v, d)} for ts, v in hist.items()
    ]
    anchor = rows[-1]["storico"]  # raccorda storico e previsione
    if anchor is not None:
        rows[-1].update({"mediana": anchor, BAND_OUTER: [anchor, anchor], BAND_INNER: [anchor, anchor]})
    q25, q75 = r.q(0.25), r.q(0.75)
    for i, ts in enumerate(r.future_index):
        rows.append(
            {
                "date": day(ts, r.freq, year),
                "mediana": kit.chart_value(r.median[i], d),
                BAND_OUTER: kit.band_point(r.lower[i], r.upper[i], d),
                BAND_INNER: kit.band_point(q25[i], q75[i], d),
            }
        )
    return kit.with_bounds(rows)


def forecast_table(r: ForecastResult) -> tuple[list[DataTableColumn], list[dict]]:
    d = kit.digits_for(r.median)
    cols = [
        DataTableColumn(key="data", header="Data"),
        DataTableColumn(key="mediana", header="Previsione", format=f"number:{d}", align="right"),
        DataTableColumn(key="basso", header="Minimo probabile", format=f"number:{d}", align="right"),
        DataTableColumn(key="alto", header="Massimo probabile", format=f"number:{d}", align="right"),
    ]
    rows = [
        {
            "data": day(ts, r.freq),
            "mediana": kit.chart_value(r.median[i], d),
            "basso": kit.chart_value(r.lower[i], d),
            "alto": kit.chart_value(r.upper[i], d),
        }
        for i, ts in enumerate(r.future_index)
    ]
    return cols, rows


def covariate_sentences(r: ForecastResult) -> list[str]:
    """Le covariate note in anticipo, dette in parole semplici («promo attiva 20–24 gen 2026»)."""
    if r.known_future is None or r.known_future.empty:
        return []
    out = []
    for col, info in describe_future_values(r.known_future, r.freq).items():
        if info["type"] == "binary":
            if not info["active_periods"]:
                out.append(f"«{col}» non è mai attiva nel periodo previsto.")
                continue
            ranges = []
            for rng in info["active_ranges"]:
                a, _, b = rng.partition("..")
                ranges.append(kit.span(pd.Timestamp(a), pd.Timestamp(b or a), r.freq))
            more = f" e altri {info['more_ranges']} intervalli" if info["more_ranges"] else ""
            out.append(
                f"«{col}» è attiva {', '.join(ranges)}{more}: {info['active_periods']} "
                f"{period_word(r.freq, info['active_periods'])} su {info['periods']}."
            )
        elif info["type"] == "numeric":
            out.append(f"«{col}» varia tra {num(info['min'])} e {num(info['max'])} nel periodo previsto.")
    return out


def forecast_steps(r: ForecastResult) -> list[tuple[str, str]]:
    steps = [
        ("Totali per mese", f"Dammi i totali mensili della previsione {r.id}."),
        ("Quanto è affidabile?", f"Fai un backtest con gli stessi parametri della previsione {r.id}."),
    ]
    if r.known_covariates:
        steps.append(("E se cambiasse…?", f"Voglio provare uno scenario what-if sulla previsione {r.id}."))
    steps.append(("Salva in Excel", f"Esporta la previsione {r.id} in Excel."))
    return steps


def forecast_app(r: ForecastResult) -> PrefabApp:
    med, lo, hi = r.median, r.lower, r.upper
    last_val = float(r.history.dropna().iloc[-1])
    last_date = r.history.last_valid_index()
    peak = int(np.argmax(med))
    end = r.future_index[-1]
    pw = period_word(r.freq, r.horizon)
    cases = kit.in_cases(r.interval)
    with kit.page(
        f"Previsione · Timestral-{r.model}",
        f"{_subject(r.target)} nei prossimi {r.horizon} {pw}",
        f"{_series_note(r.series)}Dal {day(r.future_index[0], r.freq)} al {day(end, r.freq)}, "
        f"a partire da {r.context_length} {period_word(r.freq, r.context_length)} di storico.",
    ) as view:
        kit.warnings(r.warnings)
        if sentences := covariate_sentences(r):
            with Column(gap=1, css_class="tm-panel"):
                kit.label("Già incluso nella previsione")
                for s in sentences:
                    Text(s)
        with Row(gap=6, css_class="flex-wrap"):
            Metric(label="Ultimo valore", value=num(last_val), description=day(last_date, r.freq))
            Metric(
                label=f"Previsto al {day(end, r.freq, year=False)}",
                value=num(med[-1]),
                description=f"{signed_pct(100 * (med[-1] / last_val - 1)) if last_val else 'n/d'} sull'ultimo valore",
            )
            Metric(
                label="Intervallo probabile",
                value=f"{num(lo[-1])} – {num(hi[-1])}",
                description=f"in {cases}",
            )
            Metric(
                label="Picco previsto", value=num(med[peak]), description=day(r.future_index[peak], r.freq)
            )
        with Tabs(value="grafico", variant="line"):
            with Tab("Grafico", value="grafico"), Column(gap=3):
                kit.band_chart(
                    forecast_rows(r),
                    actual_key="storico",
                    actual_label="Storico",
                    median_label="Previsione",
                    outer_label=f"Intervallo {r.interval:.0%}",
                    inner_label="Intervallo 50%",
                    height=380,
                )
                kit.note(
                    f"La linea spessa è il valore più probabile. L'area più ampia contiene il valore reale in {cases}, "
                    "quella più stretta in 5 casi su 10. Più ci si allontana nel tempo, più le aree si allargano: "
                    "è normale, l'incertezza cresce."
                )
            with Tab("Tabella", value="tabella"):
                cols, rows = forecast_table(r)
                DataTable(columns=cols, rows=rows, paginated=True, page_size=10)
        kit.explainer(
            "Come leggere questa previsione",
            [
                "Previsione: il valore centrale. Metà delle volte il valore reale sarà più alto, metà più basso.",
                f"Minimo e massimo probabile: in {cases} il valore reale cade tra i due. "
                "Usa il massimo per decidere scorte o capacità, il minimo per stimare i rischi.",
                "Prima di prendere decisioni importanti chiedi un backtest: mostra quanto il modello ha sbagliato "
                "sul tuo passato.",
            ],
        )
        kit.next_steps(forecast_steps(r))
    return kit.app(view, f"Previsione {r.target}")


def _verdict(s: dict) -> tuple[str, str]:
    """Titolo e spiegazione del giudizio sul backtest, in parole semplici."""
    skill, cov, nom = s["skill_vs_seasonal_naive_pct"], s["interval_coverage_pct"], s["interval_nominal_pct"]
    parts = []
    if skill is None:
        title = "Risultato parziale"
        parts.append("Non è stato possibile confrontare il modello con una previsione ingenua.")
    else:
        better = skill >= 0
        parts.append(
            f"Sbaglia il {num(abs(skill), 0)}% {'in meno' if better else 'in più'} di una previsione ingenua "
            "che ripete l'andamento della stagione precedente."
        )
        title = (
            "Affidabile su questi dati"
            if skill >= 20
            else "Utile, con qualche cautela"
            if skill > 0
            else "Su questi dati il modello non aggiunge valore"
        )
    if cov is not None:
        gap = cov - nom
        if abs(gap) <= 10:
            parts.append(f"Gli intervalli sono ben tarati: ci è caduto il {num(cov, 0)}% dei valori reali.")
        elif gap < 0:
            parts.append(
                f"Gli intervalli sono troppo stretti: ci è caduto solo il {num(cov, 0)}% dei valori reali "
                f"invece del {num(nom, 0)}%. Conviene usarne uno più ampio."
            )
            if skill is not None and skill >= 20:
                title = "Preciso, ma con intervalli troppo stretti"
        else:
            parts.append(f"Gli intervalli sono prudenti: ci è caduto il {num(cov, 0)}% dei valori reali.")
    return title, " ".join(parts)


def backtest_rows(b: BacktestResult) -> list[dict]:
    d = kit.digits_for([v for f in b.folds for v in f.actual])
    dates = pd.DatetimeIndex([ts for f in b.folds for ts in f.dates])
    year = _axis_year(dates) if len(dates) else False
    freq = b.folds[0].dates.freqstr if b.folds and b.folds[0].dates.freqstr else "D"
    return [
        {
            "date": day(ts, freq, year),
            "reale": kit.chart_value(f.actual[i], d),
            "mediana": kit.chart_value(f.median[i], d),
            BAND_OUTER: kit.band_point(f.lower[i], f.upper[i], d),
        }
        for f in b.folds
        for i, ts in enumerate(f.dates)
    ]


def backtest_app(b: BacktestResult) -> PrefabApp:
    s = b.summary
    freq = b.folds[0].dates.freqstr if b.folds and b.folds[0].dates.freqstr else "D"
    pw = period_word(freq, b.horizon)
    title, explanation = _verdict(s)
    skill, cov, nom = s["skill_vs_seasonal_naive_pct"], s["interval_coverage_pct"], s["interval_nominal_pct"]
    with kit.page(
        f"Backtest · Timestral-{b.model}",
        f"Quanto fidarsi delle previsioni di {b.target}",
        f"{_series_note(b.series)}Ho nascosto al modello {s['folds']} tratti già noti da {b.horizon} {pw} "
        "e gli ho chiesto di prevederli. Il confronto con i valori reali dice quanto fidarsi.",
    ) as view:
        kit.warnings(b.warnings)
        with Column(gap=2, css_class="tm-panel"):
            kit.label("Verdetto")
            Text(title, css_class="tm-verdict")
            Text(explanation, css_class="tm-sub")
        with Row(gap=6, css_class="flex-wrap"):
            if skill is not None:
                Metric(
                    label="Errore",
                    value=f"{num(abs(skill), 0)}% {'in meno' if skill >= 0 else 'in più'}",
                    description="di una previsione ingenua",
                )
            if cov is not None:
                Metric(
                    label="Valori reali nell'intervallo",
                    value=f"{num(cov, 0)}%",
                    description=f"l'ideale è {num(nom, 0)}%",
                )
            Metric(label="Prove", value=str(s["folds"]), description=f"da {b.horizon} {pw} ciascuna")
        with Column(gap=3):
            kit.band_chart(
                backtest_rows(b),
                actual_key="reale",
                actual_label="Valori reali",
                median_label="Previsione",
                outer_label=f"Intervallo {b.interval:.0%}",
                inner_label=None,
                height=340,
            )
            kit.note(
                "La linea sottile sono i valori veri, quella spessa ciò che il modello aveva previsto senza vederli. "
                "Più le due linee si somigliano e più spesso i valori veri restano dentro l'area colorata, meglio è."
            )
        kit.explainer(
            "Dettagli tecnici",
            [
                f"MASE {num(s['mase'], 2)} (previsione ingenua stagionale: {num(s['naive_mase'], 2)}). "
                "Sotto 1 il modello sbaglia meno di chi ripete l'ultimo valore.",
                f"WQL {num(s['wql'], 3)}: errore pesato su tutti i quantili, più basso è meglio.",
                *[
                    f"Prova dal {day(f.origin, freq)}: MASE {num(f.mase, 2)}, "
                    f"copertura {num(100 * f.coverage, 0) if f.coverage is not None else 'n/d'}%."
                    for f in b.folds
                ],
            ],
        )
        kit.next_steps(
            [
                (
                    "Fai la previsione",
                    f"Ora fai la previsione di {b.target} con gli stessi parametri del backtest {b.id}.",
                ),
                ("Spiegamelo", f"Spiegami in parole semplici il risultato del backtest {b.id}."),
                ("Prova un orizzonte più corto", f"Rifai il backtest {b.id} con un orizzonte più corto."),
            ]
        )
    return kit.app(view, f"Backtest {b.target}")


def batch_app(b: batchmod.BatchResult, sort_by: batchmod.SortBy = "volume") -> PrefabApp:
    t = batchmod.totals(b)
    rows = batchmod.sorted_rows(b.rows, sort_by)
    d = kit.digits_for([r["forecast_total"] for r in rows])
    table = [
        {
            "serie": r["series"],
            "prima": kit.chart_value(r["previous_total"], d) if r["previous_total"] is not None else None,
            "previsto": kit.chart_value(r["forecast_total"], d),
            "var": r["change_pct"],
            "incertezza": r["uncertainty_pct"],
            "segnalazioni": ", ".join(r["flags"]) or "—",
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
        DataTableColumn(
            key="prima", header=f"Ultimi {b.horizon}", sortable=True, format=f"number:{d}", align="right"
        ),
        DataTableColumn(
            key="previsto", header="Previsti", sortable=True, format=f"number:{d}", align="right"
        ),
        DataTableColumn(key="var", header="Variazione %", sortable=True, format="number:1", align="right"),
        DataTableColumn(
            key="incertezza", header="Incertezza %", sortable=True, format="number:1", align="right"
        ),
        DataTableColumn(key="segnalazioni", header="Da controllare"),
    ]
    if b.with_backtest:
        cols.append(DataTableColumn(key="affidabilita", header="Affidabilità", sortable=True))
    skipped = len(b.skipped) + len(b.not_run)
    with kit.page(
        f"Molte serie · Timestral-{b.model}",
        f"{_subject(b.target)} su {t['series_forecasted']} serie, prossimi {b.horizon} periodi",
        "Una riga per serie: quanto si prevede, come cambia rispetto al periodo appena passato e quanto è incerto. "
        "Ordina le colonne per trovare subito le serie che meritano attenzione.",
    ) as view:
        kit.warnings(b.warnings)
        with Row(gap=6, css_class="flex-wrap"):
            Metric(
                label="Totale previsto", value=num(t["forecast_total"]), description="somma di tutte le serie"
            )
            if t["change_pct_comparable"] is not None:
                Metric(
                    label=f"Rispetto agli ultimi {b.horizon} periodi",
                    value=signed_pct(t["change_pct_comparable"]),
                    description=f"su {t['comparable_series']} serie confrontabili",
                )
            Metric(
                label="Serie non previste",
                value=str(skipped),
                description="nessuna" if not skipped else "vedi «Da sapere»",
            )
        with Tabs(value="tabella", variant="line"):
            with Tab("Tutte le serie", value="tabella"):
                DataTable(
                    columns=cols, rows=table, search=len(table) > 15, paginated=len(table) > 15, page_size=15
                )
            with Tab("Le più grandi", value="prime"):
                BarChart(
                    data=bars,
                    series=[
                        ChartSeries(data_key="prima", label=f"Ultimi {b.horizon} periodi", color=C_MUTED),
                        ChartSeries(data_key="previsto", label="Previsti", color=C_MEDIAN),
                    ],
                    x_axis="serie",
                    height=340,
                    animate=False,
                )
            with Tab("Da sapere", value="attenzione"), Column(gap=4):
                for heading, key in (
                    ("Cali più forti", "biggest_drops"),
                    ("Crescite più forti", "biggest_rises"),
                    ("Serie grandi più incerte", "most_uncertain_large_series"),
                    ("Dati da controllare", "data_quality_flags"),
                ):
                    if att[key]:
                        with Column(gap=1):
                            kit.label(heading)
                            for x in att[key]:
                                ch = "n/d" if x["change_pct"] is None else signed_pct(x["change_pct"])
                                Text(
                                    f"{x['series']}: {num(x['previous_total'])} → {num(x['forecast_total'])} ({ch})"
                                )
                if b.skipped:
                    with Column(gap=1):
                        kit.label("Saltate")
                        for k, v in list(b.skipped.items())[:15]:
                            kit.note(f"{k}: {v}")
        kit.next_steps(
            [
                (
                    "Dettaglio di una serie",
                    f"Mostrami il grafico della serie più importante del risultato {b.id}.",
                ),
                (
                    "Quali preoccupano?",
                    f"Quali serie del risultato {b.id} devo guardare per prime, e perché?",
                ),
                ("Esporta tutto", f"Esporta in Excel il risultato {b.id}."),
            ]
        )
    return kit.app(view, f"Previsione {b.target}")


def _signed(v: float, digits: int) -> str:
    return f"{'+' if v >= 0 else '−'}{num(abs(v), digits)}"


def _window(text: str | None, freq: str) -> str:
    """«2026-01-15 → 2026-01-21» → «15–21 gen 2026»."""
    if not text or "→" not in text:
        return text or "—"
    a, b = (part.strip() for part in text.split("→"))
    try:
        return kit.span(pd.Timestamp(a), pd.Timestamp(b), freq)
    except ValueError:
        return text


def scenario_app(sr: ScenarioResult) -> PrefabApp:
    base = sr.base
    names = [BASE, *sr.scenarios]
    results = {BASE: base, **sr.scenarios}
    keys = {n: f"s{i}" for i, n in enumerate(names)}
    hist = base.history.iloc[-min(history_window(sr.horizon), 90) :]
    d = kit.digits_for(list(base.median))
    year = _axis_year(hist.index.append(base.future_index))
    rows: list[dict] = [
        {"date": day(ts, base.freq, year), "storico": kit.chart_value(v, d)} for ts, v in hist.items()
    ]
    anchor = rows[-1]["storico"]
    for n in names:
        rows[-1][keys[n]] = anchor
    for i, ts in enumerate(base.future_index):
        row = {"date": day(ts, base.freq, year)}
        for n in names:
            row[keys[n]] = kit.chart_value(results[n].median[i], d)
        rows.append(row)
    comp = [
        {
            "scenario": "Base (nessun cambio)" if r["scenario"] == BASE else r["scenario"],
            "totale": num(r["total_median"], d),
            "vs_base": "—" if r["scenario"] == BASE else _signed(r["delta_vs_base"], d),
            "finestra": _window(r.get("changed_window"), base.freq),
            "effetto": "—" if r.get("window_delta") is None else _signed(r["window_delta"], d),
            "dopo": "—" if r.get("after_window_delta") is None else _signed(r["after_window_delta"], d),
        }
        for r in sr.comparison
    ]
    pw = period_word(base.freq, sr.horizon)
    with kit.page(
        f"Scenari · Timestral-{base.model}",
        f"Cosa succede a {sr.target} se…",
        f"{_series_note(sr.series)}La previsione base a confronto con {len(sr.scenarios)} scenari nei prossimi "
        f"{sr.horizon} {pw}. Gli effetti sono stime del modello dalla storia, non prove di causa ed effetto.",
    ) as view:
        kit.warnings(sr.warnings)
        with Row(gap=6, css_class="flex-wrap"):
            for r in sr.comparison[1:]:
                delta = r["delta_vs_base"]
                Metric(
                    label=f"«{r['scenario']}»",
                    value=_signed(delta, d),
                    description=(
                        "sul totale"
                        if r["delta_vs_base_pct"] is None
                        else f"{signed_pct(r['delta_vs_base_pct'])} sul totale"
                    ),
                )
        with Column(gap=3):
            LineChart(
                data=rows,
                series=[
                    ChartSeries(data_key="storico", label="Storico", color=C_MUTED),
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
                height=360,
                show_dots=False,
                animate=False,
            )
            kit.note(
                "Ogni linea è il valore più probabile in uno scenario. Dove le linee si separano, il cambio conta."
            )
        DataTable(
            columns=[
                DataTableColumn(key="scenario", header="Scenario"),
                DataTableColumn(key="totale", header="Totale", align="right"),
                DataTableColumn(key="vs_base", header="Differenza", align="right"),
                DataTableColumn(key="finestra", header="Periodo cambiato"),
                DataTableColumn(key="effetto", header="Effetto nel periodo", align="right"),
                DataTableColumn(key="dopo", header="Effetto dopo", align="right"),
            ],
            rows=comp,
        )
        kit.next_steps(
            [
                ("Quale conviene?", "Quale di questi scenari conviene, e con quale rischio?"),
                (
                    "Prova un altro scenario",
                    "Voglio provare un altro scenario: quali covariate posso cambiare?",
                ),
            ]
        )
    return kit.app(view, f"Scenari {sr.target}")


def anomalies_app(a: AnomalyResult, max_points: int = 400) -> PrefabApp:
    pts = a.points.iloc[-max_points:]
    d = kit.digits_for(list(pts["actual"].values))
    year = _axis_year(pd.DatetimeIndex(pts.index))
    rows = [
        {
            "date": day(ts, a.freq, year),
            "reale": kit.chart_value(r["actual"], d),
            "mediana": kit.chart_value(r["median"], d),
            BAND_OUTER: kit.band_point(r["lower"], r["upper"], d),
        }
        for ts, r in pts.iterrows()
    ]
    table = [
        {
            "data": day(pd.Timestamp(r["date"]), a.freq),
            "reale": num(r["actual"], d),
            "atteso": num(r["expected_median"], d),
            "intervallo": f"{num(r['expected_range'][0], d)} – {num(r['expected_range'][1], d)}",
            "direzione": "↑ sopra" if r["direction"] == "sopra" else "↓ sotto",
            "gravita": round(float(r["score"]), 1),
            "note": "; ".join(r["notes"]) or "—",
        }
        for r in a.anomalies
    ]
    total = int(a.points["anomaly"].sum())
    chance = a.expected_by_chance
    pw = period_word(a.freq, len(a.points))
    if total == 0:
        title = "Nessun valore fuori dal normale"
    elif total <= chance * 1.5:
        title = f"{total} valori insoliti, in linea con il caso"
    else:
        title = f"{total} valori fuori dal normale"
    with kit.page(
        f"Anomalie · Timestral-{a.model}",
        f"Valori insoliti in {a.target}",
        f"{_series_note(a.series)}Ho confrontato ognuno degli ultimi {len(a.points)} {pw} con ciò che il modello "
        "si aspettava guardando solo i dati precedenti.",
    ) as view:
        kit.warnings(a.warnings)
        with Column(gap=2, css_class="tm-panel"):
            kit.label("Verdetto")
            Text(title, css_class="tm-verdict")
            Text(
                f"Fuori dall'intervallo del {a.interval:.0%}, per puro caso, ne capiterebbero circa {num(chance, 1)}. "
                + ("Ce ne sono di più: vale la pena controllarli." if total > chance * 1.5 else ""),
                css_class="tm-sub",
            )
        with Column(gap=3):
            kit.band_chart(
                rows,
                actual_key="reale",
                actual_label="Valori reali",
                median_label="Atteso",
                outer_label=f"Normale ({a.interval:.0%})",
                inner_label=None,
                height=320,
            )
            kit.note("Un valore è insolito quando la linea sottile, i valori reali, esce dall'area colorata.")
        if table:
            DataTable(
                columns=[
                    DataTableColumn(key="data", header="Data"),
                    DataTableColumn(key="reale", header="Reale", align="right"),
                    DataTableColumn(key="atteso", header="Atteso", align="right"),
                    DataTableColumn(key="intervallo", header="Normale tra"),
                    DataTableColumn(key="direzione", header="Rispetto all'atteso"),
                    DataTableColumn(key="gravita", header="Gravità", sortable=True, align="right"),
                    DataTableColumn(key="note", header="Possibile causa", cell_class="tm-wrap"),
                ],
                rows=table,
                paginated=len(table) > 8,
                page_size=8,
            )
        kit.next_steps(
            [
                ("Da cosa dipendono?", f"Aiutami a capire le cause delle anomalie {a.id}."),
                (
                    "Escludile dalla previsione",
                    "Rifai la previsione trattando queste anomalie come dati mancanti.",
                ),
            ]
        )
    return kit.app(view, f"Anomalie {a.target}")


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
    with kit.page(
        f"Covariate · Timestral-{r.model}",
        f"Quali colonne aiutano a prevedere {r.target}",
        f"{_series_note(r.series)}Ho rifatto il backtest togliendo una colonna alla volta: se senza di lei "
        "l'errore cresce, la colonna è utile. Misura l'utilità per prevedere, non la causa.",
    ) as view:
        kit.warnings(r.warnings)
        with Row(gap=6, css_class="flex-wrap"):
            if g is not None:
                Metric(
                    label="Senza nessuna covariata",
                    value=f"errore {signed_pct(g)}",
                    description="rispetto a usarle tutte",
                )
            Metric(label="Prove", value=str(r.folds), description=f"da {r.horizon} periodi ciascuna")
        with Column(gap=3):
            BarChart(
                data=bars,
                series=[ChartSeries(data_key="delta", label="Peggioramento % se la tolgo", color=C_MEDIAN)],
                x_axis="covariata",
                height=300,
                animate=False,
            )
            kit.note(
                "Barra positiva: la colonna aiuta. Barra negativa: aggiunge rumore e conviene toglierla."
            )
        DataTable(
            columns=[
                DataTableColumn(key="covariata", header="Colonna"),
                DataTableColumn(key="ruolo", header="Ruolo"),
                DataTableColumn(
                    key="delta_wql", header="Δ errore %", sortable=True, format="number:1", align="right"
                ),
                DataTableColumn(
                    key="delta_mase", header="Δ MASE %", sortable=True, format="number:1", align="right"
                ),
                DataTableColumn(key="finestre", header="Prove peggiori senza", align="right"),
                DataTableColumn(key="giudizio", header="Giudizio"),
            ],
            rows=table,
        )
        kit.next_steps(
            [("Usa solo quelle utili", f"Rifai la previsione usando solo le covariate utili secondo {r.id}.")]
        )
    return kit.app(view, f"Importanza covariate {r.target}")


def forecast_png(r: ForecastResult, history_points: int = HISTORY_POINTS) -> bytes:
    """Grafico statico con bande piene (alternativa per i client senza MCP Apps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Solo i caratteri installati: matplotlib avvisa per ogni carattere mancante della lista.
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Arial", "Helvetica", "Liberation Sans") if f in installed] + ["DejaVu Sans"]
    hist = r.history.iloc[-history_points:]
    with plt.rc_context(
        {
            "font.family": fonts,
            "text.color": theme.INK,
            "axes.labelcolor": theme.GRAPHITE,
            "xtick.color": theme.STONE,
            "ytick.color": theme.STONE,
            "axes.edgecolor": theme.SAND,
        }
    ):
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=130, facecolor=theme.PAPER)
        ax.set_facecolor(theme.PAPER)
        ax.fill_between(
            r.future_index,
            r.lower,
            r.upper,
            color=theme.ULTRAMARINE_100,
            lw=0,
            label=f"Intervallo {r.interval:.0%}",
        )
        ax.fill_between(
            r.future_index, r.q(0.25), r.q(0.75), color=theme.ULTRAMARINE_200, lw=0, label="Intervallo 50%"
        )
        ax.plot(hist.index, hist.values, color=theme.INK, lw=1.5, label="Storico")
        ax.plot(r.future_index, r.median, color=theme.ULTRAMARINE, lw=2.2, label="Previsione (mediana)")
        ax.axvline(hist.index[-1], color=theme.STONE, lw=0.8, ls=(0, (2, 4)))
        ax.set_title(
            f"{r.target} · serie '{r.series}' · Timestral-{r.model}", loc="left", fontweight="bold", pad=14
        )
        ax.grid(axis="y", color=theme.LINEN, lw=0.8)
        ax.legend(loc="upper left", frameon=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        fig.autofmt_xdate()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=theme.PAPER)
    plt.close(fig)
    return buf.getvalue()
