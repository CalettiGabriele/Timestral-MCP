"""Pezzi comuni delle MCP App: intestazione, numeri leggibili, grafico con bande, passi successivi."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import pandas as pd
from prefab_ui.actions import ToggleState
from prefab_ui.actions.mcp import SendMessage
from prefab_ui.app import PrefabApp
from prefab_ui.components import (
    Accordion,
    AccordionItem,
    Alert,
    AlertDescription,
    Button,
    Column,
    Heading,
    Row,
    Text,
)
from prefab_ui.components.charts import AreaChart, ChartSeries
from prefab_ui.rx import Rx

from timestral_mcp.ui.theme import (
    BAND_PREFIX,
    C_BAND_INNER,
    C_BAND_OUTER,
    C_HISTORY,
    C_MEDIAN,
    HIDDEN_PREFIX,
    MODE_CLASS,
    STATE_KEY,
    THEME,
)

MONTHS = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
BAND_OUTER, BAND_INNER = f"{BAND_PREFIX}", f"{BAND_PREFIX}50"
# Serie invisibili che portano nel tooltip i limiti della banda come due numeri separati.
BOUND_LO, BOUND_HI = f"{HIDDEN_PREFIX}_min", f"{HIDDEN_PREFIX}_max"


# --- numeri e date ------------------------------------------------------------------------------


def num(v: float | None, digits: int | None = None) -> str:
    """Numero all'italiana: 1.234 · 176 · 12,5 · 0,84."""
    if v is None or not np.isfinite(v):
        return "n/d"
    if digits is None:
        a = abs(v)
        digits = 0 if a >= 100 else 1 if a >= 10 else 2
    s = f"{v:,.{digits}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def signed_pct(v: float | None) -> str:
    return "n/d" if v is None or not np.isfinite(v) else f"{'+' if v >= 0 else '−'}{num(abs(v), 1)}%"


def digits_for(values: Sequence[float]) -> int:
    """Decimali adatti alla scala della serie (per tabelle e grafici)."""
    finite = [abs(v) for v in values if v is not None and np.isfinite(v)]
    top = max(finite) if finite else 0
    return 0 if top >= 100 else 1 if top >= 10 else 2


def is_intraday(freq: str) -> bool:
    return freq.lower().endswith(("h", "min", "t", "s"))


def day(ts: pd.Timestamp, freq: str = "D", year: bool = True) -> str:
    """Data leggibile: 31 dic 2025 · 31 dic 14:00 · dic 2025."""
    ts = pd.Timestamp(ts)
    if is_intraday(freq):
        return f"{ts.day} {MONTHS[ts.month - 1]} {ts:%H:%M}"
    if freq.upper().startswith(("M", "Q", "Y", "A")):
        return f"{MONTHS[ts.month - 1]} {ts.year}"
    return f"{ts.day} {MONTHS[ts.month - 1]}" + (f" {ts.year}" if year else "")


def span(start: pd.Timestamp, end: pd.Timestamp, freq: str = "D") -> str:
    """Intervallo di date compatto: 20–24 gen 2026 · 28 dic 2025 – 3 gen 2026."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not is_intraday(freq) and start.year == end.year and start.month == end.month and start != end:
        return f"{start.day}–{end.day} {MONTHS[end.month - 1]} {end.year}"
    return day(start, freq) if start == end else f"{day(start, freq)} – {day(end, freq)}"


def period_word(freq: str, n: int) -> str:
    f = freq.upper()
    words = {
        "D": ("giorno", "giorni"),
        "B": ("giorno lavorativo", "giorni lavorativi"),
        "W": ("settimana", "settimane"),
        "M": ("mese", "mesi"),
        "Q": ("trimestre", "trimestri"),
        "Y": ("anno", "anni"),
        "A": ("anno", "anni"),
        "H": ("ora", "ore"),
    }
    for key, (one, many) in words.items():
        if f.startswith(key) or (key == "H" and f.endswith("H")):
            return one if n == 1 else many
    return "periodo" if n == 1 else "periodi"


def in_cases(level: float) -> str:
    """0.8 → «8 casi su 10»; 0.95 → «95 casi su 100»."""
    pct = round(level * 100)
    return f"{pct // 10} casi su 10" if pct % 10 == 0 else f"{pct} casi su 100"


def chart_value(v: float | None, digits: int) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


# --- pagina -------------------------------------------------------------------------------------


@contextmanager
def page(kicker: str, title: str, subtitle: str | None = None) -> Iterator[Column]:
    """Contenitore della pagina con tema (chiaro di default), etichetta, titolo e sottotitolo."""
    with Column(gap=6, css_class=["tm-root", str(MODE_CLASS)]) as root:
        with Row(css_class="items-center justify-between gap-4"):
            Text(kicker, css_class="tm-kicker")
            Button(
                Rx(STATE_KEY).then("Tema chiaro", "Tema scuro"),
                variant="ghost",
                size="sm",
                on_click=ToggleState(STATE_KEY),
            )
        with Column(gap=2):
            Heading(title, level=1)
            if subtitle:
                Text(subtitle, css_class="tm-sub")
        yield root


def app(view: Column, title: str, state: dict | None = None) -> PrefabApp:
    return PrefabApp(theme=THEME, title=title, view=view, state={STATE_KEY: False, **(state or {})})


def warnings(items: Sequence[str]) -> None:
    for w in items:
        with Alert(variant="warning"):
            AlertDescription(w)


def label(text: str) -> None:
    Text(text, css_class="tm-label")


def note(text: str) -> None:
    Text(text, css_class="tm-note")


def next_steps(steps: Sequence[tuple[str, str]]) -> None:
    """Pulsanti che scrivono a Claude la domanda successiva: utili a chi usa lo strumento la prima volta."""
    with Column(gap=3):
        label("Prossimi passi")
        with Row(gap=2, css_class="flex-wrap"):
            for text, message in steps:
                Button(text, variant="outline", size="sm", on_click=SendMessage(message))


def explainer(title: str, lines: Sequence[str]) -> None:
    with Accordion(collapsible=True), AccordionItem(title=title):
        with Column(gap=2):
            for line in lines:
                Text(line, css_class="tm-note")


# --- grafico con bande --------------------------------------------------------------------------


def band_point(lo: float, hi: float, digits: int) -> list[float] | None:
    a, b = chart_value(lo, digits), chart_value(hi, digits)
    return None if a is None or b is None else [a, b]


def with_bounds(rows: list[dict]) -> list[dict]:
    """Copia i limiti della banda esterna in due chiavi numeriche, lette dal tooltip."""
    for row in rows:
        band = row.get(BAND_OUTER)
        if band and BOUND_LO not in row:
            row[BOUND_LO], row[BOUND_HI] = band
    return rows


def band_chart(
    rows: list[dict] | Rx,
    *,
    actual_key: str,
    actual_label: str,
    median_label: str,
    outer_label: str,
    inner_label: str | None,
    height: int = 360,
) -> None:
    """Storico/valori reali e mediana come linee, incertezza come bande piene (esterna e interna)."""
    series = [ChartSeries(data_key=BAND_OUTER, label=outer_label, color=C_BAND_OUTER)]
    if inner_label:
        series.append(ChartSeries(data_key=BAND_INNER, label=inner_label, color=C_BAND_INNER))
    series += [
        ChartSeries(data_key=actual_key, label=actual_label, color=C_HISTORY),
        ChartSeries(data_key="mediana", label=median_label, color=C_MEDIAN),
        ChartSeries(data_key=BOUND_LO, label="Minimo probabile", color=C_MEDIAN),
        ChartSeries(data_key=BOUND_HI, label="Massimo probabile", color=C_MEDIAN),
    ]
    if isinstance(rows, list):
        rows = with_bounds(rows)
    AreaChart(
        data=rows, series=series, x_axis="date", height=height, curve="linear", show_dots=False, animate=False
    )
