"""Dalle previsioni alle decisioni: probabilità di soglia e quantità per livello di servizio.

Tutto è ricavato dai 21 quantili marginali di ogni periodo: niente di ciò che richiede la
distribuzione congiunta tra periodi (es. "almeno un giorno sopra soglia") è esatto, e le
funzioni lo dichiarano con intervalli di valori possibili.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from timestral_mcp.analytics.forecast import ForecastResult
from timestral_mcp.analytics.summary import _date, fmt

_Z80 = NormalDist().inv_cdf(0.9)  # q90 di una normale standard


def _select(r: ForecastResult, start: str | None, end: str | None) -> np.ndarray:
    idx = r.future_index
    mask = np.ones(len(idx), dtype=bool)
    if start:
        mask &= idx >= pd.Timestamp(start)
    if end:
        mask &= idx <= pd.Timestamp(end)
    pos = np.flatnonzero(mask)
    if pos.size == 0:
        raise ValueError(
            f"Nessun periodo nell'intervallo richiesto. La previsione copre "
            f"{_date(idx[0], r.freq)} → {_date(idx[-1], r.freq)}."
        )
    return pos


def cdf_bounds(r: ForecastResult, i: int, x: float) -> tuple[float, float]:
    """(minimo, massimo) di P(Y <= x) nel periodo i, interpolando tra i quantili.

    Dentro l'intervallo dei quantili il valore è unico (minimo = massimo); oltre gli estremi
    (sotto q01 o sopra q99) si può solo limitare: [0, 0.01] oppure [0.99, 1].
    """
    q = r.quantiles[i]
    lv = np.asarray(r.levels)
    if x < q[0]:
        return 0.0, float(lv[0])
    if x > q[-1]:
        return float(lv[-1]), 1.0
    p = float(np.interp(x, q, lv))
    return p, p


def _pct_text(lo: float, hi: float) -> str:
    if lo == hi:
        return f"{100 * lo:.0f}%" if lo >= 0.095 or lo == 0 else f"{100 * lo:.1f}%"
    if lo == 0:
        return f"meno del {100 * hi:.0f}%"
    return f"più del {100 * lo:.0f}%"


def threshold_probability(
    r: ForecastResult,
    threshold: float,
    direction: Literal["below", "above"] = "below",
    start: str | None = None,
    end: str | None = None,
    max_rows: int = 200,
) -> dict[str, Any]:
    pos = _select(r, start, end)
    rows, lows, highs = [], [], []
    for i in pos:
        c_lo, c_hi = cdf_bounds(r, i, threshold)
        lo, hi = (c_lo, c_hi) if direction == "below" else (1 - c_hi, 1 - c_lo)
        lows.append(lo)
        highs.append(hi)
        rows.append(
            {
                "date": _date(r.future_index[i], r.freq),
                "median": round(float(r.median[i]), 4),
                "probability": round(lo, 4) if lo == hi else None,
                "probability_range": None if lo == hi else [round(lo, 4), round(hi, 4)],
                "text": _pct_text(lo, hi),
            }
        )
    lows_a, highs_a = np.asarray(lows), np.asarray(highs)
    worst = int(np.argmax(highs_a + lows_a))
    word = "sotto" if direction == "below" else "sopra"
    return {
        "forecast_id": r.id,
        "question": f"probabilità di essere {word} {threshold:g} in ciascun periodo",
        "threshold": threshold,
        "direction": direction,
        "periods": len(pos),
        "rows": rows[:max_rows],
        "truncated": len(rows) > max_rows,
        "max_probability_period": {"date": rows[worst]["date"], "text": rows[worst]["text"]},
        "expected_periods_beyond_threshold": [round(float(lows_a.sum()), 2), round(float(highs_a.sum()), 2)],
        "at_least_one_period": [round(float(lows_a.max()), 4), round(float(min(1.0, highs_a.sum())), 4)],
        "notes": [
            "Le probabilità sono per singolo periodo e interpolate tra i 21 quantili (da 1% a 99%); "
            "oltre gli estremi si può solo indicare un limite (es. 'meno del 1%').",
            "'expected_periods_beyond_threshold' è il numero atteso di periodi oltre la soglia (intervallo di valori possibili). "
            "'at_least_one_period' è un intervallo: la probabilità che almeno un periodo superi la soglia "
            "dipende dalla correlazione tra periodi e non è calcolabile dai soli quantili.",
        ],
    }


def quantity_for_service_level(
    r: ForecastResult,
    service_level: float = 0.9,
    start: str | None = None,
    end: str | None = None,
    max_rows: int = 200,
) -> dict[str, Any]:
    lv = np.asarray(r.levels)
    if not 0.5 <= service_level <= lv[-1]:
        raise ValueError(
            f"service_level deve essere tra 0.5 e {lv[-1]:g} (es. 0.9 per coprire il 90% dei casi)."
        )
    pos = _select(r, start, end)
    per_period = np.array([np.interp(service_level, lv, r.quantiles[i]) for i in pos])
    median = r.median[pos]
    spread = np.array([np.interp(0.9, lv, r.quantiles[i]) - np.interp(0.1, lv, r.quantiles[i]) for i in pos])
    sigma = spread / (2 * _Z80)  # deviazione standard equivalente (assumendo forma normale)
    z = NormalDist().inv_cdf(service_level)
    median_total = float(median.sum())
    cautious = float(per_period.sum())  # somma dei quantili: assume tutti i periodi "sfavorevoli" insieme
    independent = float(median_total + z * np.sqrt((sigma**2).sum()))  # errori indipendenti tra periodi
    window = len(pos) > 1
    rows = [
        {
            "date": _date(r.future_index[i], r.freq),
            "median": round(float(median[k]), 4),
            f"quantity_{round(service_level * 100)}": round(float(per_period[k]), 4),
            "safety_buffer": round(float(per_period[k] - median[k]), 4),
        }
        for k, i in enumerate(pos)
    ]
    out: dict[str, Any] = {
        "forecast_id": r.id,
        "service_level": service_level,
        "first_period": _date(r.future_index[pos[0]], r.freq),
        "last_period": _date(r.future_index[pos[-1]], r.freq),
        "periods": len(pos),
        "rows": rows[:max_rows],
        "truncated": len(rows) > max_rows,
        "notes": [
            f"Quantità che, secondo il modello, copre circa il {service_level:.0%} dei casi in ciascun periodo "
            "(interpolata tra i quantili)."
        ],
    }
    if window:
        out["window"] = {
            "median_total": round(median_total, 2),
            "cautious_total": round(cautious, 2),
            "independent_total": round(independent, 2),
            "safety_stock_range": [round(independent - median_total, 2), round(cautious - median_total, 2)],
            "summary": (
                f"Per coprire ~{service_level:.0%} dei casi su {len(pos)} periodi servono tra "
                f"{fmt(independent)} e {fmt(cautious)} (totale mediano {fmt(median_total)})."
            ),
        }
        out["notes"].append(
            "Totale su più periodi: 'cautious_total' somma i quantili (assume che i periodi vadano male insieme: stima per eccesso); "
            "'independent_total' assume errori indipendenti tra periodi con forma approssimata normale (stima per difetto se i periodi "
            "sono correlati). Il valore reale sta in genere tra i due: per il lead time si scelga una quantità dentro l'intervallo."
        )
    return out
