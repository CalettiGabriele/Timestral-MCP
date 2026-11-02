"""Importanza delle covariate per ablation: quanto peggiora la previsione sul passato se ne tolgo una."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from timestral_mcp.analytics import backtest as bt
from timestral_mcp.analytics.covariates import fmt_date
from timestral_mcp.analytics.summary import fmt
from timestral_mcp.data.ingest import Dataset

MAX_COVARIATES = 10
NOTABLE_PCT = 2.0  # sotto questo peggioramento/miglioramento % la differenza non è significativa
MAJORITY = 0.6  # quota di finestre (tra quelle in cui la covariata fa differenza) che deve concordare
MIN_INFORMATIVE_FOLDS = 2  # con meno finestre 'informative' il giudizio resta incerto
ALL, NONE = "all", "none"
USEFUL, USELESS, UNCERTAIN = "utile", "inutile o dannosa", "neutra o incerta"


@dataclass
class ImportancePlan:
    ds: Dataset
    series: str
    target: str
    model: str
    horizon: int
    interval: float
    n_folds: int
    known: list[str]
    past: list[str]
    configs: list[tuple[str, list[str], list[str]]]  # (nome, known, past)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImportanceResult:
    id: str
    dataset_id: str
    series: str
    target: str
    model: str
    horizon: int
    freq: str
    folds: int
    first_origin: str
    all_covariates: dict[str, Any]
    no_covariates: dict[str, Any]
    gain_vs_none_pct: dict[str, float | None]
    rows: list[dict[str, Any]]
    not_run: list[str]
    warnings: list[str] = field(default_factory=list)


def _auto_roles(ds: Dataset, df, target: str) -> tuple[list[str], list[str]]:
    last = df[target].last_valid_index()
    known, past = [], []
    for c in ds.value_columns:
        if c == target:
            continue
        after = df[c].loc[df.index > last] if last is not None else df[c].iloc[0:0]
        (known if after.notna().any() else past).append(c)
    return known, past


def plan_importance(
    ds: Dataset,
    *,
    target: str,
    horizon: int,
    series: str | None = None,
    model: str | None = None,
    n_folds: int = 4,
    interval: float = 0.8,
    known_covariates: list[str] | None = None,
    past_covariates: list[str] | None = None,
) -> ImportancePlan:
    key, df = ds.get_series(series)
    if target not in ds.value_columns:
        raise ValueError(f"Target {target!r} non trovato. Colonne numeriche: {', '.join(ds.value_columns)}")
    if known_covariates is None and past_covariates is None:
        known, past = _auto_roles(ds, df, target)
        auto = True
    else:
        known, past, auto = list(known_covariates or []), list(past_covariates or []), False
    for c in known + past:
        if c not in ds.value_columns or c == target:
            raise ValueError(f"Covariata {c!r} non valida. Colonne numeriche: {', '.join(ds.value_columns)}")
    if set(known) & set(past):
        raise ValueError(f"Covariate sia passate che note: {', '.join(set(known) & set(past))}")
    k = len(known) + len(past)
    if k == 0:
        raise ValueError(
            "Nessuna covariata da valutare: il dataset non ha altre colonne numeriche oltre al target."
        )
    if k > MAX_COVARIATES:
        raise ValueError(
            f"Troppe covariate ({k}): il massimo è {MAX_COVARIATES}, ogni covariata richiede un backtest in più. "
            "Indicane un sottoinsieme con known_covariates/past_covariates."
        )
    warnings: list[str] = []
    if auto:
        warnings.append(
            f"Covariate scelte in automatico: note in anticipo [{', '.join(known) or '-'}] (hanno righe future), "
            f"solo passate [{', '.join(past) or '-'}]."
        )
    if n_folds < 3:
        warnings.append("Con meno di 3 finestre di test le differenze sono poco affidabili.")
    configs = [(ALL, known, past), (NONE, [], [])]
    configs += [
        (f"without:{c}", [x for x in known if x != c], [x for x in past if x != c]) for c in known + past
    ]
    return ImportancePlan(
        ds=ds,
        series=key,
        target=target,
        model=model or "",
        horizon=horizon,
        interval=interval,
        n_folds=n_folds,
        known=known,
        past=past,
        configs=configs,
        warnings=warnings,
    )


def run_config(plan: ImportancePlan, known: list[str], past: list[str]) -> tuple[list[bt.Fold], list[str]]:
    bplan = bt.plan_backtest(
        plan.ds,
        target=plan.target,
        horizon=plan.horizon,
        series=plan.series,
        model=plan.model or None,
        n_folds=plan.n_folds,
        interval=plan.interval,
        past_covariates=past or None,
        known_covariates=known or None,
    )
    folds = [bt.run_fold(bplan, o) for o in bplan.origins]
    plan.model = bplan.model
    return folds, bplan.warnings


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


def _summary(folds: list[bt.Fold]) -> dict[str, Any]:
    return {
        "mase": _mean([f.mase for f in folds]),
        "wql": _mean([f.wql for f in folds]),
        "per_fold_wql": [f.wql for f in folds],
        "origins": [f.origin for f in folds],
    }


def _pct(new: float | None, base: float | None) -> float | None:
    return None if new is None or not base else 100 * (new / base - 1)


def _verdict(delta: float | None, worse: int, better: int) -> str:
    """Giudizio netto solo se la maggioranza delle finestre in cui la covariata fa differenza concorda."""
    informative = worse + better  # finestre in cui togliere la covariata cambia il risultato
    if delta is None or informative < MIN_INFORMATIVE_FOLDS:
        return UNCERTAIN
    need = math.ceil(MAJORITY * informative)
    if delta >= NOTABLE_PCT and worse >= need:
        return USEFUL
    if delta <= -NOTABLE_PCT and better >= need:
        return USELESS
    return UNCERTAIN


def run_importance(
    plan: ImportancePlan,
    importance_id: str,
    *,
    max_seconds: float | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ImportanceResult:
    """Esegue le configurazioni in ordine (tutte, nessuna, una tolta alla volta) entro il tempo massimo."""
    results: dict[str, dict[str, Any]] = {}
    warnings = list(plan.warnings)
    started = time.monotonic()
    total = len(plan.configs)
    not_run: list[str] = []
    for i, (name, known, past) in enumerate(plan.configs):
        if max_seconds is not None and i >= 2 and time.monotonic() - started > max_seconds:
            not_run = [c for c, _, _ in plan.configs[i:]]
            warnings.append(
                f"Tempo massimo di {max_seconds:.0f}s raggiunto: {len(not_run)} covariate non valutate "
                f"({', '.join(n.removeprefix('without:') for n in not_run)}). Ripeti indicando solo quelle."
            )
            break
        label = {ALL: "con tutte le covariate", NONE: "senza covariate"}.get(
            name, f"senza {name.removeprefix('without:')}"
        )
        if progress:
            progress(i, total, f"Backtest {label} ({i + 1}/{total})")
        folds, bwarn = run_config(plan, known, past)
        results[name] = _summary(folds)
        for w in bwarn:
            if w not in warnings:
                warnings.append(w)
    if progress:
        progress(total, total, "Completato")

    base, none = results[ALL], results[NONE]
    n = len(base["per_fold_wql"])
    rows = []
    for name, _, _ in plan.configs[2:]:
        if name not in results:
            continue
        cov = name.removeprefix("without:")
        res = results[name]
        pairs = [
            (a, b)
            for a, b in zip(base["per_fold_wql"], res["per_fold_wql"], strict=True)
            if a is not None and b is not None
        ]
        worse = sum(1 for a, b in pairs if b > a + 1e-12)
        better = sum(1 for a, b in pairs if b < a - 1e-12)
        d_wql, d_mase = _pct(res["wql"], base["wql"]), _pct(res["mase"], base["mase"])
        rows.append(
            {
                "covariate": cov,
                "role": "nota in anticipo" if cov in plan.known else "passata",
                "wql_without": None if res["wql"] is None else round(res["wql"], 4),
                "delta_wql_pct": None if d_wql is None else round(d_wql, 1),
                "delta_mase_pct": None if d_mase is None else round(d_mase, 1),
                "folds_worse_without": f"{worse}/{len(pairs)}",
                "folds_better_without": f"{better}/{len(pairs)}",
                "informative_folds": worse + better,
                "verdict": _verdict(d_wql, worse, better),
            }
        )
    rows.sort(key=lambda r: (r["delta_wql_pct"] is None, -(r["delta_wql_pct"] or 0)))
    for r in rows:
        if r["role"] == "nota in anticipo":
            s = plan.ds.series[plan.series][r["covariate"]].dropna()
            if set(np.unique(s)) <= {0.0, 1.0} and int((s == 1).sum()) < 5:
                warnings.append(
                    f"'{r['covariate']}' è attiva solo {int((s == 1).sum())} volte nello storico: giudizio poco affidabile."
                )
    return ImportanceResult(
        id=importance_id,
        dataset_id=plan.ds.id,
        series=plan.series,
        target=plan.target,
        model=plan.model,
        horizon=plan.horizon,
        freq=plan.ds.freq,
        folds=n,
        first_origin=fmt_date(base["origins"][0], plan.ds.freq),
        all_covariates={"mase": base["mase"], "wql": base["wql"]},
        no_covariates={"mase": none["mase"], "wql": none["wql"]},
        gain_vs_none_pct={
            "wql": None
            if _pct(none["wql"], base["wql"]) is None
            else round(_pct(none["wql"], base["wql"]), 1),
            "mase": None
            if _pct(none["mase"], base["mase"]) is None
            else round(_pct(none["mase"], base["mase"]), 1),
        },
        rows=rows,
        not_run=not_run,
        warnings=warnings,
    )


def importance_summary(r: ImportanceResult) -> str:
    g = r.gain_vs_none_pct
    lines = [
        f"Importanza delle covariate {r.id} · Timestral-{r.model} · serie '{r.series}' · target '{r.target}' · "
        f"{r.folds} finestre da {r.horizon} periodi (dalla {r.first_origin})",
        "Metodo: backtest con tutte le covariate, poi togliendone una alla volta sulle stesse finestre. "
        "Importanza = peggioramento % dell'errore probabilistico (WQL) quando la covariata manca. Misura quanto la covariata "
        "aiuta a PREVEDERE, non prova che CAUSI le vendite.",
        f"Con tutte le covariate: WQL {fmt(r.all_covariates['wql'], 3)}, MASE {fmt(r.all_covariates['mase'])}. "
        f"Senza nessuna covariata: WQL {fmt(r.no_covariates['wql'], 3)}, MASE {fmt(r.no_covariates['mase'])}"
        + (
            f" → togliendole tutte l'errore (WQL) {'sale' if g['wql'] >= 0 else 'scende'} del {abs(g['wql']):.1f}%."
            if g["wql"] is not None
            else "."
        ),
        "",
        "covariata | ruolo | Δ WQL se tolta | Δ MASE se tolta | finestre peggiori senza | giudizio",
    ]
    for row in r.rows:
        dw = "n/d" if row["delta_wql_pct"] is None else f"{row['delta_wql_pct']:+.1f}%"
        dm = "n/d" if row["delta_mase_pct"] is None else f"{row['delta_mase_pct']:+.1f}%"
        lines.append(
            f"{row['covariate']} | {row['role']} | {dw} | {dm} | {row['folds_worse_without']} | {row['verdict']}"
        )
    lines.append(
        "Come leggerla: positivo = senza la covariata la previsione peggiora (covariata utile); negativo = migliora (rumore). "
        f"Sotto ±{NOTABLE_PCT:g}%, con finestre in disaccordo o con meno di {MIN_INFORMATIVE_FOLDS} finestre in cui la covariata "
        "fa differenza (es. una promozione rara) il giudizio è 'neutra o incerta'."
    )
    lines.append(
        "Cautele: con poche finestre le differenze piccole sono rumore; covariate molto correlate tra loro si sostituiscono "
        "e ciascuna sembra meno importante di quanto sia."
    )
    for w in r.warnings:
        lines.append(f"ATTENZIONE: {w}")
    return "\n".join(lines)
