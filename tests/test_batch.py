import numpy as np
import pandas as pd
import pytest

from timestral_mcp.analytics import batch as bm
from timestral_mcp.data.ingest import build_dataset


def make_batch_ds(n_days=200, products=("A", "B", "C", "D")):
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    frames = []
    rng = np.random.default_rng(3)
    for k, p in enumerate(products):
        level = 50 * (k + 1)
        y = level + 10 * np.sin(np.arange(n_days) * 2 * np.pi / 7) + rng.normal(0, 2, n_days)
        if p == "C":  # crollo nell'ultima settimana
            y[-7:] *= 0.4
        if p == "D":  # prodotto con molti zeri
            y[::2] = 0
        frames.append(pd.DataFrame({"d": idx.strftime("%Y-%m-%d"), "sku": p, "qty": y}))
    short = pd.DataFrame({"d": idx[-6:].strftime("%Y-%m-%d"), "sku": "NEW", "qty": np.arange(6) + 5.0})
    return build_dataset(
        pd.concat(frames + [short]), dataset_id="ds_b", name="b", source="t", series_column="sku"
    )


def run(ds, **kw):
    return bm.run_batch(ds, batch_id="bt_x", target="qty", horizon=30, **kw)


def test_batch_rows_flags_and_totals(fake):
    ds = make_batch_ds()
    b = run(ds)
    by = {r["series"]: r for r in b.rows}
    assert (
        set(by) == {"A", "B", "C", "D"} and "NEW" in b.skipped
    )  # 6 punti: sotto il minimo di 8 osservazioni
    assert bm.DROP_FLAG in by["C"]["flags"] and by["C"]["change_pct"] < -30
    assert all(r["uncertainty_pct"] < 1000 for r in b.rows)  # serie con molti zeri: nessun valore assurdo
    drops = [x["series"] for x in bm.attention(b)["biggest_drops"]]
    assert drops == ["C"]  # le variazioni sotto il 5% non sono segnalazioni
    assert bm.ZEROS_FLAG in by["D"]["flags"]
    t = bm.totals(b)
    assert (
        t["series_forecasted"] == 4
        and abs(t["forecast_total"] - sum(r["forecast_total"] for r in b.rows)) < 0.1
    )
    vols = [r["forecast_total"] for r in bm.sorted_rows(b.rows, "volume")]
    assert vols == sorted(vols, reverse=True)
    att = bm.attention(b)
    assert att["biggest_drops"][0]["series"] == "C"
    assert all(k.startswith("bt_x::") for k in [r.id for r in b.results.values()])


def test_summary_text_mentions_key_facts(fake):
    b = run(make_batch_ds())
    text = bm.batch_summary(b)
    assert "Serie previste: 4; saltate: 1" in text and "Maggiori cali" in text and "C (" in text
    assert "batch_series_forecast(batch_id='bt_x'" in text and "stagionalità" in text


def test_series_limit_keeps_largest_and_reports_rest(fake):
    b = run(make_batch_ds(), max_series=2)
    assert len(b.rows) == 2 and len(b.not_run) >= 2 and any("più serie del limite" in w for w in b.warnings)
    assert {r["series"] for r in b.rows} <= {"A", "B", "C", "D", "NEW"}


def test_time_budget_returns_partial_results(fake):
    clock = iter(range(0, 1000, 10))
    ticks = {"n": 0}

    def fake_monotonic():
        ticks["n"] += 1
        return next(clock)

    import time

    orig = time.monotonic
    time.monotonic = fake_monotonic
    try:
        b = run(make_batch_ds(), max_seconds=15)
    finally:
        time.monotonic = orig
    assert 0 < len(b.rows) < 4 and b.not_run and any("Tempo massimo" in w for w in b.warnings)


def test_backtest_reliability_per_series(fake):
    b = run(make_batch_ds(), with_backtest=True)
    rel = {r["series"]: r.get("reliability") for r in b.rows}
    assert set(rel.values()) <= {"alta", "media", "bassa", None} and any(v for v in rel.values())
    assert "affidabilità" in bm.batch_summary(b)
    many = make_batch_ds(products=tuple(f"P{i}" for i in range(40)))
    b2 = run(many, with_backtest=True)
    assert not b2.with_backtest and any("backtest per serie" in w for w in b2.warnings)


def test_errors_and_frames(fake):
    ds = make_batch_ds()
    with pytest.raises(ValueError, match="non trovato"):
        bm.select_series(ds, "nope", 30, None, 10)
    with pytest.raises(ValueError, match="Nessuna serie prevista"):
        run(ds, series=["NEW"])
    b = run(ds, series=["A", "B", "ZZZ"])
    assert any("non trovate" in w for w in b.warnings)
    summary, long = bm.batch_frames(b)
    assert (
        list(summary["series"]) == ["A", "B"] and set(long["series"]) == {"A", "B"} and "q10" in long.columns
    )
    assert len(long) == 2 * 30
