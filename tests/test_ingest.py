import numpy as np
import pandas as pd
import pytest

from timestral_mcp.data.frequency import describe_frequency, infer_frequency, season_length
from timestral_mcp.data.ingest import build_dataset, read_source


def build(df, **kw):
    return build_dataset(df, dataset_id="ds_t", name="t", source="test", **kw)


def test_frequency_inference_and_season():
    assert infer_frequency(pd.date_range("2024-01-01", periods=30, freq="D")) == "D"
    assert infer_frequency(pd.date_range("2024-01-01", periods=30, freq="h")) == "h"
    assert infer_frequency(pd.date_range("2024-01-01", periods=24, freq="MS")) == "MS"
    assert infer_frequency(pd.date_range("2024-01-31", periods=24, freq="ME")) == "ME"
    assert infer_frequency(pd.date_range("2024-01-07", periods=30, freq="W")).startswith("W")
    assert (season_length("D"), season_length("h"), season_length("MS"), season_length("15min")) == (
        7,
        24,
        12,
        96,
    )
    assert describe_frequency("MS") == "mensile"


def test_frequency_with_gaps():
    idx = pd.date_range("2024-01-01", periods=40, freq="D").delete([5, 6, 20])
    assert infer_frequency(idx) == "D"


def test_italian_format_and_decimal_comma(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text(
        "data;vendite\n01/02/2024;1.234,50\n02/02/2024;1.300,00\n03/02/2024;1.250,25\n04/02/2024;1.100,00\n",
        encoding="utf-8",
    )
    df, origin = read_source(str(p))
    ds = build(df)
    s = ds.series["serie"]["vendite"]
    assert s.iloc[0] == 1234.5 and s.index[1].day == 2 and s.index[0].month == 2
    assert ds.freq == "D"


def test_pasted_text():
    text = "date,sales\n2024-01-01,10\n2024-01-02,12\n2024-01-03,11\n2024-01-04,13"
    df, origin = read_source(text)
    assert origin == "testo incollato"
    assert build(df).value_columns == ["sales"]


def test_long_format_multiple_series():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.concat(
        [pd.DataFrame({"date": idx, "sku": k, "qty": np.arange(10) + i}) for i, k in enumerate(["A", "B"])]
    )
    with pytest.raises(ValueError, match="series_column"):
        build(df)
    ds = build(df, series_column="sku")
    assert set(ds.series) == {"A", "B"}
    with pytest.raises(ValueError, match="indica quale usare"):
        ds.get_series(None)


def test_gaps_are_reindexed_and_future_rows_kept():
    idx = pd.date_range("2024-01-01", periods=20, freq="D").delete([4])
    df = pd.DataFrame({"date": idx, "y": np.arange(19, dtype=float), "promo": 0.0})
    future = pd.DataFrame({"date": pd.date_range("2024-01-21", periods=3), "y": np.nan, "promo": 1.0})
    ds = build(pd.concat([df, future]))
    s = ds.series["serie"]
    assert (
        len(s) == 23 and np.isnan(s["y"].iloc[4]) and s["y"].last_valid_index() == pd.Timestamp("2024-01-20")
    )
    assert s["promo"].iloc[-1] == 1.0


def test_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="File non trovato"):
        read_source(str(tmp_path / "manca.csv"))
    with pytest.raises(ValueError, match="date_column"):
        build(pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2, 3, 4]}))
    with pytest.raises(ValueError, match="numerica"):
        build(pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4), "t": list("abcd")}))
