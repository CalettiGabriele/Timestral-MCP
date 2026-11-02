import numpy as np
import pytest

from tests.helpers import normal_forecast
from timestral_mcp.analytics.decision import quantity_for_service_level, threshold_probability


def test_probability_at_median_is_half_and_monotone():
    r = normal_forecast([100.0] * 5, sigma=10)
    out = threshold_probability(r, 100, "below")
    assert all(abs(row["probability"] - 0.5) < 1e-6 for row in out["rows"])
    above = threshold_probability(r, 100, "above")
    assert abs(above["rows"][0]["probability"] - 0.5) < 1e-6
    p_low = threshold_probability(r, 90, "below")["rows"][0]["probability"]
    p_high = threshold_probability(r, 110, "below")["rows"][0]["probability"]
    assert p_low < 0.5 < p_high
    assert abs(p_low - 0.159) < 0.02  # P(Z < -1) con interpolazione tra i quantili


def test_tail_probabilities_are_bounds_not_made_up_numbers():
    r = normal_forecast([100.0] * 3, sigma=10)
    far = threshold_probability(r, 50, "below")
    row = far["rows"][0]
    assert row["probability"] is None and row["probability_range"] == [0.0, 0.01]
    assert row["text"].startswith("meno del")
    assert far["expected_periods_beyond_threshold"] == [0.0, 0.03]
    above_far = threshold_probability(r, 150, "above")
    assert above_far["rows"][0]["probability_range"] == [0.0, 0.01]


def test_at_least_one_period_bounds_and_date_filter():
    r = normal_forecast([100.0] * 10, sigma=10)
    out = threshold_probability(r, 110, "above", start="2026-01-03", end="2026-01-06")
    assert out["periods"] == 4
    lo, hi = out["at_least_one_period"]
    p = out["rows"][0]["probability"]
    assert abs(lo - p) < 1e-6 and abs(hi - min(1, 4 * p)) < 1e-3
    with pytest.raises(ValueError, match="Nessun periodo"):
        threshold_probability(r, 1, start="2030-01-01")


def test_service_level_quantity():
    r = normal_forecast([100.0] * 7, sigma=10)
    out = quantity_for_service_level(r, 0.9)
    q90 = out["rows"][0]["quantity_90"]
    assert abs(q90 - (100 + 1.2816 * 10)) < 0.5
    w = out["window"]
    assert w["median_total"] == 700
    assert abs(w["cautious_total"] - 7 * q90) < 0.01
    assert w["median_total"] < w["independent_total"] < w["cautious_total"]
    # indipendenza: 700 + z*sqrt(7)*10
    assert abs(w["independent_total"] - (700 + 1.2816 * np.sqrt(7) * 10)) < 2


def test_single_period_has_no_window_and_invalid_level():
    r = normal_forecast([100.0] * 5, sigma=10)
    out = quantity_for_service_level(r, 0.95, start="2026-01-02", end="2026-01-02")
    assert "window" not in out and out["periods"] == 1
    with pytest.raises(ValueError, match="service_level"):
        quantity_for_service_level(r, 0.3)
    with pytest.raises(ValueError, match="service_level"):
        quantity_for_service_level(r, 0.999)
