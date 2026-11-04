"""Il tema delle app deve restare allineato ai token del brand."""

import json
from pathlib import Path

from timestral_mcp.ui import theme

TOKENS = json.loads((Path(__file__).parents[1] / "brand" / "tokens" / "tokens.json").read_text())["color"]


def test_theme_colors_match_brand_tokens():
    pairs = {
        "INK": "ink",
        "GRAPHITE": "graphite",
        "STONE": "stone",
        "SAND": "sand",
        "LINEN": "linen",
        "PAPER": "paper",
        "ULTRAMARINE": "ultramarine",
        "ULTRAMARINE_400": "ultramarine-400",
        "ULTRAMARINE_200": "ultramarine-200",
        "ULTRAMARINE_100": "ultramarine-100",
        "ULTRAMARINE_50": "ultramarine-50",
        "SIGNAL_RED": "signal-red",
    }
    for attr, token in pairs.items():
        assert getattr(theme, attr).lower() == TOKENS[token].lower(), attr


def test_theme_defines_every_chart_variable():
    css = theme.THEME.to_css()
    for var in ("--tm-history", "--tm-median", "--tm-band-outer", "--tm-band-inner", "--tm-scenario-2"):
        assert css.count(var + ":") >= 2, var  # chiaro e scuro
