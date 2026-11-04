"""Identità visiva delle MCP App e del grafico PNG.

I colori rispecchiano `brand/tokens/tokens.json` (un test controlla che coincidano).

Le app partono sempre nel tema chiaro e hanno un pulsante per passare a quello scuro: il tema non dipende
dal client. Le variabili stanno su due classi (`tm-light`, `tm-dark`) applicate al contenitore della pagina,
scelte dallo stato `tm_dark` dell'app.
"""

from __future__ import annotations

from prefab_ui.rx import Rx
from prefab_ui.themes import Theme

INK = "#141414"
GRAPHITE = "#4A463F"
STONE = "#8A8375"
SAND = "#C2BAAC"
LINEN = "#D9D2C7"
PAPER = "#E3DED5"
WHITE = "#FFFFFF"
ULTRAMARINE = "#2336BC"
ULTRAMARINE_400 = "#5464CC"
ULTRAMARINE_200 = "#9AA5E4"
ULTRAMARINE_100 = "#C9CFF2"
ULTRAMARINE_50 = "#DDE1F7"
SIGNAL_RED = "#BC2323"

FONT_DISPLAY = "Arial, 'Helvetica Neue', Helvetica, 'Liberation Sans', sans-serif"
FONT_BODY = "Calibri, Carlito, 'Segoe UI', Arial, sans-serif"
FONT_MONO = "'Courier New', Courier, 'Liberation Mono', monospace"

# Colori delle serie nei grafici interattivi: variabili CSS, così seguono il tema chiaro/scuro.
C_HISTORY = "var(--tm-history)"
C_MEDIAN = "var(--tm-median)"
C_BAND_OUTER = "var(--tm-band-outer)"
C_BAND_INNER = "var(--tm-band-inner)"
C_MUTED = "var(--tm-muted-line)"
# Un solo accento: gli scenari alternativi usano toni di oltremare e neutri.
SCENARIO_COLORS = [C_MEDIAN, "var(--tm-scenario-2)", "var(--tm-scenario-3)", "var(--tm-scenario-4)"]

# Nei grafici ad area, solo le serie il cui nome inizia con questo prefisso vengono riempite (le bande).
BAND_PREFIX = "banda"
# Serie nascoste: né area né linea né legenda, solo nel tooltip (es. minimo e massimo probabile).
HIDDEN_PREFIX = "tmx"

STATE_KEY = "tm_dark"
MODE_CLASS = Rx(STATE_KEY).then("tm-dark", "tm-light")

_LIGHT = {
    "background": WHITE,
    "foreground": INK,
    "card": WHITE,
    "card-foreground": INK,
    "popover": WHITE,
    "popover-foreground": INK,
    "primary": ULTRAMARINE,
    "primary-foreground": WHITE,
    "secondary": "#F2EFEA",
    "secondary-foreground": INK,
    "muted": "#F2EFEA",
    "muted-foreground": GRAPHITE,
    "accent": ULTRAMARINE_50,
    "accent-foreground": ULTRAMARINE,
    "destructive": SIGNAL_RED,
    "border": LINEN,
    "input": SAND,
    "ring": ULTRAMARINE,
    "chart-1": ULTRAMARINE,
    "chart-2": INK,
    "chart-3": ULTRAMARINE_200,
    "chart-4": STONE,
    "chart-5": ULTRAMARINE_100,
    "radius": "0rem",
    "tm-surface": "#F2EFEA",
    "tm-label": STONE,
    "tm-history": INK,
    "tm-median": ULTRAMARINE,
    "tm-band-outer": ULTRAMARINE_50,
    "tm-band-inner": ULTRAMARINE_100,
    "tm-muted-line": STONE,
    "tm-scenario-2": GRAPHITE,
    "tm-scenario-3": ULTRAMARINE_400,
    "tm-scenario-4": STONE,
}

_DARK = {
    **_LIGHT,
    "background": INK,
    "foreground": PAPER,
    "card": "#1E1D1B",
    "card-foreground": PAPER,
    "popover": "#282624",
    "popover-foreground": PAPER,
    "secondary": "#232220",
    "secondary-foreground": PAPER,
    "muted": "#232220",
    "muted-foreground": SAND,
    "accent": "#222C78",
    "accent-foreground": ULTRAMARINE_50,
    "border": "#34312C",
    "input": GRAPHITE,
    "ring": ULTRAMARINE_200,
    "chart-1": ULTRAMARINE_200,
    "chart-2": PAPER,
    "chart-3": ULTRAMARINE_400,
    "chart-5": ULTRAMARINE,
    "tm-surface": "#232220",
    "tm-history": SAND,
    "tm-median": ULTRAMARINE_50,
    "tm-band-outer": "#1F2766",
    "tm-band-inner": "#2B3A9E",
    "tm-scenario-2": SAND,
    "tm-scenario-3": ULTRAMARINE_200,
}


def _vars(tokens: dict[str, str]) -> str:
    return " ".join(f"--{k}: {v};" for k, v in tokens.items())


_CSS = f"""
.pf-app-root {{ padding: 0 !important; font-family: {FONT_BODY}; }}
.tm-light {{ {_vars(_LIGHT)} }}
.tm-dark {{ {_vars(_DARK)} color-scheme: dark; }}
.tm-root {{
  background: var(--background); color: var(--foreground);
  min-height: 100vh; padding: 28px 32px 36px; transition: background .2s, color .2s;
}}
.tm-root h1, .tm-root h2, .tm-root h3, .tm-root h4 {{
  font-family: {FONT_DISPLAY}; font-weight: 700; letter-spacing: -0.025em; color: var(--foreground);
}}
.tm-root label, .tm-root th, .tm-kicker, .tm-label {{
  font-family: {FONT_MONO}; text-transform: uppercase; letter-spacing: 0.16em; font-size: 11px;
  color: var(--tm-label); font-weight: 400;
}}
.tm-root label[id^="switch-"] {{
  font-family: {FONT_BODY}; text-transform: none; letter-spacing: 0; font-size: 14px; color: var(--foreground);
}}
:root:has(.tm-dark), body:has(.tm-dark), .pf-app-root:has(.tm-dark) {{ background: {INK}; }}
.tm-kicker::before {{
  content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 10px;
  background: var(--primary); vertical-align: 0;
}}
.tm-sub {{ color: var(--muted-foreground); font-size: 15px; line-height: 1.5; max-width: 68ch; }}
.tm-note {{ color: var(--muted-foreground); font-size: 13px; line-height: 1.5; max-width: 72ch; }}
.tm-panel {{ background: var(--tm-surface); padding: 18px 20px; }}
.tm-verdict {{ font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 22px; letter-spacing: -0.02em; }}
.tm-root td {{ font-variant-numeric: tabular-nums; }}
.tm-wrap {{ white-space: normal !important; min-width: 220px; }}
.tm-root * {{ box-shadow: none !important; }}

/* Grafici: le aree riempite sono solo le bande di incertezza; le altre serie restano linee. */
.tm-root .recharts-area-area {{ fill-opacity: 0 !important; }}
.tm-root .recharts-area-area[fill*="--color-{BAND_PREFIX}"] {{ fill-opacity: 1 !important; }}
.tm-root .recharts-area-curve[stroke*="--color-{BAND_PREFIX}"] {{ stroke: none !important; }}
.tm-root .recharts-area-curve[stroke*="--color-mediana"],
.tm-root .recharts-area-curve[stroke*="--color-s0"] {{ stroke-width: 2.5px; }}
.tm-root .recharts-area-curve[stroke*="--color-{HIDDEN_PREFIX}"] {{ stroke: none !important; }}
.tm-root .recharts-active-dot:has([fill*="--color-{HIDDEN_PREFIX}"]),
.tm-root .recharts-active-dot:has([fill*="--color-{BAND_PREFIX}"]) {{ display: none; }}
.tm-root .recharts-legend-wrapper div:has(> div[style*="--color-{HIDDEN_PREFIX}"]) {{ display: none; }}
.tm-root .recharts-tooltip-wrapper div:has(> div[style*="--color-{BAND_PREFIX}"]) {{ display: none; }}
.tm-root .recharts-cartesian-grid line {{ stroke: var(--border); }}
.tm-root .recharts-text {{ fill: var(--tm-label); font-size: 12px; }}
"""

# `mode="light"` ignora il tema del client: il cambio avviene solo con il pulsante dell'app.
THEME = Theme(light_css=_vars(_LIGHT), css=_CSS, gradient=False, mode="light")
