"""Identità visiva delle MCP App e del grafico PNG.

I colori rispecchiano `brand/tokens/tokens.json` (un test controlla che coincidano).
"""

from __future__ import annotations

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
C_BOUND = "var(--tm-bound)"
C_MUTED = "var(--tm-muted-line)"
# Un solo accento: gli scenari alternativi usano toni di oltremare e neutri.
SCENARIO_COLORS = [C_MEDIAN, "var(--tm-scenario-2)", "var(--tm-scenario-3)", "var(--tm-scenario-4)"]

_LIGHT = f"""
--background: {PAPER}; --foreground: {INK};
--card: {WHITE}; --card-foreground: {INK};
--popover: {WHITE}; --popover-foreground: {INK};
--primary: {ULTRAMARINE}; --primary-foreground: {WHITE};
--secondary: {LINEN}; --secondary-foreground: {INK};
--muted: {LINEN}; --muted-foreground: {GRAPHITE};
--accent: {ULTRAMARINE_50}; --accent-foreground: {ULTRAMARINE};
--destructive: {SIGNAL_RED};
--border: {SAND}; --input: {SAND}; --ring: {ULTRAMARINE};
--chart-1: {ULTRAMARINE}; --chart-2: {INK}; --chart-3: {ULTRAMARINE_200}; --chart-4: {STONE}; --chart-5: {ULTRAMARINE_100};
--radius: 0rem;
--tm-history: {INK}; --tm-median: {ULTRAMARINE}; --tm-bound: {ULTRAMARINE_200}; --tm-muted-line: {STONE};
--tm-scenario-2: {GRAPHITE}; --tm-scenario-3: {ULTRAMARINE_400}; --tm-scenario-4: {STONE};
"""

_DARK = f"""
--background: {INK}; --foreground: {PAPER};
--card: #1E1D1B; --card-foreground: {PAPER};
--popover: #282624; --popover-foreground: {PAPER};
--primary: {ULTRAMARINE}; --primary-foreground: {WHITE};
--secondary: #282624; --secondary-foreground: {PAPER};
--muted: #282624; --muted-foreground: {SAND};
--accent: #222C78; --accent-foreground: {ULTRAMARINE_50};
--destructive: {SIGNAL_RED};
--border: {GRAPHITE}; --input: {GRAPHITE}; --ring: {ULTRAMARINE_200};
--chart-1: {ULTRAMARINE_200}; --chart-2: {PAPER}; --chart-3: {ULTRAMARINE_400}; --chart-4: {STONE}; --chart-5: {ULTRAMARINE};
--radius: 0rem;
--tm-history: {PAPER}; --tm-median: {ULTRAMARINE_200}; --tm-bound: {ULTRAMARINE_400}; --tm-muted-line: {STONE};
--tm-scenario-2: {SAND}; --tm-scenario-3: {ULTRAMARINE_100}; --tm-scenario-4: {STONE};
"""

# Titoli nel carattere display, etichette in maiuscoletto monospaziato, nessuna ombra.
_CSS = f"""
.pf-app-root {{ font-family: {FONT_BODY}; }}
.pf-app-root h1, .pf-app-root h2, .pf-app-root h3, .pf-app-root h4 {{
  font-family: {FONT_DISPLAY}; font-weight: 700; letter-spacing: -0.02em;
}}
.pf-app-root label, .pf-app-root th {{
  font-family: {FONT_MONO}; text-transform: uppercase; letter-spacing: 0.14em; font-size: 11px;
}}
.pf-app-root * {{ box-shadow: none !important; }}
"""

THEME = Theme(
    light_css=" ".join(_LIGHT.split()),
    dark_css=" ".join(_DARK.split()),
    css=_CSS,
    gradient=False,
)
