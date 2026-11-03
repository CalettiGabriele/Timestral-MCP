"""Generate the Timestral MCP brand assets (SVG) from code.

Run from the repository root:  python3 brand/tools/generate.py
Standard library only. The jitter is seeded, so the output is reproducible.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
TOKENS = json.loads((ROOT / "tokens" / "tokens.json").read_text())
C = TOKENS["color"]

FONT_DISPLAY = "Arial, 'Helvetica Neue', Helvetica, 'Liberation Sans', sans-serif"
FONT_MONO = "'Courier New', Courier, 'Liberation Mono', monospace"
FONT_BODY = "Calibri, Carlito, 'Segoe UI', Arial, sans-serif"


# --- geometry -----------------------------------------------------------------------------------


def jitter(points: list[tuple[float, float]], step: float, amp: float, seed: int) -> str:
    """Hand-drawn stroke: resample a polyline every `step` px and shake it sideways."""
    rnd = random.Random(seed)
    out: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(length / step))
        nx, ny = -(y1 - y0) / (length or 1), (x1 - x0) / (length or 1)
        for i in range(n):
            t = i / n
            d = rnd.gauss(0, amp)
            out.append((x0 + (x1 - x0) * t + nx * d, y0 + (y1 - y0) * t + ny * d))
    out.append(points[-1])
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in out)


def series(x0: float, x1: float, base: float, trend: float, n: int = 60) -> list[tuple[float, float]]:
    """A seasonal history: weekly-like wiggle on a gentle trend (y grows downward)."""
    pts = []
    for i in range(n + 1):
        t = i / n
        y = base - trend * t - 22 * math.sin(t * 2 * math.pi * 4.5) - 8 * math.sin(t * 2 * math.pi * 11)
        pts.append((x0 + (x1 - x0) * t, y))
    return pts


def fan_cells(x0: float, y_med: float, slope: float, cols: int, cell: float, gap: float):
    """Pixel quantile fan: one column per future step, rows coloured by quantile band.

    The five bands echo the stacked stripes of the Mistral mark, re-coloured in ultramarine tints.
    """
    tints = [C["white"], C["ultramarine-100"], C["ultramarine-200"], C["ultramarine-400"], C["ultramarine-500"]]
    pitch = cell + gap
    cells = []
    for i in range(cols):
        cx = x0 + i * pitch
        centre = round((y_med - slope * i * pitch) / pitch)
        half = 1 + round(i * 0.55)
        for k in range(-half, half + 1):
            band = min(len(tints) - 1, round(abs(k) / half * (len(tints) - 1))) if half else 0
            cells.append((i, cx, (centre + k) * pitch, tints[band]))
    return cells


# --- pieces -------------------------------------------------------------------------------------


def grain_filter(fid: str, alpha: float) -> str:
    return (
        f'<filter id="{fid}" x="0" y="0" width="100%" height="100%">'
        '<feTurbulence type="fractalNoise" baseFrequency=".85" numOctaves="2" stitchTiles="stitch"/>'
        '<feColorMatrix type="saturate" values="0"/>'
        f'<feComponentTransfer><feFuncA type="table" tableValues="0 {alpha}"/></feComponentTransfer>'
        "</filter>"
    )


def panel(x: float, y: float, size: float, animate: bool, label: bool) -> tuple[str, str]:
    """The ultramarine square with the white hand-drawn trace. Returns (svg, css)."""
    s = size / 512
    hist = series(40, 268, 318, 80)
    now_x = 276
    fan = fan_cells(292, 244, 0.22, 8, 17, 4)
    g = [f'<g transform="translate({x} {y}) scale({s:.5f})">', f'<rect width="512" height="512" fill="{C["ultramarine"]}"/>']
    g.append(f'<rect width="512" height="512" filter="url(#grain-panel)"/>')
    g.append(
        f'<path class="tm-rule" pathLength="1" d="{jitter([(24, 448), (488, 438)], 5, 0.7, 7)}" '
        'fill="none" stroke="#fff" stroke-opacity=".45" stroke-width="1.6" stroke-linecap="round"/>'
    )
    g.append(
        f'<path class="tm-hist" pathLength="1" d="{jitter(hist, 3.2, 0.9, 11)}" '
        'fill="none" stroke="#fff" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    g.append(
        f'<path class="tm-now" pathLength="1" d="{jitter([(now_x, 56), (now_x, 470)], 4, 0.8, 23)}" '
        'fill="none" stroke="#fff" stroke-opacity=".8" stroke-width="2" stroke-linecap="round"/>'
    )
    for i, cx, cy, fill in fan:
        g.append(f'<rect class="tm-c{i}" x="{cx:.1f}" y="{cy:.1f}" width="17" height="17" fill="{fill}"/>')
    if label:
        g.append(
            f'<text x="40" y="74" fill="#fff" font-family="{FONT_MONO}" font-size="22" letter-spacing="1">timestral</text>'
        )
        g.append(
            f'<text x="472" y="486" text-anchor="end" fill="#fff" font-family="{FONT_MONO}" font-size="17" '
            'letter-spacing="2.5">LESS NOISE. MORE SIGNAL.</text>'
        )
    g.append("</g>")

    css = ""
    if animate:
        # One 10 s cycle shared by every element, so the loop never drifts out of sync.
        css = (
            ".tm-hist,.tm-now,.tm-rule{stroke-dasharray:1;animation:tm-draw-h 10s ease-in-out infinite}"
            ".tm-now{animation-name:tm-draw-n}.tm-rule{animation-name:tm-draw-r}"
            "@keyframes tm-draw-h{0%{stroke-dashoffset:1}22%{stroke-dashoffset:0}90%{stroke-dashoffset:0;opacity:1}97%,100%{stroke-dashoffset:0;opacity:0}}"
            "@keyframes tm-draw-n{0%,22%{stroke-dashoffset:1}28%{stroke-dashoffset:0}90%{stroke-dashoffset:0;opacity:1}97%,100%{stroke-dashoffset:0;opacity:0}}"
            "@keyframes tm-draw-r{0%{stroke-dashoffset:1}30%{stroke-dashoffset:0}90%{stroke-dashoffset:0;opacity:1}97%,100%{stroke-dashoffset:0;opacity:0}}"
        )
        for i in range(8):
            a = 30 + i * 3.5
            css += (
                f".tm-c{i}{{animation:tm-p{i} 10s steps(1,end) infinite}}"
                f"@keyframes tm-p{i}{{0%,{a:.1f}%{{opacity:0}}{a + 0.1:.1f}%,90%{{opacity:1}}"
                f"{90 + i * 0.8:.1f}%,100%{{opacity:0}}}}"
            )
        css += "@media (prefers-reduced-motion: reduce){*{animation:none!important}}"
    return "".join(g), css


def svg(width: int, height: int, body: str, css: str = "", title: str = "", extra_defs: str = "") -> str:
    style = f"<style>{css}</style>" if css else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{title}"><title>{title}</title>'
        f"<defs>{grain_filter('grain-panel', 0.10)}{grain_filter('grain-paper', 0.07)}{extra_defs}</defs>"
        f"{style}{body}</svg>\n"
    )


# --- assets -------------------------------------------------------------------------------------


def mark(animate: bool) -> str:
    body, css = panel(0, 0, 512, animate, label=False)
    return svg(512, 512, body, css, "Timestral MCP mark")


def favicon() -> str:
    # Too small for jitter: a clean polyline and a three-column pixel fan.
    u = C["ultramarine"]
    cells = "".join(
        f'<rect x="{x}" y="{y}" width="5" height="5" fill="{f}"/>'
        for x, y, f in [
            (36, 26, "#fff"),
            (43, 22, C["ultramarine-200"]), (43, 28, "#fff"), (43, 34, C["ultramarine-200"]),
            (50, 16, C["ultramarine-400"]), (50, 22, C["ultramarine-200"]), (50, 28, "#fff"),
            (50, 34, C["ultramarine-200"]), (50, 40, C["ultramarine-400"]),
        ]
    )
    body = (
        f'<rect width="64" height="64" fill="{u}"/>'
        '<polyline points="8,40 13,34 17,38 22,30 26,35 31,28" fill="none" stroke="#fff" stroke-width="3" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<line x1="33" y1="9" x2="33" y2="55" stroke="#fff" stroke-opacity=".75" stroke-width="1.5"/>{cells}'
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" '
        f'aria-label="Timestral MCP"><title>Timestral MCP</title>{body}</svg>\n'
    )


def wordmark(dark: bool) -> str:
    ink = C["paper"] if dark else C["ink"]
    muted = C["stone"]
    p, _ = panel(0, 0, 96, False, label=False)
    body = (
        f"{p}"
        f'<text x="124" y="66" fill="{ink}" font-family="{FONT_DISPLAY}" font-weight="700" font-size="64" '
        'letter-spacing="-2">timestral</text>'
        f'<text x="128" y="92" fill="{muted}" font-family="{FONT_MONO}" font-size="14" letter-spacing="3.5">'
        "FORECASTING · MCP SERVER</text>"
    )
    return svg(452, 100, body, title="timestral, forecasting MCP server")


def banner(dark: bool, animate: bool = True) -> str:
    w, h = 1280, 440
    bg = C["ink"] if dark else C["paper"]
    ink = C["paper"] if dark else C["ink"]
    sub = C["sand"] if dark else C["graphite"]
    label = C["stone"]
    p, css = panel(864, 52, 336, animate, label=True)
    body = (
        f'<rect width="{w}" height="{h}" fill="{bg}"/>'
        f'<rect width="{w}" height="{h}" filter="url(#grain-paper)"/>'
        f'<text x="80" y="104" fill="{label}" font-family="{FONT_MONO}" font-size="15" letter-spacing="3.5">'
        "PROBABILISTIC FORECASTING · MCP SERVER</text>"
        f'<text x="74" y="214" fill="{ink}" font-family="{FONT_DISPLAY}" font-weight="700" font-size="112" '
        'letter-spacing="-4">timestral mcp</text>'
        f'<text x="80" y="266" fill="{sub}" font-family="{FONT_BODY}" font-size="27">'
        "Time series forecasts for Claude,</text>"
        f'<text x="80" y="302" fill="{sub}" font-family="{FONT_BODY}" font-size="27">'
        "with the uncertainty included.</text>"
        f'<rect x="80" y="334" width="300" height="4" fill="{ink}"/>'
        f'<rect x="80" y="364" width="14" height="14" fill="{C["ultramarine"] if not dark else C["ultramarine-200"]}"/>'
        f'<text x="108" y="376" fill="{label}" font-family="{FONT_MONO}" font-size="14" letter-spacing="3">'
        "21 QUANTILES · 4 MODEL SIZES · ZERO-SHOT</text>"
        f"{p}"
    )
    return svg(w, h, body, css, "timestral mcp — probabilistic forecasting MCP server")


def social() -> str:
    """GitHub social preview (1280×640, static)."""
    w, h = 1280, 640
    p, _ = panel(752, 96, 448, False, label=True)
    body = (
        f'<rect width="{w}" height="{h}" fill="{C["paper"]}"/>'
        f'<rect width="{w}" height="{h}" filter="url(#grain-paper)"/>'
        f'<text x="80" y="176" fill="{C["stone"]}" font-family="{FONT_MONO}" font-size="18" letter-spacing="4">'
        "PROBABILISTIC FORECASTING · MCP SERVER</text>"
        f'<text x="72" y="300" fill="{C["ink"]}" font-family="{FONT_DISPLAY}" font-weight="700" font-size="128" '
        'letter-spacing="-5">timestral</text>'
        f'<text x="80" y="362" fill="{C["graphite"]}" font-family="{FONT_BODY}" font-size="32">'
        "Time series forecasts for Claude,</text>"
        f'<text x="80" y="404" fill="{C["graphite"]}" font-family="{FONT_BODY}" font-size="32">'
        "with the uncertainty included.</text>"
        f'<rect x="80" y="448" width="340" height="5" fill="{C["ink"]}"/>'
        f'<rect x="80" y="484" width="16" height="16" fill="{C["ultramarine"]}"/>'
    )
    return svg(w, h, body + p, title="timestral — probabilistic forecasting MCP server")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    files = {
        "mark.svg": mark(animate=False),
        "mark-animated.svg": mark(animate=True),
        "favicon.svg": favicon(),
        "wordmark-light.svg": wordmark(dark=False),
        "wordmark-dark.svg": wordmark(dark=True),
        "banner-light.svg": banner(dark=False),
        "banner-dark.svg": banner(dark=True),
        "social-preview.svg": social(),
    }
    for name, content in files.items():
        (ASSETS / name).write_text(content)
        print(f"wrote brand/assets/{name}")


if __name__ == "__main__":
    main()
