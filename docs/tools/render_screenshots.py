"""Render the README screenshots and the app tour from real tool calls.

Runs the MCP tools in-process with a real Timestral model, renders each MCP app to HTML,
screenshots it with headless Chrome and builds an animated tour.

    uv run --with pillow python docs/tools/render_screenshots.py [--model 209M] [--chrome PATH]

Needs the model weights on disk and Google Chrome (or Chromium). Writes to docs/images/.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastmcp import Client
from PIL import Image
from prefab_ui.app import PrefabApp

from timestral_mcp.server import mcp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images"
EXAMPLE = ROOT / "examples" / "vendite_esempio.csv"
PAPER = (0xE3, 0xDE, 0xD5)
WIDTH = 1100

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
]


def stores_csv() -> str:
    """Six synthetic stores, 18 months of daily sales, long format."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-07-01", "2025-12-31", freq="D")
    t = np.arange(len(idx))
    rows = []
    for i, (store, level, trend) in enumerate(
        [
            ("Milano", 420, 0.12),
            ("Torino", 260, 0.05),
            ("Genova", 180, -0.03),
            ("Bologna", 210, 0.09),
            ("Firenze", 150, 0.02),
            ("Napoli", 300, 0.15),
        ]
    ):
        weekly = 0.18 * level * np.sin(2 * np.pi * (t + i) / 7)
        yearly = 0.12 * level * np.sin(2 * np.pi * (t - 300) / 365.25)
        y = level + trend * t + weekly + yearly + rng.normal(0, 0.06 * level, len(t))
        rows += [f"{d:%Y-%m-%d},{store},{max(v, 0):.0f}" for d, v in zip(idx, y, strict=True)]
    return "data,negozio,vendite\n" + "\n".join(rows)


def anomalies_csv() -> str:
    """The example sales history with a stock-out, a spike and a data-entry error injected."""
    df = pd.read_csv(EXAMPLE)
    df = df[df["vendite"].notna()].copy()
    n = len(df)
    df.loc[df.index[n - 60 : n - 55], "vendite"] = 0  # stock-out
    df.loc[df.index[n - 35], "vendite"] *= 2.6  # unrecorded event
    df.loc[df.index[n - 12], "vendite"] *= 0.15  # data-entry error
    return df[["data", "vendite"]].to_csv(index=False)


async def collect(model: str) -> dict[str, tuple[dict, str]]:
    """Call the tools and return {name: (app json, text summary)}."""
    apps: dict[str, tuple[dict, str]] = {}

    async def call(name: str, tool: str, args: dict) -> dict:
        res = await c.call_tool(tool, args)
        apps[name] = (res.structured_content, res.content[0].text if res.content else "")
        return res.structured_content

    async with Client(mcp) as c:
        ds = (
            await c.call_tool("load_dataset", {"source": str(EXAMPLE), "name": "vendite"})
        ).structured_content
        ds_id = ds["dataset_id"]
        common = {"dataset_id": ds_id, "target": "vendite", "model": model}

        await call("forecast", "forecast", {**common, "horizon": 30, "known_covariates": ["promo"]})
        fc_text = apps["forecast"][1]
        fc_id = next(w.strip(".,;:()") for w in fc_text.split() if w.startswith("fc_"))
        png = await c.call_tool("forecast_chart_image", {"forecast_id": fc_id})
        (OUT / "forecast-png.png").write_bytes(base64.b64decode(png.content[0].data))

        await call(
            "backtest", "backtest", {**common, "horizon": 30, "known_covariates": ["promo"], "n_folds": 4}
        )
        await call(
            "scenario",
            "scenario_forecast",
            {
                **common,
                "horizon": 30,
                "known_covariates": ["promo"],
                "scenarios": {
                    "senza promo": [{"column": "promo", "value": 0}],
                    "promo 15-21 gen": [
                        {"column": "promo", "value": 1, "start": "2026-01-15", "end": "2026-01-21"}
                    ],
                },
            },
        )

        stores = (
            await c.call_tool(
                "load_dataset", {"source": stores_csv(), "name": "negozi", "series_column": "negozio"}
            )
        ).structured_content
        await call(
            "batch",
            "forecast_many",
            {"dataset_id": stores["dataset_id"], "target": "vendite", "horizon": 60, "model": model},
        )

        anom = (
            await c.call_tool("load_dataset", {"source": anomalies_csv(), "name": "vendite_anomalie"})
        ).structured_content
        await call(
            "anomalies",
            "detect_anomalies",
            {"dataset_id": anom["dataset_id"], "target": "vendite", "model": model, "lookback": 120},
        )

        studio = await c.call_tool("forecast_studio", {"dataset_id": ds_id})
        apps["studio"] = (studio.structured_content, "")
    return apps


def find_chrome(explicit: str | None) -> str:
    for cand in [explicit, *CHROME_CANDIDATES]:
        if cand and (Path(cand).exists() or shutil.which(cand)):
            return cand
    raise SystemExit("Chrome/Chromium not found: pass --chrome PATH")


def screenshot(chrome: str, html: str, height: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        page, shot = Path(tmp) / "app.html", Path(tmp) / "shot.png"
        page.write_text(html)
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-prefers-reduced-motion",
                f"--window-size={WIDTH},{height}",
                "--virtual-time-budget=20000",
                f"--screenshot={shot}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        img = Image.open(shot).convert("RGB")
        img.load()
    return trim(img)


def trim(img: Image.Image, pad: int = 28) -> Image.Image:
    """Crop the empty paper below and around the app."""
    arr = np.asarray(img).astype(int)
    diff = np.abs(arr - np.array(PAPER)).sum(axis=2) > 24
    rows, cols = np.where(diff.any(axis=1))[0], np.where(diff.any(axis=0))[0]
    if not len(rows):
        return img
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad, img.height)
    left, right = max(cols[0] - pad, 0), min(cols[-1] + pad, img.width)
    return img.crop((left, top, right, bottom))


def tour(frames: list[tuple[Image.Image, str]], path: Path) -> None:
    """Crossfading GIF of the apps, each with a small caption bar."""
    from PIL import ImageDraw, ImageFont

    w = 900
    scaled = []
    for img, caption in frames:
        img = img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)
        scaled.append((img, caption))
    h = max(i.height for i, _ in scaled) + 44
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    def canvas(img: Image.Image, caption: str) -> Image.Image:
        c = Image.new("RGB", (w, h), PAPER)
        d = ImageDraw.Draw(c)
        d.rectangle((0, 0, w, 43), fill=(0x23, 0x36, 0xBC))
        d.text((20, 13), caption.upper(), fill=(255, 255, 255), font=font, spacing=4)
        c.paste(img, (0, 44))
        return c

    stills = [canvas(i, c) for i, c in scaled]
    seq, durations = [], []
    for k, still in enumerate(stills):
        seq.append(still)
        durations.append(2600)
        nxt = stills[(k + 1) % len(stills)]
        for a in (0.25, 0.5, 0.75):
            seq.append(Image.blend(still, nxt, a))
            durations.append(70)
    palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in seq]
    palette[0].save(path, save_all=True, append_images=palette[1:], duration=durations, loop=0, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="209M")
    ap.add_argument("--chrome")
    ap.add_argument("--summaries", help="folder where to save the text the model received (optional)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome(args.chrome)

    apps = asyncio.run(collect(args.model))
    heights = {
        "forecast": 900,
        "backtest": 900,
        "scenario": 950,
        "batch": 600,
        "anomalies": 760,
        "studio": 1000,
    }
    shots = {}
    for name, (wire, text) in apps.items():
        html = PrefabApp.from_json(wire).html()
        shots[name] = screenshot(chrome, html, heights.get(name, 900))
        shots[name].save(OUT / f"app-{name}.png", optimize=True)
        if text and args.summaries:
            Path(args.summaries, f"{name}.txt").write_text(text)
        print(f"wrote docs/images/app-{name}.png")

    tour(
        [
            (shots["forecast"], "forecast · 30 days ahead, with uncertainty"),
            (shots["backtest"], "backtest · how far to trust it"),
            (shots["scenario"], "scenario_forecast · what if we run a promotion?"),
            (shots["batch"], "forecast_many · six stores at once"),
            (shots["anomalies"], "detect_anomalies · what did not look normal"),
        ],
        OUT / "tour.gif",
    )
    print("wrote docs/images/tour.gif")


if __name__ == "__main__":
    main()
