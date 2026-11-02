# Timestral MCP

An MCP server (built with FastMCP 4) that lets Claude use Mistral's **Timestral** models for
probabilistic forecasting and time series analysis, with interactive charts in the chat.

> Status: **Milestone 1 (local MVP)**. See [docs/design.md](docs/design.md) for the design and roadmap.

## Features
- `load_dataset`: reads a local CSV/Excel file (or data pasted in the chat) and detects dates, frequency, series and value columns.
- `profile_dataset`: data quality checks (gaps, zeros, anomalies, short history).
- `backtest`: how far off the model would have been on past data (MASE, interval coverage, comparison with a baseline).
- `forecast`: forecast with an **interactive chart** (history, median, uncertainty limits) and a table; Claude receives the numbers.
- `get_forecast_details`: values per period, custom quantiles, weekly/monthly/quarterly totals, plus the list of incomplete periods and the horizon needed to complete them.
- `export_results`: saves the median and 21 quantiles to CSV/Excel (`~/TimestralMCP/exports`).
- `forecast_chart_image`: PNG chart with filled bands, for clients without interactive UI support.
- Skills served by the server: `forecasting-workflow`, `timestral-specifics`, `forecast-interpretation`, `sales-forecasting`.
- `previsione_guidata` prompt (guided forecast) for a step-by-step workflow.

## Installation
Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CalettiGabriele/Timestral-MCP.git
cd Timestral-MCP
uv sync
uv run pytest -q
```

Model weights are downloaded from Hugging Face into the local cache (209M ≈ 840 MB, 123M ≈ 490 MB, 28M ≈ 110 MB, 5M ≈ 21 MB; no account needed). **The server never downloads silently**: if a model is not on disk, `forecast` and `backtest` return an error and Claude asks for your consent before retrying with `allow_download=true`. To download ahead of time:

```bash
uv run timestral-mcp download 28M 209M
```

To use a lighter model, set `TIMESTRAL_DEFAULT_MODEL=28M`.

## Connecting Claude Desktop
Add this block to `claude_desktop_config.json` (inside `mcpServers`) and restart Claude Desktop. The file is located at `~/Library/Application Support/Claude/` on macOS and `%APPDATA%\Claude\` on Windows.

```json
{
  "mcpServers": {
    "timestral": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/Timestral-MCP", "timestral-mcp"],
      "env": { "TIMESTRAL_DEFAULT_MODEL": "209M" }
    }
  }
}
```

Replace `/absolute/path/to/Timestral-MCP` with the folder where you cloned the repository. If Claude Desktop cannot find `uv`, replace `"uv"` with the full path printed by `which uv`.

Then try, for example:
> Use the "previsione_guidata" prompt with the file `/absolute/path/to/Timestral-MCP/examples/vendite_esempio.csv`.

The example file is synthetic: two years of daily sales with weekly/yearly seasonality, a recurring promotion, and 30 future rows (empty target) with the planned promotions, useful for trying `known_covariates=["promo"]`.

## Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `TIMESTRAL_DEFAULT_MODEL` | `209M` | model size used when none is specified (`5M`, `28M`, `123M`, `209M`) |
| `TIMESTRAL_DEVICE` | `cpu` | torch device (`cpu`, `mps`, `cuda`) |
| `TIMESTRAL_PRELOAD` | `1` | preload the default model in the background at startup, only if already downloaded |
| `TIMESTRAL_EXPORT_DIR` | `~/TimestralMCP/exports` | where `export_results` writes |
| `TIMESTRAL_SKILLS_DIR` | project `skills/` | skills folder |
| `TIMESTRAL_MAX_ROWS` | `2000000` | maximum rows per file |

## Development
```bash
uv run pytest -q          # tests (use a fake model: no download)
uv run ruff check .       # lint
uv run ruff format .      # formatting
```

Layout: `src/timestral_mcp/{data,analytics,ui}` (data, computation, interfaces), `server.py` (MCP tools), `skills/` (core + verticals).

## Known limitations (Milestone 1)
- Local use only (stdio): no cloud, login or Docker.
- Data sources: local file path or pasted text (URLs and buckets are planned).
- The interactive chart draws the interval limits as lines; the filled band is in the PNG (`forecast_chart_image`).
- Holidays are not generated automatically: if needed, add them to the file as a 0/1 column.
