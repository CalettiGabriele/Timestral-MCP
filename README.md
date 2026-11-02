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
- `forecast_many` / `batch_series_forecast` / `export_batch`: forecast many series (products, stores) at once, with a summary table (change, uncertainty, flags, optional per-series reliability), attention lists and a total; drill into one series; export everything.
- `forecast_studio`: **Forecast Studio**, one interactive screen (MCP App) to set the series, column to forecast, horizon, model, interval and covariates, run the forecast (with an optional backtest) and see the chart and table. Buttons send the results back to Claude or save them to Excel. Preselects as "known in advance" the covariates that have future rows.
- `upload_dataset`: drag-and-drop panel to upload CSV/Excel files straight to the server (no file path to type, nothing goes through the chat); each file becomes a dataset and Claude is told its id. Files that are not recognised automatically can be loaded with `load_dataset(source="upload:<file name>", ...)`.
- `add_calendar_features`: holidays (any country, via the `holidays` library), pre-holiday days, bridges, Black Friday, Christmas period and Easter week as known-in-advance covariates, also for future rows (daily or finer data).
- `scenario_forecast`: what-if on future covariates (e.g. "what if we run a promotion next week?"), compared with the base forecast on a chart and a table.
- `threshold_probability`: probability of being below/above a threshold in each period, interpolated from the quantiles (explicit bounds in the tails).
- `quantity_for_service_level`: quantity needed to cover a given share of cases (e.g. stock), per period and over a coverage window, as a cautious/independent range.
- `covariate_importance`: which covariates actually help the forecast, by ablation (a backtest with all covariates, with none, and with each one removed on the same windows), with a per-window agreement check and a verdict (useful / useless or harmful / neutral or uncertain). It measures usefulness for prediction, not causality.
- `detect_anomalies`: values outside the expected interval, each checked against a forecast made only from the data before it; notes stock-out-like zeros and active flags.
- Skills served by the server: `forecasting-workflow`, `timestral-specifics`, `forecast-interpretation`, `anomaly-investigation` (core) and `sales-forecasting`, `promotions-and-pricing`, `sales-anomalies`, `sales-reporting` (sales vertical); plus `covariate-selection` (core).
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
| `TIMESTRAL_MAX_UPLOAD_MB` | `25` | maximum size of a file uploaded from the interface |
| `TIMESTRAL_MAX_BATCH_SERIES` | `100` | maximum series per `forecast_many` call (the largest by recent volume are kept) |
| `TIMESTRAL_MAX_ANOMALY_WINDOWS` | `60` | maximum forecast windows per `detect_anomalies` call |

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
- The Forecast Studio and the upload panel are MCP Apps: they need a client that renders them (Claude Desktop, claude.ai, ChatGPT, VS Code...). Their rendering has been checked structurally by tests and the backend with the real model, but not visually in every client; other clients can use the normal tools. The backend tools of the apps (`studio_run`, `studio_export`, `store_upload`) are listed with `visibility: ["app"]`, so hosts hide them from the model.
- `add_calendar_features` works on daily or finer data only; seasonal sales and paydays are not generated (add them to the file as 0/1 columns).
- `forecast_many` has a time budget (`max_seconds`, default 90) and a series limit; series not run are listed so the call can be repeated with `series=[...]`.
- Probabilities and totals over several periods are approximations from the marginal quantiles; the tools say so and return ranges where the exact value is not computable.
- Long tools (`backtest`, `forecast_many`, `detect_anomalies`, `covariate_importance`) are MCP Tasks (`fastmcp[tasks]`, in-memory backend): clients on the modern protocol get a background task with progress and a 15-minute default budget; clients on the legacy protocol (e.g. Claude Desktop) get a synchronous call with a 90-second default budget and the tool reports what it did not run. The backend is in memory, so tasks do not survive a server restart (a Redis/Valkey backend is planned for the cloud milestone).
