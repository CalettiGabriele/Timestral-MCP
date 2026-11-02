# Timestral MCP: design and roadmap

An MCP server in Python (FastMCP) exposing Mistral's **Timestral** models for forecasting and time series analysis from chat clients such as Claude. The first milestone runs locally; cloud deployment is planned for later milestones.

## 1. Timestral

### 1.1 Overview
- First family of time series foundation models from Mistral, released in October 2026.
- Four sizes sharing the same architecture and training recipe:

| Model | Hugging Face repo | Weights (fp32) | Notes |
|---|---|---|---|
| Timestral-5M | `mistralai/Timestral-5M` | 21 MB | ~10 ms per forecast on 8 CPU threads |
| Timestral-28M | `mistralai/Timestral-28M` | 111 MB | ~5x faster than 209M |
| Timestral-123M | `mistralai/Timestral-123M` | 490 MB | within 0.2% of 209M, about 3/4 of the latency |
| Timestral-209M | `mistralai/Timestral-209M` | 837 MB | ~140 ms on 8 CPU threads, best accuracy |

(Latencies as reported by Mistral: one-day forecast from three weeks of 15-minute history.)

- **Licenses**: library code under Apache 2.0; weights under a *Modified MIT* license. Commercial use is allowed.
- **No hosted API** is documented, so the server **runs the model in-process** (CPU is sufficient).

### 1.2 Architecture and capabilities
- Causal, patch-based transformer (patch size 32); attention over time and across variates.
- **Context** up to 32,768 steps (older steps are truncated).
- **Horizon** up to 1,024 steps per forward pass; beyond that, blockwise forecasting feeds the median back into the context, so uncertainty beyond 1,024 steps is underestimated.
- **Probabilistic output**: 21 fixed, non-crossing quantiles: `0.01, 0.05, 0.10, 0.15 … 0.90, 0.95, 0.99`.
- **Multivariate**: target variates are forecast jointly.
- **Covariates**: past-only, known-in-the-future, categorical. Covariates condition the forecast but are not forecast.
- **Missing values**: `NaN` or `observation_mask`.
- **Zero-shot**: no training required.

### 1.3 Python API (`timestral` library)
```python
from timestral import TimestralForecaster

model = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device="cpu")
fc = model.forecast(
    context,  # (B, V, T) | (B, T) | (T,), numpy or torch, NaN = missing
    prediction_length=96,
    # past_covariates=(B, C_past, T)
    # known_covariates=(B, C_known, T + H)
    # categorical_past_covariates / categorical_known_covariates
    # observation_mask=...
)
fc.quantiles  # (B, V, H, 21)
fc.median  # (B, V, H)
fc.quantile(0.1)
```

### 1.4 What the library does not do (the server does)
- No timestamp/frequency handling or DataFrame support: it works on arrays. The server reads CSV/Excel with pandas, infers the frequency, regularizes the time grid and builds future timestamps.
- No automatic calendar features.
- No backtesting, metrics, anomaly detection or charts.
- Not published on PyPI: installed from Git, pinned to a commit in the lockfile.

## 2. FastMCP
- **FastMCP 4** (`pip install fastmcp`). Tools, resources and prompts are declared with decorators; schemas come from type hints.
- Transports: `stdio` (local) and Streamable HTTP.
- **MCP Apps** (`fastmcp[apps]`, Prefab UI): tools can return interactive components (charts, tables, metrics) rendered in a sandboxed iframe by supporting hosts. Terminal clients do not render UI, so every tool also returns a text summary.
- Other useful features: elicitation, background tasks (`fastmcp[tasks]`), session state, lifespan hooks, auth (OIDC/OAuth proxies), health checks.

## 3. Architecture

```
timestral-mcp/
├── src/timestral_mcp/
│   ├── server.py            # FastMCP server, tool/prompt/skill registration
│   ├── config.py            # settings from environment variables
│   ├── models.py            # lazy loading and caching of the model sizes
│   ├── data/                # ingestion, frequency inference, dataset store
│   ├── analytics/           # profile, covariates, calendar, forecast, backtest, scenarios, anomalies, decision, batch, summaries (pure Python, MCP-independent)
│   ├── ui/                  # charts
│   └── tools/
├── skills/
│   ├── core/                # general skills
│   └── verticals/sales/     # first vertical; more can be added
├── examples/
├── tests/
└── docs/
```

Principle: **tools are general** (valid for any time series), **skills carry the domain knowledge**. Skills do not compute anything: they tell Claude which tools to use, in which order and how to interpret the results.

### 3.1 Tool catalog
Implemented so far: wave 1 plus `add_calendar_features`, `scenario_forecast`, `detect_anomalies`, `threshold_probability`, `quantity_for_service_level` and the multi-series tools (`forecast_many`, `batch_series_forecast`, `export_batch`). Still planned: the rest of waves 2 and 3.

| Area | Tool | Description | Wave |
|---|---|---|---|
| Ingestion | `load_dataset` | CSV, Excel or pasted data → `dataset_id`; detects date, series ID and value columns | 1 |
| EDA | `profile_dataset` | length, frequency, gaps, missing values, statistics | 1 |
| EDA | `covariate_importance` | ablation-based importance with per-window agreement and verdict | 2 (done) |
| EDA | `analyze_seasonality`, `test_stationarity`, `analyze_covariates` | STL, ADF/KPSS, lagged correlations | 4 |
| Preparation | `add_calendar_features` | holiday, pre-holiday, bridge, Black Friday, Christmas and Easter covariates (daily or finer data) | 2 (done) |
| Preparation | `regularize_series`, `clean_series` | resampling, outlier handling | 2 |
| Forecast | `forecast` | probabilistic forecast with interactive chart | 1 |
| Forecast | `forecast_many`, `batch_series_forecast`, `export_batch` | many series at once with summary table, drill-down and export | 2 (done) |
| Interface | `forecast_studio`, `upload_dataset` | one-screen form to set everything and run (chart, table, backtest, send to Claude, Excel); drag-and-drop file upload | 2 (done) |
| Forecast | `scenario_forecast` | what-if on future covariates, compared with the base forecast | 2 (done) |
| Forecast | `compare_models` | model/baseline comparison | 3 |
| Evaluation | `backtest` | rolling origin: MASE, interval coverage, seasonal-naive baseline | 1 |
| Evaluation | `detect_anomalies`, `threshold_probability`, `quantity_for_service_level` | points outside the bands, threshold probabilities, quantity for a service level | 2 (done) |
| Evaluation | `summarize_forecast`, `get_forecast_details`, `export_results` | text summaries, per-period/aggregated details, CSV/Excel export | 1 |
| Visualization | `forecast_chart_image` | PNG fallback for clients without UI | 1 |

Caveat repeated in the skills: Timestral handles NaNs and normalizes internally; the series must **not** be differenced or made stationary.

### 3.2 Long-running operations
Long operations (backtests, model comparison, many series) are intended to use MCP Tasks via `fastmcp[tasks]`, with progress reporting and a configurable work budget. Claude Desktop negotiates the legacy protocol (seen in its server log), where tasks are unavailable; in that case the tool runs synchronously and reduces its workload, stating so in the result. Currently implemented: `forecast_many` stops after `max_seconds` and lists the series not run; `detect_anomalies` caps the number of windows (`TIMESTRAL_MAX_ANOMALY_WINDOWS`); `forecast_many` keeps at most `TIMESTRAL_MAX_BATCH_SERIES` series (the largest by recent volume), with per-series backtests limited to 30 series. All of them report progress. `backtest`, `forecast_many`, `detect_anomalies` and `covariate_importance` are declared `task=True` (`fastmcp[tasks]`, `TasksExtension` with the default in-memory backend). With a client on the modern protocol the call runs as a background task, progress is exposed through the `Progress` dependency (and MCP progress notifications when synchronous), and the default time budget is 900 s; with a legacy client it runs synchronously and the budget is 90 s (`progress.py`). Verified with the in-memory FastMCP client in both modes and over stdio in legacy mode; not verified against a real modern-protocol client. Tasks live in memory: a Redis/Valkey backend and a shared encryption key are needed for multi-instance deployments (Milestone 3).

### 3.3 Data sources
Passing data as a tool argument costs tokens and risks transcription errors, so sources are layered: pasted data (small datasets), local file path (stdio), then URLs and cloud buckets (planned). Datasets are stored under a `dataset_id` (in memory locally).

## 4. Roadmap

### Milestone 1: local MVP (implemented)
- `uv` project, pinned dependencies, pytest, ruff, configuration via environment variables.
- Tools: `load_dataset`, `profile_dataset`, `backtest`, `forecast`, `get_forecast_details`, `export_results`, `forecast_chart_image`, `list_session_items`.
- Interactive chart (Prefab) with history, median and interval limits, plus a PNG with filled bands as fallback.
- Every result returns a text summary and the essential numbers; full results stay available by `forecast_id`.
- Skills: `forecasting-workflow`, `timestral-specifics`, `forecast-interpretation`, `sales-forecasting`.
- The server never downloads model weights silently (`allow_download=true` is required).

Not verified: the rendering of the interactive chart in every client, and model sizes other than the ones tested.

### Milestone 2: full local experience
Implemented: multi-series forecasting, calendar/holiday features, promotion scenarios, `detect_anomalies`, threshold probabilities, stock for a given service level, and the sales skills that go with them (`promotions-and-pricing`, `sales-anomalies`, `sales-reporting`, `anomaly-investigation`).
Also implemented: the **Forecast Studio** (a `FastMCPApp`: the model-visible entry tool `forecast_studio` draws the form; the app-only backend tools `studio_run` and `studio_export` run the forecast and save results; results go to the same session stores as the normal tools, so Claude can follow up with `get_forecast_details`) and **file upload** (`upload_dataset`, with an app-only `store_upload` that turns each dropped file into a dataset and adds its id to the model context through `UpdateContext`).
Also implemented: MCP Tasks for the long tools and `covariate_importance`. Untested outside automated tests: how the Studio renders in each client, and Tasks against a real modern-protocol client.

### Milestone 3: cloud and teams
Docker image, Terraform modules for AWS (Bedrock AgentCore Runtime, with ECS Express Mode as an alternative) and Google Cloud Run, OIDC login, datasets on S3/GCS, hardening (limits, rate limiting, logging, CI), guides for claude.ai and ChatGPT.

### Milestone 4: advanced analysis
Seasonality and covariate analysis, model comparison, generative UI, reports, more skills and verticals.

## 5. Risks and open points
- **Young library**: not on PyPI; pin the commit and isolate the integration in a single module.
- **Prefab UI changes often**: pin the version; custom HTML is the fallback for richer charts.
- **Two MCP protocol eras**: elicitation and tasks behave differently; test with the real clients.
- **Many tools in the model's context**: the server exposes 19 model-facing tools (plus 3 app-only backend tools that hosts hide); if tool selection degrades, hide the advanced ones behind tags/visibility.
- **Scenario effects are model estimates**: derived from the historical relationship with a covariate, not causal; the tool warns when a covariate was active only a few times in the history.
- **Aggregated intervals are approximate**: quantiles of a sum cannot be derived from per-period quantiles; the tools return ranges and say so.
- **Image size**: CPU-only torch plus weights (up to ~1.5 GB with all sizes).
- **Horizons beyond 1,024 steps**: uncertainty is underestimated; the result warns about it.
- **Sensitive data**: a hosted deployment needs authentication, dataset TTLs and no logging of values.

## References
- Mistral, Timestral announcement: https://mistral.ai/news/timestral/
- Model card: https://huggingface.co/mistralai/Timestral-209M
- Library: https://github.com/mistralai/timestral
- FastMCP documentation: https://gofastmcp.com
- Cloud Run and MCP: https://docs.cloud.google.com/run/docs/host-mcp-servers
- AgentCore Runtime and MCP: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-mcp.html
