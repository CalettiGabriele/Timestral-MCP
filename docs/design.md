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
fc.median     # (B, V, H)
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
│   ├── analytics/           # profile, covariates, forecast, backtest, summaries (pure Python, MCP-independent)
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
Wave 1 is implemented in Milestone 1; waves 2 and 3 are planned.

| Area | Tool | Description | Wave |
|---|---|---|---|
| Ingestion | `load_dataset` | CSV, Excel or pasted data → `dataset_id`; detects date, series ID and value columns | 1 |
| EDA | `profile_dataset` | length, frequency, gaps, missing values, statistics | 1 |
| EDA | `analyze_seasonality`, `test_stationarity`, `analyze_covariates`, `covariate_importance` | STL, ADF/KPSS, lagged correlations, ablation-based importance | 2 |
| Preparation | `regularize_series`, `clean_series`, `add_calendar_features` | resampling, outliers, calendar/holiday covariates | 1–2 |
| Forecast | `forecast` | probabilistic forecast with interactive chart | 1 |
| Forecast | `scenario_forecast`, `compare_models` | what-if on future covariates, model/baseline comparison | 2–3 |
| Evaluation | `backtest` | rolling origin: MASE, interval coverage, seasonal-naive baseline | 1 |
| Evaluation | `detect_anomalies`, `threshold_probability`, `quantity_for_service_level` | points outside the bands, threshold probabilities, quantity for a service level | 1–2 |
| Evaluation | `summarize_forecast`, `get_forecast_details`, `export_results` | text summaries, per-period/aggregated details, CSV/Excel export | 1 |
| Visualization | `forecast_chart_image` | PNG fallback for clients without UI | 1 |

Caveat repeated in the skills: Timestral handles NaNs and normalizes internally; the series must **not** be differenced or made stationary.

### 3.2 Long-running operations
Long operations (backtests, model comparison, many series) are intended to use MCP Tasks via `fastmcp[tasks]`, with progress reporting and a configurable work budget. Some clients only support the legacy protocol, where tasks are unavailable; in that case the tool runs synchronously and reduces its workload, stating so in the result.

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
Forecast Studio (a FastMCPApp to set all preferences from the UI), file upload, MCP Tasks with budgets, `detect_anomalies`, calendar/holiday features, promotion scenarios, stock for a given service level.

### Milestone 3: cloud and teams
Docker image, Terraform modules for AWS (Bedrock AgentCore Runtime, with ECS Express Mode as an alternative) and Google Cloud Run, OIDC login, datasets on S3/GCS, hardening (limits, rate limiting, logging, CI), guides for claude.ai and ChatGPT.

### Milestone 4: advanced analysis
Seasonality and covariate analysis, model comparison, generative UI, reports, more skills and verticals.

## 5. Risks and open points
- **Young library**: not on PyPI; pin the commit and isolate the integration in a single module.
- **Prefab UI changes often**: pin the version; custom HTML is the fallback for richer charts.
- **Two MCP protocol eras**: elicitation and tasks behave differently; test with the real clients.
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
