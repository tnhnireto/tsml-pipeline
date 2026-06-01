# TSML — Trading System ML

Weekly machine-learning portfolio strategy with walk-forward training, regime-aware backtesting, and **eToro demo-only** execution. Designed for reproducible research and a conservative automated demo workflow (e.g. OpenClaw).

---

## 1. Project Overview

TSML ranks a fixed US equity universe each week using a calibrated logistic regression model trained with **walk-forward cross-validation**. It generates buy/hold/sell signals, backtests portfolio rules historically, and optionally submits **amount-based market orders** to an **eToro demo account**.

Core capabilities:

| Capability | Description |
|---|---|
| Walk-forward training | Expanding-window folds; OOS scores use only past data |
| Extended feature set | 27 features: momentum, vol, relative strength vs SPY/QQQ, regime |
| Regime overlay | Scales portfolio exposure when SPY is below SMA200 (backtest) |
| Weekly rebalancing | First trading day of each ISO week |
| Turnover control | Buy/sell hysteresis + minimum hold period |
| eToro demo execution | Read account, place orders, never live trading |
| Local portfolio state | `data/portfolio_state.json` tracks paper positions |
| Broker reconciliation | Local state must match broker before `--execute` |

### End-to-end workflow (high level)

```
Market data (Yahoo Finance)
        │
        ▼
Walk-forward model scores  ──►  Rank universe  ──►  Generate signals
        │                                              │
        │                                              ▼
        │                                    signals/YYYY-MM-DD.csv
        │                                              │
        ▼                                              ▼
Backtest / robustness / holdout              run_etoro_demo.py (dry-run or --execute)
        │                                              │
        │                                              ▼
        │                                    eToro demo API (orders)
        │                                              │
        ▼                                              ▼
Reports (metrics, plots)                     sync_state_from_broker.py
                                             reconcile_broker.py
```

**Research path:** signal → backtest → robustness → parameter sweep → holdout validation.

**Live demo path:** verify API → reconcile → signal → dry-run plan → human-approved execute → sync state → reconcile again.

---

## 2. Architecture

### Major modules

| Area | Location | Role |
|---|---|---|
| **Features** | `src/tsml/features/` | Transformers, extended pipeline, benchmarks, targets |
| **Models** | `src/tsml/models/` | `CalibratedLogisticRegressionModel`, baselines |
| **Walk-forward** | `src/tsml/pipelines/train.py`, `src/tsml/validation/` | OOS probability generation, splitters |
| **Portfolio simulation** | `src/tsml/portfolio/simulator.py` | Weekly rebalance backtest, regime overlay |
| **Ranking & signals** | `src/tsml/portfolio/ranker.py`, `strategy.py` | Universe rank, buy/hold/sell rules |
| **Weekly backtest** | `src/tsml/portfolio/weekly_backtest.py` | Orchestration, metrics, comparisons |
| **Robustness** | `src/tsml/portfolio/robustness.py` | Regime breakdown, rolling metrics, calibration |
| **Parameter sweep** | `src/tsml/portfolio/parameter_sweep.py` | Grid search, stability diagnostics |
| **Holdout evaluation** | `src/tsml/portfolio/holdout_eval.py` | Dev vs holdout OOS validation |
| **Regime overlay** | `src/tsml/portfolio/regime_overlay.py` | SPY SMA200 / vol exposure scaling |
| **Broker client** | `src/tsml/broker/etoro_client.py` | Demo API: account, positions, orders |
| **Execution** | `src/tsml/broker/execution.py` | Signals → proposed orders → plan → submit |
| **Risk** | `src/tsml/broker/risk.py` | Demo-only, no leverage, universe, cash buffer |
| **Reconciliation** | `src/tsml/broker/reconcile.py` | Local vs broker position/cash check |
| **State sync** | `src/tsml/broker/sync.py` | Build `portfolio_state.json` from broker |
| **Reporting** | `src/tsml/reporting/` | Equity curves, exposure, robustness, sweep plots |

### ASCII architecture diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  YFinanceLoader  →  data/raw/*.parquet  →  features/pipeline.py     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      ML / VALIDATION LAYER                           │
│  WalkForwardSplit  →  run_walk_forward_proba  →  per-symbol scores  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│ run_weekly_   │     │ simulator.py    │     │ robustness / sweep  │
│ signal.py     │     │ weekly_backtest │     │ holdout_eval        │
│ (live signals)│     │ (backtests)     │     │ (validation)        │
└───────┬───────┘     └─────────────────┘     └─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER (demo only)                     │
│  run_etoro_demo.py  →  risk.py  →  EtoroClient.place_order          │
│  reconcile (gate)   →  logs/orders/*.jsonl                            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                      STATE LAYER                                       │
│  data/portfolio_state.json  ↔  sync_state_from_broker.py              │
└─────────────────────────────────────────────────────────────────────┘
```

### Project layout

```
Python/
├── run_weekly_signal.py      # Weekly signal generation (entry point)
├── run_etoro_demo.py         # Demo order planning / execution
├── scripts/
│   ├── verify_etoro_api.py   # API pre-flight
│   ├── reconcile_broker.py   # Local vs broker check
│   ├── sync_state_from_broker.py
│   ├── weekly_backtest.py
│   ├── robustness_report.py
│   ├── parameter_sweep.py
│   ├── holdout_report.py
│   └── weekly_job.py         # Automated weekly dry-run pipeline
├── src/tsml/                 # Library code
├── signals/                  # Generated signal CSVs
├── data/
│   ├── raw/                  # Cached OHLCV (gitignored)
│   └── portfolio_state.json  # Local paper state
├── logs/orders/              # Order JSONL logs
├── logs/jobs/                # weekly_job logs
├── reports/                  # Backtest & analysis outputs
└── tests/                    # pytest suite
```

---

## 3. Environment Setup

### Python version

**Python 3.11+** (see `pyproject.toml`).

### Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Development (tests, linting)
pip install -e ".[dev]"
```

Primary dependencies are listed in `requirements.txt`. Full project metadata (including dev extras) is in `pyproject.toml`.

Run tests:

```bash
pytest
```

### Environment variables

Create a `.env` file in the project root (never commit it) or export in your shell:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ETORO_API_KEY` | For demo execution | — | eToro **Public API Key** (Settings → Trading → API Key Management) |
| `ETORO_USER_KEY` | For demo execution | — | eToro **demo (virtual) user key** from the API portal |
| `ETORO_ACCOUNT_MODE` | Yes (if using broker) | `demo` | Must be `demo`. Real mode is rejected by the client |
| `TSML_MAX_LIVE_ORDER_AMOUNT` | No | `1000` | Max USD notional per live demo order. **Recommend `500` for first execution** |
| `TSML_COMMIT_STATE` | No | unset | Set to `1` to let `scripts/weekly_job.py` update `portfolio_state.json` after dry-run |

**Notes:**

- `ETORO_API_KEY` and `ETORO_USER_KEY` are **different** keys — do not swap them.
- `TSML_MAX_LIVE_ORDER_AMOUNT` is enforced in `EtoroClient.place_order` and applied upfront to proposed BUY sizes when using `--execute` or `--use-live-cap`.
- `TSML_COMMIT_STATE=1` only affects the automated weekly job dry-run path; it does **not** submit orders.

---

## 4. Initial Verification

Before any broker interaction, verify credentials and read-only endpoints:

```bash
python scripts/verify_etoro_api.py
```

Optional diagnostics:

```bash
python scripts/verify_etoro_api.py --diagnose
```

**Expected result** (when credentials are valid):

```
ALL CHECKS PASSED
```

Checks performed: `get_account()`, `get_positions()`, `get_instrument("AAPL")`. **No orders are placed.**

---

## 5. Broker State Management

### `data/portfolio_state.json`

Local JSON file tracking:

- Cash balance (approximate)
- Open position symbols
- Last rebalance date

Used by `run_weekly_signal.py` (current holdings) and `run_etoro_demo.py` (dry-run account snapshot). It is **not** the source of truth after live execution — the broker is.

### `scripts/sync_state_from_broker.py`

Reads the live demo account and builds a fresh state file.

```bash
# Preview only (no write)
python scripts/sync_state_from_broker.py

# Write data/portfolio_state.json
python scripts/sync_state_from_broker.py --confirm
```

Never places orders.

### `scripts/reconcile_broker.py`

Compares local `portfolio_state.json` against the broker demo account (cash + open positions).

```bash
python scripts/reconcile_broker.py
```

Exit code `0` = passed, `1` = mismatch or broker error.

### Why reconciliation is mandatory before execution

`run_etoro_demo.py --execute` runs reconciliation **automatically** and **aborts** if:

- Local cash differs from broker cash beyond tolerance
- Local position symbols differ from broker open positions

This prevents submitting orders based on stale or incorrect local state (e.g. manual broker trades, missed sync, partial fills). **Never bypass a failed reconciliation.**

---

## 6. Weekly Signal Generation

```bash
python run_weekly_signal.py
```

### What it does

1. Loads OHLCV for the universe (cached under `data/raw/`)
2. Runs **walk-forward** calibrated logistic regression per symbol
3. Uses **`feature_set="extended"`** (27 features)
4. Ranks symbols by P(up) score
5. Applies **SMA200 context**: stricter `min_score_downtrend` when below SMA200
6. Generates **buy / hold / sell** actions with turnover-aware rules
7. Writes CSV and prints feature importance

### Key configuration (in `run_weekly_signal.py`)

| Parameter | Value |
|---|---|
| Target | `threshold` (high-conviction days, \|return\| > 0.5%) |
| Top N | 5 |
| Min score | 0.55 |
| Min score (downtrend) | 0.62 |
| Walk-forward | 5 splits, min train 252d, test 63d, gap 1 |

### Output

```
signals/YYYY-MM-DD.csv
```

Columns include: `date`, `symbol`, `score`, `action`, `reason`, context fields (`above_sma_200`, etc.).

Only **`YYYY-MM-DD.csv`** files are valid for execution. Never pass `analysis.csv` or multi-date files to `run_etoro_demo.py`.

---

## 7. Backtesting

Historical weekly portfolio simulation mirroring live rules:

```bash
python scripts/weekly_backtest.py --start 2018-01-01 --end 2024-12-31
```

### Common options

| Flag | Description |
|---|---|
| `--feature-set legacy\|extended` | Feature set (default: extended) |
| `--regime-overlay` | Enable SPY regime exposure scaling |
| `--compare-regime-overlay` | Side-by-side with/without overlay |
| `--compare-features` | Legacy vs extended comparison |
| `--buy-threshold`, `--sell-threshold`, `--min-hold-weeks` | Turnover control |
| `--no-baseline` | Skip legacy baseline comparison |
| `--output` | Equity curve PNG path |

Example:

```bash
python scripts/weekly_backtest.py --compare-regime-overlay --start 2018-01-01 --end 2024-06-30
```

### Metrics reported

| Metric | Meaning |
|---|---|
| **CAGR** | Compound annual growth rate |
| **Sharpe** | Risk-adjusted return (daily, annualized) |
| **Max Drawdown** | Worst peak-to-trough decline |
| **Turnover** | Cumulative one-way turnover at rebalances |
| **Exposure** | Average fraction of capital in equities |
| **Trades** | Count of executed buy/sell actions |

Outputs: console report + `reports/weekly_backtest_equity.png` (and `regime_exposure_timeline.png` when overlay enabled).

---

## 8. Robustness Analysis

Stress-test regime dependence, universe sensitivity, score calibration, and feature importance:

```bash
python scripts/robustness_report.py --start 2018-01-01 --end 2024-06-30
```

### Options

| Flag | Description |
|---|---|
| `--exclude-symbols NVDA,QQQ` | Run variants excluding symbols |
| `--random-subset-size N` | Random universe subsets |
| `--skip-universe-variants` | Faster run (skip universe tests) |
| `--output-dir reports` | Output directory |

### Analysis includes

- **Regime breakdown** — performance in 2018–2019, 2020, 2021, 2022 bear, 2023–2024 bull
- **Rolling metrics** — 252-day rolling Sharpe, CAGR, drawdown
- **Universe sensitivity** — exclude symbols, random subsets
- **Score calibration** — returns by score bucket
- **Feature importance** — aggregated walk-forward fold importance

### Output files

```
reports/robustness_report.txt
reports/robustness_rolling_performance.png
reports/robustness_feature_importance.csv
```

**Known finding:** Strategy is strong in bull/momentum regimes; 2022 bear market was weak without regime overlay (~−32% CAGR, Sharpe −0.64 in bear window).

---

## 9. Parameter Sweep

Grid search over portfolio rule parameters (not ML hyperparameters):

```bash
python scripts/parameter_sweep.py --start 2018-01-01 --end 2024-06-30
python scripts/parameter_sweep.py --fast
python scripts/parameter_sweep.py --max-combinations 50 --jobs 4
```

### Parameters swept

- `buy_threshold`, `sell_threshold`, `min_score`
- `bear_exposure`, `min_hold_weeks`, `vol_threshold`

Walk-forward scores are computed **once** and reused across all combinations.

### Diagnostics

- **Robustness score** = mean(Sharpe) / std(Sharpe) across grid — higher = more stable
- **Parameter importance** — Sharpe sensitivity per parameter
- **Top-N** parameter sets by Sharpe
- **Heatmaps** — pairwise parameter interactions

### Output files

```
reports/parameter_sweep/parameter_sweep_results.csv
reports/parameter_sweep/sharpe_distribution.png
reports/parameter_sweep/cagr_vs_drawdown.png
reports/parameter_sweep/parameter_importance.png
reports/parameter_sweep/heatmap_*.png
```

**Known finding (2018–2024 sample):** Sharpe clustered ~1.32–1.42 across nearby settings; `bear_exposure` showed the largest sensitivity spread — strategy is reasonably stable, not dependent on a single fragile configuration.

---

## 10. Holdout Evaluation

Out-of-sample validation with fixed parameters (no tuning on holdout):

```bash
python scripts/holdout_report.py
```

### Default split

| Period | Dates |
|---|---|
| Development | 2018-01-01 → 2022-12-31 |
| Holdout | 2023-01-01 → 2024-12-31 |

Override with `--dev-start`, `--dev-end`, `--holdout-start`, `--holdout-end`.

### Warnings

The report warns if holdout Sharpe drops materially vs development, or if holdout drawdown is significantly worse. **Holdout is for validation only — never tune parameters on it.**

### Output files

```
reports/holdout_report.txt
reports/holdout_equity.png
```

---

## 11. Demo Trading Workflow

### Mandatory order of operations

| Step | Command | Purpose |
|---|---|---|
| **1** | `python scripts/verify_etoro_api.py` | Confirm API credentials |
| **2** | `python scripts/reconcile_broker.py` | Local state matches broker |
| **3** | `python run_weekly_signal.py` | Generate weekly signals |
| **4** | `python run_etoro_demo.py --use-live-cap` | Preview capped execution plan |
| **5** | `python run_etoro_demo.py --execute` | Submit orders (human approval) |
| **6** | `python scripts/sync_state_from_broker.py --confirm` | Refresh local state from broker |
| **7** | `python scripts/reconcile_broker.py` | Confirm post-trade consistency |

### Critical rules

- **`--execute` must never run if reconciliation fails.** `run_etoro_demo.py` exits with code 1 on reconciliation failure.
- **`--commit-state` is ignored with `--execute`.** Do not update local state manually after live orders — use broker sync.
- Step 4 prints: `Live order cap applied: $500.00 per order` (when `TSML_MAX_LIVE_ORDER_AMOUNT=500`).
- On execution failure, orders stop on **first failure** (no batch continuation).
- After successful execution, a reminder is printed to run `sync_state_from_broker.py --confirm`.

### Dry-run vs execute

| Mode | Command | HTTP orders |
|---|---|---|
| Dry-run (default) | `python run_etoro_demo.py` | No |
| Dry-run with cap preview | `python run_etoro_demo.py --use-live-cap` | No |
| Live demo | `python run_etoro_demo.py --execute` | Yes (demo account) |

---

## 12. Automation Workflow (OpenClaw)

Recommended weekly schedule (automated, no human intervention):

```
1. python scripts/verify_etoro_api.py
2. python scripts/reconcile_broker.py
3. python run_weekly_signal.py
4. python run_etoro_demo.py --use-live-cap
```

Or use the bundled job runner (never executes orders):

```bash
python scripts/weekly_job.py
# Optional local state update after dry-run:
# TSML_COMMIT_STATE=1 python scripts/weekly_job.py
```

### Human approval required before

```
python run_etoro_demo.py --execute
```

An operator must review:

- Signal CSV actions
- Dry-run / `--use-live-cap` execution plan
- Reconciliation report
- `TSML_MAX_LIVE_ORDER_AMOUNT` setting

### After human-approved execution

```
python scripts/sync_state_from_broker.py --confirm
python scripts/reconcile_broker.py
```

### OpenClaw checklist

- [ ] Verify API (`ALL CHECKS PASSED`)
- [ ] Reconciliation passed (exit 0)
- [ ] Signal file exists: `signals/YYYY-MM-DD.csv`
- [ ] Dry-run plan reviewed; live cap applied
- [ ] **Human approval** for `--execute`
- [ ] Post-trade sync + reconciliation
- [ ] Never set `ETORO_ACCOUNT_MODE` to anything other than `demo`
- [ ] Never pass `--execute` from `weekly_job.py` (hardcoded absent)

---

## 13. Safety Rules

1. **Never trade a live account.** Only `ETORO_ACCOUNT_MODE=demo` is supported.
2. **Default recommended cap for first execution:** `TSML_MAX_LIVE_ORDER_AMOUNT=500`
3. **Always reconcile before `--execute`.** Built into `run_etoro_demo.py`; do not bypass.
4. **Never ignore failed reconciliation.** Fix drift via `sync_state_from_broker.py` or manual investigation first.
5. **Never update local state after execute** except via `sync_state_from_broker.py --confirm` after broker confirms fills.
6. **No leverage, no shorts.** Risk layer enforces leverage=1; SELL only for held positions.
7. **Approved universe only.** Orders outside the configured universe are rejected.
8. **Stop on first order failure** during live execution.
9. **`weekly_job.py` never uses `--execute`.** Regression-tested.

---

## 14. Current Strategy Configuration

Defaults used across signal generation, backtest, and demo execution:

| Parameter | Value |
|---|---|
| `feature_set` | `"extended"` |
| `target` | `"threshold"` |
| `top_n` | 5 |
| `buy_threshold` | 0.58 |
| `sell_threshold` | 0.52 |
| `min_hold_weeks` | 2 |
| `min_score` | 0.55 |
| `min_score_downtrend` | 0.62 |
| `bear_exposure` | 0.25 (regime overlay) |
| `cash_buffer` | 0.05 |
| `costs_bps` | 5 |
| Walk-forward | n_splits=5, min_train=252, test=63, gap=1 |

### Universe (15 symbols)

`SPY`, `QQQ`, `AAPL`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`, `META`, `TSLA`, `JPM`, `JNJ`, `XOM`, `V`, `GS`, `NFLX`

---

## 15. Current Validation Results

Summary of latest backtests and validation runs (2018–2024, extended features, turnover control). Re-run scripts to refresh.

### Full-period backtest (extended vs legacy)

| Config | CAGR | Sharpe | Notes |
|---|---|---|---|
| Legacy features | ~12% | ~0.63 | Baseline feature set |
| Extended features | ~33% | ~1.14 | Current production feature set |

### Regime overlay (2018–2024, compare mode)

| | No overlay | With overlay |
|---|---|---|
| CAGR | +34.4% | +30.0% |
| Sharpe | 1.25 | **1.51** |
| Max drawdown | −38.0% | **−23.6%** |
| Avg exposure | 70.6% | 58.2% |

Regime overlay **reduces drawdown** (especially in 2022 bear) at a modest CAGR trade-off.

### Holdout evaluation (dev 2018–2022 / holdout 2023–mid-2024)

| Period | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Development | +20.4% | 1.05 | −24.5% |
| Holdout | +63.3% | 2.53 | −14.1% |
| SPY (holdout) | +29.1% | 2.14 | −10.0% |

No holdout degradation warnings in latest run. Holdout period coincides with strong AI/momentum bull — interpret with appropriate caution.

### Parameter sweep stability

- Sharpe across sampled configs: ~**1.32–1.42**
- Robustness score (8-combo sample): ~**33.5**
- Most sensitive parameter: **`bear_exposure`**
- Nearby settings form a **cluster of good regions** — not a single fragile optimum

### Robustness highlights

- Strong in: 2021 bull, 2023–2024 AI bull, momentum regimes
- Weak in: **2022 bear** without overlay
- Key regime features: `spy_above_sma200`, `spy_vol_20d`

---

## 16. Weekly Operations Runbook

This section describes the exact operational workflow that should run every week.

### Weekly Research Cycle

Run every weekend before trading:

```bash
python run_weekly_signal.py

python scripts/weekly_backtest.py \
  --feature-set extended \
  --regime-overlay

python scripts/robustness_report.py

python scripts/holdout_report.py
```

Review:

* Signal rankings
* Feature importance
* Backtest performance
* Robustness report
* Holdout report

Do not execute trades if:

* Holdout Sharpe deteriorates significantly
* Robustness report shows major instability
* API verification fails
* Reconciliation fails

---

### Weekly Demo Trading Cycle

Step 1

```bash
python scripts/verify_etoro_api.py
```

Must report:

```text
ALL CHECKS PASSED
```

Step 2

```bash
python scripts/reconcile_broker.py
```

Must report:

```text
RECONCILIATION PASSED
```

Step 3

```bash
python run_weekly_signal.py
```

Verify generated signals.

Step 4

```bash
python run_etoro_demo.py --use-live-cap
```

Review proposed orders.

Step 5

Human approval required.

Step 6

```bash
python run_etoro_demo.py --execute
```

Step 7

Wait for fills to settle.

Step 8

```bash
python scripts/sync_state_from_broker.py --confirm
```

Step 9

```bash
python scripts/reconcile_broker.py
```

Must report:

```text
RECONCILIATION PASSED
```

---

### Failure Recovery

If execution fails:

1. Stop all further automation.
2. Run:

```bash
python scripts/reconcile_broker.py
```

3. Run:

```bash
python scripts/sync_state_from_broker.py --confirm
```

4. Re-run reconciliation.

Never manually edit:

```text
data/portfolio_state.json
```

Broker state is always authoritative.

---

## Quick reference

```bash
# Setup
pip install -e ".[dev]"
pytest

# Research
python run_weekly_signal.py
python scripts/weekly_backtest.py --compare-regime-overlay
python scripts/robustness_report.py --start 2018-01-01
python scripts/parameter_sweep.py --fast
python scripts/holdout_report.py

# Demo trading (safe order)
python scripts/verify_etoro_api.py
python scripts/reconcile_broker.py
python run_weekly_signal.py
python run_etoro_demo.py --use-live-cap
# human approval →
python run_etoro_demo.py --execute
python scripts/sync_state_from_broker.py --confirm
python scripts/reconcile_broker.py
```

---

## License & disclaimer

This project is for **research and demo trading education**. Past backtest performance does not guarantee future results. Financial time series are non-stationary. Use at your own risk.
