"""
Multi-Symbol Walk-Forward Demo
-------------------------------
Runs three models across four symbols and three date ranges, then prints
a single flat comparison table sorted by symbol and period.

Models:
  - AlwaysLong            (buy-and-hold baseline, hard predictions)
  - CalibratedLR          (logistic regression + Platt scaling)
  - RandomForest          (ensemble, 200 trees)

Probability models use threshold=0.50.

Splitter parameters adapt to the available row count so that short periods
(e.g. 2022-2023, ~500 rows) still produce meaningful out-of-sample folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tsml.backtest import run_backtest
from tsml.data_loader import YFinanceLoader
from tsml.features.pipeline import make_dataset
from tsml.models.baselines import (
    AlwaysLong,
    CalibratedLogisticRegressionModel,
    RandomForestModel,
)
from tsml.pipelines import evaluate, run_walk_forward, run_walk_forward_proba
from tsml.reporting import plot_equity_curves
from tsml.validation import WalkForwardSplit

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = ["SPY", "QQQ", "MSFT", "NVDA"]

DATE_RANGES = [
    ("2015-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31"),
]

COSTS_BPS  = 5.0
THRESHOLD  = 0.50
TEST_SIZE  = 63    # ~1 quarter per fold (constant)
GAP        = 1     # 1-day embargo near fold boundaries


def make_splitter(n_rows: int) -> WalkForwardSplit:
    """Choose splitter parameters that fit the available row count.

    Rules:
      - min_train_size: 1 year (252) for short periods, 2 years (504) otherwise.
      - n_splits: as many quarterly folds as fit, capped at 8.
    """
    min_train = 252 if n_rows < 800 else 504
    n_splits  = min(8, (n_rows - min_train - GAP) // TEST_SIZE)
    n_splits  = max(1, n_splits)
    return WalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train,
        test_size=TEST_SIZE,
        gap=GAP,
    )


# ── Result row ────────────────────────────────────────────────────────────────

@dataclass
class Row:
    symbol:       str
    period:       str
    model:        str
    sharpe:       float
    total_return: float
    max_drawdown: float
    turnover:     float
    long_pct:     float


# ── Helpers ───────────────────────────────────────────────────────────────────

loader = YFinanceLoader(cache_dir="data/raw")


def _to_row(symbol, period, model_name, report, long_pct) -> Row:
    s = report["strategy"]
    return Row(
        symbol       = symbol,
        period       = period,
        model        = model_name,
        sharpe       = s["sharpe"],
        total_return = s["total_return"],
        max_drawdown = s["max_drawdown"],
        turnover     = s["turnover"],
        long_pct     = long_pct,
    )


# ── Main experiment loop ──────────────────────────────────────────────────────

# Equity-curve plot will be saved for this one case.
PLOT_SYMBOL = "NVDA"
PLOT_PERIOD = ("2020-01-01", "2023-12-31")

rows: list[Row] = []
_plot_backtests: dict[str, pd.DataFrame] = {}   # collected during the loop

for symbol in SYMBOLS:
    for start, end in DATE_RANGES:
        period = f"{start[:4]}-{end[:4]}"
        print(f"[{symbol}  {period}] loading ...", end="  ", flush=True)

        df = loader.load(symbol, start, end)
        _, y = make_dataset(df, target="direction")
        splitter = make_splitter(len(df))
        n_folds  = splitter.n_splits

        print(f"{len(df):,} rows  |  {n_folds} folds")

        save_plot = (symbol == PLOT_SYMBOL and (start, end) == PLOT_PERIOD)

        # ── AlwaysLong ────────────────────────────────────────────────────
        preds  = run_walk_forward(df, AlwaysLong(), splitter, target="direction")
        bt_al  = run_backtest(preds, df["close"], costs_bps=COSTS_BPS)
        y_true = y.loc[preds.index]
        report = evaluate(preds, y_true, bt_al)
        rows.append(_to_row(symbol, period, "AlwaysLong", report, long_pct=100.0))
        if save_plot:
            _plot_backtests["AlwaysLong"] = bt_al

        # ── CalibratedLR ─────────────────────────────────────────────────
        model  = CalibratedLogisticRegressionModel(
            C=1.0, method="sigmoid", cv=5, random_state=42
        )
        probas = run_walk_forward_proba(df, model, splitter, target="direction")
        preds  = (probas > THRESHOLD).astype(int).rename("prediction")
        bt_clr = run_backtest(preds, df["close"], costs_bps=COSTS_BPS)
        y_true = y.loc[preds.index]
        report = evaluate(preds, y_true, bt_clr)
        long_pct = 100.0 * float((probas > THRESHOLD).mean())
        rows.append(_to_row(symbol, period, "CalibratedLR", report, long_pct))
        if save_plot:
            _plot_backtests["CalibratedLR"] = bt_clr

        # ── RandomForest ──────────────────────────────────────────────────
        model  = RandomForestModel(
            n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42
        )
        probas = run_walk_forward_proba(df, model, splitter, target="direction")
        preds  = (probas > THRESHOLD).astype(int).rename("prediction")
        bt_rf  = run_backtest(preds, df["close"], costs_bps=COSTS_BPS)
        y_true = y.loc[preds.index]
        report = evaluate(preds, y_true, bt_rf)
        long_pct = 100.0 * float((probas > THRESHOLD).mean())
        rows.append(_to_row(symbol, period, "RandomForest", report, long_pct))
        if save_plot:
            _plot_backtests["RandomForest"] = bt_rf

# ── Print comparison table ────────────────────────────────────────────────────

C = {            # column widths
    "symbol":  6,
    "period":  9,
    "model":  12,
    "sharpe":  7,
    "ret":    10,
    "dd":      9,
    "turn":    9,
    "long":    7,
}

HEADER = (
    f"{'Symbol':<{C['symbol']}}  "
    f"{'Period':<{C['period']}}  "
    f"{'Model':<{C['model']}}  "
    f"{'Sharpe':>{C['sharpe']}}  "
    f"{'TotalRet':>{C['ret']}}  "
    f"{'MaxDD':>{C['dd']}}  "
    f"{'Turnover':>{C['turn']}}  "
    f"{'Long%':>{C['long']}}"
)
SEP = "-" * len(HEADER)

print()
print("=" * len(HEADER))
print(f"  MULTI-SYMBOL COMPARISON  (threshold={THRESHOLD}, costs={COSTS_BPS} bps)")
print("=" * len(HEADER))
print(HEADER)
print(SEP)

prev_key = None
for r in rows:
    key = (r.symbol, r.period)
    if prev_key and key != prev_key:
        print(SEP)
    prev_key = key

    print(
        f"{r.symbol:<{C['symbol']}}  "
        f"{r.period:<{C['period']}}  "
        f"{r.model:<{C['model']}}  "
        f"{r.sharpe:>{C['sharpe']}.3f}  "
        f"{r.total_return:>{C['ret']}.3f}  "
        f"{r.max_drawdown:>{C['dd']}.3f}  "
        f"{r.turnover:>{C['turn']}.3f}  "
        f"{r.long_pct:>{C['long']}.1f}%"
    )

print("=" * len(HEADER))
print()

# ── Equity curve plot ─────────────────────────────────────────────────────────

if _plot_backtests:
    plot_title  = f"{PLOT_SYMBOL} {PLOT_PERIOD[0][:4]}-{PLOT_PERIOD[1][:4]}  |  walk-forward equity curves"
    output_file = f"reports/{PLOT_SYMBOL.lower()}_{PLOT_PERIOD[0][:4]}_{PLOT_PERIOD[1][:4]}_equity.png"
    saved = plot_equity_curves(_plot_backtests, title=plot_title, output_path=output_file)
    print(f"Equity curve saved to: {saved}")
