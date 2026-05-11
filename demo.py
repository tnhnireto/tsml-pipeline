"""
Multi-Symbol Walk-Forward Demo
-------------------------------
Compares three experiment configurations across five symbols and three
date ranges, then prints a single flat comparison table.

Experiment configs
------------------
  1. target="direction",    holding_period=1   baseline: daily signal, hold 1 day
  2. target="direction_5d", holding_period=5   medium-term: 5-day label, hold 5 days
  3. target="threshold",    holding_period=1   conviction-only: high |return| days

Models: AlwaysLong (baseline), CalibratedLR, RandomForest.
Probability signal threshold: 0.50.  Transaction costs: 5 bps.

One equity-curve plot is saved for the most interesting case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from tsml.backtest import run_backtest
from tsml.data_loader import YFinanceLoader
from tsml.features import make_dataset
from tsml.models.baselines import (
    AlwaysLong,
    CalibratedLogisticRegressionModel,
    RandomForestModel,
)
from tsml.pipelines import evaluate, run_walk_forward, run_walk_forward_proba
from tsml.reporting import plot_equity_curves
from tsml.validation import WalkForwardSplit

# ── Global configuration ───────────────────────────────────────────────────────

SYMBOLS = ["SPY", "QQQ", "MSFT", "NVDA", "GOOGL"]

DATE_RANGES = [
    ("2015-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31"),
]

class Config(NamedTuple):
    target:         str
    holding_period: int
    label:          str   # short label used in the table

CONFIGS = [
    Config("direction",    1, "dir/hp1"),
    Config("direction_5d", 5, "5d/hp5"),
    Config("threshold",    1, "thr/hp1"),
]

COSTS_BPS  = 5.0
THRESHOLD  = 0.50    # probability cut-off for long signal
TEST_SIZE  = 63      # ~1 quarter per fold
GAP        = 1       # 1-day embargo near fold boundaries

# Single equity-curve plot saved for this (symbol, start, end, config index).
PLOT_CASE = ("NVDA", "2020-01-01", "2023-12-31", 0)   # config index 0 = direction/hp1


# ── Splitter factory ───────────────────────────────────────────────────────────

def make_splitter(n_rows: int) -> WalkForwardSplit | None:
    """Adapt splitter parameters to available row count.

    Returns None when the dataset is too small for even a single fold,
    signalling the caller to skip this (symbol, period, config) combination.
    """
    min_train = 252 if n_rows < 800 else 504
    if n_rows < min_train + TEST_SIZE + GAP:
        return None
    n_splits = min(8, (n_rows - min_train - GAP) // TEST_SIZE)
    return WalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train,
        test_size=TEST_SIZE,
        gap=GAP,
    )


# ── Result row ─────────────────────────────────────────────────────────────────

@dataclass
class Row:
    symbol:         str
    period:         str
    config_label:   str
    model:          str
    accuracy:       float
    total_return:   float
    sharpe:         float
    max_drawdown:   float
    turnover:       float
    long_pct:       float


# ── Helpers ────────────────────────────────────────────────────────────────────

loader = YFinanceLoader(cache_dir="data/raw")


def _run_one(
    df: pd.DataFrame,
    splitter: WalkForwardSplit,
    y: pd.Series,
    cfg: Config,
    model_name: str,
    preds: pd.Series,
    bt: pd.DataFrame,
    long_pct: float,
) -> Row:
    """Evaluate a single (model, backtest, config) combination."""
    y_true = y.loc[y.index.intersection(preds.index)]
    report = evaluate(preds, y_true, bt)
    return Row(
        symbol       = "",          # filled by caller
        period       = "",          # filled by caller
        config_label = cfg.label,
        model        = model_name,
        accuracy     = report["ml"]["accuracy"],
        total_return = report["strategy"]["total_return"],
        sharpe       = report["strategy"]["sharpe"],
        max_drawdown = report["strategy"]["max_drawdown"],
        turnover     = report["strategy"]["turnover"],
        long_pct     = long_pct,
    )


# ── Main experiment loop ───────────────────────────────────────────────────────

rows:          list[Row]                         = []
_plot_backtests: dict[str, pd.DataFrame]         = {}   # model → backtest df

for symbol in SYMBOLS:
    for start, end in DATE_RANGES:
        period = f"{start[:4]}-{end[:4]}"
        print(f"[{symbol}  {period}] loading ...", end="  ", flush=True)

        df = loader.load(symbol, start, end)
        print(f"{len(df):,} rows")

        for cfg_idx, cfg in enumerate(CONFIGS):
            X, y     = make_dataset(df, target=cfg.target)
            # Build splitter from cleaned row count: the threshold target
            # drops ~50 % of rows, so we must size folds after dropping NaNs.
            splitter  = make_splitter(len(X))
            if splitter is None:
                print(f"  [{cfg.label}]  {len(X):,} rows  |  SKIP (too few rows)")
                continue
            save_plot = (symbol, start, end, cfg_idx) == PLOT_CASE
            print(f"  [{cfg.label}]  {len(X):,} rows  |  {splitter.n_splits} folds")

            # ── AlwaysLong ────────────────────────────────────────────────
            preds = run_walk_forward(df, AlwaysLong(), splitter, target=cfg.target)
            bt    = run_backtest(preds, df["close"],
                                 costs_bps=COSTS_BPS,
                                 holding_period=cfg.holding_period)
            row = _run_one(df, splitter, y, cfg, "AlwaysLong", preds, bt, 100.0)
            row.symbol, row.period = symbol, period
            rows.append(row)
            if save_plot:
                _plot_backtests["AlwaysLong"] = bt

            # ── CalibratedLR ──────────────────────────────────────────────
            model  = CalibratedLogisticRegressionModel(
                C=1.0, method="sigmoid", cv=5, random_state=42
            )
            probas = run_walk_forward_proba(df, model, splitter, target=cfg.target)
            preds  = (probas > THRESHOLD).astype(int).rename("prediction")
            bt     = run_backtest(preds, df["close"],
                                  costs_bps=COSTS_BPS,
                                  holding_period=cfg.holding_period)
            long_pct = 100.0 * float((probas > THRESHOLD).mean())
            row = _run_one(df, splitter, y, cfg, "CalibratedLR", preds, bt, long_pct)
            row.symbol, row.period = symbol, period
            rows.append(row)
            if save_plot:
                _plot_backtests["CalibratedLR"] = bt

            # ── RandomForest ──────────────────────────────────────────────
            model  = RandomForestModel(
                n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=42
            )
            probas = run_walk_forward_proba(df, model, splitter, target=cfg.target)
            preds  = (probas > THRESHOLD).astype(int).rename("prediction")
            bt     = run_backtest(preds, df["close"],
                                  costs_bps=COSTS_BPS,
                                  holding_period=cfg.holding_period)
            long_pct = 100.0 * float((probas > THRESHOLD).mean())
            row = _run_one(df, splitter, y, cfg, "RandomForest", preds, bt, long_pct)
            row.symbol, row.period = symbol, period
            rows.append(row)
            if save_plot:
                _plot_backtests["RandomForest"] = bt

# ── Print comparison table ─────────────────────────────────────────────────────

W = {
    "sym":    6,
    "per":    9,
    "cfg":    8,
    "mdl":   12,
    "acc":    6,
    "ret":    8,
    "sh":     7,
    "dd":     8,
    "to":     8,
    "lp":     6,
}

HEADER = (
    f"{'Symbol':<{W['sym']}}  "
    f"{'Period':<{W['per']}}  "
    f"{'Config':<{W['cfg']}}  "
    f"{'Model':<{W['mdl']}}  "
    f"{'Acc':>{W['acc']}}  "
    f"{'TotRet':>{W['ret']}}  "
    f"{'Sharpe':>{W['sh']}}  "
    f"{'MaxDD':>{W['dd']}}  "
    f"{'Turnover':>{W['to']}}  "
    f"{'Long%':>{W['lp']}}"
)
SEP  = "-" * len(HEADER)
SEP2 = "=" * len(HEADER)

print()
print(SEP2)
print(f"  CONFIG x MODEL COMPARISON  (signal_threshold={THRESHOLD}, costs={COSTS_BPS} bps)")
print(SEP2)
print(HEADER)
print(SEP)

prev_key = None
for r in rows:
    key = (r.symbol, r.period)
    if prev_key is not None and key != prev_key:
        print(SEP)
    prev_key = key
    print(
        f"{r.symbol:<{W['sym']}}  "
        f"{r.period:<{W['per']}}  "
        f"{r.config_label:<{W['cfg']}}  "
        f"{r.model:<{W['mdl']}}  "
        f"{r.accuracy:>{W['acc']}.3f}  "
        f"{r.total_return:>{W['ret']}.3f}  "
        f"{r.sharpe:>{W['sh']}.3f}  "
        f"{r.max_drawdown:>{W['dd']}.3f}  "
        f"{r.turnover:>{W['to']}.3f}  "
        f"{r.long_pct:>{W['lp']}.1f}%"
    )

print(SEP2)
print()

# ── Equity curve plot ──────────────────────────────────────────────────────────

if _plot_backtests:
    sym, start, end, cfg_idx = PLOT_CASE
    cfg = CONFIGS[cfg_idx]
    title       = (f"{sym} {start[:4]}-{end[:4]}  |  "
                   f"target={cfg.target}  hp={cfg.holding_period}")
    output_file = (f"reports/{sym.lower()}_{start[:4]}_{end[:4]}_"
                   f"{cfg.target}_hp{cfg.holding_period}_equity.png")
    saved = plot_equity_curves(_plot_backtests, title=title, output_path=output_file)
    print(f"Equity curve saved to: {saved}")

# ── Summary: best result per symbol/period ─────────────────────────────────────

# Group rows by (symbol, period).
from collections import defaultdict
_groups: dict[tuple, list[Row]] = defaultdict(list)
for r in rows:
    _groups[(r.symbol, r.period)].append(r)

SW = {
    "sym":  6,
    "per":  9,
    "cfg":  8,
    "mdl": 12,
    "sh":   7,
    "ret":  8,
    "dd":   8,
    "to":   8,
    "beat": 7,
}

SHEADER = (
    f"{'Symbol':<{SW['sym']}}  "
    f"{'Period':<{SW['per']}}  "
    f"{'Config':<{SW['cfg']}}  "
    f"{'BestModel':<{SW['mdl']}}  "
    f"{'Sharpe':>{SW['sh']}}  "
    f"{'TotRet':>{SW['ret']}}  "
    f"{'MaxDD':>{SW['dd']}}  "
    f"{'Turnover':>{SW['to']}}  "
    f"{'Beat AL':>{SW['beat']}}"
)
SSEP  = "-" * len(SHEADER)
SSEP2 = "=" * len(SHEADER)

print()
print(SSEP2)
print("  BEST RESULT PER SYMBOL / PERIOD")
print(SSEP2)
print(SHEADER)
print(SSEP)

beat_count  = 0
total_groups = 0

for (sym, per), group in _groups.items():
    total_groups += 1

    # Best AlwaysLong Sharpe for this group (baseline reference).
    al_sharpes = [r.sharpe for r in group if r.model == "AlwaysLong"
                  and not (r.sharpe != r.sharpe)]  # exclude NaN
    al_best_sharpe = max(al_sharpes) if al_sharpes else float("-inf")

    # Best overall row by Sharpe (skip NaN Sharpe rows).
    valid = [r for r in group if r.sharpe == r.sharpe]   # NaN != NaN
    if not valid:
        continue
    best = max(valid, key=lambda r: r.sharpe)

    # Did a TRAINED model (non-AlwaysLong) beat AlwaysLong?
    trained_sharpes = [r.sharpe for r in group
                       if r.model != "AlwaysLong" and r.sharpe == r.sharpe]
    trained_beat = bool(trained_sharpes and max(trained_sharpes) > al_best_sharpe)
    if trained_beat:
        beat_count += 1

    beat_str = "YES" if trained_beat else "no"

    print(
        f"{sym:<{SW['sym']}}  "
        f"{per:<{SW['per']}}  "
        f"{best.config_label:<{SW['cfg']}}  "
        f"{best.model:<{SW['mdl']}}  "
        f"{best.sharpe:>{SW['sh']}.3f}  "
        f"{best.total_return:>{SW['ret']}.3f}  "
        f"{best.max_drawdown:>{SW['dd']}.3f}  "
        f"{best.turnover:>{SW['to']}.3f}  "
        f"{beat_str:>{SW['beat']}}"
    )

print(SSEP2)
print()
print(f"  Trained model beat AlwaysLong : {beat_count} / {total_groups} groups")
print(f"  AlwaysLong remained best      : {total_groups - beat_count} / {total_groups} groups")
print()
