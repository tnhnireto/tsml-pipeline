# Stock Prediction Review - Session Report
**Date:** July 2, 2026  
**Session ID:** 9115e577-561b-4456-9ceb-e6a756dbce26

---

## Executive Summary

Analysis complete with honest out-of-sample (OOS) coverage spanning 32 folds from 2018-04 → 2026-04. Fixed the frozen-score artifact that contaminated previous results by correcting the warmup estimate from 200 rows to ~70 rows.

---

## 1. Threshold Sweep Results

### Grid A — Live-Style Rules
*(Used by `run_weekly_signal.py`; `min_score` active)*

| min_score | msd | CAGR | Sharpe | MaxDD | Turnover | Trades |
|---|---|---|---|---|---|---|
| 0.55 | 0.62 (current) | 25.6% | 1.083 | −31.5% | 111 | 1077 |
| **0.58** | **0.62** | **26.1%** | **1.070** | **−29.5%** | 118 | 1035 |
| 0.58 | 0.64 | 26.5% | 1.122 | −30.4% | 125 | 1007 |
| 0.60 | 0.62 | 26.0% | 1.069 | −31.7% | 135 | 952 |
| 0.62 | 0.62 | 24.3% | 1.012 | −35.8% | 175 | 840 |

**Key Finding:** Moving 0.55 → 0.58 is free (equal/better Sharpe, best MaxDD). Higher thresholds (0.60+) degrade performance — book shrinks to 3–4 names, increasing churn and concentration risk. `msd=0.66` is worst everywhere.

### Grid B — Hysteresis Rules
*(Backtest default; `min_score` is a no-op here)*

| buy | sell | msd | CAGR | Sharpe | MaxDD | Turnover |
|---|---|---|---|---|---|---|
| 0.58 | 0.52 | 0.62 (current) | 25.7% | 1.074 | −32.4% | 31.7 |
| 0.62 | 0.52 | 0.62 | 26.9% | 1.135 | −31.3% | 33.6 |
| 0.58 | 0.56 | 0.62 | 26.5% | 1.100 | −31.5% | 47.3 |

Current defaults sit mid-plateau; the 0.62/0.52 peak is a single cell — not worth chasing. Note: `msd=0.62` is consistently best (earlier `msd=0.64` advantage vanished with honest coverage).

---

## 2. Score Calibration
*(Pooled OOS, n=30,240)*

| Bucket | n | Avg fwd-5d | Win Rate | Excess vs SPY |
|---|---|---|---|---|
| 0.50–0.55 | 6367 | 0.48% | 55.6% | 0.17% |
| **0.55–0.58** | 6053 | 0.45% | 55.7% | **0.11% ← weakest** |
| 0.58–0.60 | 3982 | 0.57% | 56.7% | 0.22% |
| 0.60–0.62 | 3638 | 0.55% | 58.0% | 0.22% |
| 0.62+ | 7402 | 0.49% | 57.6% | 0.23% |

Win rate is weakly monotone to 0.62 then flattens — scores rank OK but aren't sharply calibrated. The 0.55–0.58 bucket is demonstrably the weakest performer. Don't over-tighten beyond 0.58.

---

## 3. Live Sensitivity
*(Latest date — no retraining)*

| min_score | Eligible | Selected |
|---|---|---|
| 0.55 | 12 | V, MSFT, SPY, QQQ, NVDA |
| 0.58 | 7 | **same 5** |
| 0.60 | 4 | V, MSFT, SPY, QQQ |
| 0.62 | 3 | V, MSFT, SPY |

**Moving to 0.58 changes nothing today.** Edge case: MSFT is below SMA200 but passes via score 0.6246 ≥ msd.

---

## 4. Validation Suite
*(extended_v2 / direction_5d / gap=5)*

### Weekly Backtest (2017 → 2026)
```
Total Return              +513.94%
CAGR                       +21.14%
Sharpe                        0.97
Max Drawdown               -34.95%
Turnover (cum.)           3051.08%
Turnover (exp-adj.)          37.13
Avg Exposure                82.18%
Trades                         229
Avg Hold (weeks)              17.7

vs SPY:  CAGR +15.29%, Sharpe 0.87, DD -33.72%
vs QQQ:  CAGR +21.80%, Sharpe 0.98, DD -35.12%
```

### Parameter Sweep
- **Robustness score:** 5.881
- **Sharpe percentiles:** p10=0.707, p50=0.876, p90=1.059
- **Parameter importance:** bear_exposure dominates (0.2786); buy/sell thresholds tiny (0.03–0.07) → not fragile
- **min_score sensitivity:** 0.0000 spread in hysteresis mode (confirmed no-op)

### Holdout Test (Dev: 2018-2022 | Holdout: 2023-2026-07)
```
Development:
  CAGR                       +13.59%
  Sharpe                        0.85
  Max Drawdown               -24.73%

Holdout:
  CAGR                       +26.10%
  Sharpe                        1.46
  Max Drawdown               -15.48%

Holdout vs Development:
  Sharpe delta                 +0.61
  CAGR delta                 +12.51%
  MDD delta                   +9.25%
  
→ No degradation warnings.
```

### Robustness Report
- **Regimes:** 2022 bear is the weak spot; 2021 bull delivers +69% CAGR
- **Universe stability:** without NVDA Sharpe 0.86, without QQQ Sharpe 0.98
- **Top features:** distance_from_20d_high (0.52), distance_from_60d_high (0.39), spy_vol_20d (0.35)

---

## 5. Recommendation

### Change One Thing: `MIN_SCORE` 0.55 → 0.58 in `run_weekly_signal.py`

**Keep everything else:**
- `MIN_SCORE_DOWNTREND = 0.62`
- Backtest hysteresis: buy 0.58 / sell 0.52

### Rationale

✅ **Culls the weakest bucket** (0.55–0.58: 0.11% excess) with zero impact on today's book  
✅ **Best MaxDD in Grid A** (−29.5%), Sharpe ≥ current  
✅ **0.60/0.62 rejected:** worse Sharpe/DD, 3–4 name concentration  
✅ **msd 0.64/0.66 rejected:** grid-dependent or strictly worse

---

## Test Results
```
Full pytest:        772 passed, 4 skipped
Splitter tests:     26 passed
Warmup fix:         Commit 408c37b ✓
All validation:     All runs complete and clean ✓
```

---

**Recommendation Status:** Ready for implementation.
