"""Bar-by-bar backtest engine using predicted returns for signals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(preds: pd.DataFrame,
                 fee_bps: float = 10,
                 slippage_bps: float = 2,
                 entry_threshold: float = 0.003,
                 exit_threshold: float = 0.001,
                 stop_atr_mult: float = 2.0,
                 min_hold: int = 3,
                 ema_span: int = 3) -> pd.DataFrame:
    """Bar-by-bar backtest driven by predicted returns.

    Goes long when predicted_return > +entry_threshold,
    short when predicted_return < -entry_threshold,
    exits when |predicted_return| < exit_threshold.

    Args:
        entry_threshold: minimum |predicted_return| to enter a position (default 0.3%).
        exit_threshold: |predicted_return| below which we exit (default 0.1%).
        stop_atr_mult: stop-loss distance in ATR multiples.
        min_hold: minimum bars to hold before allowing exit.
        ema_span: EMA span to smooth predictions (reduces noise).
    """
    cost = (fee_bps + slippage_bps) / 10_000

    close = preds["close"].values
    high = preds["high"].values
    low = preds["low"].values
    raw_pred = preds["predicted_return"].values
    pred = pd.Series(raw_pred).ewm(span=ema_span).mean().values if ema_span > 1 else raw_pred
    atr = preds["atr_raw"].values

    n = len(close)
    positions = np.zeros(n)
    equity = np.ones(n)
    costs_arr = np.zeros(n)

    cur_pos = 0
    stop_px = 0.0
    hold_count = 0
    n_trades = 0

    for i in range(1, n):
        prev = close[i - 1]
        if prev <= 0:
            positions[i] = cur_pos
            equity[i] = equity[i - 1]
            hold_count += 1
            continue

        bar_ret = (close[i] - prev) / prev
        realized = 0.0
        bar_cost = 0.0
        stopped = False

        # --- Check stop-loss ---
        if cur_pos == 1 and low[i] <= stop_px:
            realized = (stop_px - prev) / prev
            bar_cost += cost
            cur_pos = 0
            hold_count = 0
            stopped = True
            n_trades += 1
        elif cur_pos == -1 and high[i] >= stop_px:
            realized = -((stop_px - prev) / prev)
            bar_cost += cost
            cur_pos = 0
            hold_count = 0
            stopped = True
            n_trades += 1

        if not stopped:
            realized = cur_pos * bar_ret

            # --- Check signal for position change ---
            if hold_count >= min_hold:
                p = pred[i]
                new_pos = cur_pos

                if cur_pos == 0:
                    if p > entry_threshold:
                        new_pos = 1
                    elif p < -entry_threshold:
                        new_pos = -1
                elif cur_pos == 1:
                    if p < exit_threshold:
                        new_pos = 0
                    if p < -entry_threshold:
                        new_pos = -1
                elif cur_pos == -1:
                    if p > -exit_threshold:
                        new_pos = 0
                    if p > entry_threshold:
                        new_pos = 1

                if new_pos != cur_pos:
                    bar_cost += abs(new_pos - cur_pos) * cost
                    cur_pos = new_pos
                    n_trades += 1
                    hold_count = 0
                    if cur_pos != 0:
                        stop_px = close[i] - cur_pos * stop_atr_mult * atr[i]

        positions[i] = cur_pos
        costs_arr[i] = bar_cost
        pnl = realized - bar_cost
        equity[i] = equity[i - 1] * (1 + pnl)
        hold_count += 1

    out = preds[["close"]].copy()
    out["position"] = positions
    out["equity"] = equity
    out["costs"] = costs_arr
    out["bar_return"] = pd.Series(close).pct_change().values
    out["buyhold_equity"] = (1 + out["bar_return"].fillna(0)).cumprod().values
    out["n_trades"] = n_trades
    return out


def compute_metrics(df: pd.DataFrame, bars_per_year: float = 2190) -> dict:
    """Compute metrics. Default bars_per_year=2190 for 4h bars (365*6)."""
    strat_rets = pd.Series(df["equity"].values).pct_change().dropna()
    bh_rets = pd.Series(df["buyhold_equity"].values).pct_change().dropna()

    def sharpe(rets: pd.Series) -> float:
        if rets.std() == 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(bars_per_year))

    def max_dd(equity: pd.Series) -> float:
        peak = equity.cummax()
        return float(((equity - peak) / peak).min())

    pos = df["position"]
    eq = pd.Series(df["equity"].values)
    bh = pd.Series(df["buyhold_equity"].values)

    return {
        "total_return": float(eq.iloc[-1] - 1),
        "buyhold_return": float(bh.iloc[-1] - 1),
        "sharpe": sharpe(strat_rets),
        "buyhold_sharpe": sharpe(bh_rets),
        "max_drawdown": max_dd(eq),
        "buyhold_max_dd": max_dd(bh),
        "n_trades": int(df["n_trades"].iloc[0]),
        "total_costs": float(df["costs"].sum()),
        "long_pct": float((pos == 1).mean()),
        "short_pct": float((pos == -1).mean()),
        "flat_pct": float((pos == 0).mean()),
    }


def print_metrics(m: dict) -> None:
    print("\n=== BACKTEST RESULTS ===")
    print(f"  {'strategy return':25s}: {m['total_return']:+.2%}")
    print(f"  {'buy & hold return':25s}: {m['buyhold_return']:+.2%}")
    print(f"  {'strategy sharpe':25s}: {m['sharpe']:+.2f}")
    print(f"  {'buy & hold sharpe':25s}: {m['buyhold_sharpe']:+.2f}")
    print(f"  {'max drawdown':25s}: {m['max_drawdown']:+.2%}")
    print(f"  {'buy & hold max DD':25s}: {m['buyhold_max_dd']:+.2%}")
    print(f"  {'total trades':25s}: {m['n_trades']}")
    print(f"  {'total costs':25s}: {m['total_costs']:.4f}")
    print(f"  {'time long':25s}: {m['long_pct']:.1%}")
    print(f"  {'time short':25s}: {m['short_pct']:.1%}")
    print(f"  {'time flat':25s}: {m['flat_pct']:.1%}")
