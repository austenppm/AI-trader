#!/usr/bin/env python
"""Run trend-following strategy with ML regime overlay on ETHUSDT."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ai_trader.data.binance import load_klines
from ai_trader.strategy import (
    trend_signals, train_regime_model, compute_positions,
)
from ai_trader.paths import DATA_ROOT


def backtest(df: pd.DataFrame, positions: pd.Series,
             fee_bps: float = 10, slippage_bps: float = 2,
             stop_atr_mult: float = 2.5, min_hold: int = 3) -> pd.DataFrame:
    """Bar-by-bar backtest with stops."""
    cost = (fee_bps + slippage_bps) / 10_000
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    target_pos = positions.values

    # ATR for stops
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)),
                   np.abs(low - np.roll(close, 1)))
    )
    atr = pd.Series(tr).rolling(14).mean().values

    n = len(close)
    actual_pos = np.zeros(n)
    equity = np.ones(n)
    costs_arr = np.zeros(n)

    cur = 0
    stop_px = 0.0
    hold = 0
    n_trades = 0

    for i in range(1, n):
        prev = close[i - 1]
        if prev <= 0 or np.isnan(prev):
            actual_pos[i] = cur
            equity[i] = equity[i - 1]
            hold += 1
            continue

        bar_ret = (close[i] - prev) / prev
        realized = 0.0
        bar_cost = 0.0
        stopped = False

        # Stop check
        if cur == 1 and low[i] <= stop_px:
            realized = (stop_px - prev) / prev
            bar_cost += cost
            cur = 0
            hold = 0
            stopped = True
            n_trades += 1
        elif cur == -1 and high[i] >= stop_px:
            realized = -((stop_px - prev) / prev)
            bar_cost += cost
            cur = 0
            hold = 0
            stopped = True
            n_trades += 1

        if not stopped:
            realized = cur * bar_ret
            if hold >= min_hold:
                new = int(target_pos[i])
                if new != cur:
                    bar_cost += abs(new - cur) * cost
                    cur = new
                    n_trades += 1
                    hold = 0
                    if cur != 0 and not np.isnan(atr[i]):
                        stop_px = close[i] - cur * stop_atr_mult * atr[i]

        actual_pos[i] = cur
        costs_arr[i] = bar_cost
        equity[i] = equity[i - 1] * (1 + realized - bar_cost)
        hold += 1

    out = pd.DataFrame({
        "close": close,
        "position": actual_pos,
        "equity": equity,
        "costs": costs_arr,
    }, index=df.index[:n])
    bh_ret = pd.Series(close).pct_change().fillna(0)
    out["buyhold_equity"] = (1 + bh_ret).cumprod().values
    out["n_trades"] = n_trades
    return out


def metrics(df: pd.DataFrame, bars_per_year: float = 2190) -> dict:
    strat = pd.Series(df["equity"].values).pct_change().dropna()
    bh = pd.Series(df["buyhold_equity"].values).pct_change().dropna()

    def sharpe(r):
        return float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 0 else 0.0

    def mdd(eq):
        peak = eq.cummax()
        return float(((eq - peak) / peak).min())

    eq = pd.Series(df["equity"].values)
    bheq = pd.Series(df["buyhold_equity"].values)
    pos = df["position"]

    years = len(df) / bars_per_year
    total_ret = eq.iloc[-1] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else 0

    return {
        "total_return": float(total_ret),
        "cagr": float(cagr),
        "buyhold_return": float(bheq.iloc[-1] - 1),
        "sharpe": sharpe(strat),
        "buyhold_sharpe": sharpe(bh),
        "max_drawdown": mdd(eq),
        "buyhold_max_dd": mdd(bheq),
        "n_trades": int(df["n_trades"].iloc[0]),
        "total_costs": float(df["costs"].sum()),
        "long_pct": float((pos == 1).mean()),
        "short_pct": float((pos == -1).mean()),
        "flat_pct": float((pos == 0).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval", default="4h")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--min-train-months", type=int, default=6)
    p.add_argument("--fee-bps", type=float, default=10)
    p.add_argument("--slippage-bps", type=float, default=2)
    p.add_argument("--regime-threshold", type=float, default=0.45)
    p.add_argument("--trend-threshold", type=float, default=0.001)
    p.add_argument("--stop-atr", type=float, default=2.5)
    p.add_argument("--min-hold", type=int, default=3)
    args = p.parse_args()

    print(f"Loading {args.symbol} {args.interval}...")
    df = load_klines(args.symbol, args.interval)
    print(f"  {len(df):,} bars  |  {df.index.min()} -> {df.index.max()}")

    print("\nComputing trend signals...")
    signals = trend_signals(df)
    print(f"  {len(signals.columns)} signal components")

    print("\nTraining regime detector (walk-forward)...\n")
    regime = train_regime_model(
        df, signals["trend_signal"],
        min_train_months=args.min_train_months, horizon=args.horizon,
    )
    if regime.empty:
        print("No predictions.")
        return

    # Merge and compute positions
    combined = regime.copy()
    positions = compute_positions(combined,
                                  regime_threshold=args.regime_threshold,
                                  trend_threshold=args.trend_threshold)

    # Align df to prediction period
    df_bt = df.loc[positions.index]
    print(f"\n  {len(positions):,} bars  |  {positions.index.min()} -> {positions.index.max()}")

    print("\nRunning backtest...")
    results = backtest(df_bt, positions,
                       fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                       stop_atr_mult=args.stop_atr, min_hold=args.min_hold)
    m = metrics(results)

    print("\n=== RESULTS ===")
    print(f"  {'strategy return':25s}: {m['total_return']:+.2%}")
    print(f"  {'CAGR':25s}: {m['cagr']:+.2%}")
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

    # Save
    out = DATA_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / f"{args.symbol}_{args.interval}_strategy.parquet")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                 sharex=True, gridspec_kw={"hspace": 0.08})
        ax = axes[0]
        ax.plot(results.index, results["equity"], label="Strategy", linewidth=1)
        ax.plot(results.index, results["buyhold_equity"], label="Buy & Hold", alpha=0.6)
        ax.set_ylabel("Equity ($1 start)")
        ax.set_title(f"{args.symbol} Trend + Regime  |  Sharpe {m['sharpe']:+.2f}  |  CAGR {m['cagr']:+.1%}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        ax2 = axes[1]
        pos = results["position"]
        ax2.fill_between(results.index, pos, 0, where=pos > 0, color="green", alpha=0.3, label="Long")
        ax2.fill_between(results.index, pos, 0, where=pos < 0, color="red", alpha=0.3, label="Short")
        ax2.set_ylabel("Position")
        ax2.set_ylim(-1.5, 1.5)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        chart = out / f"{args.symbol}_{args.interval}_strategy.png"
        fig.savefig(str(chart), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nChart: {chart}")
    except Exception as e:
        print(f"\n(chart skipped: {e})")

    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
