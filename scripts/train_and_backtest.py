#!/usr/bin/env python
"""Train a walk-forward regression model and backtest with dynamic positions."""
from __future__ import annotations

import argparse

from ai_trader.data.binance import load_klines
from ai_trader.features import build_features
from ai_trader.model import walk_forward
from ai_trader.backtest import run_backtest, compute_metrics, print_metrics
from ai_trader.paths import DATA_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval", default="4h")
    p.add_argument("--horizon", type=int, default=6,
                   help="prediction horizon in bars (default 6 = 24h for 4h bars)")
    p.add_argument("--min-train-months", type=int, default=6)
    p.add_argument("--fee-bps", type=float, default=10)
    p.add_argument("--slippage-bps", type=float, default=2)
    p.add_argument("--entry-threshold", type=float, default=0.003,
                   help="min |predicted_return| to enter (default 0.3%%)")
    p.add_argument("--exit-threshold", type=float, default=0.001,
                   help="|predicted_return| below which we exit (default 0.1%%)")
    p.add_argument("--stop-atr", type=float, default=2.0)
    p.add_argument("--min-hold", type=int, default=3,
                   help="min bars to hold (default 3 = 12h for 4h bars)")
    p.add_argument("--ema-span", type=int, default=3,
                   help="EMA span to smooth predictions")
    args = p.parse_args()

    print(f"Loading {args.symbol} {args.interval} klines...")
    df = load_klines(args.symbol, args.interval)
    print(f"  {len(df):,} bars  |  {df.index.min()} -> {df.index.max()}")

    print(f"\nBuilding features (horizon={args.horizon})...")
    feat = build_features(df, horizon=args.horizon)
    n_features = len([c for c in feat.columns
                      if c not in {"close", "high", "low", "atr_raw",
                                   "future_ret", "target"}])
    print(f"  {len(feat):,} rows  |  {n_features} features")

    print(f"\nWalk-forward training (min {args.min_train_months} months)...\n")
    preds, top_features = walk_forward(feat, min_train_months=args.min_train_months)
    if preds.empty:
        print("No predictions produced — not enough data.")
        return
    print(f"\n  {len(preds):,} predictions  |  {preds.index.min()} -> {preds.index.max()}")

    print("\nRunning bar-by-bar backtest...")
    results = run_backtest(
        preds,
        fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_atr_mult=args.stop_atr, min_hold=args.min_hold,
        ema_span=args.ema_span,
    )
    metrics = compute_metrics(results)
    print_metrics(metrics)

    print(f"\n  Top features (cumulative importance):")
    for i, name in enumerate(top_features[:15], 1):
        print(f"    {i:2d}. {name}")

    # Timeframe breakdown
    tf_totals: dict[str, int] = {}
    for name in top_features[:50]:
        if "_" in name and name.split("_")[0] in ("1d", "1w"):
            tf = name.split("_")[0]
        elif name.startswith(("hour_", "dow_", "sharpe_", "vol_pctile", "trend_")):
            tf = "regime/time"
        else:
            tf = "native"
        tf_totals[tf] = tf_totals.get(tf, 0) + 1
    print(f"\n  Timeframe usage (top 50): {tf_totals}")

    # Save
    out = DATA_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / f"{args.symbol}_{args.interval}_backtest.parquet")

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
        ax.set_title(
            f"{args.symbol} {args.interval} Regression Backtest  |  "
            f"Sharpe {metrics['sharpe']:+.2f}  |  "
            f"Trades {metrics['n_trades']}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        pos = results["position"]
        ax2.fill_between(results.index, pos, 0,
                         where=pos > 0, color="green", alpha=0.3, label="Long")
        ax2.fill_between(results.index, pos, 0,
                         where=pos < 0, color="red", alpha=0.3, label="Short")
        ax2.set_ylabel("Position")
        ax2.set_ylim(-1.5, 1.5)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        chart_path = out / f"{args.symbol}_{args.interval}_equity.png"
        fig.savefig(str(chart_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nEquity chart: {chart_path}")
    except Exception as e:
        print(f"\n(chart skipped: {e})")

    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
