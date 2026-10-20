#!/usr/bin/env python
"""Fetch historical OHLCV klines from Binance vision into the parquet cache."""
from __future__ import annotations

import argparse
from datetime import date

from ai_trader.data.binance import fetch_monthly, load_klines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval", default="1m",
                   choices=["1s", "1m", "3m", "5m", "15m", "30m",
                            "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo"])
    p.add_argument("--start", type=date.fromisoformat, default=date(2017, 8, 1),
                   help="ISO date, e.g. 2017-08-01 (ETHUSDT on Binance starts Aug 2017)")
    p.add_argument("--end", type=date.fromisoformat, default=date.today())
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the month is already cached")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = fetch_monthly(args.symbol, args.interval, args.start, args.end, force=args.force)
    df = load_klines(args.symbol, args.interval)
    print(f"cached files: {len(paths)}")
    print(f"total rows:   {len(df):,}")
    if len(df):
        print(f"range:        {df.index.min()} -> {df.index.max()}")
        print(df.tail(3)[["open", "high", "low", "close", "volume"]])


if __name__ == "__main__":
    main()
