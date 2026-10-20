"""Fetch historical OHLCV klines from Binance's public data dump (data.binance.vision).

No API key needed; the dump is free and rate-limit-free.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from typing import Iterator

import httpx
import pandas as pd
from tqdm import tqdm

from ..paths import RAW

VISION_MONTHLY = "https://data.binance.vision/data/spot/monthly/klines"

# Binance kline CSV schema (no header in historical files; some 2025+ files add one).
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
NUMERIC = ["open", "high", "low", "close", "volume",
           "quote_volume", "taker_buy_base", "taker_buy_quote"]


def _url(symbol: str, interval: str, year: int, month: int) -> str:
    return f"{VISION_MONTHLY}/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"


def _iter_months(start: date, end: date) -> Iterator[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            m, y = 1, y + 1


def _cache_path(symbol: str, interval: str, year: int, month: int) -> Path:
    return RAW / "binance" / symbol / interval / f"{year:04d}-{month:02d}.parquet"


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    first = raw.split(b"\n", 1)[0].decode(errors="ignore").lower()
    has_header = "open_time" in first or "open time" in first
    if has_header:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    else:
        df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS)
    return df


def _infer_time_unit(series: pd.Series) -> str:
    """Binance vision uses ms pre-2025 and microseconds from 2025-01 onward."""
    sample = pd.to_numeric(series, errors="coerce").dropna()
    if sample.empty:
        return "ms"
    # ms in 2020-2030 ~ 1.6e12-1.9e12; us in same range ~ 1.6e15-1.9e15.
    return "us" if sample.iloc[0] >= 1e14 else "ms"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in ("ignore",) if c in df.columns])
    unit = _infer_time_unit(df["open_time"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit=unit, utc=True)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "trades" in df.columns:
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    return df


def _download_month(client: httpx.Client, symbol: str, interval: str,
                    year: int, month: int) -> pd.DataFrame | None:
    r = client.get(_url(symbol, interval, year, month))
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    return _normalize(_read_csv_bytes(raw))


def fetch_monthly(symbol: str, interval: str, start: date, end: date,
                  force: bool = False) -> list[Path]:
    """Download monthly klines into a parquet cache. Returns written/cached paths."""
    written: list[Path] = []
    months = list(_iter_months(start, end))
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for y, m in tqdm(months, desc=f"{symbol} {interval}"):
            out = _cache_path(symbol, interval, y, m)
            if out.exists() and not force:
                written.append(out)
                continue
            df = _download_month(client, symbol, interval, y, m)
            if df is None or df.empty:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, compression="zstd")
            written.append(out)
    return written


def load_klines(symbol: str, interval: str,
                start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Load cached klines as one sorted DataFrame indexed by open_time (UTC)."""
    root = RAW / "binance" / symbol / interval
    files = sorted(root.glob("*.parquet")) if root.exists() else []
    if not files:
        return pd.DataFrame(columns=[c for c in COLUMNS if c != "ignore"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    if start is not None:
        df = df[df["open_time"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["open_time"] < pd.Timestamp(end, tz="UTC")]
    return df.set_index("open_time")
