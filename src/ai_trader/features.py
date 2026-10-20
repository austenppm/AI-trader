"""Multi-timeframe feature engineering for OHLCV kline data.

Computes core technical indicators at five resolutions (1m, 5m, 15m, 1h, 4h),
adds market-regime features and cyclical time encoding. All features are strictly
backward-looking — no future information leaks into any computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Core technical indicators for an OHLCV DataFrame."""
    out = pd.DataFrame(index=df.index)
    p = prefix
    c = df["close"]

    # --- Returns ---
    for n in [1, 5, 20]:
        out[f"{p}ret_{n}"] = c.pct_change(n)

    # --- Volatility ---
    ret = c.pct_change()
    for w in [5, 20]:
        out[f"{p}vol_{w}"] = ret.rolling(w).std()

    # --- SMA / EMA ratios ---
    for w in [5, 20]:
        out[f"{p}sma_ratio_{w}"] = c / c.rolling(w).mean()
        out[f"{p}ema_ratio_{w}"] = c / c.ewm(span=w, adjust=False).mean()
    out[f"{p}sma_cross"] = c.rolling(5).mean() / c.rolling(20).mean()

    # --- RSI (14, exponential) ---
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    out[f"{p}rsi_14"] = 100 - 100 / (1 + gain / loss)

    # --- MACD histogram (normalised by price) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out[f"{p}macd_hist"] = (macd - signal) / c

    # --- Bollinger Bands %B and bandwidth ---
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_range = (upper - lower).replace(0, np.nan)
    out[f"{p}bb_pctb"] = (c - lower) / band_range
    out[f"{p}bb_width"] = band_range / sma20

    # --- ATR (normalised by price) ---
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out[f"{p}atr_14"] = tr.rolling(14).mean() / c

    # --- Stochastic %K, %D ---
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    out[f"{p}stoch_k"] = 100 * (c - low14) / denom
    out[f"{p}stoch_d"] = out[f"{p}stoch_k"].rolling(3).mean()

    # --- ADX (14) + directional indicator diff ---
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr14 = tr.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    out[f"{p}adx_14"] = dx.ewm(span=14, adjust=False).mean()
    out[f"{p}di_diff"] = plus_di - minus_di

    # --- OBV z-score ---
    if "volume" in df.columns:
        obv = (np.sign(c.diff()) * df["volume"]).cumsum()
        obv_std = obv.rolling(20).std().replace(0, np.nan)
        out[f"{p}obv_z"] = (obv - obv.rolling(20).mean()) / obv_std
        out[f"{p}vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # --- Buy pressure ---
    if "taker_buy_base" in df.columns and "volume" in df.columns:
        out[f"{p}buy_pressure"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)

    # --- Normalised HL spread ---
    out[f"{p}hl_spread"] = (df["high"] - df["low"]) / c

    return out


def _resample_safe(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV, shifted by one period to prevent lookahead."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    resampled = df[list(agg.keys())].resample(freq).agg(agg)
    if "taker_buy_base" in df.columns:
        resampled["taker_buy_base"] = df["taker_buy_base"].resample(freq).sum()
    return resampled.shift(1).dropna()


def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Market regime detection (trend vs range, high vs low vol)."""
    out = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change()

    for w in [60, 240, 1440]:
        std = ret.rolling(w).std().replace(0, np.nan)
        out[f"sharpe_{w}"] = ret.rolling(w).mean() / std

    # Current vol relative to recent history
    vol60 = ret.rolling(60).std()
    out["vol_pctile"] = vol60.rolling(1440).rank(pct=True)

    # Trend strength = |drift| / noise
    out["trend_strength"] = (
        ret.rolling(60).mean().abs() / ret.rolling(60).std().replace(0, np.nan)
    )

    return out


TIMEFRAMES_1M = [("5min", "5m_"), ("15min", "15m_"), ("1h", "1h_"), ("4h", "4h_")]
TIMEFRAMES_4H = [("1D", "1d_"), ("1W", "1w_")]


def build_features(df: pd.DataFrame, horizon: int = 60,
                   higher_timeframes: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Build multi-timeframe feature matrix + target.

    Args:
        df: OHLCV DataFrame indexed by open_time (UTC).
        horizon: prediction horizon in bars.
        higher_timeframes: list of (pandas_freq, prefix) for multi-timeframe features.
            Defaults to TIMEFRAMES_1M for minute data, TIMEFRAMES_4H for 4h+ data.

    Returns:
        DataFrame with features + close, high, low, atr_raw, future_ret, target.
    """
    if higher_timeframes is None:
        # Auto-detect based on median bar spacing
        median_gap = df.index.to_series().diff().median()
        if median_gap >= pd.Timedelta(hours=1):
            higher_timeframes = TIMEFRAMES_4H
        else:
            higher_timeframes = TIMEFRAMES_1M

    parts: list[pd.DataFrame] = []

    # Native-resolution indicators
    parts.append(_indicators(df, prefix=""))

    # Multi-timeframe indicators
    for freq, label in higher_timeframes:
        resampled = _resample_safe(df, freq)
        tf = _indicators(resampled, prefix=label)
        tf = tf.reindex(df.index, method="ffill")
        parts.append(tf)

    # Regime
    parts.append(_regime_features(df))

    # Time (cyclical)
    time_df = pd.DataFrame(index=df.index)
    hour = df.index.hour + df.index.minute / 60
    time_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    time_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = df.index.dayofweek
    time_df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    time_df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    parts.append(time_df)

    out = pd.concat(parts, axis=1)

    # Target
    out["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1
    out["target"] = (out["future_ret"] > 0).astype("Int64")

    # Carry forward OHLC + raw ATR for the backtest (stop-loss, execution)
    out["close"] = df["close"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr_raw"] = tr.rolling(14).mean()

    return out.dropna()
