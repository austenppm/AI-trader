"""Trend-following strategy with ML risk overlay.

Core idea:
  - Direction comes from simple, robust momentum/trend rules (no ML).
  - ML model predicts REGIME (trending vs choppy) and scales exposure.
  - Trend rules capture crypto's strong momentum effect.
  - ML overlay avoids drawdowns by reducing position in choppy markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


# ---------------------------------------------------------------------------
# 1. Trend signals (rules-based, no ML)
# ---------------------------------------------------------------------------

def trend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend-following signals from OHLCV data.

    Returns a DataFrame with signal columns (all between -1 and +1).
    """
    c = df["close"]
    out = pd.DataFrame(index=df.index)

    # Dual moving average crossover (fast/slow)
    for fast, slow in [(5, 20), (10, 40), (20, 60)]:
        sma_f = c.rolling(fast).mean()
        sma_s = c.rolling(slow).mean()
        # Normalized distance between MAs (positive = uptrend)
        out[f"ma_{fast}_{slow}"] = (sma_f - sma_s) / c

    # Price momentum (rate of change)
    for lookback in [6, 12, 24]:
        out[f"mom_{lookback}"] = c.pct_change(lookback)

    # Breakout: close relative to rolling high/low channel
    for window in [20, 50]:
        high_w = df["high"].rolling(window).max()
        low_w = df["low"].rolling(window).min()
        channel = (high_w - low_w).replace(0, np.nan)
        out[f"channel_{window}"] = (c - low_w) / channel * 2 - 1  # -1 to +1

    # Composite signal: average of all normalized signals
    signal_cols = [col for col in out.columns]
    out["trend_signal"] = out[signal_cols].mean(axis=1)

    return out


# ---------------------------------------------------------------------------
# 2. ML regime detector
# ---------------------------------------------------------------------------

REGIME_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_samples=20,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    objective="binary",
)


def _regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for predicting whether trend-following will work."""
    c = df["close"]
    ret = c.pct_change()
    out = pd.DataFrame(index=df.index)

    # Volatility at multiple scales
    for w in [6, 12, 24, 48]:
        out[f"vol_{w}"] = ret.rolling(w).std()

    # Volatility ratio (short/long) — rising vol = regime change
    out["vol_ratio"] = ret.rolling(6).std() / ret.rolling(48).std().replace(0, np.nan)

    # Autocorrelation proxy — trending markets have positive autocorr
    for lag in [1, 3, 6]:
        ret_lag = ret.shift(lag)
        cov = (ret * ret_lag).rolling(24).mean() - ret.rolling(24).mean() * ret_lag.rolling(24).mean()
        var = ret.rolling(24).var().replace(0, np.nan)
        out[f"autocorr_{lag}"] = cov / var

    # ADX (trend strength)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    atr14 = tr.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    out["adx"] = dx.ewm(span=14, adjust=False).mean()

    # Variance ratio (trending > 1, mean-reverting < 1, random walk = 1)
    var_base = ret.rolling(48).var().replace(0, np.nan)
    for w in [12, 24]:
        ret_w = c.pct_change(w)
        var_w = ret_w.rolling(48).var()
        out[f"var_ratio_{w}"] = (var_w / (w * var_base)).replace([np.inf, -np.inf], np.nan)

    # Volume regime
    if "volume" in df.columns:
        out["vol_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Recent trend-following performance (meta-feature)
    # Did the MA crossover work over the last N bars?
    sma_fast = c.rolling(10).mean()
    sma_slow = c.rolling(40).mean()
    trend_pos = (sma_fast > sma_slow).astype(float) * 2 - 1
    tf_return = (trend_pos.shift(1) * ret).rolling(24).sum()
    out["tf_recent_pnl"] = tf_return

    return out


def _create_regime_target(df: pd.DataFrame, trend_signal: pd.Series,
                          horizon: int = 6) -> pd.Series:
    """Target: did trend-following WORK over the next `horizon` bars?

    1 = trend signal and actual return had the same sign (trend worked).
    0 = they had opposite signs (trend failed / choppy market).
    """
    future_ret = df["close"].shift(-horizon) / df["close"] - 1
    signal_sign = np.sign(trend_signal)
    return_sign = np.sign(future_ret)
    worked = (signal_sign == return_sign).astype("Int64")
    return worked


def train_regime_model(df: pd.DataFrame, trend_signal: pd.Series,
                       min_train_months: int = 6,
                       horizon: int = 6) -> pd.DataFrame:
    """Walk-forward training for the regime classifier.

    Predicts: probability that the trend signal will be correct over the next horizon bars.
    Returns DataFrame with 'regime_proba' column (high = trending, low = choppy).
    """
    features = _regime_features(df)
    target = _create_regime_target(df, trend_signal, horizon)

    combined = features.copy()
    combined["_target"] = target
    combined["_trend_signal"] = trend_signal
    combined = combined.dropna()

    feat_cols = [c for c in features.columns if c in combined.columns]
    combined["_ym"] = combined.index.to_period("M")
    months = sorted(combined["_ym"].unique())

    results: list[pd.DataFrame] = []

    for i in range(min_train_months, len(months)):
        if i < 2:
            continue
        train_mask = combined["_ym"].isin(months[:i - 1])
        val_mask = combined["_ym"] == months[i - 1]
        test_mask = combined["_ym"] == months[i]

        X_train = combined.loc[train_mask, feat_cols]
        y_train = combined.loc[train_mask, "_target"]
        X_val = combined.loc[val_mask, feat_cols]
        y_val = combined.loc[val_mask, "_target"]
        test_df = combined.loc[test_mask]

        if len(X_train) < 200 or len(X_val) < 30 or len(test_df) < 6:
            continue

        model = lgb.LGBMClassifier(**REGIME_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        X_test = test_df[feat_cols]
        proba = model.predict_proba(X_test)[:, 1]

        chunk = pd.DataFrame({
            "regime_proba": proba,
            "trend_signal": test_df["_trend_signal"].values,
        }, index=test_df.index)
        results.append(chunk)

        acc = ((proba > 0.5) == test_df["_target"].values).mean()
        print(f"  {months[i]}  regime_acc={acc:.3f}  trees={model.best_iteration_}")

    return pd.concat(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Combined strategy
# ---------------------------------------------------------------------------

def compute_positions(signals: pd.DataFrame,
                      regime_threshold: float = 0.45,
                      trend_threshold: float = 0.001) -> pd.Series:
    """Combine trend signal + regime overlay into a final position.

    Logic:
      - If regime_proba < regime_threshold → flat (choppy market, sit out)
      - If trend_signal > +trend_threshold → long
      - If trend_signal < -trend_threshold → short
      - Otherwise → flat
    """
    pos = pd.Series(0, index=signals.index, dtype=float)

    trending = signals["regime_proba"] >= regime_threshold
    long_sig = signals["trend_signal"] > trend_threshold
    short_sig = signals["trend_signal"] < -trend_threshold

    pos[trending & long_sig] = 1
    pos[trending & short_sig] = -1

    return pos
