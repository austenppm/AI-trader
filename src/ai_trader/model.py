"""Walk-forward model training with LightGBM regression."""
from __future__ import annotations

import pandas as pd
import lightgbm as lgb

LGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    objective="mae",
)

META_COLS = frozenset({
    "close", "high", "low", "atr_raw", "future_ret", "target",
})


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def walk_forward(df: pd.DataFrame,
                 min_train_months: int = 6) -> tuple[pd.DataFrame, list[str]]:
    """Expanding-window walk-forward CV with LightGBM regression.

    Predicts future_ret (raw return) for every bar in each test month.
    Returns (predictions_df, top_features).
    """
    feat_cols = _feature_cols(df)
    df = df.copy()
    df["_ym"] = df.index.to_period("M")
    months = sorted(df["_ym"].unique())

    results: list[pd.DataFrame] = []
    importance_acc: dict[str, float] = {}

    for i in range(min_train_months, len(months)):
        if i < 2:
            continue
        train_mask = df["_ym"].isin(months[:i - 1])
        val_mask = df["_ym"] == months[i - 1]
        test_mask = df["_ym"] == months[i]

        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, "future_ret"]
        X_val = df.loc[val_mask, feat_cols]
        y_val = df.loc[val_mask, "future_ret"]
        test_df = df.loc[test_mask]

        if len(X_train) < 200 or len(X_val) < 30 or len(test_df) < 6:
            continue

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        X_test = test_df[feat_cols]
        predicted = model.predict(X_test)

        chunk = test_df[["close", "high", "low", "atr_raw",
                         "future_ret", "target"]].copy()
        chunk["predicted_return"] = predicted
        results.append(chunk)

        # Feature importance
        for feat, imp in zip(feat_cols, model.feature_importances_):
            importance_acc[feat] = importance_acc.get(feat, 0) + imp

        # Directional accuracy (does the predicted sign match actual sign?)
        dir_acc = ((predicted > 0) == (test_df["future_ret"].values > 0)).mean()
        print(f"  {months[i]}  train={len(X_train):>7,}  "
              f"bars={len(chunk):>4}  dir_acc={dir_acc:.3f}  trees={model.best_iteration_}")

    top = sorted(importance_acc.items(), key=lambda x: -x[1])[:20]
    top_names = [name for name, _ in top]

    preds = pd.concat(results) if results else pd.DataFrame()
    return preds, top_names
