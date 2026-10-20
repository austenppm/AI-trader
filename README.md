# AI-Trader

Machine-learning crypto trading research in Python. It downloads OHLCV data from Binance, engineers technical and regime features, trains a LightGBM model with walk-forward validation (no look-ahead), and backtests dynamic positions with fees, slippage and ATR stops.

The longer-term plan, a genetic population of models plus news and on-chain data, is in [VISION.md](VISION.md).

## Usage

```bash
pip install -e ".[dev]"
python scripts/fetch_ohlcv.py            # cache Binance klines locally
python scripts/train_and_backtest.py --symbol ETHUSDT --interval 4h
pytest
```

Stack: pandas, pyarrow, LightGBM, scikit-learn, httpx, matplotlib.
