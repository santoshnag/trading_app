"""Entry point script to run the intraday trading pipeline.

This script performs the end‑to‑end workflow: data loading, feature
computation, labeling, model training with walk‑forward validation, and
backtesting based on the model's probability forecasts.  Results for
each ticker are summarised and printed to the console.

To use, adjust the configuration in ``config.py`` to select the tickers,
date range, model type and strategy parameters.  Then run:

```bash
python run.py
```

The script will download minute‑bar data if not already cached, compute
features using the library in ``features.py``, generate labels via the
triple barrier method, train a model using walk‑forward validation and
evaluate predictive and trading performance.  The final section prints
a summary of predictive metrics (ROC AUC and Brier score) along with
backtest statistics such as total return, win rate and simple Sharpe
ratio for each ticker.
"""

from __future__ import annotations

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import modules with fallback
try:
    import config
    from data_pipeline import load_data_for_tickers
    from features import compute_features
    from labeling import generate_labels_for_dataframe
    from model import walk_forward_predict, evaluate_probabilities
    from backtest import backtest_from_predictions
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def run_pipeline() -> None:
    import sys
    print(f"Starting pipeline with tickers: {config.TICKERS}", flush=True)
    sys.stdout.flush()
    tickers = config.TICKERS
    # Load data for all tickers
    print("Loading data...", flush=True)
    sys.stdout.flush()
    data = load_data_for_tickers(tickers)
    results: list[dict[str, object]] = []
    for ticker in tickers:
        df = data[ticker]
        print(f"\nProcessing {ticker}...")
        # Compute features and labels
        print(f"Computing features for {ticker}...")
        features = compute_features(df)
        print(f"Features shape: {features.shape}, NaN count: {features.isna().sum().sum()}")
        labels = generate_labels_for_dataframe(df)
        print(f"Labels shape: {labels.shape}, NaN count: {labels.isna().sum().sum()}")
        # Align features and labels; shift features by 1 bar to avoid leakage
        features_shifted = features.shift(1)
        common_index = features_shifted.index.intersection(labels.index)
        X = features_shifted.loc[common_index]
        y = labels.loc[common_index]

        # Clean infinite values introduced by shifting
        for col in X.columns:
            if not np.isfinite(X[col]).all():
                # Replace inf/-inf with reasonable bounds based on column type
                if col in ['return_5m', 'return_15m', 'return_60m', 'momentum_accel']:
                    X[col] = X[col].replace([np.inf, -np.inf], [2.0, -2.0]).fillna(0).clip(-2.0, 2.0)
                elif col in ['rsi_14', 'stochastic_k', 'cci_14', 'bollinger_pct']:
                    X[col] = X[col].replace([np.inf, -np.inf], [100, -100]).fillna(50).clip(-100, 100)
                elif col in ['macd', 'atr_14', 'vwap_dist', 'hurst_120m', 'realized_volatility_30m']:
                    X[col] = X[col].replace([np.inf, -np.inf], [10, -10]).fillna(0).clip(-10, 10)
                elif col in ['rel_volume_5m', 'volume_ma_ratio_5m', 'volume_ma_ratio_15m', 'volume_momentum']:
                    X[col] = X[col].replace([np.inf, -np.inf], [10, 0.1]).fillna(1.0).clip(0.1, 10)
                elif col in ['price_volume_trend', 'accumulation_distribution']:
                    X[col] = X[col].replace([np.inf, -np.inf], [1000, -1000]).fillna(0).clip(-1000, 1000)
                elif col in ['sin_time', 'cos_time']:
                    X[col] = X[col].replace([np.inf, -np.inf], [1.0, -1.0]).fillna(0).clip(-1.1, 1.1)
                else:
                    # Default cleaning
                    X[col] = X[col].replace([np.inf, -np.inf], [1e6, -1e6]).fillna(0).clip(-1e6, 1e6)

        print(f"After alignment - X: {X.shape}, y: {y.shape}")
        # Drop NaN labels
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        timestamps = X.index
        # Walk‑forward training and prediction
        preds, true_labels = walk_forward_predict(X, y, timestamps)
        pred_metrics = evaluate_probabilities(preds, true_labels)
        print(f"Predictive metrics for {ticker}: {pred_metrics}")
        # Backtest
        backtest_res = backtest_from_predictions(df.loc[preds.index], preds)
        stats = backtest_res["stats"]
        print(f"Backtest stats for {ticker}: {stats}")
        results.append({
            "ticker": ticker,
            "pred_metrics": pred_metrics,
            "backtest": stats,
        })
    # Summarise across tickers
    print("\nSummary across tickers:", flush=True)
    sys.stdout.flush()
    df_results = pd.DataFrame([
        {
            "Ticker": r["ticker"],
            "ROC AUC": r["pred_metrics"].get("roc_auc"),
            "Brier": r["pred_metrics"].get("brier"),
            "Total Return": r["backtest"].get("total_return"),
            "Sharpe (simple)": r["backtest"].get("sharpe_simple"),
            "Win Rate": r["backtest"].get("win_rate"),
        }
        for r in results
    ])
    print(df_results.to_string(index=False, float_format="{:.3f}".format), flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    run_pipeline()