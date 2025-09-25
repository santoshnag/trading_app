#!/usr/bin/env python3

import sys
import os
import traceback

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("Starting debug test...")

try:
    print("1. Importing config...")
    import config
    print(f"   Tickers: {config.TICKERS}")
    print(f"   Start date: {config.START_DATE}")
    print(f"   Model type: {config.MODEL_TYPE}")

    print("2. Importing data_pipeline...")
    from data_pipeline import load_data_for_tickers

    print("3. Testing data loading...")
    tickers = ["SPY"]  # Just test one ticker
    data = load_data_for_tickers(tickers)
    print(f"   Data loaded successfully for {len(data)} tickers")

    if "SPY" in data:
        df = data["SPY"]
        print(f"   SPY data shape: {df.shape}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")

        print("4. Testing feature computation...")
        from features import compute_features
        features = compute_features(df)
        print(f"   Features shape: {features.shape}")
        print(f"   NaN count: {features.isna().sum().sum()}")

        print("5. Testing labeling...")
        from labeling import generate_labels_for_dataframe
        labels = generate_labels_for_dataframe(df)
        print(f"   Labels shape: {labels.shape}")
        print(f"   Label distribution: {labels.value_counts()}")

        print("6. Testing model...")
        from model import walk_forward_predict, evaluate_probabilities
        from backtest import backtest_from_predictions

        # Align features and labels
        features_shifted = features.shift(1)
        common_index = features_shifted.index.intersection(labels.index)
        X = features_shifted.loc[common_index]
        y = labels.loc[common_index]

        # Drop NaN labels
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]

        print(f"   Training data: X={X.shape}, y={y.shape}")

        preds, true_labels = walk_forward_predict(X, y, X.index)
        print(f"   Predictions shape: {preds.shape}")

        pred_metrics = evaluate_probabilities(preds, true_labels)
        print(f"   Predictive metrics: {pred_metrics}")

        backtest_res = backtest_from_predictions(df.loc[preds.index], preds)
        stats = backtest_res["stats"]
        print(f"   Backtest stats: {stats}")

    print("All tests passed!")

except Exception as e:
    print(f"Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
