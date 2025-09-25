#!/usr/bin/env python3
"""Test the short-term (daily) trading system for 70%+ win rates."""

import sys
import os
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("TESTING SHORT-TERM (DAILY) TRADING SYSTEM FOR 70%+ WIN RATES")
print("=" * 70)

try:
    import config
    from data_pipeline import load_data_for_tickers
    from features import compute_features
    from labeling import generate_labels_for_dataframe
    from model import walk_forward_predict, evaluate_probabilities
    from backtest import backtest_from_predictions

    print("\nConfiguration (Short-Term Trading):")
    print(f"   Data Interval: {config.INTERVAL}")
    print(f"   Date Range: {config.START_DATE} to {config.END_DATE}")
    print(f"   Profit Take: {config.PROFIT_TAKE*100:.1f}%")
    print(f"   Stop Loss: {config.STOP_LOSS*100:.1f}%")
    print(f"   Risk/Reward: {(config.PROFIT_TAKE/config.STOP_LOSS):.1f}:1")
    print(f"   Max Hold: {getattr(config, 'MAX_HOLD_DAYS', 20)} days")
    print(f"   Signal Threshold: {config.SIGNAL_THRESHOLD}")
    print(f"   Features: {len(config.FEATURE_LIST)} indicators")
    print(f"   Model: {config.MODEL_TYPE}")

    print("\nLoading daily data for SPY...")
    data = load_data_for_tickers(['SPY'])
    df = data['SPY']
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    print("\nComputing advanced technical indicators...")
    features = compute_features(df, config.FEATURE_LIST)
    print(f"   Features shape: {features.shape}")
    print(f"   NaN count: {features.isna().sum().sum()}")

    print("\nGenerating triple barrier labels (daily)...")
    labels = generate_labels_for_dataframe(df)
    print(f"   Label distribution: {labels.value_counts().to_dict()}")

    # Quick alignment
    features_shifted = features.shift(1)
    common_index = features_shifted.index.intersection(labels.index)
    X = features_shifted.loc[common_index]
    y = labels.loc[common_index]

    # Clean any infinite values
    for col in X.columns:
        if not hasattr(X[col], 'replace'): continue
        X[col] = X[col].replace([float('inf'), -float('inf')], [1e6, -1e6]).fillna(0)

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    print(f"\nTraining on {len(X)} daily samples...")

    # Use simple train/test split for more predictions
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    # Split data (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

    # Convert to binary for training
    y_train_binary = (y_train == 1).astype(int)
    y_test_binary = (y_test == 1).astype(int)

    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # Train ensemble model
    if config.MODEL_TYPE == "ensemble":
        # Use Logistic Regression for better performance on this dataset
        model = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=1000, random_state=42
        )
    else:
        model = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=1000, random_state=42
        )

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train_binary)

    # Get predictions
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test_scaled)
        preds = pd.Series(proba[:, 1], index=X_test.index)
        true_labels = y_test_binary
    else:
        preds = pd.Series(model.decision_function(X_test_scaled), index=X_test.index)
        true_labels = y_test_binary

    pred_metrics = evaluate_probabilities(preds, true_labels)
    print("\nModel Performance:")
    print(f"   ROC AUC: {pred_metrics.get('roc_auc', 0):.4f}")
    print(f"   Brier Score: {pred_metrics.get('brier', 0):.4f}")

    # Analyze predictions
    high_prob_signals = (preds > config.SIGNAL_THRESHOLD).sum()
    print(f"\nPrediction Analysis:")
    print(f"   Predictions > {config.SIGNAL_THRESHOLD}: {high_prob_signals} out of {len(preds)} ({high_prob_signals/len(preds):.1%})")
    print(f"   Average prediction: {preds.mean():.3f}")
    print(f"   Max prediction: {preds.max():.3f}")

    backtest_res = backtest_from_predictions(df.loc[preds.index], preds)
    stats = backtest_res["stats"]

    win_rate = stats.get('win_rate', 0)
    print("\nBacktest Results (Daily Trading):")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Total Trades: {stats.get('n_trades', 0)}")
    print(f"   Total Return: {stats.get('total_return', 0):.2%}")
    print(f"   Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print(f"   Sharpe Ratio: {stats.get('sharpe_simple', 0):.3f}")

    if win_rate >= 0.70:
        print("\nSUCCESS: Achieved 70%+ win rate!")
        print("   Short-term trading system is highly profitable!")
    elif win_rate >= 0.65:
        print("\nEXCELLENT: 65%+ win rate!")
        print("   Very close to target, minor optimizations needed!")
    elif win_rate >= 0.60:
        print("\nGOOD: 60%+ win rate")
        print("   Solid performance, can be improved!")
    elif win_rate >= 0.55:
        print("\nDECENT: 55%+ win rate")
        print("   Reasonable performance, needs tuning!")
    else:
        print("\nNEEDS IMPROVEMENT: Win rate below 55%")
        print("   Additional feature engineering required!")

    print(f"\n📊 SUMMARY:")
    print(f"   Daily trading with {len(config.FEATURE_LIST)} technical indicators")
    print(f"   {config.PROFIT_TAKE*100:.1f}% profit target, {config.STOP_LOSS*100:.1f}% stop loss")
    print(f"   {getattr(config, 'MAX_HOLD_DAYS', 20)}-day max holding period")
    print(f"   Final Win Rate: {win_rate:.1%}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
