#!/usr/bin/env python3
"""Quick test to check if the optimized system achieves high win rates."""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("TESTING OPTIMIZED SYSTEM FOR 70%+ WIN RATES")
print("=" * 60)

try:
    import config
    from data_pipeline import load_data_for_tickers
    from features import compute_features
    from labeling import generate_labels_for_dataframe
    from model import walk_forward_predict, evaluate_probabilities
    from backtest import backtest_from_predictions

    print("\nConfiguration (Optimized for 70%+ Win Rates):")
    print(f"   Profit Take: {config.PROFIT_TAKE*100:.1f}%")
    print(f"   Stop Loss: {config.STOP_LOSS*100:.1f}%")
    print(f"   Risk/Reward: {(config.PROFIT_TAKE/config.STOP_LOSS):.1f}:1")
    print(f"   Max Hold: {config.MAX_HOLD_MINUTES} min")
    print(f"   Signal Threshold: {config.SIGNAL_THRESHOLD}")
    print(f"   Model: {config.MODEL_TYPE}")

    print("\nLoading SPY data...")
    data = load_data_for_tickers(['SPY'])
    df = data['SPY']
    print(f"   Shape: {df.shape}")

    print("\nComputing features...")
    features = compute_features(df, config.FEATURE_LIST)
    print(f"   Features shape: {features.shape}")

    print("\nGenerating labels...")
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

    print(f"\nTraining on {len(X)} samples...")

    # Use simple train/test split for more predictions and trades
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Convert to binary for training
    y_train_binary = (y_train == 1).astype(int)
    y_test_binary = (y_test == 1).astype(int)

    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if config.MODEL_TYPE == "xgboost":
        model = XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, gamma=0.2,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=2.0,
            scale_pos_weight=3.0, random_state=42
        )
    elif config.MODEL_TYPE == "logistic":
        model = LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=500, random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")
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

    # Analyze predictions
    high_prob_signals = (preds > config.SIGNAL_THRESHOLD).sum()
    print(f"\nPrediction Analysis:")
    print(f"   Predictions > {config.SIGNAL_THRESHOLD}: {high_prob_signals} out of {len(preds)} ({high_prob_signals/len(preds):.1%})")
    print(f"   Average prediction: {preds.mean():.3f}")
    print(f"   Max prediction: {preds.max():.3f}")

    backtest_res = backtest_from_predictions(df.loc[preds.index], preds)
    stats = backtest_res["stats"]

    win_rate = stats.get('win_rate', 0)
    print("\nBacktest Results:")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Total Trades: {stats.get('n_trades', 0)}")
    print(f"   Total Return: {stats.get('total_return', 0):.2%}")

    if win_rate >= 0.70:
        print("\nSUCCESS: Achieved 70%+ win rate!")
    elif win_rate >= 0.60:
        print("\nGOOD: 60%+ win rate - close to target")
    elif win_rate >= 0.50:
        print("\nMODERATE: 50%+ win rate - needs more tuning")
    else:
        print("\nLOW: Win rate below 50% - significant tuning needed")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
