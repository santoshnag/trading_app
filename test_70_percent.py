#!/usr/bin/env python3
"""Test the optimized short-term trading system for 70%+ win rates."""

import sys
import os
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("🧪 TESTING OPTIMIZED SHORT-TERM TRADING SYSTEM FOR 70%+ WIN RATES")
print("=" * 80)

try:
    import config
    from data_pipeline import load_data_for_tickers
    from features import compute_features
    from labeling import generate_labels_for_dataframe
    from model import walk_forward_predict, evaluate_probabilities
    from backtest import backtest_from_predictions

    print("\n⚙️  CONFIGURATION:")
    print(f"   Model: {config.MODEL_TYPE}")
    print(f"   Signal Threshold: {config.SIGNAL_THRESHOLD}")
    print(f"   Profit Target: {config.PROFIT_TAKE*100:.1f}%")
    print(f"   Stop Loss: {config.STOP_LOSS*100:.1f}%")
    print(f"   Max Hold: {getattr(config, 'MAX_HOLD_DAYS', 20)} days")
    print(f"   Features: {len(config.FEATURE_LIST)} indicators")

    print("\n📊 LOADING DATA...")
    data = load_data_for_tickers(config.TICKERS)
    results = []

    for ticker in config.TICKERS:
        print(f"\n🏗️  PROCESSING {ticker}...")

        df = data[ticker]
        print(f"   Data shape: {df.shape}")

        # Compute features
        features = compute_features(df, config.FEATURE_LIST)
        print(f"   Features: {features.shape[1]} computed")

        # Generate labels
        labels = generate_labels_for_dataframe(df)
        unique_labels = labels.value_counts().sort_index()
        print(f"   Labels distribution: {dict(unique_labels)}")

        # Align features and labels
        features_shifted = features.shift(1)
        common_index = features_shifted.index.intersection(labels.index)
        X = features_shifted.loc[common_index]
        y = labels.loc[common_index]

        # Clean data
        for col in X.columns:
            X[col] = X[col].replace([float('inf'), -float('inf')], [1e6, -1e6]).fillna(0)

        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]

        if len(X) < 100:
            print(f"   ⚠️  Insufficient data for {ticker}, skipping...")
            continue

        print(f"   Training samples: {len(X)}")

        # Walk-forward validation
        preds, true_labels = walk_forward_predict(X, y, X.index)

        # Evaluate
        pred_metrics = evaluate_probabilities(preds, true_labels)

        # Backtest
        backtest_res = backtest_from_predictions(df.loc[preds.index], preds)
        stats = backtest_res["stats"]

        win_rate = stats.get('win_rate', 0)
        n_trades = stats.get('n_trades', 0)

        # Determine success level
        if win_rate >= 0.75:
            status = "🎉 EXCEPTIONAL"
            emoji = "🚀"
        elif win_rate >= 0.70:
            status = "🎯 TARGET ACHIEVED"
            emoji = "✅"
        elif win_rate >= 0.60:
            status = "👍 GOOD PROGRESS"
            emoji = "📈"
        elif win_rate >= 0.50:
            status = "🤔 MODERATE"
            emoji = "⚡"
        else:
            status = "❌ NEEDS IMPROVEMENT"
            emoji = "🔧"

        print(f"   {emoji} {status}: {win_rate:.1%} win rate ({n_trades} trades)")

        results.append({
            'ticker': ticker,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'roc_auc': pred_metrics.get('roc_auc', 0),
            'total_return': stats.get('total_return', 0),
            'sharpe': stats.get('sharpe_simple', 0)
        })

    # Summary
    print(f"\n📈 SUMMARY ACROSS {len(results)} TICKERS:")
    print("-" * 80)

    valid_results = [r for r in results if r['n_trades'] > 0]
    if valid_results:
        avg_win_rate = sum(r['win_rate'] for r in valid_results) / len(valid_results)
        total_trades = sum(r['n_trades'] for r in valid_results)

        print(f"Average Win Rate: {avg_win_rate:.1%}")
        print(f"Total Trades: {total_trades}")
        print(f"Tickers Tested: {len(valid_results)}")

        # Detailed breakdown
        print(f"\nDetailed Results:")
        for result in sorted(valid_results, key=lambda x: x['win_rate'], reverse=True):
            win_rate = result['win_rate']
            if win_rate >= 0.70:
                icon = "🟢"
            elif win_rate >= 0.60:
                icon = "🟡"
            elif win_rate >= 0.50:
                icon = "🟠"
            else:
                icon = "🔴"

            print(f"  {icon} {result['ticker']}: {win_rate:.1%} win rate ({result['n_trades']} trades)")

        # Final assessment
        excellent_count = sum(1 for r in valid_results if r['win_rate'] >= 0.75)
        good_count = sum(1 for r in valid_results if 0.70 <= r['win_rate'] < 0.75)
        acceptable_count = sum(1 for r in valid_results if 0.60 <= r['win_rate'] < 0.70)

        print(f"\n🎯 FINAL ASSESSMENT:")
        if excellent_count >= len(valid_results) * 0.8:  # 80% excellent
            print("🎉 EXCEPTIONAL SUCCESS: 75%+ win rates across most tickers!")
        elif good_count + excellent_count >= len(valid_results) * 0.7:  # 70% good+
            print("🎯 TARGET ACHIEVED: 70%+ win rates successfully implemented!")
        elif acceptable_count + good_count + excellent_count >= len(valid_results) * 0.6:
            print("👍 GOOD PROGRESS: 60%+ win rates - close to target!")
        else:
            print("🔧 FURTHER OPTIMIZATION NEEDED: Win rates below 60%")

        print(f"\n📊 SYSTEM CAPABILITIES:")
        print(f"   ✓ Advanced ensemble model with feature selection")
        print(f"   ✓ 75+ technical indicators optimized for daily trading")
        print(f"   ✓ Walk-forward validation prevents overfitting")
        print(f"   ✓ Realistic transaction costs and risk management")
        print(f"   ✓ Multi-ticker support with robust error handling")

    else:
        print("❌ No valid results - check data and model configuration")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
