#!/usr/bin/env python3
"""Quick test for short-term trading system."""

import sys
import os
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("QUICK SHORT-TERM TRADING TEST")
print("=" * 40)

try:
    import config
    from data_pipeline import load_data_for_tickers
    from features import compute_features
    from labeling import generate_labels_for_dataframe

    print("Loading SPY data...")
    data = load_data_for_tickers(['SPY'])
    df = data['SPY']
    print(f"Data shape: {df.shape}")

    print("Computing features...")
    # Use just a few key features for quick testing
    simple_features = [
        "return_5d", "return_10d", "return_20d",
        "sma_20", "ema_20",
        "rsi_14", "macd",
        "bb_upper_20", "bb_lower_20"
    ]
    features = compute_features(df, simple_features)
    print(f"Features shape: {features.shape}")

    print("Generating labels...")
    labels = generate_labels_for_dataframe(df)
    print(f"Label distribution: {labels.value_counts().to_dict()}")

    print("\n✅ SUCCESS: Short-term trading system components working!")
    print(f"   - {len(simple_features)} technical indicators computed")
    print(f"   - {len(labels)} labeled trading opportunities")
    print(f"   - Ready for 70%+ win rate optimization")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
