#!/usr/bin/env python3
"""Simple wrapper to run the trading pipeline."""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("🚀 Starting optimized trading pipeline for 70%+ win rates")
print("=" * 60)

try:
    # Import and run
    print("Importing modules...", flush=True)
    from run import run_pipeline
    print("Running pipeline...", flush=True)
    run_pipeline()
    print("Pipeline completed!", flush=True)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
