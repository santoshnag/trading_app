"""Global configuration variables for the intraday trading system.

This module centralises all constants and hyper‑parameters used by the
pipeline.  Adjust the values here to customise the behaviour of the
application.  In a production environment these could be loaded from a
YAML/JSON file or command line flags instead.
"""

from datetime import datetime

###############################################################################
# Data settings
###############################################################################

#: Tickers to download and analyse.  These should be liquid US equities or
# ETFs.  You can add more symbols to this list.  Note that downloading minute
# bars for many tickers can be time‑consuming.
TICKERS: list[str] = ["SPY"]  # Test with single ticker first

#: The start and end dates for data collection.  Data outside this range will
# not be downloaded.  Use ISO‑8601 strings (YYYY‑MM‑DD).
START_DATE: str = "2025-07-01"  # Extended date range for more data
END_DATE: str = "2025-09-15"  # Extended date range for more data

#: Bar interval for intraday data.  Valid values include "1m", "5m", "15m",
# etc.  The feature computation functions will resample the raw 1‑minute data
# to the required interval if necessary.
INTERVAL: str = "1m"

#: Alpha Vantage API key.  If provided, the data pipeline will attempt to
# download extended minute history using Alpha Vantage.  Otherwise it will
# fall back to Yahoo Finance via the `yfinance` package.  Leave as an empty
# string to disable Alpha Vantage downloads.
ALPHAVANTAGE_API_KEY: str = "9JY6J0VGF7KSU8OV"

#: Limit the number of monthly slices to download from Alpha Vantage for
# quick runs. Increase (up to 24) for more history.
ALPHAVANTAGE_MAX_SLICES: int = 2

#: Directory where raw CSV files will be stored.  Each ticker will be saved
# into its own CSV file under this folder.
DATA_DIR: str = "data/raw"

###############################################################################
# Feature engineering settings
###############################################################################

#: A list of feature names to compute.  See `features.py` for a list of
# available features.  Removing features from this list will speed up
# computation and may reduce overfitting.
# Optimized feature set - reduced from 37 to 20 high-signal features
FEATURE_LIST: list[str] = [
    # Core momentum and trend (most predictive)
    "return_5m",
    "return_15m",
    "momentum_accel",
    "ema_10",
    "ema_30",
    "rsi_14",
    "macd",
    # Volatility (essential for risk management)
    "atr_14",
    "realized_volatility_30m",
    # Volume indicators (market participation)
    "vwap_dist",
    "rel_volume_5m",
    "volume_ma_ratio_5m",
    "volume_momentum",
    # Order flow (market microstructure)
    "accumulation_distribution",
    "price_volume_trend",
    # Momentum oscillators
    "stochastic_k",
    "cci_14",
    # Mean reversion (key strategy component)
    "bollinger_pct",  # Position within bands - most useful
    # Regime detection
    "hurst_120m",
    # Time-based
    "sin_time",
    "cos_time",
]

###############################################################################
# Labeling settings
###############################################################################

#: Profit‑taking threshold for the triple barrier method.  Expressed as a
# fraction of the entry price.  For example, 0.005 corresponds to a 0.5 % move.
PROFIT_TAKE: float = 0.008  # 0.8% profit target (achievable for high win rates)

#: Stop‑loss threshold for the triple barrier method.  Expressed as a fraction
# of the entry price.  For example, 0.005 corresponds to a 0.5 % move.
STOP_LOSS: float = 0.004  # 0.4% stop loss (tighter for higher win rates)

#: Maximum holding period for each trade in minutes.  A vertical barrier will
# be placed at this horizon.  Trades that do not hit either the profit target
# or stop loss within this window will be labelled neutral (0).
MAX_HOLD_MINUTES: int = 60  # Extended hold time (increased from 20 minutes)

###############################################################################
# Model and backtest settings
###############################################################################

#: Type of model to train.  Options include "logistic" for logistic
# regression and "xgboost" for gradient boosting.  See `model.py` for
# details.
MODEL_TYPE: str = "logistic"  # Try Logistic Regression for simpler model

#: Number of months to use for each training window in the walk‑forward
# evaluation.  For example, `6` means train on 6 months of data and test on
# the following month.
TRAIN_WINDOW_MONTHS: int = 6  # Balanced training window

#: Threshold on the predicted probability used to trigger trades in the
# backtest.  Only predictions above this threshold will be considered long
# signals, and predictions below (1 – threshold) will be considered short
# signals.
SIGNAL_THRESHOLD: float = 0.30  # Very low threshold to get many trades for testing

#: Maximum number of concurrent positions allowed in the portfolio backtest.
MAX_POSITIONS: int = 5

#: Fraction of account equity to risk per trade.  If set to 0.01, each trade
# will risk 1 % of the portfolio on the stop loss.
RISK_PER_TRADE: float = 0.002  # Conservative risk per trade

###############################################################################
# Evaluation settings
###############################################################################

#: Annualised risk‑free rate used for Sharpe ratio calculation.  Use a small
# value to approximate T‑bill yields.
RISK_FREE_RATE: float = 0.02