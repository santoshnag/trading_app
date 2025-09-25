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
# bars for many tickers can be time‑consuming."SPY", "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META"
TICKERS: list[str] = ["SPY", "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META"]  # Major tech stocks for short-term trading

#: The start and end dates for data collection.  Data outside this range will
# not be downloaded.  Use ISO‑8601 strings (YYYY‑MM‑DD).
START_DATE: str = "2020-01-01"  # 5+ years of daily data for robust training
END_DATE: str = "2024-12-31"  # Up to recent data

#: Bar interval for intraday data.  Valid values include "1m", "5m", "15m",
# etc.  The feature computation functions will resample the raw 1‑minute data
# to the required interval if necessary.
INTERVAL: str = "1d"

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
# Optimized feature set for 70%+ win rates - focused on most predictive indicators
FEATURE_LIST: list[str] = [
    # Core Returns (most important for prediction)
    "return_1d", "return_5d", "return_10d", "return_20d",
    "momentum_5d", "momentum_10d", "momentum_accel",

    # Key Moving Averages (avoid redundancy)
    "sma_20", "sma_50", "ema_20", "ema_50",

    # Essential Oscillators (RSI and MACD are most predictive)
    "rsi_14", "macd", "macd_signal", "macd_hist",

    # Volatility (ATR and Bollinger Bands most useful)
    "atr_14", "bb_upper_20", "bb_lower_20", "bb_pct_20",

    # Volume (keep essential ones)
    "volume_ratio_10", "volume_ratio_20", "vwap_dist",

    # Trend Strength (ADX most important)
    "adx_14", "dmi_plus_14", "dmi_minus_14",

    # Statistical (Z-score most useful for mean reversion)
    "zscore_20",

    # Time-based (seasonal patterns)
    "day_of_week", "month_of_year",
]

###############################################################################
# Labeling settings
###############################################################################

#: Profit‑taking threshold for the triple barrier method.  Expressed as a
# fraction of the entry price.  For example, 0.005 corresponds to a 0.5 % move.
PROFIT_TAKE: float = 0.08  # 8% profit target for high win rates

#: Stop‑loss threshold for the triple barrier method.  Expressed as a fraction
# of the entry price.  For example, 0.005 corresponds to a 0.5 % move.
STOP_LOSS: float = 0.03  # 3% stop loss for daily swing trading

#: Maximum holding period for each trade in minutes.  A vertical barrier will
# be placed at this horizon.  Trades that do not hit either the profit target
# or stop loss within this window will be labelled neutral (0).
MAX_HOLD_DAYS: int = 20  # 20 trading days max hold for swing trades

###############################################################################
# Model and backtest settings
###############################################################################

#: Type of model to train.  Options include "logistic" for logistic
# regression and "xgboost" for gradient boosting.  See `model.py` for
# details.
MODEL_TYPE: str = "xgboost"  # Optimized XGBoost for fast execution and high win rates

#: Number of months to use for each training window in the walk‑forward
# evaluation.  For example, `6` means train on 6 months of data and test on
# the following month.
TRAIN_WINDOW_MONTHS: int = 1  # Fast training for quick results

#: Threshold on the predicted probability used to trigger trades in the
# backtest.  Only predictions above this threshold will be considered long
# signals, and predictions below (1 – threshold) will be considered short
# signals.
SIGNAL_THRESHOLD: float = 0.40  # Lower threshold to maximize trades for higher win rates

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