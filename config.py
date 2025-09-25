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
FEATURE_LIST: list[str] = [
    # Returns and Momentum (daily basis)
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "momentum_1d", "momentum_5d", "momentum_10d", "momentum_accel",

    # Moving Averages (multiple timeframes)
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_5", "ema_10", "ema_20", "ema_50", "ema_200",
    "wma_10", "wma_20",

    # Oscillators and Momentum Indicators
    "rsi_6", "rsi_14", "rsi_21",
    "stoch_k", "stoch_d", "stoch_rsi",
    "williams_r", "cci_14", "cci_20",
    "macd", "macd_signal", "macd_hist",
    "mfi_14",

    # Volatility Indicators
    "atr_14", "atr_20",
    "bb_upper_20", "bb_middle_20", "bb_lower_20", "bb_pct_20", "bb_width_20",
    "bb_upper_10", "bb_middle_10", "bb_lower_10", "bb_pct_10", "bb_width_10",
    "keltner_upper_20", "keltner_middle_20", "keltner_lower_20", "keltner_pct_20",

    # Volume Indicators
    "volume_sma_5", "volume_sma_10", "volume_sma_20",
    "volume_ratio_5", "volume_ratio_10", "volume_ratio_20",
    "vwap", "vwap_dist",
    "accumulation_distribution", "chaikin_money_flow",
    "force_index", "ease_of_movement", "volume_price_trend",

    # Trend and Support/Resistance
    "adx_14", "dmi_plus_14", "dmi_minus_14",
    "aroon_up_14", "aroon_down_14", "aroon_osc_14",
    "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
    "pivot_point", "resistance_1", "support_1",

    # Statistical Indicators
    "hurst_100", "hurst_200",
    "zscore_20", "skew_20", "kurtosis_20",

    # Market Breadth (using SPY as proxy)
    "market_trend_5", "market_trend_20", "market_strength",

    # Time-based features
    "day_of_week", "month_of_year", "quarter",
    "days_since_high_20", "days_since_low_20",
]

###############################################################################
# Labeling settings
###############################################################################

#: Profit‑taking threshold for the triple barrier method.  Expressed as a
# fraction of the entry price.  For example, 0.005 corresponds to a 0.5 % move.
PROFIT_TAKE: float = 0.05  # 5% profit target for daily swing trading

#: Stop‑loss threshold for the triple barrier method.  Expressed as a fraction
# of the entry price.  For example, 0.005 corresponds to a 0.5 % move.
STOP_LOSS: float = 0.025  # 2.5% stop loss for daily swing trading

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
MODEL_TYPE: str = "ensemble"  # Advanced ensemble for maximum win rates

#: Number of months to use for each training window in the walk‑forward
# evaluation.  For example, `6` means train on 6 months of data and test on
# the following month.
TRAIN_WINDOW_MONTHS: int = 6  # Balanced training window

#: Threshold on the predicted probability used to trigger trades in the
# backtest.  Only predictions above this threshold will be considered long
# signals, and predictions below (1 – threshold) will be considered short
# signals.
SIGNAL_THRESHOLD: float = 0.75  # High confidence threshold for elite win rates

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