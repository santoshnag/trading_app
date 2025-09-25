"""Feature engineering for intraday data.

This module provides a function ``compute_features`` that transforms a raw
OHLCV DataFrame into a rich set of numeric features suitable for machine
learning models.  The feature library blends classical technical indicators
with novel measures derived from volatility estimators, volume analytics,
entropy and cyclical time encodings.  All computations are vectorised with
NumPy and pandas for performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import ta  # technical analysis library (https://github.com/bukosabino/ta)

# Import config with fallback for both package and script execution
try:
    from . import config
except ImportError:
    import config


def _calculate_ema(series: Series, span: int) -> Series:
    return series.ewm(span=span, adjust=False).mean()


def _calculate_parkinson_vol(high: Series, low: Series, window: int) -> Series:
    """Compute the Parkinson volatility estimator over a rolling window.

    The Parkinson estimator uses the high–low range to estimate the
    volatility of a Brownian motion without drift【154707002975751†L180-L195】.

    σ_P(T) = sqrt(1/(4 * N * ln(2)) * Σ_{i=1..N} (ln(H_i / L_i))^2)

    Parameters
    ----------
    high : Series
        High prices.
    low : Series
        Low prices.
    window : int
        Lookback window length in bars.

    Returns
    -------
    Series
        Rolling Parkinson volatility.
    """
    log_hl = np.log(high / low)
    squared = log_hl ** 2
    coef = 1.0 / (4.0 * np.log(2))
    return (coef * squared.rolling(window=window).sum() / window).pow(0.5)


def _calculate_garman_klass_vol(open_: Series, high: Series, low: Series, close: Series, window: int) -> Series:
    """Compute the Garman–Klass volatility estimator over a rolling window.

    The Garman–Klass estimator improves on Parkinson by incorporating the
    open and close prices【154707002975751†L200-L215】.  The formula is:

    σ_GK(T) = sqrt(1/N Σ [0.5 * (ln(H_i/L_i))^2 - (2 ln 2 - 1) * (ln(C_i/O_i))^2])

    Parameters
    ----------
    open_ : Series
        Opening prices.
    high : Series
        High prices.
    low : Series
        Low prices.
    close : Series
        Close prices.
    window : int
        Lookback window length.

    Returns
    -------
    Series
        Rolling Garman–Klass volatility.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    term1 = 0.5 * (log_hl ** 2)
    term2 = (2 * np.log(2) - 1) * (log_co ** 2)
    estimator = term1 - term2
    return (estimator.rolling(window=window).sum() / window).pow(0.5)


def _calculate_hurst_exponent(series: Series, window: int) -> Series:
    """Estimate the Hurst exponent over a rolling window.

    This function implements a simple rescaled range (R/S) algorithm.  For
    each window, the cumulative deviation of the demeaned series is computed
    and normalised by the standard deviation.  A linear fit in log–log space
    between the range and window size yields the Hurst exponent; however,
    because we are using a fixed window, we approximate the exponent by
    computing the slope of log(range) against log(window).  Values greater than
    0.5 indicate persistent (trend) behaviour while values less than 0.5
    indicate mean reversion【135328787340722†L270-L296】.
    """
    def hurst_window(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        y = x - x.mean()
        cumulative = np.cumsum(y)
        r = cumulative.max() - cumulative.min()
        s = y.std()
        if s == 0:
            return 0.5
        return np.log(r / s) / np.log(len(x))
    values = series.to_numpy()
    hurst_vals = np.full_like(values, np.nan, dtype=float)
    for i in range(window, len(values) + 1):
        window_data = values[i - window:i]
        hurst_vals[i - 1] = hurst_window(window_data)
    return pd.Series(hurst_vals, index=series.index)


def _shannon_entropy(series: Series, window: int, bins: int = 10) -> Series:
    """Compute the Shannon entropy of a time series over a rolling window.

    The entropy is computed on the distribution of values within the window.
    A lower entropy suggests more predictable structure, whereas higher values
    indicate randomness.  The result is normalised by log(bins) so that the
    entropy lies in [0, 1].

    Parameters
    ----------
    series : Series
        Input series (e.g. returns).
    window : int
        Lookback window length.
    bins : int, optional
        Number of bins for the histogram.  Default is 10.

    Returns
    -------
    Series
        Rolling entropy values.
    """
    values = series.to_numpy()
    entropies = np.full_like(values, np.nan, dtype=float)
    for i in range(window, len(values) + 1):
        window_data = values[i - window:i]
        hist, _ = np.histogram(window_data, bins=bins, density=True)
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        p = hist / hist.sum()
        entropy = -np.sum(p * np.log(p))
        entropies[i - 1] = entropy / np.log(bins)
    return pd.Series(entropies, index=series.index)


def _compute_vwap_distance(df: DataFrame) -> Series:
    """Compute the distance of the current price from the session VWAP.

    VWAP is calculated as the cumulative sum of price × volume divided by the
    cumulative sum of volume for each trading day.  The distance is
    expressed as (close − VWAP) / VWAP.
    """
    df = df[["close", "volume"]].copy()
    # Ensure 1D Series
    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    volume = df["volume"]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    pv = close * volume
    # Group by daily sessions using a timezone-aware date Series
    est_index = df.index.tz_convert("America/New_York")
    dates = pd.Series(est_index.normalize(), index=df.index)
    pv_cum = pv.groupby(dates).cumsum()
    vol_cum = volume.groupby(dates).cumsum()
    vwap = pv_cum / vol_cum
    dist = (close - vwap) / vwap
    return pd.Series(dist.values, index=df.index)


def _compute_rel_volume(df: DataFrame, window: int) -> Series:
    """Compute relative volume as the ratio of recent volume to its long‑term
    average.

    Here we define relative volume as the volume in the current bar divided
    by the rolling mean of volume over a larger window (e.g. a full day).  A
    value greater than 1 indicates above‑average activity, whereas values
    below 1 indicate quieter periods.
    """
    long_term_window = window * 20  # approximate 20× window for baseline
    rolling_mean = df["volume"].rolling(long_term_window, min_periods=1).mean()
    recent_sum = df["volume"].rolling(window, min_periods=1).sum()
    rel_vol = recent_sum / (rolling_mean * window)
    return rel_vol


def _sin_cos_time_of_day(index: pd.DatetimeIndex) -> tuple[Series, Series]:
    """Create cyclical encodings for the intraday time.

    The trading day has 390 minutes (from 9:30 to 16:00 Eastern).  For each
    timestamp we compute the number of minutes since the session open and map
    it onto a circle using sine and cosine.  This allows a model to learn
    periodic patterns within the day.
    """
    # Convert timestamp to US/Eastern to compute minutes since open
    # We avoid using pytz to reduce dependencies; pandas will handle tz
    # conversion if the index has timezone information.
    est = index.tz_convert("America/New_York")
    # Vectorized hour/minute accessors are efficient on DatetimeIndex
    minutes_since_midnight = est.hour * 60 + est.minute
    # Trading session runs from 9:30 AM (570 minutes) to 4:00 PM (960 minutes)
    session_start = 9 * 60 + 30
    session_end = 16 * 60
    session_length = session_end - session_start
    minutes_into_session = (minutes_since_midnight - session_start).to_numpy()
    # Clip values to [0, session_length] to avoid negative values during pre‑market
    minutes_into_session = np.clip(minutes_into_session, 0, session_length)
    frac = minutes_into_session.astype(float) / float(session_length)
    sin_time = np.sin(2 * np.pi * frac)
    cos_time = np.cos(2 * np.pi * frac)
    return pd.Series(sin_time, index=index), pd.Series(cos_time, index=index)


def compute_features(df: DataFrame, feature_list: list[str] | None = None) -> DataFrame:
    """Compute a suite of technical and novel features from raw OHLCV data.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with columns ["open", "high", "low", "close", "volume"].
        The index should be a timezone‑aware DatetimeIndex.
    feature_list : list of str, optional
        List of feature names to compute.  If omitted, all features defined in
        `config.FEATURE_LIST` will be calculated.

    Returns
    -------
    DataFrame
        New DataFrame with the same index as the input and columns for each
        requested feature.  NaN values are dropped for rows where any feature
        could not be computed (e.g. at the start of the series).
    """
    if feature_list is None:
        feature_list = config.FEATURE_LIST

    features = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    # Ensure Series (not (n,1) DataFrames) after possible provider quirks
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(open_, pd.DataFrame):
        open_ = open_.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    # Enhanced Returns with momentum (daily basis)
    if "return_1d" in feature_list:
        ret_1d = close.pct_change(1)
        features["return_1d"] = ret_1d.replace([np.inf, -np.inf], [2.0, -2.0]).fillna(0).clip(-2.0, 2.0)
    if "return_3d" in feature_list:
        ret_3d = close.pct_change(3)
        features["return_3d"] = ret_3d.replace([np.inf, -np.inf], [2.0, -2.0]).fillna(0).clip(-2.0, 2.0)
    if "return_5d" in feature_list:
        ret_5d = close.pct_change(5)
        features["return_5d"] = ret_5d.replace([np.inf, -np.inf], [2.0, -2.0]).fillna(0).clip(-2.0, 2.0)
    if "return_10d" in feature_list:
        ret_10d = close.pct_change(10)
        features["return_10d"] = ret_10d.replace([np.inf, -np.inf], [2.0, -2.0]).fillna(0).clip(-2.0, 2.0)
    if "return_20d" in feature_list:
        ret_20d = close.pct_change(20)
        features["return_20d"] = ret_20d.replace([np.inf, -np.inf], [2.0, -2.0]).fillna(0).clip(-2.0, 2.0)

    # Momentum features (daily basis)
    if "momentum_1d" in feature_list:
        features["momentum_1d"] = close / close.shift(1) - 1
    if "momentum_5d" in feature_list:
        features["momentum_5d"] = close / close.shift(5) - 1
    if "momentum_10d" in feature_list:
        features["momentum_10d"] = close / close.shift(10) - 1

    # Momentum acceleration (rate of change of returns)
    if "return_5d" in feature_list and "momentum_accel" in feature_list:
        accel = features["return_5d"].diff(5)  # Acceleration of 5d returns
        features["momentum_accel"] = accel.replace([np.inf, -np.inf], 0).fillna(0).clip(-1.0, 1.0)

    # Simple Moving Averages
    if "sma_5" in feature_list:
        features["sma_5"] = close.rolling(window=5).mean()
    if "sma_10" in feature_list:
        features["sma_10"] = close.rolling(window=10).mean()
    if "sma_20" in feature_list:
        features["sma_20"] = close.rolling(window=20).mean()
    if "sma_50" in feature_list:
        features["sma_50"] = close.rolling(window=50).mean()
    if "sma_200" in feature_list:
        features["sma_200"] = close.rolling(window=200).mean()

    # Exponential Moving Averages
    if "ema_5" in feature_list:
        features["ema_5"] = _calculate_ema(close, span=5)
    if "ema_10" in feature_list:
        features["ema_10"] = _calculate_ema(close, span=10)
    if "ema_20" in feature_list:
        features["ema_20"] = _calculate_ema(close, span=20)
    if "ema_50" in feature_list:
        features["ema_50"] = _calculate_ema(close, span=50)
    if "ema_200" in feature_list:
        features["ema_200"] = _calculate_ema(close, span=200)

    # Weighted Moving Averages
    if "wma_10" in feature_list:
        weights = np.arange(1, 11)
        features["wma_10"] = close.rolling(10).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=False)
    if "wma_20" in feature_list:
        weights = np.arange(1, 21)
        features["wma_20"] = close.rolling(20).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=False)

    # RSI (multiple timeframes)
    if "rsi_6" in feature_list:
        rsi = ta.momentum.RSIIndicator(close=close, window=6)
        features["rsi_6"] = rsi.rsi()
    if "rsi_14" in feature_list:
        rsi = ta.momentum.RSIIndicator(close=close, window=14)
        features["rsi_14"] = rsi.rsi()
    if "rsi_21" in feature_list:
        rsi = ta.momentum.RSIIndicator(close=close, window=21)
        features["rsi_21"] = rsi.rsi()

    # Stochastic Oscillators
    if "stoch_k" in feature_list or "stoch_d" in feature_list:
        stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        if "stoch_k" in feature_list:
            features["stoch_k"] = stoch.stoch()
        if "stoch_d" in feature_list:
            features["stoch_d"] = stoch.stoch_signal()

    # Williams %R
    if "williams_r" in feature_list:
        williams = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14)
        features["williams_r"] = williams.williams_r()

    # Commodity Channel Index (CCI)
    if "cci_14" in feature_list:
        cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=14)
        features["cci_14"] = cci.cci()
    if "cci_20" in feature_list:
        cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20)
        features["cci_20"] = cci.cci()

    # MACD
    if "macd" in feature_list or "macd_signal" in feature_list or "macd_hist" in feature_list:
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        if "macd" in feature_list:
            features["macd"] = macd.macd()
        if "macd_signal" in feature_list:
            features["macd_signal"] = macd.macd_signal()
        if "macd_hist" in feature_list:
            features["macd_hist"] = macd.macd_diff()

    # Money Flow Index (MFI)
    if "mfi_14" in feature_list:
        mfi = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume, window=14)
        features["mfi_14"] = mfi.money_flow_index()

    # Volatility Indicators - ATR
    if "atr_14" in feature_list:
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        features["atr_14"] = atr.average_true_range()
    if "atr_20" in feature_list:
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=20)
        features["atr_20"] = atr.average_true_range()

    # Bollinger Bands
    if any(x in feature_list for x in ["bb_upper_20", "bb_middle_20", "bb_lower_20", "bb_pct_20", "bb_width_20"]):
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        if "bb_upper_20" in feature_list:
            features["bb_upper_20"] = bb.bollinger_hband()
        if "bb_middle_20" in feature_list:
            features["bb_middle_20"] = bb.bollinger_mavg()
        if "bb_lower_20" in feature_list:
            features["bb_lower_20"] = bb.bollinger_lband()
        if "bb_pct_20" in feature_list:
            features["bb_pct_20"] = bb.bollinger_pband()
        if "bb_width_20" in feature_list:
            features["bb_width_20"] = bb.bollinger_wband()

    if any(x in feature_list for x in ["bb_upper_10", "bb_middle_10", "bb_lower_10", "bb_pct_10", "bb_width_10"]):
        bb = ta.volatility.BollingerBands(close=close, window=10, window_dev=2)
        if "bb_upper_10" in feature_list:
            features["bb_upper_10"] = bb.bollinger_hband()
        if "bb_middle_10" in feature_list:
            features["bb_middle_10"] = bb.bollinger_mavg()
        if "bb_lower_10" in feature_list:
            features["bb_lower_10"] = bb.bollinger_lband()
        if "bb_pct_10" in feature_list:
            features["bb_pct_10"] = bb.bollinger_pband()
        if "bb_width_10" in feature_list:
            features["bb_width_10"] = bb.bollinger_wband()

    # Keltner Channels
    if any(x in feature_list for x in ["keltner_upper_20", "keltner_middle_20", "keltner_lower_20", "keltner_pct_20"]):
        keltner = ta.volatility.KeltnerChannel(high=high, low=low, close=close, window=20, window_atr=10)
        if "keltner_upper_20" in feature_list:
            features["keltner_upper_20"] = keltner.keltner_channel_hband()
        if "keltner_middle_20" in feature_list:
            features["keltner_middle_20"] = keltner.keltner_channel_mband()
        if "keltner_lower_20" in feature_list:
            features["keltner_lower_20"] = keltner.keltner_channel_lband()
        if "keltner_pct_20" in feature_list:
            features["keltner_pct_20"] = keltner.keltner_channel_pband()

    # ADX and DMI
    if "adx_14" in feature_list or "dmi_plus_14" in feature_list or "dmi_minus_14" in feature_list:
        adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        if "adx_14" in feature_list:
            features["adx_14"] = adx_ind.adx()
        if "dmi_plus_14" in feature_list:
            features["dmi_plus_14"] = adx_ind.adx_pos()
        if "dmi_minus_14" in feature_list:
            features["dmi_minus_14"] = adx_ind.adx_neg()

    # Aroon Oscillator
    if "aroon_up_14" in feature_list or "aroon_down_14" in feature_list or "aroon_osc_14" in feature_list:
        aroon = ta.trend.AroonIndicator(close=close, window=14)
        if "aroon_up_14" in feature_list:
            features["aroon_up_14"] = aroon.aroon_up()
        if "aroon_down_14" in feature_list:
            features["aroon_down_14"] = aroon.aroon_down()
        if "aroon_osc_14" in feature_list:
            features["aroon_osc_14"] = aroon.aroon_indicator()

    # Ichimoku Cloud
    if any(x in feature_list for x in ["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b"]):
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
        if "ichimoku_tenkan" in feature_list:
            features["ichimoku_tenkan"] = ichimoku.ichimoku_conversion_line()
        if "ichimoku_kijun" in feature_list:
            features["ichimoku_kijun"] = ichimoku.ichimoku_base_line()
        if "ichimoku_senkou_a" in feature_list:
            features["ichimoku_senkou_a"] = ichimoku.ichimoku_a()
        if "ichimoku_senkou_b" in feature_list:
            features["ichimoku_senkou_b"] = ichimoku.ichimoku_b()

    # Pivot Points
    if "pivot_point" in feature_list or "resistance_1" in feature_list or "support_1" in feature_list:
        # Calculate pivot point using previous day's data
        pp = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        if "pivot_point" in feature_list:
            features["pivot_point"] = pp
        if "resistance_1" in feature_list:
            features["resistance_1"] = 2 * pp - low.shift(1)
        if "support_1" in feature_list:
            features["support_1"] = 2 * pp - high.shift(1)

    # Volume Indicators
    if "volume_sma_5" in feature_list:
        features["volume_sma_5"] = volume.rolling(window=5).mean()
    if "volume_sma_10" in feature_list:
        features["volume_sma_10"] = volume.rolling(window=10).mean()
    if "volume_sma_20" in feature_list:
        features["volume_sma_20"] = volume.rolling(window=20).mean()

    if "volume_ratio_5" in feature_list:
        vol_sma = volume.rolling(window=5).mean()
        features["volume_ratio_5"] = volume / vol_sma.replace(0, 1)
    if "volume_ratio_10" in feature_list:
        vol_sma = volume.rolling(window=10).mean()
        features["volume_ratio_10"] = volume / vol_sma.replace(0, 1)
    if "volume_ratio_20" in feature_list:
        vol_sma = volume.rolling(window=20).mean()
        features["volume_ratio_20"] = volume / vol_sma.replace(0, 1)

    # VWAP
    if "vwap" in feature_list:
        features["vwap"] = _compute_vwap(high, low, close, volume)

    # Additional Volume Indicators
    if "accumulation_distribution" in feature_list:
        features["accumulation_distribution"] = ta.volume.AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()
    if "chaikin_money_flow" in feature_list:
        features["chaikin_money_flow"] = ta.volume.ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume).chaikin_money_flow()
    if "force_index" in feature_list:
        features["force_index"] = ta.volume.ForceIndexIndicator(close=close, volume=volume).force_index()
    if "ease_of_movement" in feature_list:
        features["ease_of_movement"] = ta.volume.EaseOfMovementIndicator(high=high, low=low, volume=volume).ease_of_movement()
    if "volume_price_trend" in feature_list:
        features["volume_price_trend"] = ta.volume.VolumePriceTrendIndicator(close=close, volume=volume).volume_price_trend()

    # Statistical Indicators
    if "hurst_100" in feature_list:
        features["hurst_100"] = _calculate_hurst_exponent(close, window=100)
    if "hurst_200" in feature_list:
        features["hurst_200"] = _calculate_hurst_exponent(close, window=200)

    if "zscore_20" in feature_list:
        features["zscore_20"] = (close - close.rolling(window=20).mean()) / close.rolling(window=20).std()
    if "skew_20" in feature_list:
        features["skew_20"] = close.rolling(window=20).skew()
    if "kurtosis_20" in feature_list:
        features["kurtosis_20"] = close.rolling(window=20).kurt()

    # Market Breadth (using SPY as proxy - simplified)
    if "market_trend_5" in feature_list:
        features["market_trend_5"] = close.rolling(window=5).mean() / close.rolling(window=5).mean().shift(5) - 1
    if "market_trend_20" in feature_list:
        features["market_trend_20"] = close.rolling(window=20).mean() / close.rolling(window=20).mean().shift(20) - 1
    if "market_strength" in feature_list:
        features["market_strength"] = (close - close.rolling(window=20).min()) / (close.rolling(window=20).max() - close.rolling(window=20).min())

    # Time-based features
    if hasattr(close.index, 'dayofweek'):
        if "day_of_week" in feature_list:
            features["day_of_week"] = close.index.dayofweek / 6.0  # Normalize to 0-1
        if "month_of_year" in feature_list:
            features["month_of_year"] = close.index.month / 12.0  # Normalize to 0-1
        if "quarter" in feature_list:
            features["quarter"] = close.index.quarter / 4.0  # Normalize to 0-1

    if "days_since_high_20" in feature_list:
        rolling_max = close.rolling(window=20).max()
        features["days_since_high_20"] = ((close - rolling_max).abs() / rolling_max).rolling(window=20).argmin()
    if "days_since_low_20" in feature_list:
        rolling_min = close.rolling(window=20).min()
        features["days_since_low_20"] = ((close - rolling_min).abs() / rolling_min).rolling(window=20).argmin()
        features["vol_of_vol_30m"] = vol_series.rolling(window=30, min_periods=1).std().fillna(0)

    # Volume & flow
    if "obv" in feature_list:
        obv_ind = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
        features["obv"] = obv_ind.on_balance_volume()
    if "vwap_dist" in feature_list:
        features["vwap_dist"] = _compute_vwap_distance(df)
    if "rel_volume_5m" in feature_list:
        features["rel_volume_5m"] = _compute_rel_volume(df, window=5)
    if "rel_volume_15m" in feature_list:
        features["rel_volume_15m"] = _compute_rel_volume(df, window=15)
    if "volume_ma_ratio_5m" in feature_list:
        vol_ma = volume.rolling(window=5).mean()
        # Avoid division by zero, replace with 1.0 for neutral ratio
        vol_ma_safe = vol_ma.replace(0, 1.0)
        ratio = volume / vol_ma_safe
        features["volume_ma_ratio_5m"] = ratio.clip(0.1, 10.0)  # Reasonable bounds
    if "volume_ma_ratio_15m" in feature_list:
        vol_ma = volume.rolling(window=15).mean()
        # Avoid division by zero, replace with 1.0 for neutral ratio
        vol_ma_safe = vol_ma.replace(0, 1.0)
        ratio = volume / vol_ma_safe
        features["volume_ma_ratio_15m"] = ratio.clip(0.1, 10.0)  # Reasonable bounds

    if "volume_momentum" in feature_list:
        # Volume momentum: rate of change of volume
        vol_change = volume.pct_change(5)
        # Handle infinite values from division by zero
        features["volume_momentum"] = vol_change.replace([np.inf, -np.inf], 0).fillna(0)

    # Order flow indicators
    if "price_volume_trend" in feature_list:
        features["price_volume_trend"] = ((close - close.shift(1)) / close.shift(1)) * volume
        features["price_volume_trend"] = features["price_volume_trend"].fillna(0).cumsum()
    if "accumulation_distribution" in feature_list:
        ad_ind = ta.volume.AccDistIndexIndicator(high=high, low=low, close=close, volume=volume)
        features["accumulation_distribution"] = ad_ind.acc_dist_index()
    if "chaikin_money_flow" in feature_list:
        cmf_ind = ta.volume.ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume, window=20)
        features["chaikin_money_flow"] = cmf_ind.chaikin_money_flow()

    # Momentum indicators
    if "stochastic_k" in feature_list or "stochastic_d" in feature_list:
        stoch_ind = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        if "stochastic_k" in feature_list:
            features["stochastic_k"] = stoch_ind.stoch()
        if "stochastic_d" in feature_list:
            features["stochastic_d"] = stoch_ind.stoch_signal()
    if "williams_r" in feature_list:
        williams_ind = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14)
        features["williams_r"] = williams_ind.williams_r()
    if "cci_14" in feature_list:
        cci_ind = ta.trend.CCIIndicator(high=high, low=low, close=close, window=14)
        features["cci_14"] = cci_ind.cci()

    # Mean reversion indicators - Custom implementation to avoid NaN issues
    if "bollinger_upper" in feature_list or "bollinger_lower" in feature_list or "bollinger_middle" in feature_list or "bollinger_pct" in feature_list:
        # Custom Bollinger Bands calculation
        window = 20
        std_dev = 2
        rolling_mean = close.rolling(window=window, min_periods=1).mean()
        rolling_std = close.rolling(window=window, min_periods=1).std()

        if "bollinger_upper" in feature_list:
            features["bollinger_upper"] = rolling_mean + (rolling_std * std_dev)
        if "bollinger_lower" in feature_list:
            features["bollinger_lower"] = rolling_mean - (rolling_std * std_dev)
        if "bollinger_middle" in feature_list:
            features["bollinger_middle"] = rolling_mean
        if "bollinger_pct" in feature_list:
            # Position within bands: (price - lower) / (upper - lower)
            upper = rolling_mean + (rolling_std * std_dev)
            lower = rolling_mean - (rolling_std * std_dev)
            band_width = upper - lower

            # Calculate percentage position safely
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = (close - lower) / band_width

            # Replace inf/-inf/NaN with reasonable values
            pct = pct.replace([np.inf, -np.inf], [2.0, -2.0]).fillna(0.0)
            # Clip to reasonable bounds (-2 to +2 is more than enough)
            features["bollinger_pct"] = pct.clip(-2.0, 2.0)

    if "keltner_upper" in feature_list or "keltner_lower" in feature_list or "keltner_pct" in feature_list:
        keltner_ind = ta.volatility.KeltnerChannel(high=high, low=low, close=close, window=20, window_atr=10)
        if "keltner_upper" in feature_list:
            features["keltner_upper"] = keltner_ind.keltner_channel_hband()
        if "keltner_lower" in feature_list:
            features["keltner_lower"] = keltner_ind.keltner_channel_lband()
        if "keltner_pct" in feature_list:
            # Position within Keltner channel
            middle = (keltner_ind.keltner_channel_hband() + keltner_ind.keltner_channel_lband()) / 2
            features["keltner_pct"] = (close - middle) / (keltner_ind.keltner_channel_hband() - middle)

    # Regime detection
    if "hurst_120m" in feature_list:
        # Use logarithmic returns for Hurst exponent
        log_ret = np.log(close / close.shift(1))
        features["hurst_120m"] = _calculate_hurst_exponent(log_ret.fillna(0), window=120)
    if "shannon_entropy_30m" in feature_list:
        log_ret = np.log(close / close.shift(1))
        features["shannon_entropy_30m"] = _shannon_entropy(log_ret.fillna(0), window=30, bins=10)
    if "realized_volatility_30m" in feature_list:
        log_ret = np.log(close / close.shift(1)).fillna(0)
        features["realized_volatility_30m"] = (log_ret.rolling(window=30, min_periods=1).std() * np.sqrt(252)).fillna(0)  # Annualized
    if "jump_variation_30m" in feature_list:
        # Jump variation: difference between realized variance and bipower variation
        log_ret = np.log(close / close.shift(1)).fillna(0)
        rv = (log_ret ** 2).rolling(window=30, min_periods=1).sum()
        bv = (abs(log_ret).rolling(window=30, min_periods=1).sum() ** 2) / 30
        features["jump_variation_30m"] = (rv - bv).fillna(0)

    # Time of day
    if "sin_time" in feature_list or "cos_time" in feature_list:
        sin_time, cos_time = _sin_cos_time_of_day(df.index)
        if "sin_time" in feature_list:
            features["sin_time"] = sin_time
        if "cos_time" in feature_list:
            features["cos_time"] = cos_time
    if "market_phase" in feature_list:
        # Market phase: 0=pre-market, 1=regular hours, 2=after-hours
        est = df.index.tz_convert("America/New_York")
        hour = est.hour
        minute = est.minute
        time_minutes = hour * 60 + minute
        market_phase = np.zeros(len(df), dtype=int)
        market_phase[(time_minutes >= 570) & (time_minutes <= 960)] = 1  # Regular hours 9:30-16:00
        market_phase[(time_minutes >= 960) & (time_minutes <= 1200)] = 2  # After hours
        features["market_phase"] = pd.Series(market_phase, index=df.index)

    # Handle NaN values robustly
    # Fill NaN values with appropriate strategies
    for col in features.columns:
        if features[col].isna().any():
            # For momentum/oscillator features, fill with 0 (neutral)
            if any(keyword in col.lower() for keyword in ['stoch', 'williams', 'cci', 'rsi']):
                features[col] = features[col].fillna(0)
            # For volatility features, forward fill then fill remaining with 0
            elif any(keyword in col.lower() for keyword in ['vol', 'hurst', 'entropy']):
                features[col] = features[col].ffill().fillna(0)
            # For volume-based features, fill with 1 (neutral ratio)
            elif any(keyword in col.lower() for keyword in ['volume', 'rel_vol', 'vwap']):
                features[col] = features[col].fillna(1.0)
            # For price-based features, forward fill
            elif any(keyword in col.lower() for keyword in ['ema', 'bollinger', 'keltner']):
                features[col] = features[col].ffill()
            # Default: fill with 0
            else:
                features[col] = features[col].fillna(0)

    # Aggressive cleanup of infinite values throughout features
    def clean_feature_series(series, name, bounds=(-10, 10)):
        """Clean a feature series by replacing inf/NaN with reasonable values"""
        if not np.isfinite(series).all():
            # Replace inf with bounds
            series = series.replace([np.inf, -np.inf], [bounds[1], bounds[0]])
            print(f"Warning: Cleaned infinite values in {name}")
        # Fill NaN with 0
        series = series.fillna(0)
        # Clip to reasonable bounds
        series = series.clip(bounds[0], bounds[1])
        return series

    # Clean each feature column aggressively
    for col in features.columns:
        if col in ['return_5m', 'return_15m', 'return_60m', 'momentum_accel']:
            features[col] = clean_feature_series(features[col], col, (-2.0, 2.0))
        elif col in ['rsi_14', 'stochastic_k', 'cci_14', 'bollinger_pct']:
            features[col] = clean_feature_series(features[col], col, (-100, 100))
        elif col in ['macd', 'atr_14', 'vwap_dist', 'hurst_120m', 'realized_volatility_30m']:
            features[col] = clean_feature_series(features[col], col, (-10, 10))
        elif col in ['rel_volume_5m', 'volume_ma_ratio_5m', 'volume_ma_ratio_15m', 'volume_momentum']:
            features[col] = clean_feature_series(features[col], col, (0.1, 10.0))
        elif col in ['price_volume_trend', 'accumulation_distribution']:
            features[col] = clean_feature_series(features[col], col, (-1000, 1000))
        elif col in ['sin_time', 'cos_time']:
            features[col] = clean_feature_series(features[col], col, (-1.1, 1.1))
        else:
            # Default cleaning for any remaining features
            features[col] = clean_feature_series(features[col], col, (-100, 100))

    # Final validation - ensure no infinite values remain
    inf_cols = []
    for col in features.columns:
        if not np.isfinite(features[col]).all():
            inf_cols.append(col)

    if inf_cols:
        print(f"ERROR: Infinite values still present in columns: {inf_cols}")
        # Emergency fix - replace any remaining inf with 0
        features = features.replace([np.inf, -np.inf], 0)

    # Fill any remaining NaN values
    features = features.fillna(0)

    # Ensure we have at least some data
    if features.empty:
        raise ValueError("No valid features could be computed after cleanup")

    # Debug: Check for any remaining infinite values
    final_inf_cols = []
    for col in features.columns:
        if not np.isfinite(features[col]).all():
            final_inf_cols.append(col)
            print(f"CRITICAL: Column {col} still has {np.isinf(features[col]).sum()} infinite values after cleanup!")

    if not final_inf_cols:
        print("SUCCESS: All features are finite after cleanup")
    else:
        print(f"FAILED: {len(final_inf_cols)} columns still have infinite values")

    print(f"Feature cleanup complete. Final shape: {features.shape}")
    return features