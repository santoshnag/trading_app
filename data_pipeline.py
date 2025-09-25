"""Data ingestion and management for the intraday trading system.

This module defines utility functions to download and persist intraday bar data
from various free data providers.  Whenever possible, data is cached on disk
under the `DATA_DIR` specified in `config.py`.  If an API key for Alpha Vantage
is supplied, the extended intraday endpoint will be used to obtain up to two
years of minute bars.  Otherwise the code falls back to Yahoo Finance via the
`yfinance` package.  All data is returned as a pandas DataFrame indexed by
timestamp in UTC.
"""

from __future__ import annotations

import os
import time
import csv
import io
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

# Import config with fallback for both package and script execution
try:
    from . import config
except ImportError:
    import config


def _ensure_data_dir() -> None:
    """Create the data directory on disk if it does not already exist."""
    os.makedirs(config.DATA_DIR, exist_ok=True)


def download_yfinance_data(ticker: str,
                           start_date: str,
                           end_date: str,
                           interval: str = "1m") -> pd.DataFrame:
    """Download intraday price data from Yahoo Finance using yfinance.

    Parameters
    ----------
    ticker : str
        The ticker symbol (e.g. "AAPL" or "SPY").
    start_date : str
        Start date (inclusive) in YYYY‑MM‑DD format.
    end_date : str
        End date (exclusive) in YYYY‑MM‑DD format.
    interval : str, optional
        Bar interval (e.g. "1m", "5m", "15m", etc.).  Minute intervals
        shorter than one day are only available for the past 30 days on Yahoo
        Finance.

    Returns
    -------
    pd.DataFrame
        A DataFrame of OHLCV bars indexed by UTC timestamps.  Columns are
        ["open", "high", "low", "close", "volume"].  The index is named
        "timestamp".
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except TypeError as e:
        # yfinance may raise if the interval is not supported for the requested span
        if interval == "1m":
            print(f"Yahoo 1m not available for full span. Falling back to 5m for {ticker}...")
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="5m",
                auto_adjust=False,
                progress=False,
            )
        else:
            raise
    # If empty (e.g., 1m beyond 8-day limit), retry with 5m
    if df.empty and interval == "1m":
        print(f"Yahoo 1m empty for span. Falling back to 5m for {ticker}...")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="5m",
            auto_adjust=False,
            progress=False,
        )
    if df.empty:
        raise RuntimeError(f"No data returned from Yahoo Finance for {ticker}")
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df.index.name = "timestamp"
    # Convert timezone to UTC (yfinance returns localised timestamps already in
    # local timezone; convert to UTC for consistency).
    if df.index.tz is not None:
        df = df.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC")
    return df[["open", "high", "low", "close", "volume"]]


def download_alpha_vantage_intraday(
    ticker: str,
    api_key: str,
    interval: str = "1min",
    max_slices: int = 24,
) -> pd.DataFrame:
    """Download extended intraday data from Alpha Vantage.

    Alpha Vantage offers 24 monthly slices of 30‑day 1‑minute bars via the
    `TIME_SERIES_INTRADAY_EXTENDED` endpoint.  This function loops through
    slices and concatenates them into a single DataFrame.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    api_key : str
        Your Alpha Vantage API key.  Free keys allow five requests per minute
        and 500 requests per day.
    interval : str, optional
        Bar interval.  Only "1min", "5min", "15min", "30min" and "60min" are
        supported by Alpha Vantage.
    max_slices : int, optional
        Maximum number of slices to download.  Alpha Vantage returns up to 24
        slices (24 × 30 days ≈ 2 years) for the 1‑minute interval.  For longer
        intervals fewer slices may be needed.

    Returns
    -------
    pd.DataFrame
        A DataFrame of OHLCV bars indexed by UTC timestamps.
    """
    base_url = "https://www.alphavantage.co/query"
    # Session with retries
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    all_frames: list[pd.DataFrame] = []
    for i in range(1, max_slices + 1):
        slice_str = f"year{((i - 1) // 12) + 1}month{((i - 1) % 12) + 1}"
        params = {
            "function": "TIME_SERIES_INTRADAY_EXTENDED",
            "symbol": ticker,
            "interval": interval,
            "slice": slice_str,
            "apikey": api_key,
            "adjusted": "true",
            "datatype": "csv",
        }
        response = session.get(base_url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Alpha Vantage request failed: status {response.status_code}"
            )
        csv_bytes = response.content
        # Load CSV into a DataFrame
        chunk = pd.read_csv(io.BytesIO(csv_bytes))
        if chunk.empty:
            # When we reach further in the past there may be no data, so break
            break
        chunk.rename(columns={
            "time": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }, inplace=True)
        # Convert timestamp to datetime and set as index
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True)
        chunk.set_index("timestamp", inplace=True)
        chunk.sort_index(inplace=True)
        all_frames.append(chunk[["open", "high", "low", "close", "volume"]])
        # Sleep to respect rate limits (5 calls per minute)
        time.sleep(12)
    if not all_frames:
        raise RuntimeError(f"No data downloaded for {ticker} from Alpha Vantage")
    df = pd.concat(all_frames).sort_index()
    return df


def load_or_download_data(ticker: str) -> pd.DataFrame:
    """Load data for a ticker from disk, downloading it if necessary.

    The function checks for an existing CSV file under `DATA_DIR`.  If the file
    does not exist, it attempts to download data using Alpha Vantage (if
    configured) and falls back to Yahoo Finance.  The resulting DataFrame is
    saved to disk for future reuse.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    pd.DataFrame
        Minute bar data with columns ["open", "high", "low", "close", "volume"]
        and a UTC DatetimeIndex named "timestamp".
    """
    _ensure_data_dir()
    # Prefer Parquet for faster IO and smaller size; fall back to CSV if needed
    parquet_path = os.path.join(config.DATA_DIR, f"{ticker}_{config.INTERVAL}.parquet")
    csv_path = os.path.join(config.DATA_DIR, f"{ticker}_{config.INTERVAL}.csv")
    # If file exists, load it
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        df.index.name = "timestamp"
        return df
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    # Try to download using Alpha Vantage if API key provided
    if config.ALPHAVANTAGE_API_KEY:
        try:
            df = download_alpha_vantage_intraday(
                ticker,
                api_key=config.ALPHAVANTAGE_API_KEY,
                interval=config.INTERVAL,
                max_slices=getattr(config, "ALPHAVANTAGE_MAX_SLICES", 24),
            )
        except Exception as e:
            print(f"Alpha Vantage download failed for {ticker}: {e}")
            df = None
        if df is not None and not df.empty:
            # Save both CSV (compat) and Parquet (fast)
            df.to_csv(csv_path)
            try:
                df.to_parquet(parquet_path, compression="snappy")
            except Exception:
                pass
            return df
    # Fallback: download from Yahoo Finance
    try:
        df = download_yfinance_data(
            ticker,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            interval=config.INTERVAL,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {ticker}: {e}")
    df.to_csv(csv_path)
    try:
        df.to_parquet(parquet_path, compression="snappy")
    except Exception:
        pass
    return df


def load_data_for_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load data for multiple tickers.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols to load.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary mapping each ticker to its DataFrame of minute bars.
    """
    data = {}
    for ticker in tickers:
        print(f"Loading data for {ticker}...")
        df = load_or_download_data(ticker)
        print(f"Data loaded for {ticker}: shape {df.shape}, date range {df.index.min()} to {df.index.max()}")
        # Restrict to the configured date range
        df = df.loc[(df.index >= config.START_DATE) & (df.index < config.END_DATE)]
        print(f"After date filtering: shape {df.shape}")
        data[ticker] = df
    return data