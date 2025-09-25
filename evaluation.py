`"""Evaluation utilities for strategy performance.

This module provides functions to compute common performance metrics
used in quantitative finance.  These include cumulative return,
compound annual growth rate (CAGR), Sharpe ratio, maximum drawdown and
the MAR ratio.  Metrics operate on an equity curve or on return
series.  Note that care should be taken to align the frequency of
returns with the annualisation factor when computing Sharpe ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import Series

from . import config


def cumulative_return(equity_curve: Series) -> float:
    """Compute total return from an equity curve.

    Parameters
    ----------
    equity_curve : Series
        Series of equity values over time (must start at initial equity).

    Returns
    -------
    float
        Final equity divided by initial equity minus one.
    """
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def max_drawdown(equity_curve: Series) -> float:
    """Compute the maximum drawdown of an equity curve.

    Drawdown is the percentage decline from a historical peak.  The
    maximum drawdown is the worst such decline over the series.

    Parameters
    ----------
    equity_curve : Series
        Series of equity values.

    Returns
    -------
    float
        Maximum drawdown as a positive number (e.g. 0.15 for 15 %).
    """
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return -drawdowns.min()


def sharpe_ratio(returns: Series, risk_free_rate: float | None = None, periods_per_year: int = 252) -> float:
    """Compute the annualised Sharpe ratio of a return series.

    Parameters
    ----------
    returns : Series
        Series of periodic returns (e.g. daily or trade returns).
    risk_free_rate : float, optional
        Annualised risk‑free rate.  If provided, it will be converted
        to the periodic rate based on ``periods_per_year`` and subtracted
        from the mean return.  Defaults to ``config.RISK_FREE_RATE``.
    periods_per_year : int, optional
        Number of return periods per year (e.g. 252 for daily, 52 for
        weekly).  Default is 252.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE
    # Drop NaNs
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    if std_ret == 0:
        return np.nan
    # Convert risk free rate to per period rate
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_mean = mean_ret - rf_per_period
    return (excess_mean / std_ret) * np.sqrt(periods_per_year)


def mar_ratio(equity_curve: Series, risk_free_rate: float | None = None, periods_per_year: int = 252) -> float:
    """Compute the MAR ratio (CAGR divided by maximum drawdown).

    Parameters
    ----------
    equity_curve : Series
        Series of equity values over time.
    risk_free_rate : float, optional
        Annualised risk‑free rate.  Defaults to ``config.RISK_FREE_RATE``.
    periods_per_year : int, optional
        Number of periods per year.  Default is 252.

    Returns
    -------
    float
        MAR ratio (CAGR / MaxDrawdown).
    """
    # Compute total return and approximate CAGR by assuming constant return per period
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return np.nan
    annual_factor = periods_per_year / n_periods
    cagr = (1 + total_ret) ** annual_factor - 1
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return np.nan
    return cagr / mdd