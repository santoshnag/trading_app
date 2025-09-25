"""Label generation using the triple barrier method.

This module implements the triple barrier labeling scheme described in
López de Prado's *Advances in Financial Machine Learning*【18715623638126†L335-L355】.  Given a
series of prices, it assigns a label of +1, –1 or 0 to each time point
depending on which of three barriers is hit first: an upper barrier
(profit target), a lower barrier (stop loss) or a vertical barrier
(maximum holding period).  The barriers are expressed as fractional
distance from the entry price.  If the price moves up by ``profit_take``
before it falls by ``stop_loss`` within ``max_minutes`` bars, the label
is +1.  If it falls by ``stop_loss`` first, the label is –1.  If neither
barrier is reached, the label is 0.  This labeling scheme ensures that
labels capture both magnitude and timing of moves and prevent lookahead
bias by fixing a finite horizon【18715623638126†L335-L355】.

Example
-------
>>> import pandas as pd
>>> from trading_app.labeling import triple_barrier_labels
>>> prices = pd.Series([100, 100.5, 101, 99.8, 100.2],
...                   index=pd.date_range("2023-01-01", periods=5, freq="T"))
>>> triple_barrier_labels(prices, 0.01, 0.01, 2)
2023-01-01 00:00:00    0
2023-01-01 00:01:00    0
2023-01-01 00:02:00   -1
2023-01-01 00:03:00   -1
2023-01-01 00:04:00   -1
Freq: T, dtype: int64
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import Series

# Import config with fallback for both package and script execution
try:
    from . import config
except ImportError:
    import config

try:  # Optional acceleration
    from numba import njit
except Exception:  # pragma: no cover
    njit = None  # type: ignore


def _triple_barrier_numba(prices: np.ndarray, profit_take: float, stop_loss: float, max_minutes: int) -> np.ndarray:
    """Numba-accelerated triple barrier labeling over a price array.

    Returns float array with values {1.0, 0.0, -1.0, nan}.
    """
    # Define inner compiled function only if numba is available
    if njit is None:
        raise RuntimeError("Numba not available")

    @njit(cache=True, fastmath=True)  # type: ignore
    def _kernel(prices_: np.ndarray, pt: float, sl: float, horizon: int) -> np.ndarray:
        n = prices_.shape[0]
        out = np.empty(n, dtype=np.float64)
        for idx in range(n):
            entry = prices_[idx]
            end_idx = idx + horizon if horizon > 0 else n - 1
            if end_idx >= n:
                end_idx = n - 1
            if idx + 1 > end_idx:
                out[idx] = np.nan
                continue
            t_pt = n + 1
            t_sl = n + 1
            for j in range(idx + 1, end_idx + 1):
                r = (prices_[j] - entry) / entry
                if pt > 0 and r >= pt:
                    t_pt = j
                    break
                if sl > 0 and r <= -sl:
                    t_sl = j
                    break
            # If neither hit, keep scanning for the other until end
            if t_pt == n + 1 and t_sl == n + 1:
                out[idx] = 0.0
            elif t_pt <= t_sl:
                out[idx] = 1.0
            else:
                out[idx] = -1.0
        return out

    return _kernel(prices, profit_take, stop_loss, max_minutes)


def triple_barrier_labels(
    price: Series,
    profit_take: float | None = None,
    stop_loss: float | None = None,
    max_minutes: int | None = None,
) -> Series:
    """Generate triple barrier labels for a price series.

    Parameters
    ----------
    price : Series
        Series of prices indexed by timestamps.  The index must be a
        DatetimeIndex with regular frequency (e.g. minute bars).
    profit_take : float, optional
        Profit‐taking threshold expressed as a fraction of the entry price.
        If ``None`` or 0, the upper barrier is disabled.  Default uses
        ``config.PROFIT_TAKE``.
    stop_loss : float, optional
        Stop‐loss threshold expressed as a fraction of the entry price.  If
        ``None`` or 0, the lower barrier is disabled.  Default uses
        ``config.STOP_LOSS``.
    max_minutes : int, optional
        Maximum holding period in bars/minutes.  If ``None``, uses
        ``config.MAX_HOLD_MINUTES``.  If zero, no vertical barrier is
        applied and labels will be NaN at the end where no data exists.

    Returns
    -------
    Series
        A series of integer labels (+1, –1 or 0) indexed like the input.
        Rows near the end that do not have sufficient future data to
        determine the barrier outcome are assigned NaN and should be
        dropped prior to model training.
    """
    # Use defaults from config if not provided
    if profit_take is None:
        profit_take = config.PROFIT_TAKE
    if stop_loss is None:
        stop_loss = config.STOP_LOSS
    if max_minutes is None:
        max_minutes = config.MAX_HOLD_MINUTES

    prices = price.to_numpy(dtype=float)
    try:
        if njit is not None:
            labels = _triple_barrier_numba(prices, float(profit_take), float(stop_loss), int(max_minutes))
        else:
            raise RuntimeError
    except Exception:
        # Fallback Python implementation
        n = len(prices)
        labels = np.full(n, np.nan, dtype=float)
        for idx in range(n):
            entry_price = prices[idx]
            end_idx = idx + max_minutes if max_minutes > 0 else n
            if end_idx > n:
                end_idx = n
            if idx + 1 >= end_idx:
                labels[idx] = np.nan
                continue
            future_prices = prices[idx + 1 : end_idx]
            returns = (future_prices - entry_price) / entry_price
            t_pt = np.inf
            t_sl = np.inf
            if profit_take is not None and profit_take > 0:
                hits = np.where(returns >= profit_take)[0]
                if hits.size > 0:
                    t_pt = hits[0]
            if stop_loss is not None and stop_loss > 0:
                hits = np.where(returns <= -stop_loss)[0]
                if hits.size > 0:
                    t_sl = hits[0]
            if np.isfinite(t_pt) and t_pt <= t_sl:
                labels[idx] = 1
            elif np.isfinite(t_sl) and t_sl < t_pt:
                labels[idx] = -1
            else:
                labels[idx] = 0
    return pd.Series(labels, index=price.index)


def generate_labels_for_dataframe(df: pd.DataFrame) -> Series:
    """Apply triple barrier labeling to a DataFrame containing OHLCV data.

    This helper extracts the close price from the DataFrame and calls
    ``triple_barrier_labels`` with the global configuration thresholds.  It
    returns a series of labels aligned with the DataFrame's index.

    Parameters
    ----------
    df : DataFrame
        DataFrame with a ``close`` column and a DatetimeIndex.

    Returns
    -------
    Series
        Triple barrier labels for each timestamp.
    """
    price = df["close"]
    return triple_barrier_labels(
        price,
        profit_take=config.PROFIT_TAKE,
        stop_loss=config.STOP_LOSS,
        max_minutes=config.MAX_HOLD_MINUTES,
    )