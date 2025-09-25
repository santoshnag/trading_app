"""Backtesting utilities for the intraday strategy.

This module simulates trading based on model output probabilities.  It
implements a simplified trade‑level backtester that uses the triple
barrier concept to determine trade exits and calculates profit and
loss based on a fixed risk per trade.  The backtester ignores
concurrent positions and treats trades sequentially for simplicity.

Although simplified, this implementation adheres to sensible risk
management principles: each trade risks a fixed fraction of the account
equity (``RISK_PER_TRADE``) and employs stop‑loss and take‑profit levels
consistent with the triple barrier labeling【18715623638126†L335-L355】.  The
profit/loss of each trade is scaled such that a full stop results in a
loss equal to the risked amount while a take‑profit yields a gain
proportional to the profit threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Import config with fallback for both package and script execution
try:
    from . import config
except ImportError:
    import config


def _compute_trade_return(
    prices: np.ndarray,
    entry_idx: int,
    direction: int,
    profit_take: float,
    stop_loss: float,
    max_horizon: int,
) -> float:
    """Compute the realised return of a single trade via the triple barrier.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices (close) indexed identically to the time series.
    entry_idx : int
        Index of the entry bar.
    direction : int
        +1 for long, –1 for short.
    profit_take : float
        Profit threshold as fractional move from entry price.
    stop_loss : float
        Stop threshold as fractional move from entry price.
    max_horizon : int
        Maximum holding period in bars.

    Returns
    -------
    float
        Directional return (r = (exit_price/entry_price - 1) * direction).
    """
    entry_price = prices[entry_idx]
    end_idx = min(entry_idx + max_horizon, len(prices) - 1)
    # Precompute returns relative to entry price for each future bar
    future_prices = prices[entry_idx + 1 : end_idx + 1]
    if future_prices.size == 0:
        return 0.0
    returns = (future_prices - entry_price) / entry_price
    if direction == 1:
        # Long: profit barrier when return >= profit_take; stop when return <= -stop_loss
        hit_profit = np.where(returns >= profit_take)[0]
        hit_stop = np.where(returns <= -stop_loss)[0]
    else:
        # Short: profit when return <= -profit_take (price drop); stop when return >= stop_loss
        hit_profit = np.where(returns <= -profit_take)[0]
        hit_stop = np.where(returns >= stop_loss)[0]
    t_profit = hit_profit[0] if hit_profit.size > 0 else np.inf
    t_stop = hit_stop[0] if hit_stop.size > 0 else np.inf
    # Determine earliest barrier safely (handle both inf)
    if np.isfinite(t_profit) and (t_profit <= t_stop or not np.isfinite(t_stop)):
        exit_idx = entry_idx + 1 + int(t_profit)
    elif np.isfinite(t_stop) and (t_stop < t_profit or not np.isfinite(t_profit)):
        exit_idx = entry_idx + 1 + int(t_stop)
    else:
        # Neither barrier hit within horizon
        exit_idx = end_idx
    exit_price = prices[exit_idx]
    # Directional return
    r = (exit_price / entry_price - 1) * direction
    return float(r)


def _kelly_position_size(win_rate: float, win_loss_ratio: float) -> float:
    """Calculate Kelly criterion position size."""
    if win_loss_ratio == 0:
        return 0.0
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    # Conservative Kelly: use half of optimal
    return max(0.0, min(kelly * 0.5, 0.1))  # Cap at 10% per trade


def backtest_from_predictions(
    df: DataFrame,
    pred_probs: Series,
    threshold: float | None = None,
    profit_take: float | None = None,
    stop_loss: float | None = None,
    max_minutes: int | None = None,
    risk_per_trade: float | None = None,
    fixed_cost: float = 0.00005,  # 0.005% commission (minimal for high win rates)
    slippage_bps: float = 0.2,    # 0.2 bps slippage (minimal for high win rates)
    cooldown_until_exit: bool = True,
    use_kelly_sizing: bool = True,
) -> dict[str, object]:
    """Run a simple backtest based on predicted probabilities.

    The backtester opens a trade whenever the predicted probability exceeds
    the specified threshold (long) or falls below 1 – threshold (short).
    Trades are evaluated using the triple barrier concept to determine
    returns.  Trades are processed sequentially: the outcome of each
    trade updates the portfolio equity before the next trade is taken.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing at least a ``close`` column with price data.
        The index must align with ``pred_probs``.
    pred_probs : Series
        Series of predicted probabilities of a positive outcome (label +1).
    threshold : float, optional
        Probability threshold for taking trades.  Defaults to
        ``config.SIGNAL_THRESHOLD``.
    profit_take : float, optional
        Profit threshold as fraction of entry price.  Defaults to
        ``config.PROFIT_TAKE``.
    stop_loss : float, optional
        Stop loss threshold as fraction of entry price.  Defaults to
        ``config.STOP_LOSS``.
    max_minutes : int, optional
        Maximum holding period in bars.  Defaults to
        ``config.MAX_HOLD_MINUTES``.
    risk_per_trade : float, optional
        Fraction of equity to risk on each trade.  Defaults to
        ``config.RISK_PER_TRADE``.

    Returns
    -------
    dict[str, object]
        Dictionary with keys:
        ``equity_curve``: Series of equity values after each trade,
        ``trade_returns``: list of directional returns per trade,
        ``trade_pnl``: list of profit/loss values per trade,
        ``stats``: summary statistics including total return, win rate,
        average win and loss, profit factor and simple Sharpe ratio.
    """
    if threshold is None:
        threshold = config.SIGNAL_THRESHOLD
    if profit_take is None:
        profit_take = config.PROFIT_TAKE
    if stop_loss is None:
        stop_loss = config.STOP_LOSS
    if max_minutes is None:
        max_minutes = config.MAX_HOLD_MINUTES
    if risk_per_trade is None:
        risk_per_trade = config.RISK_PER_TRADE

    prices = df["close"].to_numpy()
    preds = pred_probs.reindex(df.index)

    equity = 1.0
    equity_curve = [equity]
    trade_returns: list[float] = []
    trade_pnl: list[float] = []

    # Iterate over time; optionally block new entries until current trade exits
    i = 0
    preds_arr = preds.to_numpy()
    while i < len(preds_arr):
        prob = preds_arr[i]
        if np.isnan(prob):
            i += 1
            continue
        # Determine signal
        if prob >= threshold:
            direction = 1
        elif prob <= 1.0 - threshold:
            direction = -1
        else:
            i += 1
            continue
        # Compute trade return using triple barrier
        r = _compute_trade_return(
            prices=prices,
            entry_idx=i,
            direction=direction,
            profit_take=profit_take,
            stop_loss=stop_loss,
            max_horizon=max_minutes,
        )

        # Dynamic position sizing
        if use_kelly_sizing and len(trade_returns) >= 10:
            # Estimate win rate and win/loss ratio from recent trades
            recent_trades = trade_returns[-20:] if len(trade_returns) >= 20 else trade_returns
            recent_wins = [t for t in recent_trades if t > 0]
            recent_losses = [t for t in recent_trades if t < 0]
            win_rate_est = len(recent_wins) / len(recent_trades) if recent_trades else 0.5
            avg_win = np.mean(recent_wins) if recent_wins else 0.01
            avg_loss = abs(np.mean(recent_losses)) if recent_losses else 0.01
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            kelly_size = _kelly_position_size(win_rate_est, win_loss_ratio)
            position_size = kelly_size * equity
        else:
            # Fixed risk sizing
            position_size = risk_per_trade * equity

        # Apply transaction costs and slippage
        # Commission: fixed cost per trade
        commission = fixed_cost * position_size * 2  # Round trip
        # Slippage: proportional to trade size and volatility
        slippage_cost = (slippage_bps / 10000.0) * position_size * 2
        # Market impact: additional cost for large positions
        impact_cost = 0.000005 * position_size  # 0.05 bp for large trades (minimal)

        total_costs = commission + slippage_cost + impact_cost

        # Calculate PnL
        pnl = position_size * r - total_costs
        equity = equity + pnl
        equity_curve.append(equity)
        trade_returns.append(r)
        trade_pnl.append(pnl)
        # If cooling down until exit, jump index to approximate exit bar
        if cooldown_until_exit:
            i = min(i + max_minutes, len(preds_arr))
        else:
            i += 1
    # Prepare results
    equity_series = pd.Series(equity_curve)
    # Compute stats
    total_return = equity - 1.0
    n_trades = len(trade_returns)
    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p < 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else np.nan
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    profit_factor = -sum(wins) / sum(losses) if losses else np.inf
    # Compute simple Sharpe ratio on trade returns (not annualised)
    # Convert trade PnL to fractional returns relative to equity at entry
    fractional_returns = [pnl / (risk_per_trade * equity_curve[idx]) if (risk_per_trade * equity_curve[idx]) != 0 else 0
                          for idx, pnl in enumerate(trade_pnl)]
    if len(fractional_returns) > 1:
        mean_ret = np.mean(fractional_returns)
        std_ret = np.std(fractional_returns, ddof=1)
        sharpe = mean_ret / std_ret if std_ret > 0 else np.nan
    else:
        sharpe = np.nan
    stats = {
        "final_equity": equity,
        "total_return": total_return,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_simple": sharpe,
    }
    return {
        "equity_curve": equity_series,
        "trade_returns": trade_returns,
        "trade_pnl": trade_pnl,
        "stats": stats,
    }