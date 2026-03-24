import pandas as pd
import quantstats as qs
import numpy as np

def calculate_trade_metrics(y_true, y_pred, realised_return, prediction_window, set_name="train"):
    """
    Calculate trading metrics using QuantStats.

    Parameters:
    - y_true:          True labels (pd.Series, DatetimeIndex)
    - y_pred:          Predicted labels (pd.Series, DatetimeIndex)
    - realised_return:  Log returns from triple_barrier (pd.Series, DatetimeIndex)
    - set_name:        'train' or 'test' for metric naming

    Returns:
    - dict of trading metrics
    """
    top_barrier_pred = (y_pred == 1)
    long_returns = realised_return[top_barrier_pred]  # returns on trades we entered
    long_true = y_true[top_barrier_pred]          # actual outcomes of those trades
    
    trade_count = long_returns.count()
    trade_frequency = len(long_returns) / len(realised_return)
    win_rate = qs.stats.win_rate(long_returns)
    profit_factor = qs.stats.profit_factor(long_returns)
    avg_win = qs.stats.avg_win(long_returns)
    avg_loss = abs(qs.stats.avg_loss(long_returns))
    expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Series with returns on all days, traded or not.
    full_index = realised_return.index
    daily_long_signal_returns = pd.Series(0.0, index=full_index)
    daily_long_signal_returns [long_returns.index] = long_returns.values
    
    cumulative_return = qs.stats.comp(daily_long_signal_returns) * 100
    sharpe_ratio = qs.stats.sharpe(daily_long_signal_returns, periods = 365)
    sortino_ratio = qs.stats.sortino(daily_long_signal_returns, periods = 365)
    max_drawdown = qs.stats.max_drawdown(daily_long_signal_returns)
    calmar_ratio = qs.stats.calmar(daily_long_signal_returns)
    
    # Custom metric: Avg Loss When Wrong
    bought_but_price_not_rise_mask = (long_true != 1)
    returns_on_wrong_trades = long_returns[bought_but_price_not_rise_mask]
    avg_loss_when_wrong = abs(returns_on_wrong_trades[returns_on_wrong_trades < 0].mean())
    
    trade_metrics = {
        f"{set_name}_trade_count": trade_count,
        f"{set_name}_trade_frequency_pct": trade_frequency * 100,
        f"{set_name}_win_rate": win_rate,
        f"{set_name}_profit_factor": profit_factor,
        f"{set_name}_avg_win_pct": avg_win * 100,
        f"{set_name}_avg_loss_pct": avg_loss * 100,
        f"{set_name}_cum_return_pct": cumulative_return * 100,
        f"{set_name}_sharpe_ratio": sharpe_ratio,
        f"{set_name}_sortino_ratio": sortino_ratio,
        f"{set_name}_max_drawdown_pct": max_drawdown * 100,
        f"{set_name}_calmar_ratio": calmar_ratio,
        f"{set_name}_expected_value": expected_value,
        f"{set_name}_avg_loss_when_wrong_pct": avg_loss_when_wrong * 100
    }
    
    return trade_metrics, daily_long_signal_returns