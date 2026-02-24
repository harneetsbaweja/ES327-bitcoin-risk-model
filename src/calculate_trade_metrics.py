import os
import pandas as pd
import quantstats as qs
import numpy as np

def calculate_trade_metrics(y_true, y_pred, actual_returns, set_name="train"):
    """
    Calculate trading-specific metrics using QuantStats.
    
    Parameters:
    - y_true: true labels
    - y_pred: predicted labels (from predictions_df)
    - actual_returns: actual 7-day returns
    - set_name: "train" or "test" for metric naming
    
    Returns:
    - dict of trading metrics
    """
    traded_mask = (y_pred != 0)
    traded_returns = actual_returns[traded_mask] / 100
    traded_pred = y_pred[traded_mask]
    traded_true = y_true[traded_mask]
    
    win_rate = qs.stats.win_rate(traded_returns)
    profit_factor = qs.stats.profit_factor(traded_returns)
    avg_win = qs.stats.avg_win(traded_returns)
    avg_loss = abs(qs.stats.avg_loss(traded_returns))
    max_drawdown = qs.stats.max_drawdown(traded_returns)
    sharpe_ratio = qs.stats.sharpe(traded_returns)
    expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Custom metric: Avg Loss When Wrong
    wrong_predictions = (traded_pred != np.sign(traded_true))
    wrong_returns = traded_returns[wrong_predictions]
    avg_loss_when_wrong = abs(wrong_returns[wrong_returns < 0].mean())
    
    trade_metrics = {
        f"{set_name}_win_rate": win_rate,
        f"{set_name}_profit_factor": profit_factor,
        f"{set_name}_avg_win_pct": avg_win * 100,
        f"{set_name}_avg_loss_pct": avg_loss * 100,
        f"{set_name}_max_drawdown": max_drawdown,
        f"{set_name}_sharpe_ratio": sharpe_ratio,
        f"{set_name}_expected_value": expected_value,
        f"{set_name}_avg_loss_when_wrong_pct": avg_loss_when_wrong * 100
    }
    
    return trade_metrics