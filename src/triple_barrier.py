"""
Label each day as Oversold (+1), Neutral (0), or Overbought (-1)
based on triple barrier method (López de Prado, 2018).

Implementation follows:
López de Prado, M. (2018). "Advances in Financial Machine Learning"
Wiley. Chapter 3: Labeling.

Args:
    price_series: Close prices (pd.Series)
    volatility_series: Rolling volatility (pd.Series)
    holding_period: Maximum days to hold position (int)
    profit_mult: Profit target multiplier (float)
    stop_mult: Stop loss multiplier (float)
    min_ret_threshold: Minimum return to consider non-neutral (float)

Returns:
    Tuple of (labels, actual_returns, barrier_hit_times)
"""

import pandas as pd
import numpy as np
import sys

def triple_barrier(price_series, volatility_series, holding_period, 
                            profit_mult, stop_mult, min_ret_threshold):

    labels = []
    actual_returns = []
    barrier_hit_times = []
    
    for i in range(len(price_series) - holding_period):
        entry_price = price_series.iloc[i]
        volatility = volatility_series.iloc[i]
        entry_date = price_series.index[i]
        
        # Skip if volatility not yet calculated (NaN in early rows)
        if pd.isna(volatility) or pd.isna(entry_price) or volatility == 0:
            labels.append(np.nan)
            actual_returns.append(np.nan)
            barrier_hit_times.append(pd.NaT)
            continue
        
        # Define barriers based on volatility (López de Prado method)
        profit_target = entry_price * (1 + profit_mult * volatility)
        stop_loss = entry_price * (1 - stop_mult * volatility)
        
        # Vertical barrier (time limit)
        vertical_barrier_date = entry_date + pd.Timedelta(days=holding_period)
        
        # Check which barrier hit first in next holding_period days
        future_prices = price_series.iloc[i+1:i+holding_period+1]
        
        # Check profit target hit
        profit_hits = future_prices >= profit_target
        profit_hit = profit_hits.any()
        if profit_hit:
            profit_hit_idx = profit_hits.idxmax()
        else:
            profit_hit_idx = None
        
        # Check stop loss hit
        loss_hits = future_prices <= stop_loss
        loss_hit = loss_hits.any()
        if loss_hit:
            loss_hit_idx = loss_hits.idxmax()
        else:
            loss_hit_idx = None
        
        # Determine which barrier hit FIRST
        if profit_hit and loss_hit:
            # Both hit - check which came first (compare index positions)
            profit_pos = future_prices.index.get_loc(profit_hit_idx)
            loss_pos = future_prices.index.get_loc(loss_hit_idx)
            
            if profit_pos < loss_pos:
                label = +1  # Oversold (profit hit first)
                actual_return = (profit_target - entry_price) / entry_price
                hit_time = profit_hit_idx
            else:
                label = -1  # Overbought (loss hit first)
                actual_return = (stop_loss - entry_price) / entry_price
                hit_time = loss_hit_idx
                
        elif profit_hit:
            label = +1  # Oversold
            actual_return = (profit_target - entry_price) / entry_price
            hit_time = profit_hit_idx
            
        elif loss_hit:
            label = -1  # Overbought
            actual_return = (stop_loss - entry_price) / entry_price
            hit_time = loss_hit_idx
            
        else:
            # Neither barrier hit - exit at vertical barrier (time limit)
            label = 0  # Neutral (time limit hit)
            exit_price = future_prices.iloc[-1]  # Price at end of holding period
            actual_return = (exit_price - entry_price) / entry_price
            hit_time = vertical_barrier_date
            
            # # Apply minimum return threshold for neutral classification
            # if abs(actual_return) < min_ret_threshold:
            #     label = 0  # Neutral (insufficient movement)
            # elif actual_return > 0:
            #     label = +1  # Oversold (small gain)
            # else:
            #     label = -1  # Overbought (small loss)
        
        labels.append(label)
        actual_returns.append(actual_return)
        barrier_hit_times.append(hit_time)
    
    # Fill remaining rows (last holding_period days) with NaN
    labels.extend([np.nan] * holding_period)
    actual_returns.extend([np.nan] * holding_period)
    barrier_hit_times.extend([pd.NaT] * holding_period)
    
    return labels, actual_returns, barrier_hit_times