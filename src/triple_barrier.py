"""
Label each day as Overbought (+1), Neutral (0), or Oversold (-1)
based on triple barrier method.

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
    pd.DataFrame with columns: labels, returns, exit_prices, barrier_hit_times
"""

import pandas as pd
import numpy as np

def triple_barrier(price_series, volatility_series, holding_period, 
                            profit_mult, stop_mult, min_ret_threshold, apply_min_ret_threshold = False):
    
    n = len(price_series)
    labels = np.full(n, np.nan)
    returns = np.full(n, np.nan)
    exit_prices = np.full(n, np.nan)
    barrier_hit_times = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')
    
    for i in range(n - holding_period):
        entry_date = price_series.index[i]
        entry_price = price_series.iloc[i]
        volatility = volatility_series.iloc[i]
        
        # Skip if volatility not yet calculated (NaN in early rows)
        if pd.isna(volatility) or pd.isna(entry_price) or volatility == 0:
            continue
        
        # Define barriers based on volatility (López de Prado method)
        profit_target = entry_price * (1 + profit_mult * volatility)
        stop_loss = entry_price * (1 - stop_mult * volatility)
        
        # Find prices of the next 7 days
        future_prices = price_series.iloc[i+1:i+holding_period+1]
        
        # Vertical barrier (time limit)
        vertical_barrier_date = entry_date + pd.Timedelta(days=holding_period)
        
        # Check which barrier hit first in next holding_period days
        # Check profit target hit
        profit_hits = future_prices >= profit_target
        profit_hit = profit_hits.any()
        if profit_hit:
            profit_hit_idx = profit_hits.idxmax() # idxmax gives the first occurence of maximum value - 
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
            if profit_hit_idx < loss_hit_idx:
                label = +1  # Overbought (profit hit first)
                hit_time = profit_hit_idx
                exit_price = future_prices.loc[hit_time]
                actual_return = (exit_price - entry_price) / entry_price
                
            else:
                label = -1  # Oversold (loss hit first)
                hit_time = loss_hit_idx
                exit_price = future_prices.loc[hit_time]
                actual_return = (exit_price - entry_price) / entry_price
                
        elif profit_hit:
            label = +1  # Overbought
            hit_time = profit_hit_idx
            exit_price = future_prices.loc[hit_time]
            actual_return = (exit_price - entry_price) / entry_price
            
        elif loss_hit:
            label = -1  # Oversold
            hit_time = loss_hit_idx
            exit_price = future_prices.loc[hit_time]
            actual_return = (exit_price - entry_price) / entry_price
            
        else:
            # Neither barrier hit - exit at vertical barrier (time limit)
            label = 0  # Neutral (time limit hit)
            hit_time = vertical_barrier_date
            exit_price = future_prices.iloc[-1]  # Price at end of holding period
            actual_return = (exit_price - entry_price) / entry_price
            
            # Apply minimum return threshold for neutral classification
            if apply_min_ret_threshold:
                if abs(actual_return) < min_ret_threshold:
                    label = 0  # Neutral (insufficient movement)
                elif actual_return > 0:
                    label = +1  # Overbought (small gain)
                else:
                    label = -1  # Oversold (small loss)
        
        labels[i] = label
        returns[i] = actual_return
        exit_prices[i] = exit_price
        barrier_hit_times[i]   = hit_time
    
    return pd.DataFrame({
        'exit_prices': exit_prices,
        'returns':     returns,
        'barrier_hit_times':   barrier_hit_times,
        'labels':      labels
    }, index = price_series.index)