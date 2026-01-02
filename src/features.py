"""
Feature Engineering Module
Calculates all technical indicators and ML features from raw Bitcoin data
"""
# ========== CONFIG ==========
rsi_period = 14
macd_fast_period = 12
macd_slow_period = 26
macd_signal_period = 9
bollinger_bands_period = 20
bollinger_bands_std_deviations = 2

halving_dates = [
    '2012-11-28',  # First halving: 50 → 25 BTC
    '2016-07-09',  # Second halving: 25 → 12.5 BTC
    '2020-05-11',  # Third halving: 12.5 → 6.25 BTC
    '2024-04-19'   # Fourth halving: 6.25 → 3.125 BTC
]
# ============================

import pandas as pd
import numpy as np
import talib

def calculate_all_features(bitcoin_historical_data, fear_greed_index_data):
    """
    Calculate ALL features in one go - both for visualization and machine learning
    
    This function:
    1. Calculates technical indicators (RSI, MACD, Bollinger Bands)
    2. Adds Fear & Greed Index
    3. Creates ML features (moving averages, volatility, momentum)
    4. Calculates Cycle Low Multiple (halving cycle indicator)
    
    Args:
        bitcoin_historical_data: DataFrame with OHLCV data (multi-index columns)
        fear_greed_index_data: DataFrame with Fear & Greed Index values
    
    Returns:
        DataFrame with all original data plus calculated features
    """
    bitcoin_data_copy = bitcoin_historical_data.copy()
    bitcoin_close_prices = bitcoin_data_copy[('Close', 'BTC-USD')].values
    
    # RSI - measures if Bitcoin is overbought (>70) or oversold (<30)
    bitcoin_data_copy['RSI'] = talib.RSI(bitcoin_close_prices, timeperiod=rsi_period)
    
    # MACD - shows momentum and trend direction
    macd_line, macd_signal_line, macd_histogram = talib.MACD(
        bitcoin_close_prices,
        fastperiod=macd_fast_period,
        slowperiod=macd_slow_period,
        signalperiod=macd_signal_period
    )
    bitcoin_data_copy['MACD'] = macd_histogram
    bitcoin_data_copy['MACD_Signal'] = macd_signal_line
    
    # Fear & Greed Index - market sentiment indicator (0=Extreme Fear, 100=Extreme Greed)
    bitcoin_data_copy['Fear_and_Greed_Index'] = fear_greed_index_data.set_index('Date')['value']
    
    # Bollinger Bands - shows price volatility and potential breakouts
    upper_band, middle_band, lower_band = talib.BBANDS(
        bitcoin_close_prices,
        timeperiod=bollinger_bands_period,
        nbdevup=bollinger_bands_std_deviations,
        nbdevdn=bollinger_bands_std_deviations
    )
    bitcoin_data_copy['BB_Upper'] = upper_band
    bitcoin_data_copy['BB_Middle'] = middle_band
    bitcoin_data_copy['BB_Lower'] = lower_band
    
    # Daily price changes
    bitcoin_data_copy['Daily_Return'] = bitcoin_data_copy[('Close', 'BTC-USD')].pct_change()
    bitcoin_data_copy['Log_Return'] = np.log(
        bitcoin_data_copy[('Close', 'BTC-USD')] / bitcoin_data_copy[('Close', 'BTC-USD')].shift(1)
    )
    
    # ===== CYCLE LOW MULTIPLE (Optimized Vectorized Implementation) =====
    # Ratio of current price to minimum price in halving cycle
    # Cycle duration = (current_halving - prev_halving), split symmetrically
    bitcoin_timestamps = bitcoin_data_copy.index
    
    # Convert halving dates and calculate cycle durations
    halving_timestamps = pd.to_datetime(halving_dates).sort_values().values
    num_halvings = len(halving_timestamps)
    
    cycle_durations_days = np.diff(halving_timestamps).astype('timedelta64[D]').astype(int)
    cycle_durations_days = np.concatenate([[1461], cycle_durations_days])  # First cycle: 4 years default
    
    days_before_halving = np.ceil(cycle_durations_days / 2).astype(int)
    days_after_halving = np.floor(cycle_durations_days / 2).astype(int)
    
    cycle_start_dates = halving_timestamps - days_before_halving.astype('timedelta64[D]')
    cycle_end_dates = halving_timestamps + days_after_halving.astype('timedelta64[D]')
    
    # Vectorized cycle membership: broadcast to find which dates belong to which cycles
    timestamps_2d = bitcoin_timestamps.values[:, np.newaxis]  # Shape: (n_dates, 1)
    cycle_start_2d = cycle_start_dates[np.newaxis, :]  # Shape: (1, n_cycles)
    cycle_end_2d = cycle_end_dates[np.newaxis, :]  # Shape: (1, n_cycles)
    date_in_cycle_mask = (timestamps_2d >= cycle_start_2d) & (timestamps_2d <= cycle_end_2d)
    
    # Initialize result
    cycle_low_multiple_values = np.full(len(bitcoin_close_prices), np.nan)
    
    # Calculate cycle low multiple for each cycle
    for cycle_index in range(num_halvings):
        dates_in_current_cycle = date_in_cycle_mask[:, cycle_index]
        
        if not dates_in_current_cycle.any():
            continue
        
        prices_in_current_cycle = bitcoin_close_prices[dates_in_current_cycle]
        cycle_minimum_price = np.nanmin(prices_in_current_cycle)
        
        cycle_low_multiple_values[dates_in_current_cycle] = (
            prices_in_current_cycle / cycle_minimum_price
        )
    
    bitcoin_data_copy['Cycle_Low_Multiple'] = cycle_low_multiple_values
    # ===== END CYCLE LOW MULTIPLE =====
    
    # Moving averages (trend indicators for different time periods)
    moving_average_windows = [7, 14, 30, 50, 200]  # 1 week to ~7 months
    for num_days in moving_average_windows:
        # Simple Moving Average (equal weight to all days)
        bitcoin_data_copy[f'SMA_{num_days}'] = bitcoin_data_copy[('Close', 'BTC-USD')].rolling(window=num_days).mean()
        # Exponential Moving Average (more weight to recent days)
        bitcoin_data_copy[f'EMA_{num_days}'] = bitcoin_data_copy[('Close', 'BTC-USD')].ewm(span=num_days).mean()
    
    # Volatility measures (how much price fluctuates)
    bitcoin_data_copy['Volatility_7day'] = bitcoin_data_copy['Daily_Return'].rolling(window=7).std()
    bitcoin_data_copy['Volatility_30day'] = bitcoin_data_copy['Daily_Return'].rolling(window=30).std()
    
    # Volume indicators (trading activity level)
    bitcoin_data_copy['Volume_SMA_7day'] = bitcoin_data_copy[('Volume', 'BTC-USD')].rolling(window=7).mean()
    bitcoin_data_copy['Volume_SMA_30day'] = bitcoin_data_copy[('Volume', 'BTC-USD')].rolling(window=30).mean()
    bitcoin_data_copy['Volume_Daily_Change'] = bitcoin_data_copy[('Volume', 'BTC-USD')].pct_change()
    
    # Momentum indicators (rate of price change)
    momentum_windows = [7, 14, 30]
    for num_days in momentum_windows:
        # Absolute momentum (price difference)
        bitcoin_data_copy[f'Momentum_{num_days}day'] = (
            bitcoin_data_copy[('Close', 'BTC-USD')] - bitcoin_data_copy[('Close', 'BTC-USD')].shift(num_days)
        )
        # Rate of Change (percentage change)
        bitcoin_data_copy[f'ROC_{num_days}day'] = bitcoin_data_copy[('Close', 'BTC-USD')].pct_change(periods=num_days)
    
    # Bollinger Band Width (measures volatility)
    bitcoin_data_copy['BB_Width'] = (bitcoin_data_copy['BB_Upper'] - bitcoin_data_copy['BB_Lower']) / bitcoin_data_copy['BB_Middle']
    
    return bitcoin_data_copy
