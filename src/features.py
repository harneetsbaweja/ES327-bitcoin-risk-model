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
adx_period = 14
plus_di_period = 14
minus_di_period = 14
atr_period = 14
natr_period = 14
stochrsi_period = 14
stoch_period = 14
stoch_fast_k = 3
stoch_fast_d = 3
willr_period = 14
obv_period = 14
mfi_period = 14
cci_period = 20
cmo_period = 14
kama_period = 10
ultosc_period_1 = 7
ultosc_period_2 = 14
ultosc_period_3 = 28
ppo_fast_period = 12
ppo_slow_period = 26
trix_period = 15
linearreg_period = 20
sar_af = 0.02
sar_af_max = 0.2
bollinger_bands_std_deviations = 2
ewm_span = 100
include_fgi = False  # Whether to include Fear & Greed Index as a feature

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

def calculate_all_features(bitcoin_historical_data, fear_greed_index_data, downloaded_metrics):
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
    high = bitcoin_data_copy['High'].values.astype(np.float64)
    low = bitcoin_data_copy['Low'].values.astype(np.float64)
    close = bitcoin_data_copy['Close'].values.astype(np.float64)
    volume = bitcoin_data_copy['Volume'].values.astype(np.float64)
    
    # Fear & Greed Index - market sentiment indicator (0=Extreme Fear, 100=Extreme Greed)
    if include_fgi:
        bitcoin_data_copy['Fear_and_Greed_Index'] = fear_greed_index_data.set_index('Date')['value']
        
    # Coin Metrics loaded metrics
    for column in downloaded_metrics.columns:
        if column != 'time':
            bitcoin_data_copy[column] = downloaded_metrics.set_index('time')[column]
            
    # RSI - measures if Bitcoin is overbought (>70) or oversold (<30)
    bitcoin_data_copy['RSI'] = talib.RSI(close, timeperiod=rsi_period)
    
    # MACD - shows momentum and trend direction
    macd_line, macd_signal_line, macd_histogram = talib.MACD(
        close,
        fastperiod=macd_fast_period,
        slowperiod=macd_slow_period,
        signalperiod=macd_signal_period
    )
    bitcoin_data_copy['MACD'] = macd_histogram
    bitcoin_data_copy['MACD_Signal'] = macd_signal_line
    
    # Bollinger Bands - shows price volatility and potential breakouts
    upper_band, middle_band, lower_band = talib.BBANDS(
        close,
        timeperiod=bollinger_bands_period,
        nbdevup=bollinger_bands_std_deviations,
        nbdevdn=bollinger_bands_std_deviations
    )
    bitcoin_data_copy['BB_Upper'] = upper_band
    bitcoin_data_copy['BB_Middle'] = middle_band
    bitcoin_data_copy['BB_Lower'] = lower_band
    
    # Daily price changes
    bitcoin_data_copy['Daily_Return'] = bitcoin_data_copy['Close'].pct_change()
    bitcoin_data_copy['Log_Return'] = np.log(
        bitcoin_data_copy['Close'] / bitcoin_data_copy['Close'].shift(1)
    )
    
    # Moving averages (trend indicators for different time periods)
    moving_average_windows = [7, 14, 30, 50, 200]  # 1 week to ~7 months
    for num_days in moving_average_windows:
        # Simple Moving Average (equal weight to all days)
        bitcoin_data_copy[f'SMA_{num_days}'] = bitcoin_data_copy['Close'].rolling(window=num_days).mean()
        # Exponential Moving Average (more weight to recent days)
        bitcoin_data_copy[f'EMA_{num_days}'] = bitcoin_data_copy['Close'].ewm(span=num_days).mean()
    
    # Volatility measures (how much price fluctuates)
    bitcoin_data_copy['Volatility_7day'] = bitcoin_data_copy['Daily_Return'].rolling(window=7).std()
    bitcoin_data_copy['Volatility_30day'] = bitcoin_data_copy['Daily_Return'].rolling(window=30).std()
    bitcoin_data_copy['Volatility_EWMA'] = bitcoin_data_copy['Daily_Return'].ewm(span=ewm_span).std()
    
    # Volume indicators (trading activity level)
    bitcoin_data_copy['Volume_SMA_7day'] = bitcoin_data_copy['Volume'].rolling(window=7).mean()
    bitcoin_data_copy['Volume_SMA_30day'] = bitcoin_data_copy['Volume'].rolling(window=30).mean()
    bitcoin_data_copy['Volume_Daily_Change'] = bitcoin_data_copy['Volume'].pct_change()
    
    # Momentum indicators (rate of price change)
    momentum_windows = [7, 14, 30]
    for num_days in momentum_windows:
        # Absolute momentum (price difference)
        bitcoin_data_copy[f'Momentum_{num_days}day'] = (
            bitcoin_data_copy['Close'] - bitcoin_data_copy['Close'].shift(num_days)
        )
        # Rate of Change (percentage change)
        bitcoin_data_copy[f'ROC_{num_days}day'] = bitcoin_data_copy['Close'].pct_change(periods=num_days)
    
    # Bollinger Band Width (measures volatility)
    bitcoin_data_copy['BB_Width'] = (bitcoin_data_copy['BB_Upper'] - bitcoin_data_copy['BB_Lower']) / bitcoin_data_copy['BB_Middle']
    
    # ADX — trend strength (0-100, regardless of direction)
    bitcoin_data_copy['ADX'] = talib.ADX(high, low, close, timeperiod=adx_period)
    bitcoin_data_copy['ADXR'] = talib.ADXR(high, low, close, timeperiod=adx_period)
    bitcoin_data_copy['PLUS_DI']  = talib.PLUS_DI(high, low, close, timeperiod=plus_di_period)
    bitcoin_data_copy['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=minus_di_period)
    bitcoin_data_copy['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=plus_di_period)
    
    # Volatility indicators — how much price moves, regardless of direction
    bitcoin_data_copy['ATR']      = talib.ATR(high, low, close, timeperiod=atr_period)
    bitcoin_data_copy['NATR']     = talib.NATR(high, low, close, timeperiod=natr_period)
    bitcoin_data_copy['TRANGE']   = talib.TRANGE(high, low, close)
    bitcoin_data_copy['STOCHRSI'] = talib.STOCHRSI(close, timeperiod=stochrsi_period)[0]  # fastk
    bitcoin_data_copy['WILLR']    = talib.WILLR(high, low, close, timeperiod=willr_period)
    bitcoin_data_copy['OBV']      = talib.OBV(close, volume)
    
    # Stochastic Oscillator — overbought-oversold (slower than STOCHRSI)
    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=stoch_period, slowk_period=stoch_fast_k, slowd_period=stoch_fast_d)
    bitcoin_data_copy['STOCH_K'] = stoch_k
    bitcoin_data_copy['STOCH_D'] = stoch_d
    
    # Money Flow Index — volume-weighted RSI-like indicator
    bitcoin_data_copy['MFI'] = talib.MFI(high, low, close, volume, timeperiod=mfi_period)
    
    # Accumulation/Distribution Line — volume-weighted price action
    bitcoin_data_copy['AD'] = talib.AD(high, low, close, volume)
    
    # Chaikin A/D Oscillator — short/long EMA of A/D line (volume momentum)
    bitcoin_data_copy['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    
    # Commodity Channel Index — mean reversion / cyclical extremes
    bitcoin_data_copy['CCI'] = talib.CCI(high, low, close, timeperiod=cci_period)
    
    # Parabolic SAR — trend reversal and stop-loss levels
    bitcoin_data_copy['SAR'] = talib.SAR(high, low, acceleration=sar_af, maximum=sar_af_max)
    
    # Kaufman Adaptive Moving Average — noise-adaptive trend smoothing
    bitcoin_data_copy['KAMA'] = talib.KAMA(close, timeperiod=kama_period)
    
    # Adaptive Moving Averages — fast/slow adaptive trend followers
    bitcoin_data_copy['MAMA'], bitcoin_data_copy['FAMA'] = talib.MAMA(close)
    
    # Ultimate Oscillator — multi-timeframe momentum (7, 14, 28 period weighted)
    bitcoin_data_copy['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=ultosc_period_1, timeperiod2=ultosc_period_2, timeperiod3=ultosc_period_3)
    
    # Chande Momentum Oscillator — momentum with extended range
    bitcoin_data_copy['CMO'] = talib.CMO(close, timeperiod=cmo_period)
    
    # Percentage Price Oscillator — normalised MACD (useful for comparing instruments)
    bitcoin_data_copy['PPO'] = talib.PPO(close, fastperiod=ppo_fast_period, slowperiod=ppo_slow_period)
    
    # TRIX — Triple Exponential Moving Average momentum (momentum of momentum)
    bitcoin_data_copy['TRIX'] = talib.TRIX(close, timeperiod=trix_period)
    
    # Linear Regression Slope — trend strength via linear regression coefficient
    bitcoin_data_copy['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=linearreg_period)

    return bitcoin_data_copy
