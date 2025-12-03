import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .config import INDICATORS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD


def calculate_indicators(bitcoin_historical_data, bitcoin_fgi_data):
    """Calculate all technical indicators"""
    btc_close = bitcoin_historical_data[('Close', 'BTC-USD')].values
    
    # RSI
    bitcoin_historical_data['RSI'] = talib.RSI(btc_close, timeperiod=RSI_PERIOD)
    
    # MACD
    macd_line, macd_signal, macd_hist = talib.MACD(
        btc_close, 
        fastperiod=MACD_FAST, 
        slowperiod=MACD_SLOW, 
        signalperiod=MACD_SIGNAL
    )
    bitcoin_historical_data['MACD'] = macd_hist
    bitcoin_historical_data['MACD_Signal'] = macd_signal
    
    # Fear & Greed Index
    bitcoin_historical_data['Fear_and_Greed_Index'] = bitcoin_fgi_data.set_index('Date')['value']
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        btc_close, 
        timeperiod=BB_PERIOD, 
        nbdevup=BB_STD, 
        nbdevdn=BB_STD
    )
    bitcoin_historical_data['BB_Upper'] = bb_upper
    bitcoin_historical_data['BB_Middle'] = bb_middle
    bitcoin_historical_data['BB_Lower'] = bb_lower
    
    return bitcoin_historical_data


def transform_features(bitcoin_historical_data, indicators=None):
    """Transform features using MinMax and Standard scaling"""
    if indicators is None:
        indicators = INDICATORS
    
    transformed_features = {
        'absolute_values': {}, 
        'normalised_values': {}, 
        'standardised_values': {}
    }
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    for indicator in indicators:
        # Store absolute values
        transformed_features['absolute_values'][indicator] = bitcoin_historical_data[indicator].values
        
        # Get valid mask for non-NaN values
        valid_mask = bitcoin_historical_data[[indicator]].notna().values.flatten()
        
        # Normalise only valid values
        normalised = np.full(len(bitcoin_historical_data), np.nan)
        if valid_mask.any():
            valid_data = bitcoin_historical_data.loc[valid_mask, [indicator]].values
            normalised[valid_mask] = minmax_scaler.fit_transform(valid_data).flatten()
        transformed_features['normalised_values'][indicator] = normalised
        
        # Standardise only valid values
        standardised = np.full(len(bitcoin_historical_data), np.nan)
        if valid_mask.any():
            valid_data = bitcoin_historical_data.loc[valid_mask, [indicator]].values
            standardised[valid_mask] = standard_scaler.fit_transform(valid_data).flatten()
        transformed_features['standardised_values'][indicator] = standardised
    
    return transformed_features, minmax_scaler, standard_scaler
