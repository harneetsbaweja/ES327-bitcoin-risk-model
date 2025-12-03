# Configuration constants
INDICATORS = [('Close', 'BTC-USD'), 'RSI', 'MACD', 'MACD_Signal', 'Fear_and_Greed_Index']

HALVING_DATES = [
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-19'
]

INDICATOR_COLORS = {
    ('Close', 'BTC-USD'): 'blue',
    'RSI': 'red',
    'MACD': 'green',
    'MACD_Signal': 'orange',
    'Fear_and_Greed_Index': 'purple'
}

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
