"""
Download Data
Historical price and Fear and Greed Index, then caches them locally for future use.
"""

import yfinance as yf
import pandas as pd
from fear_and_greed import FearAndGreedIndex
from datetime import datetime
import os

# ============ CONFIG ============
CACHE_DIR = 'data_cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'bitcoin_data.pkl')
START_DATE = pd.to_datetime('2017-01-01')
END_DATE = pd.to_datetime('2025-11-05')

os.makedirs(CACHE_DIR, exist_ok=True)

# ============ DOWNLOAD ============
if __name__ == '__main__':
    print("Downloading Bitcoin price data.")
    btc_data = yf.download('BTC-USD', start=START_DATE, end=END_DATE, progress=False)

    print("Downloading Fear and Greed Index.")
    fgi = FearAndGreedIndex()
    fgi_raw = fgi.get_historical_data(START_DATE)
    fgi_df = pd.DataFrame(fgi_raw)
    fgi_df['Date'] = pd.to_datetime(fgi_df['timestamp'].astype(int), unit='s')

    # Convert Fear and Greed values to integers (0-100 scale)
    fgi_df['value'] = fgi_df['value'].astype(int)

    # ============ SAVE CACHE ============
    cache_data = {
        'bitcoin_data': btc_data,
        'fgi_data': fgi_df,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'cache_date': datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    pd.to_pickle(cache_data, CACHE_FILE)
    print(f"    Cached to: {CACHE_FILE}")
    print(f"    Timestamp: {cache_data['cache_date']}")
    print(f"    Range: {START_DATE.date()} to {END_DATE.date()}")