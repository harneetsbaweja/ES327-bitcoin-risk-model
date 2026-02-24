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
chache_dir = '../data_cache'
cache_file_name = os.path.join(chache_dir, 'bitcoin_data.pkl')
start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2026-01-01')

os.makedirs(chache_dir, exist_ok=True)

# ============ DOWNLOAD ============
if __name__ == '__main__':
    bitcoin_price_history = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)

    fgi = FearAndGreedIndex()
    fgi_raw = fgi.get_historical_data(start_date)
    fgi_df = pd.DataFrame(fgi_raw)
    fgi_df['Date'] = pd.to_datetime(fgi_df['timestamp'].astype(int), unit='s')

    # Convert Fear and Greed values to integers (0-100 scale)
    fgi_df['value'] = fgi_df['value'].astype(int)

    # ============ SAVE CACHE ============
    cache_data = {
        'bitcoin_price_history': bitcoin_price_history,
        'fgi_data': fgi_df,
        'start_date': start_date,
        'end_date': end_date,
        'cache_date': datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    pd.to_pickle(cache_data, cache_file_name)
    print(f"    Cached to: {cache_file_name}")
    print(f"    Range: {start_date.date()} to {end_date.date()}")