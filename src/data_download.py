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
chache_dir = 'data_cache'
cache_file_name = os.path.join(chache_dir, 'bitcoin_data.pkl')
start_date = pd.to_datetime('2014-09-17') #First available date: 2014-09-17
end_date = pd.to_datetime('2026-01-01')
drop_cols = [
    'AssetCompletionTime', 'AssetEODCompletionTime',
    'BlkCnt', 'IssTotNtv', 'IssTotUSD', 'SplyExpFut10yr', # deterministic
    'FlowInExUSD', 'FlowOutExUSD', 'SplyExUSD', # redundant with Native values
    'PriceBTC',                        # always 1
    'PriceUSD','ROI30d', 'volume_reported_spot_usd_1d', # already have it
    'ReferenceRate', 'ReferenceRateUSD', 'ReferenceRateETH', 'ReferenceRateEUR', # empty util 21/06/2019
    'CapMrktEstUSD' # empty until 22/06/2019
]
archive_url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

os.makedirs(chache_dir, exist_ok=True)

# ============ DOWNLOAD ============
if __name__ == '__main__':
    # Download Bitcoin price history
    bitcoin_price_history = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    
    # Flatten MultiIndex if present
    if isinstance(bitcoin_price_history.columns, pd.MultiIndex):
        bitcoin_price_history.columns = bitcoin_price_history.columns.get_level_values(0)
        
    # Download Fear and Greed Index data
    fgi = FearAndGreedIndex()
    fgi_raw = fgi.get_historical_data(start_date)
    fgi_df = pd.DataFrame(fgi_raw)
    fgi_df['Date'] = pd.to_datetime(fgi_df['timestamp'].astype(int), unit='s')
    # Convert Fear and Greed values to integers (0-100 scale)
    fgi_df['value'] = fgi_df['value'].astype(int)

    # On-chain data from Coin Metrics
    metrics_df = pd.read_csv(archive_url, index_col='time', parse_dates=True, low_memory=False)
    metrics_df = metrics_df[metrics_df.index >= start_date].reset_index()
    metrics_df = metrics_df.drop(columns=drop_cols)
    
# ============ SAVE CACHE ============
    cache_data = {
        'bitcoin_price_history': bitcoin_price_history,
        'fgi_data': fgi_df,
        'metrics_data': metrics_df,
        'start_date': start_date,
        'end_date': end_date,
        'cache_date': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    pd.to_pickle(cache_data, cache_file_name)
    print(f"    Cached to: {cache_file_name}")
    print(f"    Range: {start_date.date()} to {end_date.date()}")