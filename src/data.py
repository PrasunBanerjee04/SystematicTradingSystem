from . import utils
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone

## ============ STATISTICAL ARBITRAGE / PAIRS TRADING ============ ## 

def get_latest_candles(session, symbol, n_candles=1000, category="linear", interval='15'):
    try:
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

        response = session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            end=end_ts,
            limit=n_candles
        )
        if response.get('retCode') != 0:
            print(f"API Error: {response.get('retMsg', 'Unknown error')}")
            return pd.DataFrame()

        result_list = response.get('result', {}).get('list', [])
        if not result_list:
            print(f"No data returned from API. Is the interval '{interval}' valid?")
            return pd.DataFrame()

        columns = ['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover']
        df = pd.DataFrame(result_list, columns=columns)

        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col])
        df['startTime'] = pd.to_datetime(df['startTime'].astype(int), unit='ms', utc=True)
        df = df.iloc[::-1].reset_index(drop=True)
        df.set_index('startTime', inplace=True)
        print(f"Successfully fetched {len(df)} candles: {interval} minute interval.")
        return df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def get_current_price(session, symbol, category="linear"):
    try:
        response = session.get_tickers(category=category, symbol=symbol)
        if response.get('retCode') != 0:
            print(f"API Error: {response.get('retMsg', 'Unknown error')}")
            return None
        result_list = response.get('result', {}).get('list', [])
        if result_list:
            last_price = result_list[0].get('lastPrice')
            return float(last_price)
        else:
            print(f"Error: No ticker data returned for symbol {symbol}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def stat_arb_data(session, symbol1, symbol2, category="linear", W=1000, I='15'):
    CACHE_DIR = "../cache"
    CACHE_FILE = os.path.join(CACHE_DIR, "close.csv")

    current_price_A = get_current_price(session, symbol1)
    current_price_B = get_current_price(session, symbol2)
    
    use_cache = os.path.exists(CACHE_FILE)

    if use_cache:
        close = pd.read_csv(CACHE_FILE, index_col='startTime', parse_dates=True)
        if close.index.tz is None:
            close.index = close.index.tz_localize('UTC')
        latest_cached_time = close.index[-1]
        current_utc_time = datetime.now(timezone.utc)
        
        if (current_utc_time - latest_cached_time) > timedelta(minutes=20):
            print("Cache is stale (>10 minutes old). Triggering a full data refresh.")
            use_cache = False

    if use_cache:
        print("Cache found and is fresh. Updating with the latest candle...")
        last_candle_A = get_latest_candles(session, symbol1, category=category, n_candles=1, interval=I)
        last_candle_B = get_latest_candles(session, symbol2, category=category, n_candles=1, interval=I)

        if not (last_candle_A.empty or last_candle_B.empty):
            new_timestamp = last_candle_A.index[0]
            if new_timestamp > close.index[-1]:
                new_row = pd.DataFrame({
                    symbol1: [last_candle_A['closePrice'].iloc[0]],
                    symbol2: [last_candle_B['closePrice'].iloc[0]]
                }, index=[new_timestamp])
                new_row.index.name = 'startTime'
                close = pd.concat([close.iloc[1:], new_row])
    else:
        print("No cache found or cache was stale. Performing initial full data fetch...")
        rolling_window_A = get_latest_candles(session, symbol1, n_candles=W, interval=I)
        rolling_window_B = get_latest_candles(session, symbol2, n_candles=W, interval=I)

        if rolling_window_A.empty or rolling_window_B.empty:
            print("Error: Could not retrieve initial historical data.")
            return current_price_A, current_price_B, pd.DataFrame()
            
        close_A = rolling_window_A['closePrice']
        close_B = rolling_window_B['closePrice']
        close = pd.concat([close_A, close_B], axis=1)
        close.columns = [symbol1, symbol2]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    close.to_csv(CACHE_FILE)
    print("Cache file has been updated.")
    return current_price_A, current_price_B, close

## =============================================================== ## 

