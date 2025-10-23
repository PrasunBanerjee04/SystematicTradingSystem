import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import glob
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LassoCV
from pybit.unified_trading import HTTP
from itertools import combinations
from datetime import datetime, timezone
from typing import Union

def plot_prices(X, Y, relative=True):
    first_index = X.index[0]
    if relative:
        plt.plot(X / X[first_index])
        plt.plot(Y / Y[first_index])
    else:
        plt.plot(X)
        plt.plot(Y)

def fetch_bybit_klines(session, symbol, start_date_str, end_date_str, category="linear", interval=1):
    start_ts = int(datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_data = []
    current_end_ts = end_ts
    last_oldest_timestamp = None
    print(f"Fetching {symbol} {interval}-minute klines from {start_date_str} to {end_date_str}...")
    while True:
        try:
            response = session.get_kline(
                category=category,
                symbol=symbol,
                interval=interval,
                end=current_end_ts,
                limit=1000
            )

            if response['retCode'] != 0:
                print(f"API Error: {response['retMsg']}")
                break

            result_list = response['result']['list']
            if not result_list:
                print("No more data found.")
                break

            all_data.extend(result_list)
            oldest_timestamp_in_batch = int(result_list[-1][0])
            
            print(f"Fetched {len(result_list)} candles. Oldest candle at: {datetime.fromtimestamp(oldest_timestamp_in_batch / 1000, tz=timezone.utc)}")

            if last_oldest_timestamp == oldest_timestamp_in_batch:
                print("Stuck on the same timestamp. Reached the beginning of available data.")
                break
            
            if oldest_timestamp_in_batch < start_ts:
                print("Reached desired start date.")
                break

            current_end_ts = oldest_timestamp_in_batch
            last_oldest_timestamp = oldest_timestamp_in_batch
            time.sleep(0.2)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(10)
            
    if not all_data:
        print("Could not fetch any data.")
        return pd.DataFrame()

    columns = ['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover']
    df = pd.DataFrame(all_data, columns=columns)
    numeric_cols = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    df['startTime'] = pd.to_datetime(df['startTime'].astype(int), unit='ms', utc=True)
    df = df.iloc[::-1]
    df.drop_duplicates(subset='startTime', inplace=True)
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    df = df[(df['startTime'] >= start_dt) & (df['startTime'] < end_dt)]
    df.set_index('startTime', inplace=True) 
    print(f"\nSuccessfully fetched a total of {len(df)} candles.")
    return df

def regression(X,Y):
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    residuals = results.resid
    alpha, beta = results.params
    return residuals, alpha, beta

def lasso_regression_sm(X, Y, alpha):
    X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(Y, X)
    results = model.fit_regularized(
        method='elastic_net', 
        alpha=alpha, 
        L1_wt=1.0
    )
    intercept = results.params['const']
    beta = results.params.drop('const')
    residuals = Y - results.predict(X)
    return residuals, intercept, beta

def find_optimal_alpha(X, y):
    if isinstance(X, pd.Series):
        X = X.to_frame()
    lasso_cv = LassoCV(cv=10, random_state=42)
    lasso_cv.fit(X, y)
    return lasso_cv.alpha_

def dataframe_subset(df, start=None, end=None):
    if start and not end:
        return df.loc[start:]
    elif not start and end:
        return df.loc[:end]
    return df.loc[start:end]

def print_adf_test(series):
    result = adfuller(series)  
    test_statistic = result[0]
    p_value = result[1]
    lags_used = result[2]
    critical_values = result[4]
    print("\nAugmented Dickey-Fuller (ADF) Test")
    print(f"Test Statistic: {test_statistic:.4f}")
    print(f"p-value       : {p_value:.4f}")
    print(f"Lags Used     : {lags_used}")
    print("Null Hypothesis: Non-stationary (has unit root)") 
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f'   {key}: {value:.4f}')

def resample_prices(df: pd.DataFrame, rule: str, timestamp_col: Union[str, None] = None, price_cols: Union[list, None] = None, tz: Union[str, None] = None, label: str = "right", closed: str = "right", dropna: bool = True) -> pd.DataFrame:
    if price_cols is None:
        price_cols = ["OPUSDT", "ATOMUSDT"]
    df = df.copy()
    if timestamp_col is not None:
        df.index = pd.to_datetime(df[timestamp_col])
        df = df.drop(columns=[timestamp_col], errors="ignore")
    else:
        df.index = pd.to_datetime(df.index)
    if tz is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)
    present_price_cols = [c for c in price_cols if c in df.columns]
    agg = {c: "last" for c in present_price_cols}
    other_cols = [c for c in df.columns if c not in agg]
    for c in other_cols:
        agg[c] = "last"
    res = df.resample(rule, label=label, closed=closed).agg(agg)
    if dropna and len(present_price_cols) > 0:
        res = res.dropna(subset=present_price_cols, how="all")
    return res