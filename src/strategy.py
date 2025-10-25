import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import statsmodels.api as sm

from . import data 

## ============ STATISTICAL ARBITRAGE / PAIRS TRADING ============ ## 

def stat_arb_strategy_sm(df, current_price_1, current_price_2, symbol1, symbol2, entry_threshold=2.5, exit_threshold=0.2):
    if df.isnull().values.any() or df.empty:
        print("Warning: DataFrame contains NaN values or is empty. No signal generated.")
        return None, None
 
    Y = df[symbol2]
    X = sm.add_constant(df[symbol1])

    model = sm.OLS(Y, X).fit()
    alpha = model.params['const']
    beta = model.params[symbol1]
    historical_residuals = model.resid
    mean_resid = historical_residuals.mean()
    std_resid = historical_residuals.std()

    if std_resid == 0:
        print("Warning: Standard deviation of residuals is zero. Cannot calculate z-score.")
        return None, None
    current_residual = current_price_2 - (alpha + beta * current_price_1)
    current_z_score = (current_residual - mean_resid) / std_resid

    signal = None
    if current_z_score > entry_threshold:
        signal = -1 
    elif current_z_score < -entry_threshold:
        signal = 1 
    elif abs(current_z_score) < exit_threshold:
        signal = 0

    return signal, current_z_score
## =============================================================== ## 