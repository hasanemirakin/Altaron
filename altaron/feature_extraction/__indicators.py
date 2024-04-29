import pandas as pd
import numpy as np

def SMA(
        df,
        source="Close",
        window=20
):
    
    data = df.copy()

    return data[source].rolling(window).mean()

def EMA(
        df: pd.DataFrame,
        source="Close",
        window=20,
        adjust=False
):
    
    return df[source].ewm(span=window, adjust=adjust).mean()

def VWMA(
        df,
        source="Close",
        window=20
):

    data = df.copy()

    return (data[source]*data["Volume"]).rolling(window).mean()/data["Volume"].rolling(window).mean()

def AvgBarRange(
        df,
        window=20
):
    
    data = df.copy()

    return (100*(data["High"]/data["Low"] - 1)).rolling(window).mean()

def ATR(
        df,
        window=12
):

    data = df.copy()

    tr = np.where(
            abs(data["High"]/data["Low"] - 1) - abs(data["Returns"]-1) >= 0,
            abs(data["High"]/data["Low"] - 1),
            abs(data["Returns"]-1)
    )

    return pd.Series(tr, index=data.index).rolling(window).mean()

def ATRBands(
        df,
        source="MidPrice",
        window=12,
        atr_mult=3
):
    
    data = df.copy()

    atr = ATR(data, window=window)
    upper_band = data[source]*(1+atr_mult*atr)
    lower_band = data[source]*(1-atr_mult*atr)
    
    return pd.concat((upper_band, lower_band), axis=1)

def SuperTrend(
        df,
        source="MidPrice",
        window=12,
        atr_mult=3
):
    
    data = df.copy()

    bands = ATRBands(data, source=source, window=window, atr_mult=atr_mult)

    u,l = list(bands.columns)
    
    upper_trend = []
    lower_trend = []

    super_trend = []
    
    trend = 1

    for i in range(len(bands)):
        upper = bands[u].iloc[i]
        lower = bands[l].iloc[i]
        c_pr = data["Close"].iloc[i-1]

        if np.isnan(upper):
            upper_trend.append(np.nan)
            lower_trend.append(np.nan)
            super_trend.append(np.nan)
            continue
        
        if (upper < np.nan_to_num(upper_trend[-1], nan=upper) 
            or 
            c_pr > np.nan_to_num(upper_trend[-1], nan=upper)):
            upper_trend.append(upper)
        else:
            upper_trend.append(np.nan_to_num(upper_trend[-1], nan=upper))

        if (lower > np.nan_to_num(lower_trend[-1], nan=lower) 
            or 
            c_pr < np.nan_to_num(lower_trend[-1], nan=lower)):
            
            lower_trend.append(lower)
        else:
            lower_trend.append(np.nan_to_num(lower_trend[-1], nan=lower))

        if trend == 1 and data["Close"].iloc[i] < lower_trend[-1]:
            trend = 0
        elif trend == 0 and data["Close"].iloc[i] > upper_trend[-1]:
            trend = 1
        
        if bool(trend):
            super_trend.append(lower_trend[-1])
        else:
            super_trend.append(upper_trend[-1])
        
    return pd.Series(np.array(super_trend, dtype=np.float32), index=data.index)