import pandas as pd
import numpy as np

def parkinson_hl_volatility_estimator(
        df,
        h_source="High",
        l_source="Low", 
        window=20
):

    data = df.copy()

    k1 = 4*np.log(2)
    k2 = np.sqrt(8/np.pi)

    k1_vol = np.sqrt((np.square(np.log(data[h_source]/data[l_source]))).rolling(window).mean()/k1)
    k2_vol = np.log(data[h_source]/data[l_source]).rolling(window).mean()/k2

    return pd.concat((k1_vol, k2_vol), axis=1)

def corwin_schultz_spread_estimator(
        df,
        h_source="High",
        l_source="Low", 
        beta_length=2
):
    
    data = df.copy()

    beta = np.square(np.log(data[h_source]/data[l_source])).rolling(2).sum()
    beta = beta.rolling(beta_length).mean()

    gamma = np.square(np.log(data[h_source].rolling(2).max()/data[l_source].rolling(2).min()))
    
    den = 3 - np.sqrt(8)
    alpha = np.sqrt(beta)*(np.sqrt(2)-1)/den - np.sqrt(gamma/den)
    alpha[alpha < 0] = 0

    estimated_spread = 2*(np.exp(alpha) - 1)/(1+np.exp(alpha))

    return estimated_spread