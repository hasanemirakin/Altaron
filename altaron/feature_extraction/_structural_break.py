import pandas as pd
import numpy as np
from scipy.stats import linregress

def cusum_filter(
        df,
        source="LogPrice",
        expected_value="previous",
        event_threshold=0.1,
        event_mult=5.,
        roll = 0,
):
    
    data = df.copy()

    if expected_value not in ["mean", "min", "previous"]:
        assert(isinstance(expected_value, float) and expected_value >= 0.), "Wrong arg"
        expected = pd.Series([expected_value for x in range(len(df))])
    else:
        if roll == 0:
            expected = {
                "mean": data[source].expanding().mean(),
                "min": data[source].expanding().min(),
                "previous": data[source].copy(),
            }[expected_value]
        else:
            expected = {
                "mean": data[source].rolling(roll).mean(),
                "min": data[source].rolling(roll).min(),
                "previous": data[source].copy(),
            }[expected_value]

    if event_threshold in ["std", "mean", "previous"]:
        assert(isinstance(event_mult, float) and event_mult > 0), "Wrong arg mult"
        if roll == 0:
            event_threshold = {
                "mean": data[source].expanding().mean(),
                "std": data[source].expanding().std(),
                "previous": data[source].copy(),
            }[event_threshold]*event_mult
        else:
            event_threshold = {
                "mean": data[source].rolling(roll).mean(),
                "std": data[source].rolling(roll).std(),
                "previous": data[source].copy(),
            }[event_threshold]*event_mult
    else:
        assert(isinstance(event_threshold, float) and event_threshold > 0.), "Wrong arg"
        event_threshold = pd.Series([event_threshold for x in range(len(df))])

    s_pos, s_neg, events = [0],[0],[0]

    for i in range(1,len(data)):
        y_t = data[source].iloc[i]
        e_t = expected.iloc[i-1]

        cur_pos = max(0, s_pos[-1] + y_t-e_t)
        cur_neg = min(0, s_neg[-1] + y_t-e_t)

        if cur_neg < -event_threshold.iloc[i]:
            cur_neg = 0
            events.append(-1)
        
        elif cur_pos > event_threshold.iloc[i]:
            cur_pos = 0
            events.append(1)

        else:
            events.append(0)

        s_pos.append(cur_pos)
        s_neg.append(cur_neg)

    return pd.concat(
            (
            pd.Series(s_pos, index=data.index),
            pd.Series(s_neg, index=data.index),
            pd.Series(events, index=data.index),
        ), axis=1
    )

def bde_cusum_test(
        df,
        target="LogPrice",
        features=[],
        k = 10,    
):

    data = df.copy()
    data = data.fillna(method="bfill")
    
    assert(isinstance(features, list) and features != []), "Provide features to run regression"

    X = data[features].values
    square_X = np.linalg.inv(np.matmul(np.transpose(X),X))

    betas = [[np.nan for x in range(len(features))] for i in range(k-1)]

    y = data[target].iloc[:k].values
    x = X[:k]

    try:
        betas.append(np.linalg.lstsq(x, y, rcond=None)[0])
    except:
        betas.append(betas[-1])

    omegas = []
    
    for i in range(k,len(data)):
        y = data[target].iloc[:i+1].values
        x = X[:i+1]

        try:
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
        except:
            beta = betas[-1]
            
        errs = y - np.matmul(x, beta)
        err_var = np.var(errs)
        
        x_t = x[-1, :]
        y_t = y[-1]

        f = err_var*(1 + np.matmul(np.matmul(x_t, square_X), np.transpose(x_t)))

        omega = (y_t - np.matmul(x_t, betas[-1]))/(np.sqrt(f) + 1e-6)

        betas.append(beta)
        omegas.append(omega)
    
    omegas = np.array(omegas)
    std_w = omegas.std()

    cusum_series = [np.nan for i in range(k)]

    for i in range(k, len(data)):
        cusum_series.append(np.sum(omegas[i-k:i+1]/std_w))

    return pd.concat(
            (
            pd.Series(cusum_series, index=data.index),
            #####Betas are unstable in multiprocessing, should be able to fix later ####
            #pd.DataFrame(
            #            np.array(betas).reshape(len(df), len(features)), 
            #            index=data.index, 
            #            columns=[f"BETA_{i}" for i in range(len(features))]
            #        )
        ), axis=1
    )

def csw_cusum_test(
    df,
    source="LogPrice",
    max_reference_lookback=100,
):

    data = df.copy()
    diffed_series = data[source] - data[source].shift(1)

    critical_series = np.zeros(len(data))*np.nan
    cusum_series = np.zeros(len(data))*np.nan

    for n in range(0, len(data)-1):
        for i in range(n+1, n+max_reference_lookback):  
            if i >= len(df):
                break  
            std_t = np.sqrt(np.nansum(np.square(diffed_series.iloc[:i+1].values))/(i) + 1e-6)
        
            s_nt = (data[source].iloc[i] - data[source].iloc[n])/(std_t*np.sqrt(i-n))

            cusum_series[i] = max(np.nan_to_num(cusum_series[i], s_nt-1), s_nt)

            if cusum_series[i] == s_nt:
                critical_series[i] = np.sqrt(4.6 + np.log(i-n))
        
    return pd.concat(
            (
            pd.Series(cusum_series, index=data.index),
            pd.Series(critical_series, index=data.index).fillna(method="ffill")
        ), axis=1   
    ) 