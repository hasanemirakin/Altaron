import pandas as pd
import numpy as np
from scipy.stats import linregress
from qntml.base.__feature_extractor import FeatureExtractor

class StructuralBreakFE(FeatureExtractor):

    def __init__(
            self,
            fe_config: dict,
            **kwargs
    ):
        
        super().__init__(fe_config=fe_config, **kwargs)
    
    def cusum_filter(
            self,
            df,
            source="LogPrices",
            expected_value="previous",
            event_threshold=0.1,
            event_mult=5.,
    ):
        
        data = df.copy()

        if expected_value not in ["mean", "min", "previous"]:
            assert(isinstance(expected_value, float) and expected_value >= 0.), "Wrong arg"
            expected = pd.Series([expected_value for x in range(len(df))])
        else:
            expected = {
                "mean": data[source].expanding().mean(),
                "min": data[source].expanding().min(),
                "previous": data[source].copy(),
            }[expected_value]

        if event_threshold == "std":
            assert(isinstance(event_mult, float) and event_mult > 0), "Wrong arg"
            event_threshold = data[source].expanding().std()*event_mult
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
            self,
            df,
            target="LogPrice",
            features=[],
            k = 10,    
    ):
        
        data = df.copy()
        
        assert(isinstance(features, list) and features != []), "Provide features to run regression"
        X = data[features].values
        square_X = np.linalg.inv(np.matmul(np.transpose(X),X))

        betas = [[np.nan for x in range(len(features))] for i in range(k-1)]

        y = data[target].iloc[:k].values
        x = X[:k]
        betas.append(np.linalg.lstsq(x, y, rcond=None)[0])

        omegas = []
        
        for i in range(k,len(data)):
            y = data[target].iloc[:i+1].values
            x = X[:i+1]

            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            errs = y - np.matmul(x, beta)
            err_var = np.var(errs)
            
            x_t = x[-1, :]
            y_t = y[-1]

            f = err_var*(1 + np.matmul(np.matmul(x_t, square_X), np.transpose(x_t)))

            omega = (y_t - np.matmul(x_t, betas[-1]))/np.sqrt(f)

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
                pd.DataFrame(
                            np.array(betas).reshape(len(df), len(features)), 
                            index=data.index, 
                            columns=[f"BETA_{i}" for i in range(len(features))]
                        )
            ), axis=1
        )

    def csw_cusum_test(
            self,
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

class MicroStructuralFE(FeatureExtractor):

    def __init__(
            self,
            fe_config: dict,
            **kwargs,
    ):
        """FML PART 4"""
        super().__init__(fe_config=fe_config, **kwargs)

    def parkinson_hl_volatility_estimator(
            self, 
            df, 
            window=20
    ):

        data = df.copy()

        k1 = 4*np.log(2)
        k2 = np.sqrt(8/np.pi)

        k1_vol = np.sqrt((np.square(np.log(data["High"]/data["Low"]))).rolling(window).mean()/k1)
        k2_vol = np.log(data["High"]/data["Low"]).rolling(window).mean()/k2

        return pd.concat((k1_vol, k2_vol), axis=1)
    
    def corwin_schultz_spread_estimator(
            self,
            df,
            beta_length=2
    ):
        
        data = df.copy()

        beta = np.square(np.log(data["High"]/data["Low"])).rolling(2).sum()
        beta = beta.rolling(beta_length).mean()

        gamma = np.square(np.log(data["High"].rolling(2).max()/data["Low"].rolling(2).min()))
        
        den = 3 - np.sqrt(8)
        alpha = np.sqrt(beta)*(np.sqrt(2)-1)/den - np.sqrt(gamma/den)
        alpha[alpha < 0] = 0

        estimated_spread = 2*(np.exp(alpha) - 1)/(1+np.exp(alpha))

        return estimated_spread