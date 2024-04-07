import pandas as pd
import numpy as np
from altaron.base.__q_class import QClass

class FeatureExtractor(QClass):

    def __init__(
                self,
                fe_config: dict,
                **kwargs
    ):
        """
        Allow multiple operations for same function
        """
        self.fe_config = fe_config
    
    def apply_feature_extraction(
            self,
            df: pd.DataFrame,
    ):

        data = self.prep_initial_features(df)

        assert(isinstance(self.fe_config, dict)), "fe_config must be a dict"
        
        for k,v in self.fe_config.items():
                        
            assert(isinstance(v, dict)), "Values of fe_config must be a dictionary containing related function arguments"
            assert("name" in v), "Values of fe_config must contain 'name' key to save the function results as"
            
            args = v.copy()
            _ = args.pop("name")
            args["df"] = data

            result = getattr(self, k)(**args)
            if len(result) < len(data):
                nan_arr = np.zeros(shape=(len(data), len(result.columns)))*np.nan
                nan_arr[-len(result):, :] = result.values
                result = pd.DataFrame(nan_arr, index=data.index, columns=result.columns)
    
            data[v["name"]] = result

        return data

    def prep_initial_features(self, df: pd.DataFrame):

        data = df.copy()

        if (len(data[data["Volume"] == 0]) == 0 and
            set(["Returns", "LogReturns", "LogPrice", 
                 "MidPrice", "LogMidPrice", "LogVolume"]
                ).issubset(data.columns)
        ): 
            return data
        
        assert (data.isna().sum().sum() == 0), "Data must not contain na values"
        assert (np.isinf(data).sum().sum() == 0), "Data must not contain inf values"

        non_zero_min_vol = data[data["Volume"] != 0]["Volume"].min()
        
        data["Volume"] = [
                        vol if vol != 0 else non_zero_min_vol 
                        for vol in data["Volume"].values
                        ]

        data["Returns"] = data["Close"]/data["Close"].shift(1)
        data["LogReturns"] = np.log(data["Returns"])
        data["LogPrice"] = np.log(data["Close"])
        data["MidPrice"] = (data["Close"]+data["Open"]+data["High"]+data["Low"])/4
        data["LogMidPrice"] = np.log(data["MidPrice"])
        data["LogVolume"] = np.log(data["Volume"])

        for col in data.columns:
            data[col] = data[col].values.astype(np.float32)

        data = data.dropna()

        return data

    def VWMA(
            self, 
            df,
            source="Close",
            window=20
    ):

        data = df.copy()

        return (data[source]*data["Volume"]).rolling(window).mean()/data["Volume"].rolling(window).mean()
    
    def AvgBarRange(
            self,
            df,
            window=20
    ):
        
        data = df.copy()

        return (100*(data["High"]/data["Low"] - 1)).rolling(window).mean()
    
    def ATR(
            self,
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
            self,
            df,
            source="MidPrice",
            window=12,
            atr_mult=3
    ):
        
        data = df.copy()

        atr = self.ATR(data, window=window)
        upper_band = data[source]*(1+atr_mult*atr)
        lower_band = data[source]*(1-atr_mult*atr)
        
        return pd.concat((upper_band, lower_band), axis=1)
    
    def SuperTrend(
            self,
            df,
            source="MidPrice",
            window=12,
            atr_mult=3
    ):
        
        data = df.copy()

        bands = self.ATRBands(data, source=source, window=window, atr_mult=atr_mult)

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

    def set_allowed_kwargs(self, kwargs: list):
        """This function sets the allowed keyword arguments for the model."""
        self.allowed_kwargs = kwargs

    def set_kwargs(self, kwargs: dict):
        """This function sets the keyword arguments for the model."""
        valid_kwargs = self.check_kwargs(kwargs)
        for k, v in valid_kwargs.items():
            self.__setattr__(k, v)
    
    def check_kwargs(self, kwargs: dict):
        return kwargs   