import pandas as pd
import numpy as np
from altaron import feature_extraction
from altaron.base.__base import AltaronBaseClass

class FeatureExtractor(AltaronBaseClass):

    def __init__(
                self,
                fe_config: list,
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

        assert(isinstance(self.fe_config, list)), "fe_config must be a list containing dictionaries"
        
        for config in self.fe_config:
                        
            assert(isinstance(config, dict)), "Elements of fe_config must be a dictionary containing related function arguments"
            assert("name" in config.keys()), "Each config must contain 'name' key to save the function results as"
            assert("func" in config.keys()), "Each config must contain 'func' key to specify whic function to apply"

            args = config.copy()
            func = getattr(feature_extraction, args["func"])
            
            #Remove keys not used by function; main two is "name" and "func"
            #And any other non_argument keys
            for arg in config.keys():
                if arg not in func.__code__.co_varnames:
                    _ = args.pop(arg)

            args["df"] = data
            results = func(**args)
            if len(results) < len(data):
                nan_arr = np.zeros(shape=(len(data), len(results.columns)))*np.nan
                nan_arr[-len(results):, :] = results.values
                results = pd.DataFrame(nan_arr, index=data.index, columns=results.columns)

            data[config["name"]] = results

        return data

    def prep_initial_features(self, df: pd.DataFrame):

        data = df.copy()

        if (len(data[data["Volume"] == 0]) == 0 and
            set(["Returns", "LogReturns", "LogPrice", 
                 "MidPrice", "LogMidPrice", "LogVolume"]
                ).issubset(data.columns)
        ): 
            return data
        
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
        
        
