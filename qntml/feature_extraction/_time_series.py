import pandas as pd
import numpy as np
from chronokit.decomposition import MSTL
from chronokit.preprocessing import AutoCorrelation
from scipy.signal import find_peaks
from qntml.base.__feature_extractor import FeatureExtractor

class TimeSeriesFE(FeatureExtractor):

    def __init__(
            self,
            fe_config: dict,
            **kwargs
    ):
        """CHRONO-KIT GITHUB"""
        super().__init__(fe_config=fe_config, **kwargs)
    
    def infer_seasonality(
            self,
            df,
            source="LogReturns",
            window = 20,
    ):
        data = df.copy()
        acf = AutoCorrelation(data[source]).acf(window)

        peaks = find_peaks(acf, distance=2)[0]
        if len(peaks) == 0:
            return window
        
        max_peak = acf[peaks].argmax()

        return peaks[max_peak]

    def decomposition(
            self,
            df,
            source="MidPrice",
            seasonalities = [],
            method="add",
            window=100,
    ):
        
        data = df.copy()
        
        if seasonalities == [] or not isinstance(seasonalities, list):
            seasonalities = [self.infer_seasonality(data.iloc[-window:])]

        trend, seasonal, remainder = MSTL(data[source].iloc[-window:].values, seasonal_periods=seasonalities,
                                          method=method)
        
        seasonal_frame = pd.DataFrame(seasonal.transpose(1,0), index=data.index[-window:], columns=[f"s{x}" for x in seasonalities])

        return pd.concat((
            pd.Series(trend, index=data.index[-window:]),
            seasonal_frame,
            pd.Series(remainder, index=data.index[-window:])
            ), axis=1
        )

