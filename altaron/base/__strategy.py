import pandas as pd
import numpy as np
from altaron.base.__base import AltaronBaseClass

class TradingStrategy(AltaronBaseClass):

    def __init__(
            self,
            **kwargs
    ):

        super().__init__(**kwargs)
    
    def get_strategy_out(
            self,
            inputs,
            ticker_positions,
            ohlcv,
    ):
        
        outs = {
            ticker : {
                "decision": {
                    "side": ticker_positions[ticker]["side"],
                    "size": ticker_positions[ticker]["size"]
                },
                "limits": {
                    "lower_limit": None,
                    "upper_limit": None,
                    "time_limit": None
                }
            }
            for ticker in ticker_positions.keys()
        }
        
        return outs
    
    