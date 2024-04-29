from binance import Client
from binance.enums import HistoricalKlinesType
import pandas as pd
import numpy as np
import datetime
from altaron.base.__base import AltaronBaseClass

class BinanceDataPuller(AltaronBaseClass):

    def __init__(
            self,
            api_secret,
            api_key,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.client = Client(api_key, api_secret)
    
    def __handle_candles(
            self,
            candles
    ):
        
        cand_arr = np.array(candles)

        data = pd.DataFrame(
            cand_arr[:, [1,2,3,4,5,8]], dtype=float, 
            columns=["Open", "High", "Low", "Close", "Volume", "Num Trades"],
            index = pd.Series(
                cand_arr[:, 0].astype(int)/1000).apply(
                    datetime.datetime.fromtimestamp
                    )
                )
        
        return data

    def get_historical_ohlcv(
            self,
            ticker,
            interval,
            start="2021-01-01 00:00:00",
            end = None,
            market_type="spot",
    ):
        
        assert(market_type in ["spot", "futures"]), "Wrong type arg"

        klines_type = {
            "spot": HistoricalKlinesType.SPOT,
            "futures": HistoricalKlinesType.FUTURES
        }[market_type]

        try:
            pull_tf = {
                "m": "MINUTE",
                "h": "HOUR",
                "d": "DAY",
                "w": "WEEK",
                "M": "MONTH"
            }[interval[-1]]

            pull_interval = "KLINE_INTERVAL_" + interval[:-1] + pull_tf

            candles = self.client.get_historical_klines(
                symbol=ticker,
                interval=getattr(Client, pull_interval),
                klines_type=klines_type,
                start_str=start,
                end_str=end
            )

        except Exception as e:
            raise ValueError(
                repr(e)
            )

        data = self.__handle_candles(candles)

        if end is None:
            #last value of candles will be innacurate since it will be pulled in real time
            #ex 1h OHLCV for 2022-01-01 03:00:00 will be pulled in 2022-01-01 03:17:12
            #which will return different values from when the bar properly closes 
            return data[:-1]
        
        return data
    
    def get_live_ohlcv(
            self,
            ticker,
            interval,
            n_candles=10,
            market_type="spot"
    ):
        
        assert(market_type in ["spot", "futures"]), "Wrong type arg"

        try:
            pull_tf = {
                "m": "MINUTE",
                "h": "HOUR",
                "d": "DAY",
                "w": "WEEK",
                "M": "MONTH"
            }[interval[-1]]

            pull_interval = "KLINE_INTERVAL_" + interval[:-1] + pull_tf

            if market_type == "spot":
                candles = self.client.get_klines(
                    symbol=ticker,
                    interval=getattr(Client, pull_interval),
                    limit=n_candles,
                )
            elif market_type == "futures":
                candles = self.client.futures_klines(
                    symbol=ticker,
                    interval=getattr(Client, pull_interval),
                    limit=n_candles,
                )

        except Exception as e:
            raise ValueError(
                repr(e)
            )

        data = self.__handle_candles(candles)

        return data