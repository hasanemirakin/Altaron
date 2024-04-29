import pandas as pd
import numpy as np
import datetime
import os
from altaron.base.__strategy import TradingStrategy
from altaron.base.__data_processor import DataProcessor
from altaron.base.__backtest import BackTestEnvironment
from altaron.mpengine import (
    prepare_jobs,
    process_jobs,
    expand_call_by_dates,
    combine_outputs_update_dictionary
)

class WalkForwardTest(BackTestEnvironment):

    def __init__(
            self,
            strategy: TradingStrategy,
            processor: DataProcessor,
            capital = 1000,
            min_bet_size = 0.01,
            max_bet_amount = 1000,
            fee = 0.0005,
            **kwargs
    ): 
        
        super().__init__(
            strategy=strategy,
            processor=processor,
            capital=capital,
            min_bet_size=min_bet_size,
            max_bet_amount=max_bet_amount,
            fee=fee,
            **kwargs
        )
    
    def __prep_date_inputs(
            self,
            start,
            end
    ):

        dates = [
            self.processor.get_index_date(ind)
            for ind in range(start, end)
        ]

        jobs = prepare_jobs(
            func=self.processor.get_date_inputs,
            data=dates,
            args={"get_labels": False},
            num_threads=None, #function will infer,
            linear_split=True
        )

        date_inputs = process_jobs(
            jobs=jobs,
            call_expansion=expand_call_by_dates,
            output_combination=combine_outputs_update_dictionary,
            num_threads=None, #function will infer
        )

        return date_inputs

    def run_backtest(
            self,
            start_date=None,
            end_date=None
    ):
        
        main_ticker = self.processor.data_dict[self.tickers[0]].copy()

        if start_date is not None:
            start_index = self.processor.get_date_index(
                main_ticker,
                date=start_date,
                earlier=False
            )

            if start_index < self.processor.cfg[self.tickers[0]]["feature_window"] - 1:
                start_index = self.processor.cfg[self.tickers[0]]["feature_window"] - 1
        else:
            start_index = self.processor.cfg[self.tickers[0]]["feature_window"] - 1

        if end_date is not None:
            end_index = self.processor.get_date_index(
                main_ticker,
                date=end_date,
                earlier=True
            )
            
            if end_index > len(main_ticker):
                end_index = len(main_ticker)
        else:
            end_index = len(main_ticker)

        if not hasattr(self, "date_inputs"):
            print("Prepping Inputs...")
            self.date_inputs = self.__prep_date_inputs(
                start=start_index,
                end=end_index
            )

        return super().run_backtest(
            self.date_inputs,
            start_index,
            end_index
        )