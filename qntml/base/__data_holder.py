import pandas as pd
import numpy as np
import torch
import datetime
from qntml.base.__q_class import QClass

class DataHolder(QClass):

    def __init__(
            self, 
            data: dict,
            **kwargs
    ):
        
        for v in data.values():
            if not isinstance(v, pd.DataFrame):
                raise TypeError(f"data must be a pandas.DataFrame")
        
        self.data_dict = {k: self.__organize_df(v) for k,v in data.items()}
        
        for k, v in self.data_dict.items():
            v.freq = pd.infer_freq(v)
            self.data_dict[k] = v
    
    def __organize_df(self, df: pd.DataFrame):
        """Method for organizing a possibly unorganized dataframe while also making sure
        all entries are convertible to tensors and keeping track of the operations done
        on the original dataframe"""

        organized_df = df.copy()

        for ind, dtype in enumerate(df.dtypes):
            col = df.dtypes.index[ind]

            if dtype == object:
                try:
                    float_vals = df[col].values.astype(np.float32)
                    organized_df[col] = float_vals

                except:  # noqa: E722
                    try:
                        dates = pd.to_datetime(df[col])
                        organized_df.pop(col)
                        organized_df = self.organized_df.set_index(dates)
                        organized_df.index.name = "Dates"
                        
                    except:  # noqa: E722
                        raise Exception(f"Could not handle entries of column: '{col}' ")
        
        try:
            organized_df = organized_df.set_index(pd.to_datetime(organized_df.index))
            organized_df.index.name = "Dates"
        except:
            raise ValueError("Index for data must be dates")

        return organized_df
    
    def get_nearest_date(
                    self, 
                    df, 
                    date,
                    earlier=True
    ):

        date = pd.to_datetime(date)

        time_deltas = df.index - date

        if earlier:
            match_index = np.where(
                            (abs(time_deltas) == abs(time_deltas).min())
                            & (time_deltas <= pd.to_timedelta("0"))
                            )[0][0]
        else:
            match_index = np.where(
                            (abs(time_deltas) == abs(time_deltas).min())
                            & (time_deltas >= pd.to_timedelta("0"))
                            )[0][0]
        
        return df.index[match_index]
    
    def get_date_index(
            self,
            df,
            date,
            earlier=True
    ):
        
        nearest_match = self.get_nearest_date(df, date, earlier)

        index = np.where(df.index == nearest_match)[0][0]

        return index
    
    def get_window_values(
                        self, 
                        date, 
                        tickers_and_windows = {},
                        match_dates=True
    ):

        if tickers_and_windows == {}:
            tickers_and_windows = {k: 10 for k in list(self.data_dict.keys())}
        
        main_ticker = list(tickers_and_windows.keys())[0]
        date = pd.to_datetime(date)
        
        assert (date in self.data_dict[main_ticker].index), "Date mismatch on main ticker"

        values = {}

        for ticker, window in tickers_and_windows.items():
            df = self.data_dict[ticker].copy()
            matching_date = self.get_nearest_date(df, date, earlier=True)

            index_iloc = np.where(df.index == matching_date)[0][0]
            values[ticker] = df.iloc[index_iloc-window:index_iloc]
        
        return values

    def get_ticker(self, ticker=None):
        
        if ticker is None:
            ticker = list(self.data_dict.keys())[0]

        return self.data_dict[ticker]
    
    def add_ticker(self, ticker, data):

        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas.DataFrame")

        self.data_dict[ticker] = self.__organize_df(data)
    
    def update_data(
            self,
            new_data: dict,
            stack=False
    ):
        
        assert(set(list(new_data.keys())) == set(list(self.data_dict.keys()))), "\
            new_data must contain all keys"

        for v in new_data.values():
            assert(isinstance(v, pd.DataFrame) or isinstance(v, type(None))), "\
                values for new_data must be a pandas.DataFrame or NoneType"
        
        updated = self.data_dict.copy()
        
        for ticker, new_val in new_data.items():
            prev_data = updated[ticker]

            if stack:
                new_data = pd.concat((prev_data, new_data), axis=0)
            else:
                if new_val is None:
                    new_data = prev_data.copy()
                else:
                    new_data = new_val.copy()
            
            updated[ticker] = new_data
        
        self.data_dict.update(updated)
                