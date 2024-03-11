import pandas as pd
import numpy as np
import datetime
from qntml.base.__data_holder import DataHolder
from qntml.base.__q_class import QClass

class DataProcessor(QClass):

    def __init__(
            self,
            config={},
            **kwargs
    ):
        """
        LABELS FML Chapter 3
        
        META LABELING: 1 IF TRADE SUCCESS 0 IF TRADE FAILS
            BINARY CLASSIFIER TO HELP DETERMINE TRADE SIZE
        CAN IMPLEMENT A FUNCTION TO DROP RARE CLASSES
        AS TO NOT INTERFERE WITH MODEL TRAINING 
        """

        assert(isinstance(config, dict)), "Config must be a dict"
        assert(config != {}), "Config must include at least one ticker"

        for k, v in config.items():
            assert(isinstance(v, dict)), f"\
                Values for config key {k} must be a dict, specifying\
                processing config for the key {k}"
            
            assert("data" in list(v.keys())), f"\
                no data found for ticker {k}"

        self.cfg = {
            k: {
                "data": None,
                "feature_extractors": [],
                "features": "all",
                "window": 1,
                "preprocessed": False,
                "label": ["fixed_horizon_label", {"h": 1, "categorical": True, "threshold": 0}]
            }
        }

        for k,v in config.items():
            self.cfg[k].update(v)

        self.data_dict = {
            k: self.__organize_df(self.cfg[k]["data"]) 
            for k in self.cfg.keys()
        }

        for k, v in self.data_dict.items():
            v.freq = pd.infer_freq(v)
            self.data_dict[k] = v
        
        for ticker in self.cfg.keys():

            max_lookback = 0

            for extractor in self.cfg[ticker]["feature_extractors"]:
                cfg = extractor.fe_config

                for v in cfg.values():
                    try:
                        feature_window = v["window"]
                    except:
                        feature_window = 0

                    if feature_window > max_lookback:
                        max_lookback = feature_window
            
            max_lookback += 1
            self.cfg[ticker]["feature_window"] = self.cfg[ticker]["window"] + max_lookback
    
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

                        if organized_df.index.name != "Dates":                
                            organized_df = self.organized_df.set_index(dates)
                            organized_df.index.name = "Dates"
                        else:
                            raise ValueError(
                                f"Found a datetime column, other than index {col}"
                                )
                        
                    except:  # noqa: E722
                        raise Exception(f"Could not handle entries of column: '{col}' ")
        
        try:
            organized_df = organized_df.set_index(pd.to_datetime(organized_df.index))
            organized_df.index.name = "Dates"
        except:
            raise ValueError("Index for data must be dates")

        return organized_df

    def fixed_horizon_label(
            self,
            ticker,
            date,
            h = 1,
            threshold = 0,
            categorical = True
    ):

        h = max(h, 1)

        df = self.get_ticker(ticker)

        date_index = self.get_date_index(df, date, earlier=False)
        #This will thrown an error if date_index + h is out of bounds
        return_t_h = df["Close"].iloc[date_index+h]/df["Close"].iloc[date_index] - 1

        if not categorical:
            return return_t_h

        if return_t_h < -threshold:
            return -1
        elif return_t_h > threshold:
            return 1
        else:
            return 0

    def tripple_barrier_label(
            self,
            ticker,
            date,
            h = 1,
            barrier_width = 0.01,
            tp_mult=1,
            sl_mult=1,
            categorical=True,
            sign_on_vertical=True
    ):

        h = max(0, h)
        barrier_width = max(1e-6, barrier_width)
        tp_mult = max(0, tp_mult)
        sl_mult = max(0, sl_mult)

        df = self.get_ticker(ticker)

        date_index = self.get_date_index(df, date, earlier=False)

        cur_close = df["Close"].iloc[date_index]

        upper_barrier = None if tp_mult == 0 else cur_close*(1 + barrier_width*tp_mult)
        lower_barrier = None if sl_mult == 0 else cur_close*(1 - barrier_width*sl_mult)

        for trade_time, close in enumerate(df["Close"].iloc[date_index:].values):
            if trade_time == 0:
                continue

            if trade_time >= h and h != 0:
                return_t_h = close/cur_close - 1

                if categorical:
                    if sign_on_vertical:
                        return -1 if return_t_h <= 0 else 1
                    else:
                        return 0
                else:
                    return return_t_h
            
            elif upper_barrier is not None and close >= upper_barrier:
                if categorical:
                    return 1
                else:
                    return upper_barrier/close - 1
            
            elif lower_barrier is not None and close <= lower_barrier:
                if categorical:
                    return -1
                else:
                    return lower_barrier/close - 1
        
        #Return None to detect loop ended without a label assigned
        if True:
            return None

    def get_ticker_date_inputs(
            self,
            ticker,
            date,
            features=None,
            preprocessed=False,
    ):
        
        df = self.get_ticker(ticker)
        date_index = self.get_date_index(df, date=date, earlier=True)

        f_window = self.cfg[ticker]["feature_window"]
        extractors = self.cfg[ticker]["feature_extractors"]

        featured_data = df.iloc[date_index-f_window+1:date_index+1].copy()

        if not preprocessed:
            assert(date_index >= f_window-1), f"\
                Not enough past values on {ticker} on date {date}\
                to apply feature extraction"
            
            for extractor in extractors:
                featured_data = extractor.apply_feature_extraction(featured_data)

        if features is None or features == "all":
            features = featured_data.columns
        
        featured_data = featured_data[features].iloc[-self.cfg[ticker]["window"]:].values

        return featured_data
    
    def get_date_inputs(
            self,
            date,
            get_labels=True
    ):
        
        x = {
            k: self.get_ticker_date_inputs(
                ticker=k, date=date, 
                features=self.cfg[k]["features"],
                preprocessed=self.cfg[k]["preprocessed"]
                )
            for k in self.cfg.keys()
        }
        
        if get_labels:
            y = {
                k: getattr(
                    self, self.cfg[k]["label"][0]
                    )(ticker=k, date=date, **self.cfg[k]["label"][1])
                if self.cfg[k]["label"] is not None else None
                for k in self.cfg.keys()
            }

            return (x, y)
        
        return x
    
    def get_data_batch(
            self,
            start_date=None,
            end_date=None,
            get_labels=True
    ):
        
        x = {k: None for k in self.cfg.keys()}
        y = x.copy()

        for ticker in x.keys():
            df = self.get_ticker(ticker)

            if start_date is None:
                start_ind = 0
            else:
                start_ind = self.get_date_index(df, start_date, earlier=False)
            
            if end_date is None:
                end_ind = len(df)-1
            else:
                end_ind = self.get_date_index(df, end_date, earlier=True)
            
            label_window = 0
            feature_window = self.cfg[ticker]["feature_window"]

            if get_labels:
                if self.cfg[ticker]["label"] is not None:
                    label_window = self.cfg[ticker]["label"][1]["h"]
            
            cur_x = []
            cur_y = []

            for ind in range(start_ind+feature_window, end_ind-label_window):
                date = df.index[ind]

                cur_vals = self.get_ticker_date_inputs(
                                    ticker, date, 
                                    features=self.cfg[ticker]["features"],
                                    preprocessed=self.cfg[ticker]["preprocessed"]
                            )
                
                cur_x.append(cur_vals)
                
                if get_labels:
                    if self.cfg[ticker]["label"] is None:
                        cur_label = np.nan
                    else:
                        cur_label = getattr(
                                self, self.cfg[ticker]["label"][0]
                                )(ticker=ticker, date=date, **self.cfg[ticker]["label"][1])
                    
                    cur_y.append(cur_label)
            
            n_window, n_features = cur_x[0].shape

            cur_x = np.array(cur_x).reshape(len(cur_x), n_window, n_features)
            cur_y = np.array(cur_y).reshape(len(cur_y), 1)

            x[ticker] = cur_x
            y[ticker] = cur_y
        
        if get_labels:
            return (x, y)
        
        return x
    
    def preprocess(self):

        self.original_data_dict = self.data_dict.copy()

        for ticker in self.data_dict:
            
            extractors = self.cfg[ticker]["feature_extractors"]
            featured_data = self.data_dict[ticker].copy()

            for extractor in extractors:
                featured_data = extractor.apply_feature_extraction(featured_data)


            self.data_dict[ticker] = featured_data.dropna()
    

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