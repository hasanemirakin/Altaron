import pandas as pd
import numpy as np
import datetime
from altaron.base.__base import AltaronBaseClass
from altaron.base.__feature_extractor import FeatureExtractor
from altaron import feature_extraction
from altaron.mpengine import (
    prepare_jobs,
    infer_nan_window,
    process_jobs,
    combine_outputs_concat_df,
    expand_call_fe,
    expand_call_parallel_tickers,
    combine_outputs_update_dictionary
)

class DataProcessor(AltaronBaseClass):

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
                "feature_extractor": FeatureExtractor([]),
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
        
        jobs = prepare_jobs(
            func=self.__get_feature_window__,
            data=list(self.cfg.keys()), #tickers
            args={},
            num_threads= len(self.cfg.keys()) #tickers
        )

        fw = process_jobs(
            jobs=jobs,
            call_expansion=expand_call_parallel_tickers,
            output_combination=combine_outputs_update_dictionary,
            num_threads= len(self.cfg.keys()) #tickers
        )
        
        for ticker, f_window in fw.items():
            self.cfg[ticker]["feature_window"] = f_window

            #Initialize features in case features is passed as "all"
            _ = self.get_ticker_date_inputs(
                ticker=ticker,
                date=self.data_dict[ticker].index[-1],
                features=self.cfg[ticker]["features"],
                preprocessed=False
            )
    
    def __get_feature_window__(
            self,
            ticker
    ):

            extractor = self.cfg[ticker]["feature_extractor"]
            cfg = extractor.fe_config
            dummy_data = self.data_dict[ticker].copy()

            for i in range(1,11):
                try:
                    max_lookback = infer_nan_window(
                        func=extractor.apply_feature_extraction,
                        args={},
                        data=dummy_data[:min(i*200, len(dummy_data))]
                    )
                    break
                except Exception as e:
                    print(repr(e))
                    continue

            return self.cfg[ticker]["window"] + max_lookback
    
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

    def preprocess(self):

        self.original_data_dict = self.data_dict.copy()

        for ticker in self.data_dict:
            
            if self.cfg[ticker]["preprocessed"]:
                continue

            extractor = self.cfg[ticker]["feature_extractor"]
            featured_data = self.__apply_feature_extraction(
                data=self.data_dict[ticker],
                ticker=ticker,
                num_threads=None #func will infer
            )

            self.data_dict[ticker] = featured_data.dropna()
            self.cfg[ticker]["preprocessed"] = True

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
        extractor = self.cfg[ticker]["feature_extractor"]

        featured_data = df.iloc[date_index-f_window+1:date_index+1].copy()

        if not preprocessed:
            assert(date_index >= f_window-1), f"\
                Not enough past values on {ticker} on date {date}\
                to apply feature extraction"
            
            #This is called without multiprocessing, since it is applied on a single window of data
            featured_data = extractor.apply_feature_extraction(featured_data)

        #Check if features is iterable
        try:
            iter(features)
            if isinstance(features, str):
                features = featured_data.columns
                self.cfg[ticker]["features"] = features
        except:
            pass

        featured_data = featured_data[features].iloc[-self.cfg[ticker]["window"]:]

        return featured_data
    
    def get_date_inputs(
            self,
            date,
            get_labels=False
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
    
    def backtest_results_to_training_data(
            self,
            actions: pd.DataFrame,
            label="categorical",
            feature_tickers = {}
    ):
        
        """

        Labels are assigned as the price change between
        actions of opposite sides

        Function of the feature tickers is this;
        
        On date inputs for date T suppose we have inputs for ticker Y;

        date_inputs[T][Y] = arr0

        We want to use the inputs for Y as additional features for X.

        Hence, features for ticker X is given as;

        date_inputs[T][X] = arr1

        features[X]_T = concatenate(arr0, arr1)
         
        """
        features = {
                ticker: []
                for ticker in actions["Ticker"].unique()
            }

        targets = {
                ticker: []
                for ticker in actions["Ticker"].unique()
            }


        for ticker in features.keys():
            ticker_actions = actions[actions["Ticker"] == ticker].copy()
            for row in range(len(ticker_actions)):
                if ticker_actions["Action"].iloc[row] != "Entry":
                    continue
                
                if row == len(ticker_actions) - 1:
                    break

                action_side = ticker_actions["Side"].iloc[row]
                action_price = ticker_actions["Price"].iloc[row]
                action_date = ticker_actions["Date"].iloc[row]

                action_features = self.get_ticker_date_inputs(
                                    ticker = ticker,
                                    date = action_date,
                                    features=self.cfg[ticker]["features"],
                                    preprocessed=self.cfg[ticker]["preprocessed"]
                                ).values

                if action_features.ndim == 2:
                    if len(action_features) == 1:
                        action_features = np.squeeze(action_features, axis=0)
                    else:
                        action_features = action_features[-1, :]
                
                if ticker in feature_tickers.keys():

                    if isinstance(feature_tickers[ticker], str):
                        additional = self.get_ticker_date_inputs(
                                        ticker = feature_tickers[ticker],
                                        date = action_date,
                                        features=self.cfg[ticker]["features"],
                                        preprocessed=self.cfg[ticker]["preprocessed"]
                                    ).values

                        if additional.ndim == 2:
                            if len(additional) == 1:
                                additional = np.squeeze(additional, axis=0)
                            else:
                                additional = additional[-1, :]
                        
                        action_features = np.concatenate((action_features, additional), axis=-1)

                    elif isinstance(feature_tickers[ticker], list):
                        
                        for v in feature_tickers[ticker]:
                            additional = self.get_ticker_date_inputs(
                                            ticker = v,
                                            date = action_date,
                                            features=self.cfg[ticker]["features"],
                                            preprocessed=self.cfg[ticker]["preprocessed"]
                                        ).values

                            if additional.ndim == 2:
                                if len(additional) == 1:
                                    additional = np.squeeze(additional, axis=0)
                                else:
                                    additional = additional[-1, :]
                            
                            action_features = np.concatenate((action_features, additional), axis=-1)

                end_loop = True
                for row2 in range(row+1, len(ticker_actions)):
                    if ticker_actions["Action"].iloc[row2] == "Entry":
                        continue

                    exit_price = ticker_actions["Price"].iloc[row2]
                    end_loop = False
                    break
                
                if end_loop:
                    break

                price_change = action_side*(exit_price/action_price - 1)

                if label == "categorical":
                    action_target = int(bool(price_change > 0))
                else:
                    action_target = price_change

                features[ticker].append(action_features)
                targets[ticker].append(action_target)
                
        return features, targets
    
    def __apply_feature_extraction(
            self,
            data,
            ticker,
            num_threads=None
    ):
        
        extractor = self.cfg[ticker]["feature_extractor"]
        df = data.copy()

        jobs = prepare_jobs(
            func=extractor.apply_feature_extraction,
            data=df,
            args={},
            num_threads=None, #function will infer based on cpu count,
            extend_parts=self.cfg[ticker]["feature_window"]
        )

        df = process_jobs(
            jobs=jobs,
            call_expansion=expand_call_fe,
            output_combination=combine_outputs_concat_df,
            num_threads=None #function will infer
        )

        return df

    def get_ticker_ohlcv(
            self,
            ticker,
            date,
            earlier = True
    ):

        data = self.get_ticker(ticker)

        date_match = self.get_nearest_date(data, date, earlier)

        return data[["Open", "High", "Low", "Close", "Volume"]].loc[date_match]

    def get_ohlcv(
           self,
           date 
    ):
        ohlcv = {
            ticker: self.get_ticker_ohlcv(ticker, date, earlier=True)
            for ticker in self.data_dict.keys()
        }
        
        return ohlcv

    def get_current_price(
            self,
            ticker,
            date,
            earlier=True
    ):
        
        data = self.get_ticker(ticker)

        date_match = self.get_nearest_date(data, date, earlier)

        return data["Close"].loc[date_match]

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
    
    def get_index_date(
            self,
            index
    ):
        
        main_ticker = list(self.data_dict.values())[0].copy()

        return main_ticker.index[index]

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