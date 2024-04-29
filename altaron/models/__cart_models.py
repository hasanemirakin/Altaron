import numpy as np
import pandas as pd
import joblib
from altaron.base.__trading_model import TradingModel

class CARTModel(TradingModel):

    def __init__(
            self,
            input_dim,
            output_dim,
            tickers = [],
            model=None,
            **kwargs,
    ):

        super().set_allowed_kwargs(list(kwargs.keys()))
        super().__init__(input_dim, output_dim, tickers, **kwargs)

        self.model = model
        self.input_shape = (input_dim)

    def train_model(
            self,
            X: dict,
            Y: dict,
    ):
        
        if self.tickers == []:

            x_inputs = np.concatenate([np.array(val) for val in X.values()], axis=0)
            y_inputs = np.concatenate([np.array(val) for val in Y.values()], axis=0)

            self.model = self.model.fit(x_inputs, y_inputs)

        else:

            for ticker, model in self.model:

                x_inputs = np.array(X[ticker])
                y_inputs = np.array(Y[ticker])

                self.model[ticker] = self.model[ticker].fit(x_inputs, y_inputs)
    
    def get_prediction(
            self,
            inputs: dict
    ):
        
        if self.tickers == []:

            preds = self.model.predict(inputs)

        else:
            preds = {}
            for ticker, model in self.model:

                x_inputs = np.array(inputs[ticker])
                preds[ticker] = self.model[ticker].predict(x_inputs)
        
        return preds
    
    def save_model(
            self,
            path
    ):
        """path must be abc.joblib"""

        _ = joblib.dump(self.model, path)
    
    def load_model(
            self,
            path
    ):
        """path must be abc.joblib"""
        self.model = joblib.load(path)


class RandomForestModel(CARTModel):

    def __new__(
            self,
            input_dim,
            output_dim,
            tickers = [],
            model="regressor",
            **kwargs,
    ):
        
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        if tickers == []:
            if model == "regressor":
                model = RandomForestRegressor(**kwargs)
            elif model == "classifier":
                model = RandomForestClassifier(**kwargs)
        else:
            model = {
                ticker: RandomForestRegressor(**kwargs) 
                if model=="regressor" else RandomForestClassifier(**kwargs)
                for ticker in tickers
            }

        return CARTModel(
            input_dim=input_dim,
            output_dim=output_dim,
            tickers=tickers,
            model=model,
            **kwargs
        )

class XGBoostModel(CARTModel):

    def __new__(
            self,
            input_dim,
            output_dim,
            tickers = [],
            model="regressor",
            **kwargs,
    ):
        
        from xgboost import XGBRegressor, XGBClassifier

        if tickers == []:
            if model == "regressor":
                model = XGBRegressor(**kwargs)
            elif model == "classifier":
                model = XGBClassifier(**kwargs)
        else:
            model = {
                ticker: XGBRegressor(**kwargs) 
                if model=="regressor" else XGBClassifier(**kwargs)
                for ticker in tickers
            }

        return CARTModel(
            input_dim=input_dim,
            output_dim=output_dim,
            tickers=tickers,
            model=model,
            **kwargs
        )

class LightGBMModel(CARTModel):

    def __new__(
            self,
            input_dim,
            output_dim,
            tickers = [],
            model="regressor",
            **kwargs,
    ):
        
        from lightgbm import LGBMRegressor, LGBMClassifier

        if tickers == []:
            if model == "regressor":
                model = LGBMRegressor(**kwargs)
            elif model == "classifier":
                model = LGBMClassifier(**kwargs)
        else:
            model = {
                ticker: LGBMRegressor(**kwargs) 
                if model=="regressor" else LGBMClassifier(**kwargs)
                for ticker in tickers
            }

        return CARTModel(
            input_dim=input_dim,
            output_dim=output_dim,
            tickers=tickers,
            model=model,
            **kwargs
        )

class AdaBoostModel(CARTModel):

    def __new__(
            self,
            input_dim,
            output_dim,
            tickers = [],
            model="regressor",
            **kwargs,
    ):
        
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

        if tickers == []:
            if model == "regressor":
                model = AdaBoostRegressor(**kwargs)
            elif model == "classifier":
                model = AdaBoostClassifier(**kwargs)
        else:
            model = {
                ticker: AdaBoostRegressor(**kwargs) 
                if model=="regressor" else AdaBoostClassifier(**kwargs)
                for ticker in tickers
            }

        return CARTModel(
            input_dim=input_dim,
            output_dim=output_dim,
            tickers=tickers,
            model=model,
            **kwargs
        )

