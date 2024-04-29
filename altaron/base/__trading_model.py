import pandas as pd
import numpy as np 
import torch
from altaron.base.__base import AltaronBaseClass

class TradingModel(AltaronBaseClass):

    def __init__(
            self, 
            input_dim,
            output_dim,
            tickers = [],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.tickers = tickers