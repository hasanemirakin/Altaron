import pandas as pd
import numpy as np
from altaron.base.__q_class import QClass
from altaron.base import DataProcessor

class QStrategy(QClass):

    def __init__(
            self,
            processor: DataProcessor,
            model_cfg = {},
            strategy_cfg = {},
            **kwargs
    ):

        self.processor = processor

        assert(isinstance(model_cfg, dict)), "Config must be a dict"
        assert(isinstance(strategy_cfg, dict)), "Config must be a dict"

        self.model_cfg = model_cfg
        self.strategy_cfg = strategy_cfg
    
    def get_strategy_out(
            self,
            inputs,
            ticker_positions,
    ):
        """Child classes will override this"""
        raise NotImplementedError("This method is not implemented yet")
    
    def get_model_out(self, inputs):
        """Child classes will override this"""
        raise NotImplementedError("This method is not implemented yet")
    
    