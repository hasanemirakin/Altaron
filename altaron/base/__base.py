import pandas as pd
import numpy as np
import torch

class AltaronBaseClass:

    def __init__(
            self,
            **kwargs
    ):
        
        self.set_kwargs(kwargs)

    def set_allowed_kwargs(self, kwargs: list):
        """This function sets the allowed keyword arguments for the model."""
        self.allowed_kwargs = kwargs

    def set_kwargs(self, kwargs: dict):
        """This function sets the keyword arguments for the model."""
        valid_kwargs = self.check_kwargs(kwargs)
        for k, v in valid_kwargs.items():
            self.__setattr__(k, v)
    
    def check_kwargs(self, kwargs: dict):
        valid_kwargs = {}

        for k,v in kwargs.items():
            if k in self.allowed_kwargs:
                valid_kwargs[k] = v

        return valid_kwargs   
    
    def to_tensor(self, series):
        """Turn series into tensors"""

        return torch.tensor(series, dtype=torch.float32).detach().clone()

    def to_numpy(self, series):
        """Turn series into numpy arrays"""

        return np.array(series, dtype=np.float32).copy()