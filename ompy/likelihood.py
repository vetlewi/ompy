import numpy as np
import numba as nb
import pandas as pd
from numpy import ndarray
from inspect import signature
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence
import scipy.stats as stats
import matplotlib.pyplot as plt

from .vector import Vector
from .library import log_interp1d
from .gsf_model import GSFmodel
from .models import (ExtrapolationModelLow, ExtrapolationModelHigh,
                     NormalizationParameters)


class Likelihood:

    def __init__(self, x: ndarray, y: ndarray, model: Callable[..., ndarray]):
        """
        Args:
            x: predictor
            y: observed
            model: Model describing the observed data.
        """

        self.x = np.array(x)
        self.y = np.array(y)
        self.model = model

    def __call__(self, *param) -> float:
        """ Wrapper for logp. """
        return self.logp(*param)

    def logp(self, *param) -> float:
        raise NotImplementedError


class NormalLikelihood(Likelihood):

    def __init__(self, x: ndarray, y: ndarray, model: Callable[..., ndarray],
                 yerr: Union[float, ndarray] = 0.3):
        """
        Args:
            x: predictor
            y: observed
            yerr: 1 sigma of the observed.
            model: Model describing the data.
        """

        super().__init__(x=x,  y=y, model=model)

        self.yerr = yerr if isinstance(yerr, ndarray) else yerr*y
        self.yerr = np.array(self.yerr)

    def logp(self, *model_args) -> float:
        logp = std_error(self.yerr)
        logp -= 0.5*error(self.y, self.model(self.x, *model_args), self.yerr)
        return logp


class OsloNormalLikelihood(NormalLikelihood):

    def transform(self, const: Union[ndarray, float],
                  alpha: Union[ndarray, float]
                  ) -> Tuple[ndarray, ndarray, ndarray]:
        norm = (const*np.exp(alpha*self.x.T)).T
        return self.x, self.y*norm, self.yerr*norm

    def logp(self, const: ndarray, alpha: ndarray, *model_args):
        x, y, yerr = self.transform(const, alpha)
        logp = std_error(yerr)
        logp -= 0.5*error(y, self.model(x, *model_args), yerr)
        return logp


@nb.jit(nopython=True)
def std_error(weights: ndarray) -> float:
    return np.sum(np.log(1/(np.sqrt(2*np.pi)*weights)))


@nb.jit(nopython=True)
def error(data: ndarray, model: ndarray,
          weights: Optional[ndarray] = None) -> float:
    """ A simple function for calculating the unweighted sum.
    Note that to avoid issues with weights, we assume 30% if weights are None.
    """
    if weights is None:
        weights = 0.1*data
    return np.sum(((data - model)/weights)**2)
