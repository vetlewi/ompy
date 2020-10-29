import numpy as np
import numba as nb
from numpy import ndarray
from scipy import stats as scistats
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence

from .stats import normal_vec as builtin_normal_vec
from .stats import truncnorm_vec as builtin_truncnorm_vec


@nb.jit(nopython=True)
def uniform(x: Union[float, ndarray], a: Union[float, ndarray],
            b: Union[float, ndarray]) -> Union[float, ndarray]:
    """ Transform uniform RV ([0,1]) to uniform [a, b] RV.
    Args:
        x: Unform RV [0,1]
        a: Lower limit
        b: Upper limit
    Returns: Random number with uniform probability.
    """
    return (b-a)*x + a

@nb.jit(nopython=True)
def exponential(x: Union[float, ndarray],
                scale: Union[float, ndarray] = 1.0) -> Union[float, ndarray]:
    return -np.log(1-x)*scale


#@nb.jit(nopython=True)
def normal(x: Union[float, ndarray], mu: Union[float, ndarray],
           sigma: Union[float, ndarray]) -> Union[float, ndarray]:
    """ Transform uniform RV ([0,1]) to Gaussian RV.
    Args:
        x: Uniform RV [0,1]
        mu: Mean of the Gaussian distribution.
        sigma: Standard diviation of the Gaussian distribution.
    Returns: Random number with Gaussian probability.
    """
    return builtin_normal_vec(x)*sigma + mu


#@nb.jit(nopython=True)
def truncnorm(x: Union[float, ndarray],
              mu: Union[float, ndarray],
              sigma: Union[float, ndarray],
              lower: Optional[Union[float, ndarray]] = -np.inf,
              upper: Optional[Union[float, ndarray]] = np.inf
              ) -> Union[float, ndarray]:
    """ Transform uniform RV ([0,1]) to truncated Gaussian RV.
    Args:
        x: Uniform RV [0,1]
        mu: Mean of the Gaussian distribution.
        sigma: Standard diviation of the Gaussian distribution.
        lower: (Optional) lower limit for the truncation.
        upper: (Optioanl) upper limit for the truncation.
    """
    return builtin_truncnorm_vec(x, (lower - mu)/sigma,
                                    (lower - mu)/sigma)*sigma + mu
