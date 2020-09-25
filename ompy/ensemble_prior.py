import numpy as np
#from numba import jit, prange
from numpy import ndarray
from scipy import stats
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence

from .vector import Vector


class EnsemblePrior:
    """
    """

    def __init__(self, A: Callable[..., any],
                 B: Callable[..., any],
                 alpha: Callable[..., any],
                 nld_param: Callable[..., any],
                 gsf_param: Callable[..., any],
                 N: int, N_nld_par: int, N_gsf_par: int) -> float:
        """
        """

        self.A, self.B, self.alpha = A, B, alpha
        self.nld_param, self.gsf_param = nld_param, gsf_param
        self.N, self.N_nld_par, self.N_gsf_par = N, N_nld_par, N_gsf_par

    def __call__(self, param):
        """
        """
        return self.evaluate(param)

    def prior(self, cube, ndim, nparam):
        """
        """
        set_par(cube, self.evaluate(cube))

    def evaluate(self, param):
        """
        """
        N, Nnld, Ngsf = self.N, self.N_nld_par, self.N_gsf_par
        pars = np.array(param[0:3*N+Nnld+Ngsf])
        pars[0:N] = self.A(pars[0:N])
        pars[N:2*N] = self.B(pars[N:2*N])
        pars[2*N:3*N] = self.alpha(pars[2*N:3*N])
        pars[3*N:3*N+Nnld] = self.nld_param(pars[3*N:3*N+Nnld])
        pars[3*N+Nnld:3*N+Nnld+Ngsf] =\
            self.gsf_param(pars[3*N+Nnld:3*N+Nnld+Ngsf])
        return pars


def uniform(x: Union[float, ndarray], lower: Union[float, ndarray] = 0,
            upper: Union[float, ndarray] = 1) -> Union[float, ndarray]:
    """ Transforms a random number from a uniform PDF between
    0 and 1 to a uniform distribution between lower and upper.
    """
    return x*(upper-lower) + lower


def normal(x:  Union[float, ndarray], loc: Union[float, ndarray] = 0.,
           scale: Union[float, ndarray] = 1.) -> Union[float, ndarray]:
    return stats.norm.ppf(x, loc=loc, scale=scale)


def exponential(x: Union[float, ndarray],
                scale: Union[float, ndarray] = 1.0) -> Union[float, ndarray]:
    return -np.log(1-x)*scale


def truncnorm(x: Union[float, ndarray], lower: Union[float, ndarray],
              upper: Union[float, ndarray], loc: Union[float, ndarray],
              scale: Union[float, ndarray]) -> Union[float, ndarray]:
    """ Wrapper for the scipy stats norm ppf.
    """
    a = (lower - loc)/scale
    b = (upper - loc)/scale
    return stats.truncnorm.ppf(x, a, b, loc, scale)


def set_par(cube, values):
    for i, value in enumerate(values):
        cube[i] = value
