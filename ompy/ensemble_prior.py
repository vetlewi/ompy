import numpy as np
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
                 spc_param: Callable[..., any],
                 N: int, N_nld_par: int, N_gsf_par: int, N_spc_par) -> float:
        """
        """

        self.A, self.B, self.alpha, self.N = A, B, alpha, N
        self.nld_param, self.N_nld_par = nld_param, N_nld_par
        self.gsf_param, self.N_gsf_par = gsf_param, N_gsf_par
        self.spc_param, self.N_spc_par = spc_param, N_spc_par

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
        N = self.N
        Nnld, Ngsf, Nspc = self.N_nld_par, self.N_gsf_par, self.N_spc_par
        pars = np.array(param[0:3*N+Nnld+Ngsf+Nspc])
        pars[0:N] = self.A(pars[0:N])
        pars[N:2*N] = self.B(pars[N:2*N])
        pars[2*N:3*N] = self.alpha(pars[2*N:3*N])
        pars[3*N:3*N+Nnld] = self.nld_param(pars[3*N:3*N+Nnld])
        pars[3*N+Nnld:3*N+Nnld+Ngsf] =\
            self.gsf_param(x=pars[3*N+Nnld:3*N+Nnld+Ngsf])
        pars[3*N+Nnld+Ngsf:3*N+Nnld+Ngsf+Nspc] =\
            self.spc_param(pars[3*N+Nnld+Ngsf:3*N+Nnld+Ngsf+Nspc])
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


class cnormal:
    def __init__(loc: Union[float, ndarray] = 0,
                 scale: Union[float, ndarray] = 1):
        self.loc = loc if isinstance(loc, float) else loc.copy()
        self.scale = scale if isinstance(scale, float) else scale.copy()

    def __call__(x: Union[float, ndarray]) -> Union[float, ndarray]:
        return stats.norm.ppf(x, loc=self.loc, scale=self.scale)


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
