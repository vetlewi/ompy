import numpy as np
import numba as nb
from numpy import ndarray
from scipy import stats
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence

from .vector import Vector

class EnsemblePriorNLD:

    def __init__(self, A: Callable[..., any],
                 alpha: Callable[..., any],
                 nld_param: Callable[..., any],
                 N: int, N_nld: int) -> None:
        """
        """

        self.A, self.alpha = A, alpha
        self.nld_param = nld_param
        self.N, self.N_nld = N, N_nld

    def __call__(self, param):
        return self.evaluate(param)

    def prior(self, cube, ndim, nparam):
        set_par(cube, self.evaluate(cube))

    def evaluate(self, param):
        N, N_nld = self.N, self.N_nld

        pars = np.array(param[0:2*N+N_nld])
        pars[0:N] = self.A(pars[0:N])
        pars[N:2*N] = self.alpha(pars[N:2*N])
        pars[2*N:2*N+N_nld] = self.nld_param(pars[2*N:2*N+N_nld])
        return pars


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


def set_par(cube, values):
    for i, value in enumerate(values):
        cube[i] = value
