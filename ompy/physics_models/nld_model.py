import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence

from .model import model
from ..prior import normal, truncnorm
from ..library import self_if_none


class NLD_model(Model):
    def __init__(self, name: str, priors: dict):
        """ Initialize GSF model. """

        self.prior_funcs = []
        # If all priors are normal dist we can improve by skiping list stuff.
        self.optimize_prior = True
        self.mu = []
        self.sigma = []
        self.prior_func = None
        super().__init__(name=name, N_free=self.setup_prior(priors))

    def setup_prior(self, priors: dict) -> int:
        n_param = 0
        for par_key in priors:
            n_param += 1
            try:
                if priors[par_key]['kind'] == 'uniform':
                    self.optimize_prior = False
                    a = priors[par_key]['a']
                    b = priors[par_key]['b']
                    self.prior_funcs.append(lambda x: uniform(x, a, b))
                elif priors[par_key]['kind'] == 'normal':
                    mu = priors[par_key]['mu']
                    sigma = priors[par_key]['sigma']
                    self.mu.append(mu)
                    self.sigma.append(sigma)
                    self.prior_funcs.append(lambda x: normal(x, mu, sigma))
                elif priors[par_key]['kind'] == 'truncnorm':
                    self.optimize_prior = False
                    mu = priors[par_key]['mu']
                    sigma = priors[par_key]['sigma']
                    a, b = -np.inf, np.inf
                    try:
                        a = priors[par_key]['lower']
                    except KeyError:
                        pass
                    try:
                        b = priors[par_key]['upper']
                    except KeyError:
                        pass
                    self.prior_funcs.append(lambda x: truncnorm(x, mu,
                                                                sigma,
                                                                a, b))
                elif priors[par_key]['kind'] == 'exponential':
                    self.optimize_prior = False
                    sigma = priors[par_key]['sigma']
                    self.prior_funcs(lambda x: exponential(x, sigma))
                else:
                    raise TypeError("Unknown distribution '%s'."
                                    % (priors[par_key]['kind']))
            except KeyError:
                mu = priors[par_key]['mu']
                sigma = priors[par_key]['sigma']
                self.mu.append(mu)
                self.sigma.append(sigma)
                self.prior_funcs.append(lambda x: normal(x, mu, sigma))

        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)
        self.prior_func = self.normal_prior if self.optimize_prior\
            else self.list_prior
        return n_param

    def normal_prior(self, param: ndarray) -> ndarray:
        return normal(param, self.mu, self.sigma)

    def list_prior(self, param: ndarray) -> ndarray:
        assert len(self.prior_funcs) == len(param), \
            "Expected %d RVs, not %d" % (len(self.prior_funcs), len(param))
        return [func(x) for func, x in zip(self.prior_funcs, param)]

    def prior(self, param: ndarray) -> ndarray:
        return self.prior_func(param)

    def mean(self, E: ndarray) -> ndarray:
        return self.__call__(E, np.ones(self.n_param)*0.5)


class Constant_temperature(NLD_model):
    def __init__(self, priors):
        super().__init__(name="Constant temperature", priors=priors)

    def __call__(self, E: Union[ndarray, float],
                 T: float, Eshift: float) -> Union[ndarray, float]:
        return np.exp((E-Eshift)/T)/T

    @staticmethod
    @nb.jit(nopython=True)
    def evaluate_jit(E: Union[ndarray, float],
                     T: float, Eshift: float) -> Union[ndarray, float]:
        return np.exp((E-Eshift)/T)/T


class Backshift_FermiGas(NLD_model):
    def __init__(self, priors, A):
        super().__init__(name="Back-shift Fermi Gas", priors=priors)
        self.A = A

    def __call__(self, E: Union[ndarray, float],
                 a: float, Eshift: float) -> Union[ndarray, float]:
        sigma = 0.0146*A**(5./3.)
        sigma = np.sqrt((1 + np.sqrt(1 + 4*a*(E-Eshift)))*sigma/(2*a))

        bsfg = np.exp(2*np.sqrt(a*(E-Eshift)))
        bsfg /= (12.*np.sqrt(2)*sigma*a**(1./4.)*(E-Eshift)**(5./4.))
        return bsfg
