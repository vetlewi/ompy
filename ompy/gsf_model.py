import numpy as np
import numba as nb
from numpy import ndarray
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence
from inspect import signature


from .ensemble_prior import normal


class GSF_model:
    def __init__(self, priors):

        # We assume normal prior for all parameters
        # We iterate through each.
        # First we expect
        self.mu = []
        self.sigma = []
        for res_key in priors:
            for par_key in priors[res_key]:
                self.mu.append(priors[res_key][par_key]['mu'])
                self.sigma.append(priors[res_key][par_key]['sigma'])
        self.mu = np.array(self.mu)
        self.sigma = np.array(self.sigma)

    def prior(self, x: Sequence) -> Sequence:
        x = normal(x, self.mu, self.sigma)
        return x

    def __call__(self, *args) -> ndarray:
        raise NotImplementedError


class GSFmodel:

    def __init__(self, prior_params: dict):
        """ Initialize the GSF model.
        Args:
            prior_params: List of parameters
        """
        self.components = []
        self.prior_m = []
        self.prior_s = []
        self.T_pos = []
        self.par_names = ['E']
        N = 0
        for res_key in prior_params:
            func = prior_params[res_key]['func']
            func_par = list(dict(signature(func).parameters).keys())
            del func_par[0]
            if func_par.count('T') > 0:
                self.T_pos.append(N+func_par.index('T'))
            func_idx = slice(N, N+len(func_par))
            N += len(func_par)
            self.components.append((func, func_idx))

            for par_key in prior_params[res_key]['prior']:
                if par_key.endswith('_err'):
                    self.prior_s.append(prior_params[res_key]['prior'][par_key])
                else:
                    self.prior_m.append(prior_params[res_key]['prior'][par_key])
                    self.par_names.append(res_key + "_" + par_key)

        self.prior_m = np.array(self.prior_m)
        self.prior_s = np.array(self.prior_s)

        # Ensure that everything is consistent.
        assert len(self.prior_m) == len(self.prior_s), \
            "Expected same number of prior means as prior sigmas"

        # There may be arguments for the functions that are not spesific for
        # the gsf model.

    def __call__(self, E: ndarray, x: Sequence) -> ndarray:
        return self.calculate(E, x)

    def calculate(self, E: ndarray, x: Sequence) -> ndarray:
        """ Calculate the gSF given the set of parameters x.
        """
        gsf = np.zeros(E.shape)
        for (func, idx) in self.components:
            gsf += func(E, *x[idx])
        return gsf

    def mean(self, E: ndarray) -> ndarray:
        """ Returns the most probable gSF based on priors.
        """

        x = list(self.prior_m)
        # Insert the temperature
        for pos in self.T_pos:
            x.insert(pos, 0.8)
        return self.calculate(E, x)

    def prior(self, x: ndarray) -> ndarray:
        return normal(x, self.prior_m, self.prior_s)

    @staticmethod
    @nb.jit(nopython=True)
    def SMLO(E: ndarray, mean: float, width: float,
             size: float, T: float) -> ndarray:
        """ The simple modified Lorentzian model.

        Args:
            E: Gamma ray energy in [MeV]
            mean: Mean position of the resonance in [MeV]
            width: Width of the resonance in [MeV]
            size: Total cross-section in [mb MeV]
            T: Temperature of the final level in [MeV]
        Returns: Gamma ray strength [MeV^(-3)]
        """

        width_ = width*(E/mean + (2*np.pi*T/mean)**2)
        smlo = 8.6737e-8*size/(1-np.exp(-E/T))
        smlo *= (2./np.pi)*E*width_
        smlo /= ((E**2 - mean**2)**2 + (E*width_)**2)
        return smlo

    @staticmethod
    @nb.jit(nopython=True)
    def GLO(E: ndarray, mean: float, width: float,
            size: float, T: float) -> ndarray:
        """ The Generalized Lorentzian model.

        Args:
            E: Gamma ray energy in [MeV]
            mean: Mean position of the resonance in [MeV]
            width: Width of the resonance in [MeV]
            size: Total cross-section in [mb]
            T: Temperature of the final level in [MeV]
        Returns: Gamma ray strength [MeV^(-3)]
        """

        width_ = width*(E**2 + (2*np.pi*T)**2)/mean**2
        glo = E*width_/((E**2 - mean**2)**2+(E*width_)**2)
        glo += 0.7*width*(2*np.pi*T)**2/mean**5
        glo *= 8.6737e-8*size*width
        return glo

    @staticmethod
    @nb.jit(nopython=True)
    def EGLO(E: ndarray, mean: float, width: float,
             size: float, T: float, k: float) -> ndarray:
        """ The enhanced Generalized Lorentzian model.

        Args:
            E: Gamma ray energy in [MeV]
            mean: Mean position of the resonance in [MeV]
            width: Width of the resonance in [MeV]
            size: Total cross-section in [mb]
            T: Temperature of the final level in [MeV]
        Returns: Gamma ray strength [MeV^(-3)]
        """

        def K(e: ndarray) -> ndarray:
            return 1 + (1-k)*(e - 4.5)/(mean - 4.5)

        def Width(e: ndarray, t: float) -> ndarray:
            return K(e)*width*((e/mean)**2 + (2*np.pi*t/mean)**2)

        eglo = 0.7*Width(0, T)/mean**3
        eglo += E*Width(E, T)/((E**2 - mean**2)**2 + (E*Width(E, T))**2)
        eglo *= 8.6737e-8*size*width
        return eglo

    @staticmethod
    @nb.jit(nopython=True)
    def SLO(E: ndarray, mean: float, width: float, size: float) -> ndarray:
        """ The standard Lorentzian.

        Args:
            E: Gamma ray energy in [MeV]
            mean: Mean position of the resonance in [MeV]
            width: Width of the resonance in [MeV]
            size: Total cross-section in [mb]
            T: Temperature of the final level in [MeV]
        Returns: Gamma ray strength [MeV^(-3)]
        """

        slo = 8.6737e-8*size*E*width**2
        slo /= ((E**2 - mean**2)**2 + (E*width)**2)
        return slo
