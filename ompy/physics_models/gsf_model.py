import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence

from .model import model
from .prior import *
from .library import self_if_none


class GSF_model(model):

    def __init__(self, name: str, priors: dict):
        """ Initialize GSF model. """
        super().__init__(priors=priors)
        self.prior_funcs = []
        self.need_T = False
        for res_key in priors:
            for par_key in priors[res_key]:
                if par_key == 'T' and priors[res_key][par_key] is None:
                    self.need_T = True
                    continue
                if priors[res_key][par_key]['kind'] == 'uniform':
                    a = priors[res_key][par_key]['a']
                    b = priors[res_key][par_key]['b']
                    self.prior_funcs.append(lambda x: uniform(x, a, b))
                elif priors[res_key][par_key]['kind'] == 'normal':
                    mu = priors[res_key][par_key]['mu']
                    sigma = priors[res_key][par_key]['sigma']
                    self.prior_funcs.append(lambda x: normal(x, mu, sigma))
                elif priors[res_key][par_key]['kind'] == 'truncnorm':
                    mu = priors[res_key][par_key]['mu']
                    sigma = priors[res_key][par_key]['sigma']
                    a, b = -np.inf, np.inf
                    try:
                        a = priors[res_key][par_key]['lower']
                    except KeyError:
                        pass
                    try:
                        b = priors[res_key][par_key]['upper']
                    except KeyError:
                        pass
                    self.prior_funcs.append(lambda x: truncnorm(x, mu, sigma,
                                                                a, b))
                elif priors[res_key][par_key]['kind'] == 'exponential':
                    sigma = priors[res_key][par_key]['sigma']
                    self.prior_funcs(lambda x: exponential(x, sigma))

        def priors(self, param: ndarray) -> ndarray:
            assert len(self.prior_funcs) == len(param), \
                "Expected %d RVs, not %d" % (len(self.prior_funcs), len(param))
            return np.array([func(x) for func, x in zip(self.prior_funcs,
                                                        param)])


class Interpolated_model(With_UB):
    def __init__(self, priors, upbend, gdr):
        super().__init__(priors=priors, upbend=upbend)
        self.gdr = gdr

    def __call__(self):
        raise NotImplementedError


class GLO_model(GSF_model):
    def __init__(self, priors, upbend):
        super().__init__(name="GLO_model", priors=priors)
        self.upbend = upbend

    def __call__(self, E, gdr_mean, gdr_width, gdr_size, T,
                 pdr_mean, pdr_width, pdr_size,
                 sf_mean, sf_width, sf_size, sm_scale):
        gsf = GSFmodel.GLO(E, gdr_mean, gdr_width, gdr_size, T)
        gsf += GSFmodel.SLO(E, pdr_mean, pdr_width, pdr_size)
        gsf += GSFmodel.SLO(E, sf_mean, sf_width, sf_size)
        gsf += self.upbend(E)*sm_scale
        return gsf


class SMLO_model(GSF_model):
    def __init__(self, priors, upbend):
        super().__init__(name="SMLO_model", priors=priors)
        self.upbend = upbend

    def __call__(self, E, gdr_mean, gdr_width, gdr_size, T,
                 pdr_mean, pdr_width, pdr_size,
                 sf_mean, sf_width, sf_size, sm_scale):
        gsf = GSFmodel.SMLO(E, gdr_mean, gdr_width, gdr_size, T)
        gsf += GSFmodel.SLO(E, pdr_mean, pdr_width, pdr_size)
        gsf += GSFmodel.SLO(E, sf_mean, sf_width, sf_size)
        gsf += self.upbend(E)*sm_scale
        return gsf


class QRPA_model(GSF_model):
    def __init__(self, priors, upbend, gdr):
        super().__init__(name="QRPA_model", priors=priors)
        self.upbend = upbend
        self.gdr = gdr

    def __call__(self, E, gdr_scale, gdr_shift,
                 pdr_mean, pdr_width, pdr_size,
                 sf_mean, sf_width, sf_size,
                 sm_scale):
        gsf = self.gdr(E-gdr_shift)*gdr_scale
        gsf += GSFmodel.SLO(E, pdr_mean, pdr_width, pdr_size)
        gsf += GSFmodel.SLO(E, sf_mean, sf_width, sf_size)
        gsf += self.upbend(E)*sm_scale
        return gsf


class QTBA_model(Interpolated_model):
    def __init__(self, priors, upbend, gdr):
        super().__init__(name="QTBA_model", priors=priors)
        self.upbend = upbend
        self.gdr = gdr

    def __call__(self, E, gdr_scale, gdr_shift,
                 sf_mean, sf_width, sf_size,
                 sm_scale):
        gsf = self.gdr(E-gdr_shift)*gdr_scale
        gsf += GSFmodel.SLO(E, sf_mean, sf_width, sf_size)
        gsf += self.upbend(E)*sm_scale
        return gsf


class QRPA_model_Goriely19(GSF_model):
    def __init__(self, priors, E1, M1, U: Optional[float] = None):
        super().__init__(name='QRPA_model_Goriely19', priors=priors)
        self.E1 = E1
        self.M1 = M1
        self.U = U

    def __call__(self, E, f0, E0, C, n,
                 U: Optional[float] = None):
        U = self.self_if_none(U)
        gsf = self.E1(E) + f0*U/(1 + np.exp(E - E0))
        gsf += self.M1(E) + C*np.exp(-n*E)
        return gsf

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)


@nb.jit(nopython=True)
def SMLO(E: Union[float, ndarray], mean: float, width: float,
         size: float, T: float) -> Union[float, ndarray]:
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


@nb.jit(nopython=True)
def UB_analytical(E: ndarray, C: float, n: float) -> ndarray:
    """ Analytical form for an exponential upbend.
    Args:
        C: Scale of the upbend [MeV^-3]
        n: Slope of the upbend [1/MeV]
    """
    return C*np.exp(-n*E)
