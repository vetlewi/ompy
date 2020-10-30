import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Any, Union, Callable, Dict, List
from inspect import signature

from .likelihood import Likelihood, OsloNormalLikelihood, NormalLikelihood
from .physics_models import model
from .vector import Vector
from .models import ResultsNormalized, NormalizationParameters


class Likelihoods:

    def __init__(self, N_free):
        self.N_free = N_free

    def logp(self, param: ndarray) -> float:
        raise NotImplementedError

    def logp_pointwise(self, param: ndarray) -> ndarray:
        raise NotImplementedError

    def WIAC(self, samples: ndarray) -> float:
        raise NotImplementedError

    def loglike(self, cube, ndim, nparams) -> float:
        param = np.array([cube[i] for i in range(ndim)])
        return self.logp(param)

    def __call__(self, param) -> float:
        return self.logp(param)


class Likelihoods_NoGamGam(Likelihoods):

    def __init__(self, nld: Vector,
                 gsf: Vector,
                 nld_ref: Vector,
                 nld_limit_low: Tuple[float, float],
                 nld_limit_high: Tuple[float, float],
                 gsf_limit: Tuple[float, float],
                 nld_model: model,
                 gsf_model: model,
                 ext_nld: Optional[Vector] = None,
                 ext_gsf: Optional[Vector] = None,
                 norm_pars: Optional[NormalizationParameters] = None) -> None:
        """
        """

        super().__init__(N_free=3+nld_model.N_free+gsf_model.N_free)

        nld_low = nld.cut(*nld_limit_low, inplace=False)
        nld_high = nld.cut(*nld_limit_high, inplace=False)
        gsf_fit = gsf.cut(*gsf_limit, inplace=False)

        self.discrete = np.array(nld_ref.cut(*nld_limit_low,
                                             inplace=False).values)

        self.nld_low_like = OsloNormalLikelihood(x=nld_low.E,
                                                 y=nld_low.values,
                                                 yerr=nld_low.std,
                                                 model=self.discrete_model)

        self.nld_high_like = OsloNormalLikelihood(x=nld_high.E,
                                                  y=nld_high.values,
                                                  yerr=nld_high.std,
                                                  model=nld_model)

        self.gsf_like = OsloNormalLikelihood(x=gsf_fit.E,
                                             y=gsf_fit.values,
                                             yerr=gsf_fit.std,
                                             model=gsf_model)

        if ext_nld is not None:
            self.ext_nld = NormalLikelihood(x=ext_nld.E, y=ext_nld.values,
                                            yerr=ext_nld.std, model=nld_model)
        else:
            self.ext_nld = lambda *par: 0

        if ext_gsf is not None:
            self.ext_gsf = NormalLikelihood(x=ext_gsf.E, y=ext_gsf.values,
                                            yerr=ext_gsf.std, model=gsf_model)
        else:
            self.ext_gsf = lambda *par: 0

        def GetSignature(func: Callable[..., any]) -> List[str]:
            return list(dict(signature(func).parameters).keys())

        nld_sig = GetSignature(nld_model)
        gsf_sig = GetSignature(gsf_model)

        # First we need to remove the first argument
        # (pressumably this is the Ex/Eg argument)
        del nld_sig[0]
        del gsf_sig[0]

        self.norm_mask = np.arange(3)

        # Setup mask
        self.nld_model_mask = np.arange(len(nld_sig)) + 3
        self.gsf_model_mask = np.arange(len(gsf_sig)) + 3 + \
            len(nld_sig)

        # Check if gSF requires temperature (T)
        if 'T' in gsf_sig and 'T' in nld_sig and gsf_model.need_T:
            self.gsf_model_mask[gsf_sig.index('T')] = \
                self.nld_model_mask[nld_sig.index('T')]
            self.gsf_model_mask[gsf_sig.index('T')+1:] -= 1
            del gsf_sig[gsf_sig.index('T')]
        elif 'T' in gsf_sig and 'T' in nld_sig and not gsf_model.need_T:
            gsf_sig[gsf_sig.index('T')] = 'T_gsf'
        self.names = ['A', 'B', 'α']
        self.names += nld_sig
        self.names += gsf_sig

    def discrete_model(self, E: ndarray) -> ndarray:
        return self.discrete

    def logp(self, param: ndarray) -> float:

        A, B, alpha = param[:3]
        nld_parameters = param[self.nld_model_mask]
        gsf_parameters = param[self.gsf_model_mask]

        logp = self.nld_low_like(A, alpha)
        logp += self.nld_high_like(A, alpha, *nld_parameters)
        logp += self.gsf_like(B, alpha, *gsf_parameters)
        logp += self.ext_nld(*nld_parameters)
        logp += self.ext_gsf(*gsf_parameters)

        if logp != logp:
            return -np.inf
        return logp

    def logp_pointwise(self, param: ndarray) -> ndarray:

        # We get the parameters
        A, B = param[self.A_mask], param[self.B_mask]
        alpha = param[self.alpha_mask]

        nld_parameters = param[self.nld_model_mask]
        gsf_parameters = param[self.gsf_model_mask]

        logp = [self.nld_low_like.logp_pointwise(A, alpha),
                self.nld_high_like.logp_pointwise(A, alpha, *nld_parameters),
                self.gsf_like(B, alpha, *gsf_parameters),
                self.ext_nld(*nld_parameters),
                self.ext_gsf(*gsf_parameters)]

        return np.concatenate(logp, axis=None)

    def WIAC(self, samples: ndarray) -> float:
        """ Compute the widely applicable information criterion of the model.
        Args:
            samples: A 2D array with each element along the first axis is the
                set of samples of the posterior.
        """
        logp_pointwise = [self.logp_pointwise(sample) for sample in samples]
        logp_pointwise = np.array(logp_pointwise)

        # Each row now contains the logp for each observed value.
        # We want to sum each of the columns.

        lppd = np.sum(np.exp(logp_pointwise), axis=0)
        lppd = np.sum(np.log(lppd/len(samples)))

        logp_mean = np.mean(logp_pointwise, axis=0, keepdims=True)
        p_WIAC = np.sum((logp_pointwise - logp_mean)**2, axis=0)
        p_WIAC = np.sum(p_WIAC/(len(samples) - 1))

        return -2.*lppd + 2*p_WIAC
