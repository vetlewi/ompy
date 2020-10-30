import numpy as np
from numpy import ndarray
from inspect import signature
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence


from .physics_models import model


class Priors:
    def __call__(self, param: ndarray) -> ndarray:
        raise NotImplementedError

    def prior(self, cube, ndim, nparam):
        param = np.array([cube[i] for i in range(ndim)])
        param = self.__call__(param)
        for i, par in enumerate(param):
            cube[i] = par
        return cube


class Priors_NoGamGam(Priors):

    def __init__(self, A: Callable[[float], float],
                 B: Callable[[float], float],
                 alpha: Callable[[float], float],
                 nld_model: model,
                 gsf_model: model):

        self.A = A
        self.B = B
        self.alpha = alpha

        self.nld_model = nld_model
        self.gsf_model = gsf_model

        def GetSignature(func: Callable[..., any]) -> List[str]:
            return list(dict(signature(func).parameters).keys())

        nld_signature = GetSignature(nld_model)
        gsf_signature = GetSignature(gsf_model)

        # First we need to remove the first argument
        # (pressumably this is the Ex/Eg argument)
        del nld_signature[0]
        del gsf_signature[0]

        if 'T' in gsf_signature and 'T' in nld_signature and gsf_model.need_T:
            del gsf_signature[nld_signature.index('T')]

        # Setup mask
        self.nld_model_mask = np.arange(len(nld_signature)) + 3
        self.gsf_model_mask = np.arange(len(gsf_signature)) + 3 + \
            len(nld_signature)

    def __call__(self, param: ndarray) -> ndarray:
        param[0] = self.A(param[0])
        param[1] = self.B(param[1])
        param[2] = self.alpha(param[2])
        param[self.nld_model_mask] = \
            self.nld_model.prior(param[self.nld_model_mask])
        param[self.gsf_model_mask] = \
            self.gsf_model.prior(param[self.gsf_model_mask])
        return param
