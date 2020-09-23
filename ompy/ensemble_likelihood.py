import numpy as np
import numba as nb
from numpy import ndarray
from inspect import signature
from typing import Optional, Tuple, Any, Union, Callable, Dict, List

from .vector import Vector


class EnsembleLikelihood:
    """ A class to make life easier when evaluating a likelihood
    function by storing all the arguments needed at creation time.
    """

    def __init__(self, nlds: List[Vector], gsfs: List[Vector],
                 nld_limit_low: Tuple[float, float],
                 nld_limit_high: Tuple[float, float],
                 gsf_limit_low: Tuple[float, float],
                 gsf_limit_high: Tuple[float, float],
                 nld_ref: Vector, gsf_ref: Vector,
                 nld_model: Callable[..., ndarray],
                 gsf_model: Callable[..., ndarray]) -> None:
        """ Setup parameters for the log likelihood.

        Args:
            nlds: List with NLD vectors
            gsfs: List with GSF vectors
            nld_limit_low: Tuple with lower and upper energy of the NLD to be
                included in the normalization fit (determination of A, alpha).
            nld_limit_high: Tuple with the lower and upper energy of the NLD to
                be included in model determination.
            gsf_limit_low: Tuple with lower and upper energy of the GSF to be
                included in the normalization fit (determination of B, alpha).
            gsf_limit_high: Tuple with the lower and upper energy of the GSF to
                be included in model determination. (Not yet implemented)
            nld_ref: Vector with the level density to which we will compare NLD
                to determine normalization parameters. Usually this will be the
                level density from discrete levels.
            gsf_ref: Vector with the strength function to which we will compare
                GSF to determine normalization parameters.
            nld_model: Callable NLD model.
            gsf_model: Callable GSF model. (Not yet implemented)

        TODO:
            - Implement gSF model fit.

        """

        assert len(nlds) == len(gsfs),\
            "Must have the same number of NLDs as GSFs."

        def as_array(vecs, cut=None):
            if cut is not None:
                vecs = [vec.cut(*cut, inplace=False) for vec in vecs]
            E = np.array([vec.E for vec in vecs])
            values = np.array([vec.values for vec in vecs])
            return E, values

        def make_idx(mat):
            N, M = mat.shape
            return np.repeat(range(N), M).reshape(N, M)

        self.nld_E, self.nld = as_array(nlds)
        self.gsf_E, self.gsf = as_array(gsfs)

        self.N, self.M = self.nld.shape

        self.nld_E_low, self.nld_low = as_array(nlds, nld_limit_low)
        self.gsf_E_low, self.gsf_low = as_array(gsfs, gsf_limit_low)
        self.nld_idx_low = make_idx(self.nld_low)
        self.gsf_idx_low = make_idx(self.gsf_low)

        self.nld_E_high, self.nld_high = as_array(nlds, nld_limit_high)
        self.gsf_E_high, self.gsf_high = as_array(gsfs, gsf_limit_high)
        self.nld_idx_high = make_idx(self.nld_high)
        self.gsf_idx_high = make_idx(self.gsf_high)

        self.nld_ref = nld_ref.copy()
        nld_low_ref = nld_ref.cut(*nld_limit_low, inplace=False)
        self.nld_low_ref = np.tile(nld_low_ref.values, (self.N, 1))

        self.gsf_ref = gsf_ref.copy()
        gsf_low_ref = gsf_ref.cut(*gsf_limit_low, inplace=False)
        self.gsf_low_ref = np.tile(gsf_low_ref.values, (self.N, 1))

        self.nld_model = nld_model
        self.gsf_model = gsf_model

        self.N_nld_par = len(signature(self.nld_model).parameters)-1
        self.N_gsf_par = len(signature(self.gsf_model).parameters)-1

    def __call__(self, param: Tuple[float]) -> float:
        return self.evaluate(param)

    def loglike(self, cube, ndim, nparams) -> float:
        """ Method to be used with pyMultiNest. Matches
        the expected function signature.
        """
        return self.evaluate(cube)

    def evaluate(self, param: Tuple[float]) -> float:
        """ Evaluate the likelihood.
        """
        N, Nnld, Ngsf = self.N, self.N_nld_par, self.N_gsf_par

        A, B, alpha = param[0:N], param[N:2*N], param[2*N:3*N]
        nld_param = param[3*N:3*N+Nnld]
        gsf_param = param[3*N+Nnld:3*N+Nnld+Ngsf]

        A, B, alpha, T, Eshift, sm_shift = self.unpack_param(param)

        nld_error_low = self.nld_error_low(A, alpha)
        gsf_error_low = self.gsf_error_low(B, alpha, gsf_param)

        nld_error_high = self.nld_error_high(A, alpha, nld_param)
        gsf_error_high = self.gsf_error_high(B, alpha, gsf_param)

        return -0.5*(nld_error_low + gsf_error_low +
                     nld_error_high + gsf_error_high)

    def nld_error_low(self, A: ndarray, alpha: ndarray) -> float:
        """
        """
        exp = self.nld_low*self.evaluate_norm_lin(self.nld_E_low,
                                                  self.nld_idx_low,
                                                  A, alpha)
        return self.error(exp, self.nld_low_ref)

    def gsf_error_low(self, B: ndarray, alpha: ndarray,
                      renorm: float = 1.0) -> float:
        """
        """
        exp = self.gsf_low*self.evaluate_norm_lin(self.gsf_E_low,
                                                  self.gsf_idx_low,
                                                  B, alpha)
        return self.error(exp, self.gsf_low_ref*renorm)

    def nld_error_high(self, A: ndarray, alpha: ndarray,
                       nld_par: any) -> float:
        """
        """
        exp = self.nld_high*self.evaluate_norm_lin(self.nld_E_high,
                                                   self.nld_idx_high,
                                                   A, alpha)
        return self.error(exp, self.nld_model(self.nld_E_high, *nld_par))

    def gsf_error_high(self, B: ndarray, alpha: ndarray,
                       gsf_par: any) -> float:
        """
        """
        # Not yet implemented. Will only return 0s
        return 0
        exp = self.gsf_high*self.evaluate_norm_lin(self.gsf_E_high,
                                                   self.gsf_idx_high,
                                                   B, alpha)
        return self.error(exp, self.nld_model(self.gsf_E_high, *gsf_par))

    @staticmethod
    def error(data: ndarray, model: ndarray,
              weights: Optional[Union[ndarray, float]] = None) -> float:
        """ A simple function for calculating the unweighted sum.
        Note that to avoid issues with weights, we assume 10 % error if weights
        are None.

        Args:
            data: "Exp." data
            model: Model prediction
            weights: Array or float. If float we assume to be the relative
                error.
        Returns: The ChiSq sum.
        """

        if weights is None:
            weights = 0.1*data
        elif isinstance(weights, float):
            weights = weights*data

        return np.sum(((data-model)/weights)**2)

    @staticmethod
    def evaluate_norm_lin(E: ndarray, idx: ndarray,
                          const: ndarray, slope: ndarray) -> ndarray:
        """ Caculate the normalization factors.
        """
        return const[idx]*np.exp(E*slope[idx])

    def unpack_param(self, param: Tuple[float]) -> Tuple[ndarray, ndarray,
                                                         ndarray, float, float,
                                                         float]:
        """ A helper function to quickly unpack the parameters.

        Args:
            param: List of floats with the model parameters.
            N: Number of realizations of the NLD & gSF.
        Returns:
            A tuple with the parameters.
        """
        N = self.N
        A = np.array(param[0:N])
        B = np.array(param[N:2*N])
        alpha = np.array(param[2*N:3*N])
        T, Eshift, sm_shift = param[3*N:3*N+3]

        return A, B, alpha, T, Eshift, sm_shift


@nb.jit(nopython=True)
def unpack_param(param: List[float], N: int) -> Tuple[ndarray, ndarray,
                                                      ndarray, float, float,
                                                      float]:
    """ A helper function to quickly unpack the parameters.

    Args:
        param: List of floats with the model parameters.
        N: Number of realizations of the NLD & gSF.
    Returns:
        A tuple with the parameters.
    """

    A = np.array(param[0:N])
    B = np.array(param[N:2*N])
    alpha = np.array(param[2*N:3*N])
    T, Eshift, sm_shift = param[3*N:3*N+3]

    return A, B, alpha, T, Eshift, sm_shift

@nb.jit(nopython=True)
def evaluate_norm_lin(E: ndarray, idx: ndarray,
                      const: ndarray, slope: ndarray) -> ndarray:
    return const[idx]*np.exp(E*slope[idx])

@nb.jit(nopython=True)
def error(data: ndarray, model: ndarray,
          weights: Optional[ndarray] = None) -> float:
    """ A simple function for calculating the unweighted sum.
    Note that to avoid issues with weights, we assume 30% if weights are None.
    """
    if weights is None:
        weights = 0.1*data
    return np.sum(((data - model)/weights)**2)




