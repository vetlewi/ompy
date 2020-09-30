import numpy as np
import numba as nb
from numpy import ndarray
from inspect import signature
from typing import Optional, Tuple, Any, Union, Callable, Dict, List
import scipy.stats as stats

from .vector import Vector
from .library import log_interp1d
from .models import (ExtrapolationModelLow, ExtrapolationModelHigh,
                     NormalizationParameters)


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
                 gsf_model: Callable[..., ndarray],
                 spc_model: Optional[Callable[..., ndarray]] = None,
                 norm_pars: Optional[NormalizationParameters] = None) -> None:
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

        cum_lim = 4.0

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
        self.nld_idx = make_idx(self.nld)
        self.gsf_idx = make_idx(self.gsf)

        self.N, self.M = self.nld.shape
        self.bin_size = np.diff(nlds[0].E)[0]

        self.nld_E_low, self.nld_low = as_array(nlds, nld_limit_low)
        self.gsf_E_low, self.gsf_low = as_array(gsfs, gsf_limit_low)
        self.nld_idx_low = make_idx(self.nld_low)
        self.gsf_idx_low = make_idx(self.gsf_low)

        self.nld_E_cum, self.nld_cum = as_array(nlds, (0, cum_lim))
        self.nld_idx_cum = make_idx(self.nld_cum)
        self.dx = np.diff(nlds[0].E)[0]

        self.nld_E_high, self.nld_high = as_array(nlds, nld_limit_high)
        self.gsf_E_high, self.gsf_high = as_array(gsfs, gsf_limit_high)
        self.nld_idx_high = make_idx(self.nld_high)
        self.gsf_idx_high = make_idx(self.gsf_high)

        self.nld_ref = nld_ref.copy()
        nld_low_ref = nld_ref.cut(*nld_limit_low, inplace=False)
        nld_cum_ref = nld_ref.cut(0, cum_lim, inplace=False)
        self.nld_low_ref = np.tile(nld_low_ref.values, (self.N, 1))
        self.nld_cum_ref = np.tile(np.cumsum(nld_cum_ref.values), (self.N, 1))
        self.nld_cum_ref *= np.diff(nld_ref.E)[0]

        self.gsf_ref = gsf_ref.copy()
        gsf_low_ref = gsf_ref.cut(*gsf_limit_low, inplace=False)
        self.gsf_low_ref = np.tile(gsf_low_ref.values, (self.N, 1))

        self.nld_model = nld_model
        self.gsf_model = gsf_model
        self.spc_model = spc_model

        self.N_nld_par = len(signature(self.nld_model).parameters)-1
        self.N_gsf_par = len(signature(self.gsf_model).parameters)-1
        self.N_spc_par = len(signature(self.spc_model).parameters)-1

        self.norm_pars = norm_pars if norm_pars is not None else None

        self.Gg0_dist = lambda x: np.sum(((self.norm_pars.Gg[0]-x)/self.norm_pars.Gg[1])**2)

        # Or
        # self.Gg0_dist = lambda x: self.truncnorm(x, 5, np.inf, 25., 250.)

        if norm_pars is not None:
            self.Sn = norm_pars.Sn[0]
            self.integrateE = np.linspace(0, norm_pars.Sn[0], 1001)
            gsfs_extrapolated = self.extrapolate(gsfs, gsf_limit_low,
                                                 gsf_limit_high,
                                                 self.Sn,
                                                 self.integrateE)

            self.gsf_E_ext, self.gsf_ext = as_array(gsfs_extrapolated)
            self.gsf_idx_ext = make_idx(self.gsf_ext)

            self.nld_extr_E = np.arange(max(nlds[0].E)+np.diff(nlds[0].E)[0],
                                        self.Sn+np.diff(nlds[0].E)[0],
                                        np.diff(nlds[0].E)[0])

            self.nld_extr_E_all = np.concatenate((nlds[0].E, self.nld_extr_E))
            self.nld_extr_E = np.tile(self.nld_extr_E, (self.N, 1))
            self.integrateE_MG = np.tile(self.integrateE, (self.N, 1))
            self.nld_int_E = np.tile(np.linspace(0, norm_pars.Sn[0], 1001),
                                     (self.N, 1))

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
        N = self.N
        Nnld, Ngsf, Nspc = self.N_nld_par, self.N_gsf_par, self.N_spc_par
        A, B, alpha = param[0:N], param[N:2*N], param[2*N:3*N]
        nld_param = param[3*N:3*N+Nnld]
        gsf_param = param[3*N+Nnld:3*N+Nnld+Ngsf]
        spc_param = param[3*N+Nnld+Ngsf:3*N+Nnld+Ngsf+Nspc]
        renorm = 0
        gsf_param_ = gsf_param
        if Ngsf == 2:
            gsf_param_ = gsf_param
        elif Ngsf > 1:
            gsf_param_ = [gsf_param[0], (gsf_param[1], nld_param[0]),
                          gsf_param[2], gsf_param[3], gsf_param[4],
                          gsf_param[5], gsf_param[6]]
            renorm = gsf_param[-1]
        else:
            renorm = gsf_param

        A, B, alpha, T, Eshift, sm_shift = self.unpack_param(param)

        nld_error_low = self.nld_error_low(A, alpha)
        #gsf_error_low = self.gsf_error_low(B, alpha, renorm)

        nld_error_high = self.nld_error_high(A, alpha, nld_param)
        gsf_error_high = self.gsf_error_high(B, alpha, gsf_param_)

        nld_error_cum = self.nld_error_cum(A, alpha)

        #print(gsf_error_low, " ", gsf_error_high)

        #Gg0_error = self.Gg_error(A, B, alpha, nld_param, spc_param)

        logp = -0.5*(nld_error_low + nld_error_cum + nld_error_high
                     + gsf_error_high)

        if logp != logp:
            return -np.inf

        return logp

    def nld_error_low(self, A: ndarray, alpha: ndarray) -> float:
        """
        """
        exp = self.nld_low*self.evaluate_norm_lin(self.nld_E_low,
                                                  self.nld_idx_low,
                                                  A, alpha)
        return self.error(exp, self.nld_low_ref, 0.3)

    def nld_error_cum(self, A: ndarray, alpha: ndarray) -> float:
        """
        """
        exp = self.nld_cum*self.evaluate_norm_lin(self.nld_E_cum,
                                                  self.nld_idx_cum,
                                                  A, alpha)
        exp = np.cumsum(exp, axis=1)*self.dx
        return self.error(exp, self.nld_cum_ref, 0.3)

    def gsf_error_low(self, B: ndarray, alpha: ndarray,
                      renorm: float = 1.0) -> float:
        """
        """
        exp = self.gsf_low*self.evaluate_norm_lin(self.gsf_E_low,
                                                  self.gsf_idx_low,
                                                  B, alpha)
        return self.error(exp, self.gsf_low_ref*renorm, 0.3)

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
        if self.N_gsf_par == 1:
            return 0
        exp = self.gsf_high*self.evaluate_norm_lin(self.gsf_E_high,
                                                   self.gsf_idx_high,
                                                   B, alpha)
        return self.error(exp, self.gsf_model(self.gsf_E_high, *gsf_par), 0.5)

    def Gg_error(self, A: ndarray, B: ndarray, alpha: ndarray,
                 nld_par: any, spc_par: any) -> float:
        """
        """

        # Normalize the gsf
        gsf = self.gsf_ext*self.evaluate_norm_lin(self.gsf_E_ext,
                                                  self.gsf_idx_ext,
                                                  B, alpha)

        # Normalize the NLD
        nld = self.nld*self.evaluate_norm_lin(self.nld_E,
                                              self.nld_idx,
                                              A, alpha)

        # Calculate the model density
        nld_model = self.nld_model(self.nld_extr_E, *nld_par)
        nld = np.concatenate((nld, nld_model), axis=1)
        interp = log_interp1d(self.nld_extr_E_all, nld, axis=1,
                              bounds_error=False,
                              fill_value=(-np.inf, -np.inf))

        nld = interp(self.Sn - self.integrateE)
        nld *= self.spc_model(np.tile(self.Sn - self.integrateE, (self.N, 1)),
                              *spc_par)

        # Perform the actual integration
        Gg0 = np.sum(nld*gsf, axis=1)*np.diff(self.integrateE)[0]  # Unitless
        Gg0 *= self.D0_from_model(nld_par, spc_par)/2  # <Gg0> in meV

        return np.sum(((self.norm_pars.Gg[0]-Gg0)/self.norm_pars.Gg[1])**2)
        #return self.Gg0_dist(Gg0)

    def D0_from_model(self, nld_par: any, spc_par: any) -> float:
        """ Calculate the D0 parameter in [meV]
        """

        rhoSn = self.nld_model(self.norm_pars.Sn[0], *nld_par)
        spin_part = self.spc_model(self.norm_pars.Sn[0], *spc_par)
        return 1e9/(rhoSn*spin_part)

    def truncnorm(self, x, a, b, loc, scale):
        a = (a-loc)/scale
        b = (b-loc)/scale
        logp = stats.truncnorm.logpdf(x, a, b, loc, scale)
        if logp != logp:
            return -np.inf
        return logp

    @staticmethod
    def extrapolate(gsfs, gsf_limit_low, gsf_limit_high, Sn,
                    grid) -> List[Vector]:

        extr_low = ExtrapolationModelLow('low')
        extr_high = ExtrapolationModelHigh('high')

        extr_low.Efit = gsf_limit_low
        extr_high.Efit = gsf_limit_high

        de = np.diff(gsfs[0].E)[0]
        e_low = np.arange(0, np.min(gsfs[0].E), de)
        e_high = np.arange(np.max(gsfs[0].E)+de, Sn+de, de)

        e_all = np.concatenate((e_low, gsfs[0].E, e_high))

        extr_gsfs = []
        for gsf in gsfs:
            extr_low.fit(gsf)
            extr_high.fit(gsf)

            low_val = extr_low.extrapolate(scaled=False, E=e_low).values
            high_val = extr_high.extrapolate(scaled=False, E=e_high).values
            values = np.concatenate((low_val, gsf.values, high_val))

            # Interpolation with grid.
            gsf_intrp = log_interp1d(e_all, values, bounds_error=False,
                                     fill_value=(-np.inf, -np.inf))

            extr_gsfs.append(Vector(E=grid, values=gsf_intrp(grid),
                                    units='MeV'))
        return extr_gsfs

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




