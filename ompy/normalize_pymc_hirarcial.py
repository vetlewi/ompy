import numpy as np
import pymc3 as pm
import theano
import copy
import logging
import termtables as tt
import json
import pymultinest
import matplotlib.pyplot as plt
import warnings
from contextlib import redirect_stdout
from numpy import ndarray
from scipy import stats
from scipy.optimize import differential_evolution, curve_fit
from typing import Optional, Tuple, Any, Union, Callable, Dict
from pathlib import Path

from scipy.stats import truncnorm
from .vector import Vector
from .library import self_if_none, log_interp1d
from .spinfunctions import SpinFunctions
from .filehandling import load_discrete
from .models import ResultsNormalized, NormalizationParameters
from .abstract_normalizer import AbstractNormalizer
from .GamGamIntegrator import GamGamIntegrator,SpinDist,GamGamFuctional, GamGam
from .gsf_functions import SLMO_model, SLO_model, UB_model
from .extractor import Extractor

def prepare_data(nlds: list[Vector], )

class NormalizerPYMC(AbstractNormalizer):
    """ A re-implementation of the NormalizeNLD class using the
        pyMC3 rather than Multinest. 
    """
    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, extractor: Optional[Extractor] = None,
                 discrete: Optional[Union[str, Vector]] = None,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers',
                 regenerate: bool = False,
                 model: str = 'CT',
                 norm_pars: Optional[NormalizationParameters] = None) -> None:
        """ Normalize NLD (and slope of gSF)

        Args:
            nld:
            discrete:
            path:
            regenerate:
            norm_pars:
        """
        super().__init__(regenerate)


        self._discrete = None
        self._cumulative = None
        self._discrete_path = None
        self._D0 = None
        self._smooth_levels_fwhm = None
        self.norm_pars = norm_pars
        self.model = model
        self.smooth_levels_fwhm = 0.1
        self.extractor = None if extractor is None else copy.deepcopy(extractor)
        self.nld = None if nld is None else nld.copy()
        self.gsf = None if gsf is None else gsf.copy()
        self.discrete = discrete

        self.res = ResultsNormalized(name="Results NLD")
        self.trace: any = None
        self.norm_pars = norm_pars

        self.limit_low = None
        self.limit_high = None

        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True, parents=True)

    def normalize(self, *, limit_low: Optional[Tuple[float, float]] = None,
                  limit_high: Optional[Tuple[float, float]] = None,
                  extractor: Optional[Extractor] = None,
                  discrete: Optional[Vector] = None,
                  norm_pars: Optional[NormalizationParameters] = None,
                  regenerate: Optional[bool] = None,
                  E_pred: ndarray = np.linspace(0, 20., 1001)
                  **kwargs) -> any:


        regenerate = self.self_if_none(regenerate)

        if not regenerate:
            try:
                self.load()
                return
            except FileNotFoundError:
                pass

        # Update internal state
        self.limit_low = self.self_if_none(limit_low)
        limit_low = self.limit_low

        self.limit_high = self.self_if_none(limit_high)
        limit_high = self.limit_high

        discrete = self.self_if_none(discrete)
        discrete.to_MeV()
        
    def setup_ct_model(self, num, data_low, data_high) -> pm.Model:

        with pm.Model() as model:

            # Set the data from the lower region fitting the discrete states
            E_low = pm.Data('E_low', data_low['E_st']) # This has shape (M,N)
            q_low = pm.Data('q_low', data_low['q_st']) # This has shape (M,N)
            qerr_low = None

            # Set the data from the higher region fitting the CT model
            E_high = pm.Data('E_high', data_high['E_st']) # This has 
            q_high = pm.Data('q_high', data_high['q_st'])
            qerr_high = None

            # Hyper-priors (ie. the physics)
            T = pm
            c0 = pm.Normal('c0', mu=0, sigma=100)
            c1 = pm.Normal('c1', mu=0, sigma=100)

            # Other solutions
            b0 = pm.Normal('b0', mu=0, sigma=100, num=num) # alpha'
            b1 = pm.Normal('b1', mu=0, sigma=100, num=num) # D'


    def sigma(self, E, a, Eshift):
        sigmaSq = 0.0146*self.norm_pars.A**(5./3.)
        sigmaSq *= (1 + pm.math.sqrt(1 + 4*a*(E-Eshift)))/(2*a)
        return pm.math.sqrt(sigmaSq)

    def rhoBSFG(self, E, a, Eshift):
            rhoBSFG = pm.math.exp(2*pm.math.sqrt(1 + 4*a*(E-Eshift)))
            rhoBSFG /= (12.*np.sqrt(2)*self.sigma(E, a, Eshift)*a**(1./4.)*(E-Eshift)**(5./4.))
            return rhoBSFG

    def plot(self, *, ax: Any = None,
             add_label: bool = True,
             trace: Any = None,
             add_figlegend: bool = True,
             plot_fitregion: bool = True,
             reset_color_cycle: bool = True,
             **kwargs) -> Tuple[Any, Any]:
        """ Plot results from the inference.
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if reset_color_cycle:
            ax.set_prop_cycle(None)

        trace = self.self_if_none(trace)

        labelNld = '_exp.'
        labelNldSn = None
        labelConf = "_68%"
        labelModel = "_model"
        labelDiscrete = "_known levels"
        if add_label:
            labelNld = 'exp.'
            labelConf = "68%"
            labelNldSn = r'$\rho(S_n)$'
            labelModel = 'model'
            labelDiscrete = "known levels"
        nld_exp = Vector(E=self.nld.E, values=trace['rho_norm'].mean(axis=0))
        nld_exp.plot(ax=ax, label=labelNld, scale='log', **kwargs)
        ax.fill_between(self.nld.E, trace['rho_norm'].mean(axis=0)-trace['rho_norm'].std(axis=0),
                        trace['rho_norm'].mean(axis=0)+trace['rho_norm'].std(axis=0), alpha=0.3, label=labelConf)

        self.discrete.plot(ax=ax, label=labelDiscrete, kind='step', c='k')

        ax.plot(self.model_range, np.quantile(trace['model'], 0.5, axis=0), label=labelModel)
        ax.fill_between(self.model_range, np.quantile(trace['model'], 0.16, axis=0),
                        np.quantile(trace['model'], 0.84, axis=0), alpha=0.2)
        ax.errorbar(self.norm_pars.Sn[0], np.quantile(trace['rhoSn'], 0.5),
                    yerr=np.array([[np.quantile(trace['rhoSn'], 0.5)-np.quantile(trace['rhoSn'], 0.16),
                                    np.quantile(trace['rhoSn'], 0.84)-np.quantile(trace['rhoSn'], 0.5)]]).T,
                    label=labelNldSn, fmt="ks", markerfacecolor='none')
        ax.semilogy()

        return fig, ax


    @property
    def discrete(self) -> Optional[Vector]:
        return self._discrete

    @property
    def cumulative(self):
        return self._cumulative

    @discrete.setter
    def discrete(self, value: Optional[Union[Path, str, Vector]]) -> None:
        if value is None:
            self._discretes = None
            self.LOG.debug("Set `discrete` to None")
        elif isinstance(value, (str, Path)):
            if self.nld is None:
                raise ValueError(f"`nld` must be set before loading levels")
            nld = self.nld.copy()
            nld.to_MeV()
            self.LOG.debug("Set `discrete` levels from file with FWHM %s",
                           self.smooth_levels_fwhm)
            self._discrete = load_levels_smooth(value, nld.E,
                                                self.smooth_levels_fwhm)
            self._discrete.units = "MeV"
            self._discrete_path = value
            self._cumulative = load_cumulative_discrete(value)

        elif isinstance(value, Vector):
            if self.nld is not None and np.any(self.nld.E != value.E):
                raise ValueError("`nld` and `discrete` must"
                                 " have same energy binning")
            self._discrete = value
            self._cumulative = value.to_cumulative(factor='de', inplace=False)
            self.LOG.debug("Set `discrete` by Vector")
        else:
            raise ValueError(f"Value {value} is not supported"
                             " for discrete levels")

    @property
    def smooth_levels_fwhm(self) -> Optional[float]:
        return self._smooth_levels_fwhm

    @smooth_levels_fwhm.setter
    def smooth_levels_fwhm(self, value: float) -> None:
        self._smooth_levels_fwhm = value
        if self._discrete_path is not None:
            self.discrete = self._discrete_path

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)

def estimate_B(gsf: Vector, model: Callable[..., any]) -> Tuple[float,float]:
    """ A handy function to get a rough estimate for the B
        parameter needed for normalizing the gSF.
    Args:
        gsf: The 'semi-normalized' gsf, meaning after applying
            the slope normalization (f*exp(alpha*Eg))
        model: A callable that estimates the roughly expected
            gsf given an energy Eg.
    Returns:
        mean: The mean B parameter
        std: The std of the B parameter
    """

    B = model(gsf.E)/gsf.values
    return B.mean(), B.std()

def estimate_upbend(gsf: Vector, model: Callable[..., any]) -> Tuple[float,float]:
    """ Estimate the approximate slope and constant parameter needed
        to best fit the upbend, fUB(Eg) = size*exp(-slope*Eg)/(3*pi^2*hbar^2*c^2).
    Args:
        gsf: A roughly normalized gsf.
        model: A callable model that doesnt include the upbend.
    Returns:
        slope: The slope of the upbend (1/MeV).
        size: The strength of the upbend (mb/MeV).
    """
    factor = 8.6737e-8
    only_ub = gsf.values - model(gsf.E)
    #gsf.values = only_ub
    #gsf.plot(scale='log')

    # Ensure only positive values are keept
    E_fit = gsf.E[only_ub > 0]
    only_ub = only_ub[only_ub > 0]

    # Estimate the parameters with a common least sq.
    p = np.polyfit(E_fit, np.log(only_ub/8.6737e-8), 1) 
    slope = -p[0]
    size = np.exp(p[1])
    return slope, size


def load_cumulative_discrete(path: Union[str, Path]) -> Vector:
    """ Load the cumulative number of levels without smoothing.
    Args:
        path: The file to load
    Returns:
        A vector with the cumulative number of levels.
    """

    energies = np.loadtxt(path)
    energies /= 1e3  # convert to MeV
    num = np.cumsum(np.ones(energies.shape))
    return Vector(values=num, E=energies)

def load_levels_discrete(path: Union[str, Path], energy: ndarray) -> Vector:
    """ Load discrete levels without smoothing

    Assumes linear equdistant binning

    Args:
        path: The file to load
        energy: The binning to use
    Returns:
        A vector describing the levels
    """
    histogram, _ = load_discrete(path, energy, 0.1)
    return Vector(values=histogram, E=energy)


def load_levels_smooth(path: Union[str, Path], energy: ndarray,
                       resolution: float = 0.1) -> Vector:
    """ Load discrete levels with smoothing

    Assumes linear equdistant binning

    Args:
        path: The file to load
        energy: The binning to use in MeV
        resolution: The resolution (FWHM) of the smoothing to use in MeV
    Returns:
        A vector describing the smoothed levels
    """
    histogram, smoothed = load_discrete(path, energy, resolution)
    return Vector(values=smoothed if resolution > 0 else histogram, E=energy)

