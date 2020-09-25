import numpy as np
import copy
import logging
import termtables as tt
import json
import pymultinest
import matplotlib.pyplot as plt
import warnings
from contextlib import redirect_stdout
from numpy import ndarray
from scipy.optimize import differential_evolution
from typing import Optional, Tuple, Any, Union, Callable, Dict, List
from pathlib import Path
from inspect import signature

import pandas as pd
from scipy.interpolate import interp1d
from .vector import Vector
from .library import self_if_none
from .spinfunctions import SpinFunctions
from .filehandling import load_discrete
from .models import ResultsNormalized, NormalizationParameters
from .abstract_normalizer import AbstractNormalizer
from .extractor import Extractor
from .ensemble_likelihood import EnsembleLikelihood
from .ensemble_prior import (EnsemblePrior, uniform,
                             normal, truncnorm, exponential)

TupleDict = Dict[str, Tuple[float, float]]


class NormalizerSim(AbstractNormalizer):
    """ This is the docstring!
    """

    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, *,
                 extractor: Optional[Extractor] = None,
                 discrete: Optional[Union[str, Vector]] = None,
                 shell_model: Optional[Union[str, Vector]] = None,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers',
                 regenerate: bool = False,
                 nld_model: str = 'CT',
                 gsf_model: str = 'SM',
                 norm_pars: NormalizationParameters) -> None:
        """ Init docstring
        """
        super().__init__(regenerate)

        self.nlds = None
        self.gsfs = None

        if extractor is not None:
            self.nlds = [nld.copy() for nld in extractor.nld]
            self.gsfs = [gsf.copy() for gsf in extractor.gsf]

        try:
            self.nld = self.nlds[0]
        except any:
            pass

        self._discrete = None
        self._shell_model = None
        self._discrete_path = None
        self._shell_model_path = None
        self._smooth_levels_fwhm = None
        self.norm_pars = norm_pars

        self.nld_limit_low = None
        self.nld_limit_high = None

        self.gsf_limit_low = None
        self.gsf_limit_high = None

        self.smooth_levels_fwhm = 0.1
        self.discrete = discrete
        self.shell_model = shell_model

        self.ub_gen = interp1d(self.shell_model.E,
                               self.shell_model.values,
                               kind='linear', bounds_error=False,
                               fill_value=(0, 0))

        if nld_model.upper() == 'CT':
            self.nld_model = self.const_temperature
            self.nld_prior = lambda x: uniform(x, np.array([0., -10.]),
                                               np.array([5., 10.]))
        elif nld_model.upper() == 'BSFG':
            self.nld_model = \
                lambda E, a, Eshift: self.back_shift_fg(E, norm_pars.A,
                                                        a, Eshift)
            self.nld_prior = lambda x: uniform(x, np.array([0., -10.]),
                                               np.array([15., 10.]))
        else:
            raise NotImplementedError("NLD model '%s' has not yet been \
                implemented." % nld_model)

        if gsf_model.upper() == 'SM':
            self.gsf_model = lambda x, renorm: None
            #self.gsf_prior = lambda x: uniform(x, 1.0, 2.0)
            self.gsf_prior = lambda x: truncnorm(x, 1.0, np.inf, 1.5, 10.)
        elif gsf_model.upper() == 'SM+SMLO':
            try:
                self.setup_slmo_prior(norm_pars)
                self.gsf_model = self.slmo_gsf_model
                self.gsf_prior = self.slmo_prior
            except KeyError:
                print("SLMO priors not found, falling back on SM only.")
                self.gsf_model = lambda x, renorm: None
                self.gsf_prior = lambda x: uniform(x, 1.0, 2.0)
        else:
            raise NotImplementedError("GSF model '%s' has not yet been \
                implemented." % gsf_model)

        self.multinest_path = Path('multinest')
        self.multinest_kwargs: dict = {"seed": 65498, "resume": False}

        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True, parents=True)

    def __call__(self, *args, **kwargs) -> any:
        """ Wrapper for normalize """
        return self.normalize(*args, **kwargs)

    def normalize(self, *, nld_limit_low: Optional[Tuple[float, float]] = None,
                  nld_limit_high: Optional[Tuple[float, float]] = None,
                  gsf_limit_low: Optional[Tuple[float, float]] = None,
                  gsf_limit_high: Optional[Tuple[float, float]] = None,
                  discrete: Optional[Vector] = None,
                  shell_model: Optional[Vector] = None,
                  nlds: Optional[List[Vector]] = None,
                  gsfs: Optional[List[Vector]] = None,
                  norm_pars: Optional[NormalizationParameters] = None) -> any:
        """ Docstring!
        """

        if not self.regenerate:
            try:
                self.load()
                return
            except FileNotFoundError:
                pass

        # Update state
        self.nld_limit_low = self.self_if_none(nld_limit_low)
        self.nld_limit_high = self.self_if_none(nld_limit_high)
        self.gsf_limit_low = self.self_if_none(gsf_limit_low)
        self.gsf_limit_high = self.self_if_none(gsf_limit_high)

        nld_limit_low = self.nld_limit_low
        nld_limit_high = self.nld_limit_high
        gsf_limit_low = self.gsf_limit_low
        gsf_limit_high = self.gsf_limit_high

        discrete = self.self_if_none(discrete)
        shell_model = self.self_if_none(shell_model)
        shell_model.to_MeV()
        discrete.to_MeV()

        self.nlds = self.self_if_none(nlds)
        self.gsfs = self.self_if_none(gsfs)

        nlds = [nld.copy() for nld in self.nlds]
        gsfs = [gsf.copy() for gsf in self.gsfs]

        for nld, gsf in zip(nlds, gsfs):
            nld.to_MeV()
            gsf.to_MeV()

        # We determine the number of NLD model parameters
        N = len(nlds)
        nld_sig = list(dict(signature(self.nld_model).parameters).keys())
        gsf_sig = list(dict(signature(self.gsf_model).parameters).keys())
        N_nld_par = len(nld_sig)-1
        N_gsf_par = len(gsf_sig)-1
        nld_names = nld_sig[1:N_nld_par+1]
        gsf_names = gsf_sig[1:N_gsf_par+1]
        Nvars = 3*N + N_nld_par + N_gsf_par

        self.norm_pars = self.self_if_none(norm_pars)
        self.norm_pars.is_changed(include=["D0", "Sn", "spincutModel",
                                           "spincutPars"])  # check that set

        # First we need to ensure that we have a meaningful priors
        lnl = EnsembleLikelihood(nlds, gsfs, nld_limit_low, nld_limit_high,
                                 gsf_limit_low, gsf_limit_high, discrete,
                                 shell_model, self.nld_model, self.gsf_model)

        def const_prior(x):
            #return normal(x, 1., 10.)
            #return truncnorm(x, 0, np.inf, 1., 10.)
            return 10**uniform(x, -3, 3)
            #return 10**normal(x, 0, 1.)

        prp = EnsemblePrior(A=const_prior,
                            B=const_prior,
                            alpha=lambda x: uniform(x, -10, 10),
                            nld_param=self.nld_prior, gsf_param=self.gsf_prior,
                            N=N, N_nld_par=N_nld_par, N_gsf_par=N_gsf_par)

        names = ['A[%d]' % i for i in range(N)]
        names += ['B[%d]' % i for i in range(N)]
        names += ['alpha[%d]' % i for i in range(N)]
        names += nld_names + gsf_names

        def prior(cube, ndim, nparams):
            prp(cube)

        def loglike(cube, ndim, nparams):
            return lnl(prp(cube))


        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / f"norm_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        self.LOG.info("Starting multinest")
        self.LOG.debug("with following keywords %s:", self.multinest_kwargs)

        self.LOG.write = lambda msg: (self.LOG.info(msg) if msg != '\n'
                                      else None)

        with redirect_stdout(self.LOG):
            pymultinest.run(lnl.loglike, prp.prior, Nvars,
                            outputfiles_basename=str(path),
                            **self.multinest_kwargs)

        json.dump(names, open(str(path) + 'params.json', 'w'))
        analyzer = pymultinest.Analyzer(Nvars,
                                        outputfiles_basename=str(path))

        mnstats = analyzer.get_stats()
        samples = analyzer.get_equal_weighted_posterior()[:, :-1]
        samples = dict(zip(names, samples.T))

        # Format the output
        popt = dict()
        vals = []
        for name, m in zip(names, mnstats['marginals']):
            lo, hi = m['1sigma']
            med = m['median']
            sigma = (hi - lo) / 2
            popt[name] = (med, sigma)
            i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join([fmt + " ± " + fmt])
            vals.append(fmts % (med, sigma))

        return popt, samples

    def slmo_gsf_model(self, E: ndarray,
                       slmo_mean: float,
                       slmo_width: Tuple[float, float],
                       slmo_size: float,
                       sf_mean: float,
                       sf_width: float,
                       sf_size: float,
                       renorm: float) -> ndarray:
        """ Calculate the GSF model we have choosen.
        """
        slmo_width, T = slmo_width
        gsf = self.SLMO(E, slmo_mean, slmo_width, slmo_size, T)
        gsf += self.SLO(E, sf_mean, sf_width, sf_size)
        gsf += self.UB_model(E, renorm)
        return gsf

    @staticmethod
    def const_temperature(E: ndarray, T: float, Eshift: float) -> ndarray:
        """ Constant Temperature NLD"""
        ct = np.exp((E - Eshift) / T) / T
        return ct

    @staticmethod
    def ct_prior(x: ndarray, lower: ndarray, upper: ndarray) -> ndarray:
        return uniform(x, lower, upper)

    @staticmethod
    def back_shift_fg(E: ndarray, A: int, a: float, Eshift: float) -> ndarray:
        """ Back-shift Fermi Gas"""
        sigma = 0.0146*A**(5./3.)
        sigma = np.sqrt((1 + np.sqrt(1 + 4*a*(E-Eshift)))*sigma/(2*a))

        bsfg = np.exp(2*np.sqrt(a*(E-Eshift)))
        bsfg /= (12.*np.sqrt(2)*sigma*a**(1./4.)*(E-Eshift)**(5./4.))
        return bsfg

    @staticmethod
    def SLMO(E: ndarray, mean: float, width: float,
             size: float, T: float) -> ndarray:
        """ The SLMO model for the gSF"""

        width_ = width*(E/mean + (2*np.pi*T/mean)**2)
        slmo = 8.6737e-8*size/(1-np.exp(-E/T))
        slmo *= (2./np.pi)*E*width_
        slmo /= ((E**2 - mean**2)**2 + (E*width_)**2)
        return slmo

    @staticmethod
    def SLO(E: ndarray, mean: float, width: float, size: float) -> ndarray:
        """ The common SLO """

        slo = 8.6737e-8*size*E*width**2
        slo /= ((E**2 - mean**2)**2 + (E*width)**2)
        return slo

    def UB_model(self, E: ndarray, renorm: float) -> ndarray:
        """ Model of the upbend. In this case we expect
        an object that returns the value of the upbend at energy E.
        """
        return renorm*self.ub_gen(E)

    def setup_slmo_prior(self, norm_pars):
        """
        """
        # In the following
        self.gsf_means = np.array([norm_pars.GSFmodelPars['GDR']['E'],
                                   norm_pars.GSFmodelPars['GDR']['G'],
                                   norm_pars.GSFmodelPars['GDR']['S'],
                                   norm_pars.GSFmodelPars['SF']['E'],
                                   norm_pars.GSFmodelPars['SF']['G'],
                                   norm_pars.GSFmodelPars['SF']['S']])

        self.gsf_sigmas = np.array([norm_pars.GSFmodelPars['GDR']['E_err'],
                                   norm_pars.GSFmodelPars['GDR']['G_err'],
                                   norm_pars.GSFmodelPars['GDR']['S_err'],
                                   norm_pars.GSFmodelPars['SF']['E_err'],
                                   norm_pars.GSFmodelPars['SF']['G_err'],
                                   norm_pars.GSFmodelPars['SF']['S_err']])

    def slmo_prior(self, x: ndarray) -> ndarray:
        """
        """
        x[0:6] = normal(x[0:6], self.gsf_means, self.gsf_sigmas)
        # x[-1] = uniform(x[-1], 1.0, 2.0)
        x[-1] = truncnorm(x[-1], 1.0, np.inf, 1.5, 0.5)
        return x

    @property
    def discrete(self) -> Optional[Vector]:
        return self._discrete

    @property
    def shell_model(self) -> Optional[Vector]:
        return self._shell_model

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

        elif isinstance(value, Vector):
            if self.nld is not None and np.any(self.nld.E != value.E):
                raise ValueError("`nld` and `discrete` must"
                                 " have same energy binning")
            self._discrete = value
            self.LOG.debug("Set `discrete` by Vector")
        else:
            raise ValueError(f"Value {value} is not supported"
                             " for discrete levels")

    @shell_model.setter
    def shell_model(self, value: Optional[Union[Path, str, Vector]]) -> None:
        if value is None:
            self._shell_models = None
            self.LOG.debug("Set `shell_model` to None")
        elif isinstance(value, (str, Path)):
            self.LOG.debug("Set `shell_model` from file %s", value)
            df_sm = pd.read_csv(value)
            df_sm = df_sm.loc[df_sm['gSF'] > 0]
            self._shell_model = Vector(E=df_sm['Eg'], values=df_sm['gSF'],
                                       units='MeV')
            self._shell_model_path = value
        elif isinstance(value, Vector):
            self._shell_model = value
            self.LOG.debug("Set `shell_model` by Vector")

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
