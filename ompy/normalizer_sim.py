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
from .library import self_if_none, log_interp1d
from .spinfunctions import SpinFunctions
from .filehandling import load_discrete
from .models import ResultsNormalized, NormalizationParameters
from .abstract_normalizer import AbstractNormalizer
from .extractor import Extractor
from .ensemble_likelihood import *
from .gsf_model import GSFmodel, GSF_model
from .ensemble_prior import (EnsemblePrior, uniform, cnormal,
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
                 gsf_model: any = 'SM',
                 spc_model: bool = False,
                 gdr_path: Optional[Union[str, pd.DataFrame]] = None,
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

        self.gsf_model = gsf_model
        self.gsf_prior = lambda x: self.gsf_model.prior(x)

        self.spc_model = self.GetSpinModel(norm_pars) \
            if spc_model else lambda x: None
        self.spc_prior = self.GetSpinPrior(norm_pars) \
            if spc_model else lambda x: None

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
                  ext_logp: Callable[..., float] = None) -> any:
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
        spc_sig = list(dict(signature(self.spc_model).parameters).keys())
        gsf_sig = []
        if isinstance(self.gsf_model, GSFmodel):
            gsf_sig = self.gsf_model.par_names
        else:
            gsf_sig = list(dict(signature(self.gsf_model).parameters).keys())

        if 'T' in nld_sig and 'T' in gsf_sig:
            del gsf_sig[gsf_sig.index('T')]

        N_nld_par = len(nld_sig)-1
        N_gsf_par = len(gsf_sig)-1
        N_spc_par = len(spc_sig)-1
        nld_names = nld_sig[1:N_nld_par+1]
        gsf_names = gsf_sig[1:N_gsf_par+1]
        spc_names = spc_sig[1:N_spc_par+1] if N_spc_par > 0 else []
        Nvars = 3*N + N_nld_par + N_gsf_par + N_spc_par

        # ext_logp.model = lambda E, param: self.gsf_model(E, param)

        # First we need to ensure that we have a meaningful priors
        """lnl = EnsembleLikelihood(nlds=nlds, gsfs=gsfs,
                                 nld_limit_low=nld_limit_low,
                                 nld_limit_high=nld_limit_high,
                                 gsf_limit_low=gsf_limit_low,
                                 gsf_limit_high=gsf_limit_high,
                                 nld_ref=discrete, gsf_ref=shell_model,
                                 nld_model=self.nld_model,
                                 gsf_model=self.gsf_model,
                                 spc_model=self.spc_model,
                                 gsf_ext_logp=ext_logp,
                                 norm_pars=self.norm_pars)"""

        lnl = ensemblelikelihood(nlds=nlds, gsfs=gsfs,
                                 nld_limit_low=nld_limit_low,
                                 nld_limit_high=nld_limit_high,
                                 gsf_limit=gsf_limit_high,
                                 nld_ref=discrete,
                                 nld_model=self.nld_model,
                                 gsf_model=self.gsf_model,
                                 ext_gsf=ext_logp)

        """lnl = OsloNormalize(nlds=nlds, gsfs=gsfs,
                            nld_limit_low=nld_limit_low,
                            nld_limit_high=nld_limit_high,
                            gsf_limit=gsf_limit_high,
                            nld_ref=discrete,
                            nld_model='CT',
                            gsf_model=self.gsf_model,
                            ext_gsf_data={'x': ext_logp.x, 'y': ext_logp.y,
                                          'yerr': ext_logp.yerr})"""

        alphas = self.estimate_alpha(nlds, discrete, nld_limit_low)
        xs = self.estimate_B(gsfs, self.gsf_model, alphas)

        Bmean = np.array(xs)

        def const_prior(x):
            return 10**uniform(x, 0, 4)

        # To get a prior for the gSF, we find the best value of B using only the priors

        prp = EnsemblePrior(A=const_prior,
                            B=lambda x: normal(x, Bmean, 10*Bmean),
                            alpha=lambda x: normal(x, 0, 10),
                            nld_param=self.nld_prior, gsf_param=self.gsf_prior,
                            spc_param=self.spc_prior, N=N, N_nld_par=N_nld_par,
                            N_gsf_par=N_gsf_par, N_spc_par=N_spc_par)

        names = ['A[%d]' % i for i in range(N)]
        names += ['B[%d]' % i for i in range(N)]
        names += ['alpha[%d]' % i for i in range(N)]
        names += nld_names + gsf_names + spc_names

        def prior(cube, ndim, nparams):
            prp(cube)

        def loglike(cube, ndim, nparams):
            return lnl(prp(cube))

        def test(cube, ndim, nparams):
            prp.prior(cube, ndim, nparams)
            return lnl.loglike(cube, ndim, nparams)

        #print(self.spc_prior([0.4, 0.45]))
        print(test(np.random.rand(Nvars), Nvars, None))
        #return


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
        samples = analyzer.get_equal_weighted_posterior()[:, :-1].T

        samples_dict = {
            'A': np.array(samples[:N,:]),
            'B': np.array(samples[N:2*N,:]),
            'alpha': np.array(samples[2*N:3*N,:])
        }

        def get_samples(names, start_idx):
            res = {}
            for i, name in enumerate(names):
                res[name] = np.array(samples[start_idx+i,:])
            return res

        samples_dict.update(get_samples(nld_names, 3*N))
        samples_dict.update(get_samples(gsf_names, 3*N+N_nld_par))
        samples_dict.update(get_samples(spc_names, 3*N+N_nld_par+N_gsf_par))

        # Next we need to get the log_likelihood
        log_like = []
        for sample in samples.T:
            log_like.append(lnl(sample))
        log_like = np.array(log_like)
        samples = samples_dict

        # Next we need to get the log_likelihood for each sample...

        # Format the output
        popt = dict()
        vals = []
        for name, m in zip(names, mnstats['marginals']):
            lo, hi = m['1sigma']
            med = m['median']
            sigma = (hi - lo) / 2
            popt[name] = (med, sigma)
            i = 2
            try:
                i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            except:
                i = 2
            fmt = '%%.%df' % i
            fmts = '\t'.join([fmt + " ± " + fmt])
            vals.append(fmts % (med, sigma))


        return popt, samples, log_like

    def estimate_B(self, gsfs: List[Vector], gsf_model, alphas):
        """ A function that estimates the best guesses for
        B and alpha.
        """

        # Define the function to maximize
        Bs = []
        for gsf, alpha in zip(gsfs, alphas):
            gsf_new = gsf.copy()
            gsf_new.to_MeV()
            model = self.gsf_model.mean(gsf_new.E)
            normed = np.log(model/gsf_new.values)

            def min_me(B):
                return np.sum((normed - alpha*gsf_new.E - np.log(B))**2)
            bounds = [(0, 1e9)]
            res = differential_evolution(min_me, bounds=bounds)
            Bs.append(res.x[0])
        return Bs

    def estimate_alpha(self, nlds: List[Vector], nld_ref: Vector, cut):
        alphas = []
        ref = nld_ref.cut(*cut, inplace=False)
        for nld in nlds:
            nld_new = nld.copy()
            nld_new.to_MeV()
            nld_new.cut(*cut, inplace=True)
            normed = np.log(ref.values/nld_new.values)

            def min_me(x):
                A, alpha = x
                return np.sum((normed - alpha*nld_new.E - np.log(A))**2)
            bounds = [(0, 1e9), (-10, 10)]
            res = differential_evolution(min_me, bounds=bounds)
            alphas.append(res.x[1])
        return alphas

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
    def GetSpinPrior(norm_pars):
        if norm_pars.spincutModel == 'MG17':
            mean = np.array([norm_pars.spincutPars['sigmaD'][0],
                             norm_pars.spincutPars['sigmaSn'][0]])
            std = np.array([norm_pars.spincutPars['sigmaD'][1],
                            norm_pars.spincutPars['sigmaSn'][1]])
            return lambda x: normal(x, mean, std)
        else:
            raise NotImplementedError("Prior for spin cut model '%s' has not \
                yet been implemented" % norm_pars.spincutModel)

    def GetSpinModel(self, norm_pars):
        Js = NormalizerSim.GetJs(norm_pars.Jtarget, norm_pars.A % 2 == 1)
        if norm_pars.spincutModel == 'MG17':
            Sn = norm_pars.Sn[0]
            Ed = norm_pars.spincutPars['Ed']
            return lambda E, sigmaD, sigmaSn:\
                self.spin_dist(self.Sigma_MG17(E, sigmaD, sigmaSn, Ed, Sn), Js)
        else:
            raise NotImplementedError("Spin-cut model '%s' has not been \
                implemented yet." % norm_pars.spincutModel)

    @staticmethod
    def GetJs(J, odd):
        Js = None
        if J == 0:
            Js = np.array([0., 1.])
        else:
            Js = np.array([J-1, J, J+1])
        if odd:
            Js += 0.5
        return Js

    @staticmethod
    def Sigma_MG17(E: Union[float, ndarray], sigmaD: float, sigmaSn: float,
                   Ed: float, Sn: float) -> Union[float, ndarray]:
        """
        """
        if isinstance(E, float):
            E = E if E > Ed else Ed
        else:
            E[E < Ed] = Ed
        sigma = sigmaD**2 + (E - Ed)*(sigmaSn**2 - sigmaD**2)/(Ed-Sn)
        sigma = np.sqrt(sigma)
        if isinstance(E, float):
            sigma = sigma if E > Ed else sigmaD**2
        else:
            sigma[E <= Ed] = sigmaD
        return sigma

    @staticmethod
    def spin_dist(sigma: Union[float, ndarray],
                  J: Union[float, ndarray]) -> Union[float, ndarray]:
        """
        """
        sigma = sigma**2
        spinDist = np.zeros(sigma.shape)

        if isinstance(J, float):
            return ((2. * J + 1.) / (2. * sigma)
                    * np.exp(-np.power(J + 0.5, 2.) / (2. * sigma)))

        for j in J:
            spinDist += ((2. * j + 1.) / (2. * sigma)
                         * np.exp(-np.power(j + 0.5, 2.) / (2. * sigma)))
        return spinDist  # return 1D if Ex or J is single entry

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


def qrpa_prior(x):
    return truncnorm(x, np.array([0., 1.]), np.array([np.inf, 1]),
                     np.array([1.0, 1.0]), np.array([0.1, 0.001]))

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
