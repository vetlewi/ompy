import numpy as np
import numba as nb
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
from .matrix import Matrix
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

OneHalfLog2Pi = float(0.5*np.log(2*np.pi))

class ModelFG:

    def __init__(self, discrete: Vector,
                 nld_model: str,
                 gsf_model: GSF_model,
                 E_change: float,
                 Ex: ndarray,
                 Eg: ndarray):

        de = np.diff(discrete.E)[0]

        self.discrete = interp1d(discrete.E, discrete.values,
                                 bounds_error=False,
                                 fill_value=(0., 0.))

        self.gsf_model = gsf_model
        self.gsf_prior = gsf_model.prior

        if nld_model.upper() == 'CT':
            self.nld_model = self.const_temperature
            self.nld_prior = self.ct_prior
        elif nld_model.upper() == 'BSFG':
            self.nld_model = lambda E, a, Eshift: self.back_shift_fg(E, 67, a, Eshift)
            self.nld_prior = self.bsfg_prior
        else:
            raise NotImplementedError("NLD model '%s' has not \
                been implemented." % (nld_model.upper()))

        # Determine the number of NLD parameters
        nld_sig = list(dict(signature(self.nld_model).parameters).keys())
        gsf_sig = list(dict(signature(self.gsf_model).parameters).keys())

        del nld_sig[0]
        del gsf_sig[0]

        self.N_nld = len(nld_sig)
        self.N_gsf = len(gsf_sig)
        self.N = self.N_nld + self.N_gsf

        self.gsf_mask = np.arange(self.N_gsf) + self.N_nld
        if 'T' in nld_sig and 'T' in gsf_sig:
            self.gsf_mask[gsf_sig.index('T')] = nld_sig.index('T')
            self.gsf_mask[gsf_sig.index('T')+1:] -= 1
            self.N -= 1
            del gsf_sig[gsf_sig.index('T')]

        self.names = nld_sig + gsf_sig

        self.Ex = Ex
        self.Eg = Eg
        self.Ef = Ex - Eg
        self.E_change = E_change
        self.where = self.Ef < self.E_change

    def __call__(self, param) -> ndarray:
        return self.GenerateFG(param)

    def GenerateFG(self, param) -> ndarray:

        par = np.array(param[0:self.N])
        rho_final = np.where(self.where, self.discrete(self.Ef),
                             self.nld_model(self.Ef, *par[:self.N_nld]))
        gsf = self.gsf_model(self.Eg, *par[self.gsf_mask])

        fg = rho_final * gsf

        # Normalize
        fg = fg/fg.sum(axis=1, keepdims=True)
        return fg_model

    def prior(self, param):
        par = np.array(param[0:self.N])
        par[:self.N_nld] = self.nld_prior(par[:self.N_nld])
        par[self.N_nld:self.N] = \
            self.gsf_model.prior(par[self.N_nld:self.N])
        for i, val in enumerate(par):
            param[i] = val
        return param

    @staticmethod
    @nb.jit(nopython=True)
    def const_temperature(E: ndarray, T: float, Eshift: float) -> ndarray:
        """ Constant Temperature NLD"""
        ct = np.exp((E - Eshift) / T) / T
        return ct

    @staticmethod
    def ct_prior(param):
        param = normal(param, np.array([0.8, 0.]), np.array([0.2, 10.]))
        return param

    @staticmethod
    @nb.jit(nopython=True)
    def back_shift_fg(E: ndarray, A: int, a: float, Eshift: float) -> ndarray:
        """ Back-shift Fermi Gas"""
        sigma = 0.0146*A**(5./3.)
        sigma = np.sqrt((1 + np.sqrt(1 + 4*a*(E-Eshift)))*sigma/(2*a))

        bsfg = np.exp(2*np.sqrt(a*(E-Eshift)))
        bsfg /= (12.*np.sqrt(2)*sigma*a**(1./4.)*(E-Eshift)**(5./4.))
        return bsfg

    @staticmethod
    def bsfg_prior(param):
        pparam = normal(param, np.array([8., 0.]), np.array([10., 10.]))
        return param

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)


class NormalizeDirect(AbstractNormalizer):
    """ We fit the model directly to the FG matrix.
    """

    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, FirstGen: Matrix,
                 discrete: Vector,
                 nld_model: str,
                 gsf_model: GSF_model,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers'):
        """
        """

        self.FirstGen = FirstGen
        self.discrete = discrete
        self.nld_model = nld_model
        self.gsf_model = gsf_model

        if path is None:
            self.path = None
        else:
            self.path = Path(path)
            self.path.mkdir(exist_ok=True, parents=True)

        self.multinest_path = Path('multinest')
        self.multinest_kwargs: dict = {"seed": 65498, "resume": False}

    def FitModel(self):
        """ At the moment we will use the uncertainties...
        """
        FirstGen = self.FirstGen.copy()
        Eg, Ex = np.meshgrid(FirstGen.Eg/1e3,
                             FirstGen.Ex/1e3)

        # Next we setup the "NLD builder"
        fg_model = ModelFG(self.discrete, self.nld_model,
                           self.gsf_model, 2.5, Eg, Ex)

        # Next we will need to normalize the FG
        FirstGen, FirstGen_err = self.NormFG(FirstGen.values)
        #FirstGen_err[FirstGen_err == 0] = -np.inf

        # Now we can get our likelihood. For now
        # we assume that these are Gaussian...
        def loglike(cube, ndim, nparams):
            fg_th = fg_model(cube)
            return logp(FirstGen, fg_th, FirstGen_err)

        def prior(cube, ndim, nparams):
            return fg_model.prior(cube)

        print(loglike(prior(0.5*np.ones(fg_model.N), None, None), None, None))
        #return None, None, None
        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / f"norm_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        self.LOG.info("Starting multinest")
        self.LOG.debug("with following keywords %s:", self.multinest_kwargs)

        self.LOG.write = lambda msg: (self.LOG.info(msg) if msg != '\n'
                                      else None)

        with redirect_stdout(self.LOG):
            pymultinest.run(loglike, prior, fg_model.N,
                            outputfiles_basename=str(path),
                            **self.multinest_kwargs)

        json.dump(fg_model.names, open(str(path) + 'params.json', 'w'))
        analyzer = pymultinest.Analyzer(fg_model.N,
                                        outputfiles_basename=str(path))

        mnstats = analyzer.get_stats()
        samples = analyzer.get_equal_weighted_posterior()[:, :-1].T

        print(mnstats['global evidence']/np.log(10))

        samples_dict = {}
        for i, name in enumerate(fg_model.names):
            samples_dict[name] = samples[i, :]

        # Next we need to get the log_likelihood
        log_like = []
        for sample in samples.T:
            log_like.append(loglike(sample, None, None))
        log_like = np.array(log_like)
        samples = samples_dict

        # Next we need to get the log_likelihood for each sample...
        # We should also generate the best and the std matrix


        # Format the output
        popt = dict()
        vals = []
        for name, m in zip(fg_model.names, mnstats['marginals']):
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

        self.samples = samples
        self.popt = popt
        self.log_like = log_like

        return popt, samples, log_like

    @staticmethod
    def NormFG(fg: ndarray) -> Tuple[ndarray, ndarray]:
        N = fg.sum(axis=1, keepdims=True)
        FG = fg/N
        FG_std = np.sqrt((1/N - fg/N**2)*fg + fg**2/N**3)
        return FG, FG_std

    def self_if_none(self, *args, **kwargs):
        """ wrapper for lib.self_if_none """
        return self_if_none(self, *args, **kwargs)


@nb.jit#(nopython=True)
def logp(data: ndarray, model: ndarray,
         weights: ndarray) -> float:
    """ A simple function for calculating the unweighted sum.
    Note that to avoid issues with weights, we assume 30% if weights are None.
    """
    diff = ((data - model)/weights)**2
    diff[weights == 0] = 0.
    logw = np.log(weights)
    logw[weights == 0] = 0.

    logp = -0.5*np.sum(diff)
    logp -= OneHalfLog2Pi
    logp -= np.sum(logw)
    return np.sum(diff)


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
