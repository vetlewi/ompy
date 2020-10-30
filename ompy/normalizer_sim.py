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
from .ensemble_prior import EnsemblePrior
from .prior import uniform, normal, truncnorm

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
                 gdr_path: Optional[Union[str, pd.DataFrame]] = None,
                 ext_logp: Dict[str, ndarray],
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

        self.nld_model_str = nld_model.upper()
        if nld_model.upper() == 'CT':
            self.nld_model = self.const_temperature
            self.nld_prior = lambda x: uniform(x, np.array([0., -10.]),
                                               np.array([10., 10.]))

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

        self.ext_logp = ext_logp

        self.multinest_path = Path('multinest')
        self.multinest_kwargs: dict = {"seed": 65498, "resume": False}

        self.samples = None
        self.popt = None
        self.log_like = None

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
                  ext_logp: Dict[str, ndarray] = None) -> any:
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

        try:
            ext_logp = self.self_if_none(ext_logp)
        except ValueError:
            pass

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
        gsf_sig = []
        if isinstance(self.gsf_model, GSFmodel):
            gsf_sig = self.gsf_model.par_names
        else:
            gsf_sig = list(dict(signature(self.gsf_model).parameters).keys())

        if 'T' in nld_sig and 'T' in gsf_sig:
            del gsf_sig[gsf_sig.index('T')]
            self.LOG.info("Temperature parameter 'T' in gSF model derived from NLD model.")

        N_nld_par = len(nld_sig)-1
        N_gsf_par = len(gsf_sig)-1
        nld_names = nld_sig[1:N_nld_par+1]
        gsf_names = gsf_sig[1:N_gsf_par+1]
        Nvars = 3*N + N_nld_par + N_gsf_par

        names = ['A[%d]' % i for i in range(N)]
        names += ['B[%d]' % i for i in range(N)]
        names += ['α[%d]' % i for i in range(N)]
        names += nld_names + gsf_names

        lnl = ensemblelikelihood(nlds=nlds, gsfs=gsfs,
                                 nld_limit_low=nld_limit_low,
                                 nld_limit_high=nld_limit_high,
                                 gsf_limit=gsf_limit_high,
                                 nld_ref=discrete,
                                 nld_model=self.nld_model,
                                 gsf_model=self.gsf_model,
                                 ext_gsf=ext_logp)

        alphas = self.estimate_alpha(nlds, discrete, nld_limit_low)
        xs = self.estimate_B(gsfs, self.gsf_model, alphas)

        Bmean = np.log10(np.array(xs))

        res = self.initial_guess(lnl, N, N_nld_par, N_gsf_par, self.nld_prior,
                                 self.gsf_prior)

        # We want to print only 5 and 5 parameters at a time.
        self.LOG.info("DE results:")
        norm_values = [[res[i], res[i+N], res[i+2*N]] for i in range(N)]
        self.LOG.info('\n%s', tt.to_string(norm_values,
                                           header=['A', 'B', 'α']))
        self.write_log(names[3*N:], res[3*N:])

        A_est = np.log(res[0])
        B_est = np.log(res[1])

        def const_prior(x):
            return 10**uniform(x, 0, 4)

        # To get a prior for the gSF, we find the best value of B using only the priors

        prp = EnsemblePrior(A=lambda x: np.exp(normal(x, A_est, A_est)),
                            B=lambda x: np.exp(normal(x, B_est, 3*B_est)),
                            alpha=lambda x: normal(x, 0, 1),
                            nld_param=self.nld_prior, gsf_param=self.gsf_prior,
                            spc_param=lambda x: 0, N=N, N_nld_par=N_nld_par,
                            N_gsf_par=N_gsf_par, N_spc_par=0)

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

        print(mnstats['global evidence']/np.log(10))

        samples_dict = {
            'A': np.array(samples[:N, :]),
            'B': np.array(samples[N:2*N, :]),
            'alpha': np.array(samples[2*N:3*N, :])
        }

        def get_samples(names, start_idx):
            res = {}
            for i, name in enumerate(names):
                res[name] = np.array(samples[start_idx+i, :])
            return res

        samples_dict.update(get_samples(nld_names, 3*N))
        samples_dict.update(get_samples(gsf_names, 3*N+N_nld_par))

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

        # First 3*N are norm parameters
        norm_values = [[vals[i], vals[i+N], vals[i+2*N]] for i in range(N)]
        self.LOG.info("Multinest results:\n%s",
                      tt.to_string(norm_values, header=['A', 'B', 'α']))
        self.write_log(names[3*N:], vals[3*N:])

        self.samples = samples
        self.popt = popt
        self.log_like = log_like

        return popt, samples, log_like, lnl

    def initial_guess(self, likelihood, N, N_nld, N_gsf, nld_prior, gsf_prior):

        bounds = [[0, 5] for n in range(N)]
        bounds += [[0, 5000] for n in range(N)]
        bounds += [[-10., 10.] for n in range(N)]

        nld_low = nld_prior(np.repeat(0.04, N_nld))
        nld_high = nld_prior(np.repeat(0.96, N_nld))
        bounds += [[low, high] for low, high in zip(nld_low, nld_high)]

        gsf_low = gsf_prior(np.repeat(0.04, N_gsf))
        gsf_high = gsf_prior(np.repeat(0.96, N_gsf))
        bounds += [[low, high] for low, high in zip(gsf_low, gsf_high)]

        # Now, we can try to normalize!
        res = differential_evolution(lambda x: -likelihood(x), bounds=bounds)
        return res.x

    def estimate_B(self, gsfs: List[Vector], gsf_model, alphas):
        """ A function that estimates the best guesses for
        B and alpha.
        """

        # Define the function to maximize
        Bs = []
        for gsf, alpha in zip(gsfs, alphas):
            gsf_new = gsf.copy()
            gsf_new.to_MeV()
            model = gsf_model.mean(gsf_new.E)
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

    def plot(self, *, ax: Tuple[Any, Any] = None,
             add_label: bool = True,
             add_figlegend: bool = True,
             plot_fitregion: bool = True,
             reset_color_cycle: bool = True,
             **kwargs) -> Tuple[Any, Any]:
        """
        """
        # Setup figure

        if ax is None:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
        else:
            fig = ax[0].figure

        # First we will plot NLD.
        ax[0].set_title('Level density')
        ax[1].set_title('$\gamma$SF')

        self.plot_nld(ax=ax[0], add_label=add_label,
                      plot_fitregion=plot_fitregion,
                      add_figlegend=add_figlegend)
        self.plot_gsf(ax=ax[1], add_label=add_label,
                      plot_fitregion=plot_fitregion,
                      add_figlegend=add_figlegend)

        ax[0].set_yscale('log')
        ax[1].set_yscale('log')

        #ax[0].legend(loc='best')

        return fig, ax

    def plot_nld(self, ax: Any = None, add_label: bool = True,
                 plot_fitregion: bool = True,
                 add_figlegend: bool = True) -> None:
        """
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        median_kwargs = {'kind': 'line', 'scale': 'log'}
        model_kwargs = {'kind': 'line', 'scale': 'log', 'marker': ''}
        model_std_kwargs = {'alpha': 0.2}
        std_kwargs = {'alpha': 0.3}
        rhoSn_kwargs = {'markersize': 3, 'marker': 'o'}
        disc_kwars = {'kind': 'step', 'scale': 'log'}
        if add_label:
            median_kwargs['label'] = 'Exp.'
            std_kwargs['label'] = 'Exp. 68 %'
            model_kwargs['label'] = self.nld_model_str
            model_std_kwargs['label'] = self.nld_model_str + " 68 %"
            rhoSn_kwargs['label'] = "$\rho$(S$_n$)"
            disc_kwars['label'] = "ca48mh1g"

        nld_median, nld_low, nld_high = self.get_nld([0.16, 0.84])
        nld_median.plot(ax=ax, **median_kwargs)
        ax.fill_between(nld_median.E, nld_low.values,
                        nld_high.values, **std_kwargs)

        E_model = np.linspace(self.nld_limit_high[0],
                              self.norm_pars.Sn[0], 1001)
        model_median, model_low, model_high = self.get_nld_model(E_model)
        model_median.plot(ax=ax, **model_kwargs)
        ax.fill_between(E_model, model_low.values, model_high.values,
                        **model_std_kwargs)

        # Next the Sn point
        rhoSn = model_median.values[-1]
        rhoSnErr = np.array([[rhoSn-model_low.values[-1]],
                             [model_high.values[-1]-rhoSn]])
        ax.errorbar(self.norm_pars.Sn[0], rhoSn, yerr=rhoSnErr, **rhoSn_kwargs)

        if plot_fitregion:
            ax.axvspan(self.nld_limit_low[0], self.nld_limit_low[1],
                       color='grey', alpha=0.1, label="fit limits")
            ax.axvspan(self.nld_limit_high[0], self.nld_limit_high[1],
                       color='grey', alpha=0.1)

        # Plot discrete
        self.discrete.plot(ax=ax, **disc_kwars)

        ax.set_yscale('log')
        ax.set_ylabel(r"Level density $\rho(E_x)~[\mathrm{MeV}^{-1}]$")
        ax.set_xlabel(r"Excitation energy $E_x~[\mathrm{MeV}]$")
        ax.set_ylim(bottom=0.5/(nld_median.E[1]-nld_median.E[0]))

        #if fig is not None and add_figlegend:
        #    fig.legend(loc=9, ncol=3, frameon=False)

    def plot_gsf(self, ax: Any = None, add_label: bool = True,
                 plot_fitregion: bool = True,
                 add_figlegend: bool = True) -> None:
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        median_kwargs = {'kind': 'line', 'scale': 'log'}
        model_kwargs = {'kind': 'line', 'scale': 'log', 'marker': ''}
        std_kwargs = {'alpha': 0.3}
        model_std_kwargs = {'alpha': 0.2}
        sm_kwars = {'kind': 'step', 'scale': 'log'}
        ext_kwargs = {'markersize': 3, 'marker': 'd', 'linestyle': ''}
        if add_label:
            median_kwargs['label'] = 'Exp.'
            std_kwargs['label'] = 'Exp. 68 %'
            model_kwargs['label'] = "Model"
            model_std_kwargs['label'] = "Model ±" + "68 %"
            sm_kwars['label'] = "ca48mh1g"
            if self.ext_logp is not None:
                try:
                    ext_kwargs['label'] = self.ext_logp['label']
                except KeyError:
                    ext_kwargs['label'] = 'Ext. data'

        gsf_median, gsf_low, gsf_high = self.get_gsf([0.16, 0.84])
        gsf_median.plot(ax=ax, **median_kwargs)
        ax.fill_between(gsf_median.E, gsf_low.values,
                        gsf_high.values, **std_kwargs)

        max_E = [max(gsf_median.E)]
        if self.ext_logp is not None:
            max_E.append(max(self.ext_logp['x']))

        E_model = np.linspace(0, max(max_E), 1001)
        model_median, model_low, model_high = self.get_gsf_model(E_model)
        model_median.plot(ax=ax, **model_kwargs)
        ax.fill_between(E_model, model_low.values, model_high.values,
                        **model_std_kwargs)

        self.shell_model.plot(ax=ax, **sm_kwars)

        # Next external data!
        if self.ext_logp is not None:
            ax.errorbar(self.ext_logp['x'], self.ext_logp['y'],
                        yerr=self.ext_logp['yerr'], **ext_kwargs)

        if plot_fitregion:
            ax.axvspan(self.gsf_limit_high[0], self.gsf_limit_high[1],
                       color='grey', alpha=0.1, label="fit limits")

        ax.set_yscale('log')
        ax.set_xlabel(rf"$\gamma$-ray energy $E_\gamma$~[MeV]")
        ax.set_ylabel(rf"$\gamma$-SF f($E_\gamma$)~[MeV$^{{-3}}$]")
        ax.set_ylim(bottom=1e-9)

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def get_normalized(self, vecs: List[Vector], const: ndarray,
                       slope: ndarray, q: Tuple[float, float]
                       ) -> Tuple[Vector, Vector, Vector]:
        def To_Vector(E, data, q):
            return Vector(E=E[0, :].flatten(),
                          values=np.quantile(data, q, axis=(1, 2)),
                          units='MeV')

        E, values, stds = [], [], []
        for vec in vecs:
            vec_ = vec.copy()
            vec_.to_MeV()
            E.append(vec_.E)
            values.append(vec_.values)
            stds.append(vec_.std/vec_.values)
        E, values, stds = np.array(E), np.array(values), np.array(stds)
        normed = self.transform(E, values, const, slope)

        median = To_Vector(E, normed, 0.5)
        median.std = stds[0]**2
        median.std += (normed.std(axis=(1, 2))/normed.mean(axis=(1, 2)))**2
        median.std = np.sqrt(median.std)*median.values
        low = To_Vector(E, normed, q[0])
        high = To_Vector(E, normed, q[1])
        return median, low, high

    def get_nld(self, q: Tuple[float, float] = (0.16, 0.84)
                ) -> Tuple[Vector, Vector, Vector]:
        """
        """ 

        assert self.samples is not None, \
            "Normalization has not yet been ran."

        return self.get_normalized(self.nlds, self.samples['A'],
                                   self.samples['alpha'], q)

    def get_gsf(self, q: Tuple[float, float] = (0.16, 0.84)
                ) -> Tuple[Vector, Vector, Vector]:
        assert self.samples is not None, \
            "Normalization has not yet been ran."

        return self.get_normalized(self.gsfs, self.samples['B'],
                                   self.samples['alpha'], q)

    def get_nld_model(self, E: ndarray,
                      q: Tuple[float, float] = (0.16, 0.84)
                      ) -> Tuple[Vector, Vector, Vector]:
        """
        """
        assert self.samples is not None, \
            "Normalization has not yet been ran."

        return self.evaluate_model(E, self.nld_model, self.samples, q)

    def get_gsf_model(self, E: ndarray,
                      q: Tuple[float, float] = (0.16, 0.84)
                      ) -> Tuple[Vector, Vector, Vector]:
        """
        """
        assert self.samples is not None, \
            "Normalization has not yet been ran."

        return self.evaluate_model(E, self.gsf_model, self.samples, q)

    def write_log(self, names, values, size=5):
        def flush_log(names, values):
            self.LOG.info("\n%s", tt.to_string([values], header=header))

        header = []
        table = []
        for name, value in zip(names, values):
            if len(header) == size:
                flush_log(header, table)
                header = []
                table = []
            header.append(name)
            table.append(value)
        flush_log(header, table)

    @staticmethod
    def evaluate_model(E: ndarray, model: Callable[..., ndarray],
                       samples: Dict[str, ndarray],
                       q: Tuple[float, float] = (0.16, 0.84)):
        def To_vec(E, data, q):
            return Vector(E=E.flatten(), values=np.quantile(data, q, axis=1),
                          units='MeV')
        E = E.reshape(len(E), 1)
        args = list(dict(signature(model).parameters).keys())
        del args[0]  # First argument should be energy. Not needed.
        params = []
        for arg in args:
            try:
                params.append(samples[arg].reshape(1, len(samples[arg])))
            except KeyError:
                raise KeyError("Missing field '%' \
                    from samples dictionary" % arg)

        model_res = model(E, *params)

        median = To_vec(E, model_res, 0.5)
        low = To_vec(E, model_res, q[0])
        high = To_vec(E, model_res, q[1])
        return median, low, high

    @staticmethod
    def transform(E: ndarray, values: ndarray,
                  const: ndarray, slope: ndarray) -> ndarray:
        """ Normalize NLD and/or gSF.
        Args:
            E: A NxM matrix with energy points
            values: A NxM matrix with the values of the NLD/gSF
            const: Constant part of the normalization eq. (NxS)
            slope: Slope part of the normalization eq. (NxS)
        """

        # First we need to ensure correct shapes
        assert E.shape == values.shape, \
            "'E' and 'values' have different shapes."
        assert const.shape == slope.shape, \
            "'const' and 'slope' have different shapes."

        # Next we make sure that they have the same first dim. shape.
        assert E.shape[0] == const.shape[0], \
            "Transformation parameters has to have the same size in the first \
            dimention."

        # Now we can reshape to the desired shape.
        E = E.T.reshape(*E.T.shape, 1)
        values = values.T.reshape(*values.T.shape, 1)
        const = const.reshape(1, *const.shape)
        slope = slope.reshape(1, *slope.shape)

        return const*values*np.exp(slope*E)

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
