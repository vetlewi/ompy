import numpy as np
import pandas as pd
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

from .vector import Vector
from .library import self_if_none, log_interp1d
from .filehandling import load_discrete
from .abstract_normalizer import AbstractNormalizer
from .physics_models import model
from .likelihoods import Likelihoods_NoGamGam, Likelihoods
from .models import ResultsNormalized, NormalizationParameters
from .prior import *
from .priors import *

TupleDict = Dict[str, Tuple[float, float]]


class NormalizerSim_tmp(AbstractNormalizer):
    """ Normalizes NLD and gSF simultaniously to data.
    """

    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, *,
                 nld_model: model,
                 gsf_model: model,
                 nld: Optional[Vector],
                 gsf: Optional[Vector],
                 discrete: Optional[Union[str, Vector]] = None,
                 shell_model: Optional[Union[str, Vector]] = None,
                 path: Optional[Union[str, Path]] = 'saved_run/normalizers',
                 regenerate: bool = False,
                 ext_gsf_data: Optional[Vector] = None,
                 ext_nld_data: Optional[Vector] = None,
                 norm_pars: NormalizationParameters):
        """ Initialize the normalizer.

        Args:
            nld_model:
            gsf_model:
            nld:
            gsf:
            discrete:
            shell_model:
            path:
            regenerate:
            ext_gamma_data:
            ext_nld_data:
            norm_pars:
        """

        super().__init__(regenerate)

        self.nld = nld.copy() if nld is not None else None
        self.gsf = gsf.copy() if gsf is not None else None

        self._discrete = None
        self._shell_model = None
        self._discrete_path = None
        self._shell_model_path = None
        self._smooth_levels_fwhm = None
        self.norm_pars = norm_pars

        self.nld_limit_low = None
        self.nld_limit_high = None
        self.gsf_limit = None

        self.smooth_levels_fwhm = 0.1
        self.discrete = discrete
        self.shell_model = shell_model

        self.res = ResultsNormalized(name="Results NLD&gSF")

        self.nld_model = nld_model
        self.gsf_model = gsf_model

        self.ext_gsf_data = ext_gsf_data
        self.ext_nld_data = ext_nld_data

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

    def normalize(self, *,
                  nld_limit_low: Optional[Tuple[float, float]] = None,
                  nld_limit_high: Optional[Tuple[float, float]] = None,
                  gsf_limit: Optional[Tuple[float, float]] = None,
                  discrete: Optional[Vector] = None,
                  shell_model: Optional[Vector] = None,
                  nld: Optional[Vector] = None,
                  gsf: Optional[Vector] = None,
                  ext_gsf_data: Optional[Vector] = None,
                  ext_nld_data: Optional[Vector] = None) -> any:
        """
        """
        if not self.regenerate:
            try:
                self.load()
                return
            except FileNotFoundError:
                pass

        nld_limit_low = self.self_if_none(nld_limit_low)
        nld_limit_high = self.self_if_none(nld_limit_high)
        gsf_limit = self.self_if_none(gsf_limit)

        try:
            ext_gsf_data = self.self_if_none(ext_gsf_data)
        except ValueError:
            pass
        try:
            ext_nld_data = self.self_if_none(ext_nld_data)
        except ValueError:
            pass

        discrete = self.self_if_none(discrete)
        shell_model = self.self_if_none(shell_model)
        shell_model.to_MeV()
        discrete.to_MeV()

        nld = self.self_if_none(nld)
        gsf = self.self_if_none(gsf)

        self.res = ResultsNormalized(name="Results NLD")

        self.LOG.info(f"\n\n---------\nNormalizing nld & gSF")
        nld = nld.copy()
        self.LOG.debug("Setting NLD, convert to MeV")
        nld.to_MeV()

        gsf = gsf.copy()
        self.LOG.debug("Setting gSF, convert to MeV")
        gsf.to_MeV()

        # Setup likelihood
        loglike = Likelihoods_NoGamGam(nld, gsf, discrete,
                                       nld_limit_low, nld_limit_high,
                                       gsf_limit, self.nld_model,
                                       self.gsf_model, ext_nld_data,
                                       ext_gsf_data, self.norm_pars)

        #  Get the prior
        prior = self.initial_guess(loglike, self.nld_model, self.gsf_model)

        #prior.prior(np.ones(15)*0.5, 15, 2)
        #return

        # Now we can actually evaluate <3
        self.popt, self.samples, evidence = self.optimize(loglike, prior)

        # And transform to the now found parameters!
        nld_transformed = self.transform(nld, self.samples['A'],
                                         self.samples['α'])
        gsf_transformed = self.transform(gsf, self.samples['B'],
                                         self.samples['α'])

        self.res.nld = nld_transformed
        self.res.gsf = gsf_transformed
        self.res.pars = self.popt
        self.res.samples = self.samples
        self.res.evidence = evidence

        return self.popt, self.samples

    def initial_guess(self,
                      loglike: Likelihoods_NoGamGam,
                      nld_model: model,
                      gsf_model: model) -> Priors_NoGamGam:
        """
        """

        bounds = [[0, 5], [0, 5000], [-10, 10]]

        nld_low = nld_model.prior(np.repeat(0.04, nld_model.N_free))
        nld_high = nld_model.prior(np.repeat(0.96, nld_model.N_free))
        bounds += [[low, high] for low, high in zip(nld_low, nld_high)]

        gsf_low = gsf_model.prior(np.repeat(0.04, gsf_model.N_free))
        gsf_high = gsf_model.prior(np.repeat(0.96, gsf_model.N_free))
        bounds += [[low, high] for low, high in zip(gsf_low, gsf_high)]

        res = differential_evolution(lambda x: -loglike(x), bounds=bounds)

        nld_sig = list(dict(signature(nld_model).parameters).keys())
        gsf_sig = list(dict(signature(gsf_model).parameters).keys())

        del nld_sig[0]
        del gsf_sig[0]

        if 'T' in nld_sig and 'T' in gsf_sig and gsf_model.need_T:
            del gsf_sig[gsf_sig.index('T')]

        names = loglike.names[3:]

        self.LOG.info("DE results:")
        norm_values = [[res.x[0], res.x[1], res.x[2]]]
        self.LOG.info('\n%s', tt.to_string(norm_values,
                                           header=['A', 'B', 'α']))
        self.write_log(names, res.x[3:])

        return Priors_NoGamGam(A=lambda x: np.exp(normal(x, np.log(res.x[0]),
                                                         np.log(res.x[0]))),
                               B=lambda x: np.exp(normal(x, np.log(res.x[1]),
                                                         5*np.log(res.x[1]))),
                               alpha=lambda x: normal(x, res.x[2], res.x[2]),
                               nld_model=nld_model, gsf_model=gsf_model)

    def optimize(self, loglike: Likelihoods, prior: Priors) -> any:
        """
        """

        self.multinest_path.mkdir(exist_ok=True)
        path = self.multinest_path / f"nld_norm_"
        assert len(str(path)) < 60, "Total path length too long for multinest"

        self.LOG.info("Starting multinest")
        self.LOG.debug("with following keywords %s:", self.multinest_kwargs)
        #  Hack where stdout from Multinest is redirected as info messages
        self.LOG.write = lambda msg: (self.LOG.info(msg) if msg != '\n'
                                      else None)
        with redirect_stdout(self.LOG):
            pymultinest.run(loglike.loglike, prior.prior, loglike.N_free,
                            outputfiles_basename=str(path),
                            **self.multinest_kwargs)

        # Save parameters for analyzer
        names = loglike.names
        json.dump(names, open(str(path) + 'params.json', 'w'))
        analyzer = pymultinest.Analyzer(loglike.N_free,
                                        outputfiles_basename=str(path))

        stats = analyzer.get_stats()

        samples = analyzer.get_equal_weighted_posterior()[:, :-1]
        samples = dict(zip(names, samples.T))

        # Format the output
        popt = dict()
        vals = []
        for name, m in zip(names, stats['marginals']):
            lo, hi = m['1sigma']
            med = m['median']
            sigma = (hi - lo) / 2
            popt[name] = (med, sigma)
            i = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % i
            fmts = '\t'.join([fmt + " ± " + fmt])
            vals.append(fmts % (med, sigma))

        self.LOG.info("Multinest results:\n%s",
                      tt.to_string([vals[:3]],
                                   header=['A', 'B', 'α [MeV⁻¹]']))
        self.write_log(names[3:], vals[3:])

        return popt, samples, (stats['global evidence'],
                               stats['global evidence error'])

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
        model_kwargs = {}#{'kind': 'line', 'scale': 'log', 'marker': ''}
        model_std_kwargs = {'alpha': 0.2}
        std_kwargs = {'alpha': 0.3}
        rhoSn_kwargs = {'markersize': 3, 'marker': 'o'}
        disc_kwars = {'kind': 'step', 'scale': 'log'}
        ext_kwargs = {'scale': 'log', 'markersize': 3,
                      'marker': 'd', 'linestyle': ''}
        if add_label:
            median_kwargs['label'] = 'Exp.'
            std_kwargs['label'] = 'Exp. 68 %'
            model_kwargs['label'] = self.nld_model.name
            model_std_kwargs['label'] = self.nld_model.name + " 68 %"
            rhoSn_kwargs['label'] = "$\rho$(S$_n$)"
            disc_kwars['label'] = "ca48mh1g"
            ext_kwargs['label'] = "External data"

        nld_median = self.res.nld
        nld_median.plot(ax=ax, **median_kwargs)

        E_model = np.linspace(self.nld_limit_high[0],
                              self.norm_pars.Sn[0], 1001)
        model_values = self.get_nld_model(E_model)
        median_values = np.mean(model_values, axis=1)#np.quantile(model_values, 0.5, axis=1)
        min_values = np.quantile(model_values, 0.16, axis=1)
        max_values = np.quantile(model_values, 0.84, axis=1)
        ax.plot(E_model, median_values, **model_kwargs)
        ax.fill_between(E_model, min_values, max_values, **model_std_kwargs)

        # Next the Sn point
        rhoSn = median_values[-1]
        rhoSnErr = [[median_values[-1] - min_values[-1]],
                    [max_values[-1] - median_values[-1]]]

        ax.errorbar(self.norm_pars.Sn[0], rhoSn, yerr=rhoSnErr, **rhoSn_kwargs)

        if plot_fitregion:
            ax.axvspan(self.nld_limit_low[0], self.nld_limit_low[1],
                       color='grey', alpha=0.1, label="fit limits")
            ax.axvspan(self.nld_limit_high[0], self.nld_limit_high[1],
                       color='grey', alpha=0.1)

        # Plot discrete
        self.discrete.plot(ax=ax, **disc_kwars)

        # Plot external data
        if self.ext_nld_data is not None:
            self.ext_nld_data.plot(ax=ax, **ext_kwargs)

        ax.set_yscale('log')
        ax.set_ylabel(r"Level density $\rho(E_x)~[\mathrm{MeV}^{-1}]$")
        ax.set_xlabel(r"Excitation energy $E_x~[\mathrm{MeV}]$")
        ax.set_ylim(bottom=0.5/(nld_median.E[1]-nld_median.E[0]))

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def plot_gsf(self, ax: Any = None, add_label: bool = True,
                 plot_fitregion: bool = True,
                 add_figlegend: bool = True) -> None:
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        median_kwargs = {'kind': 'line', 'scale': 'log'}
        model_kwargs = {}#{'kind': 'line', 'scale': 'log', 'marker': ''}
        std_kwargs = {'alpha': 0.3}
        model_std_kwargs = {'alpha': 0.2}
        sm_kwars = {'kind': 'step', 'scale': 'log'}
        ext_kwargs = {'scale': 'log', 'markersize': 3,
                      'marker': 'd', 'linestyle': ''}
        if add_label:
            median_kwargs['label'] = 'Exp.'
            std_kwargs['label'] = 'Exp. 68 %'
            model_kwargs['label'] = "Model"
            model_std_kwargs['label'] = "Model ±" + "68 %"
            sm_kwars['label'] = "ca48mh1g"
            ext_kwargs['label'] = "External data"

        gsf_median = self.res.gsf
        gsf_median.plot(ax=ax, **median_kwargs)

        max_E = [max(gsf_median.E) + 1.5]
        if self.ext_gsf_data is not None:
            max_E.append(max(self.ext_gsf_data.E) + 1.5)

        E_model = np.linspace(0, max(max_E), 1001)
        model_values = self.get_gsf_model(E_model)
        median_values = np.mean(model_values, axis=1)#np.quantile(model_values, 0.5, axis=1)
        min_values = np.quantile(model_values, 0.16, axis=1)
        max_values = np.quantile(model_values, 0.84, axis=1)
        ax.plot(E_model, median_values, **model_kwargs)
        ax.fill_between(E_model, min_values, max_values, **model_std_kwargs)

        self.shell_model.plot(ax=ax, **sm_kwars)

        # Next external data!
        if self.ext_gsf_data is not None:
            self.ext_gsf_data.plot(ax=ax, **ext_kwargs)

        if plot_fitregion:
            ax.axvspan(self.gsf_limit[0], self.gsf_limit[1],
                       color='grey', alpha=0.1, label="fit limits")

        ax.set_yscale('log')
        ax.set_xlabel(rf"$\gamma$-ray energy $E_\gamma$~[MeV]")
        ax.set_ylabel(rf"$\gamma$-SF f($E_\gamma$)~[MeV$^{{-3}}$]")
        ax.set_ylim(bottom=1e-9)

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

        return fig, ax

    def transform(self, vec: Vector, const: ndarray,
                  slope: ndarray) -> ndarray:
        E = np.atleast_2d(vec.E)
        values = np.atleast_2d(vec.values)
        const = np.atleast_2d(const)
        slope = np.atleast_2d(slope)
        trans_values = const*values.T*np.exp(slope*E.T)

        std = np.std(trans_values, axis=1)
        mean = np.mean(trans_values, axis=1)
        mean = np.quantile(trans_values, 0.5, axis=1)

        if vec.std is not None:
            std = (std/mean)**2
            std += (vec.std/vec.values)**2
            std = np.sqrt(std)*mean
        return Vector(E=vec.E, values=mean, std=std, units=vec.units)

    def get_nld_model(self, E: ndarray) -> ndarray:
        """
        """
        assert self.samples is not None, \
            "Normalization has not yet been ran."

        return self.evaluate_model(E, self.nld_model, self.samples)

    def get_gsf_model(self, E: ndarray) -> ndarray:
        """
        """
        assert self.samples is not None, \
            "Normalization has not yet been ran."

        return self.evaluate_model(E, self.gsf_model, self.samples)

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

    @staticmethod
    def evaluate_model(E: ndarray, model: Callable[..., ndarray],
                       samples: Dict[str, ndarray]):
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
        return model_res


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
