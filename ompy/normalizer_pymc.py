import numpy as np
import pymc3 as pm
import pymc3.math as pmm
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
#from .distributions import SLMO, SLO, UB


class NormalizerPYMC(AbstractNormalizer):
    """ A re-implementation of the NormalizeNLD class using the
        pyMC3 rather than Multinest.
    """
    LOG = logging.getLogger(__name__)  # overwrite parent variable
    logging.captureWarnings(True)

    def __init__(self, nld: Optional[Vector] = None,
                 gsf: Optional[Vector] = None,
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
                  nld: Optional[Vector] = None,
                  gsf: Optional[Vector] = None,
                  discrete: Optional[Vector] = None,
                  norm_pars: Optional[NormalizationParameters] = None,
                  regenerate: Optional[bool] = None,
                  sm: Optional[Vector] = None,
                  E_pred: ndarray = np.linspace(0, 20., 1001),
                  num: int = 0,
                  **kwargs) -> any:
        """ Normalize the NLD and gSF with the selected model.

        Args:
            limit_low:
            limit_high:
            nld:
            gsf:
            discrete:
            norm_pars:
            regenerate:
            E_pred:
            num:
            **kwargs:
        Returns: The pyMC trace.
        """

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
        nld = self.self_if_none(nld)
        self.nld = nld.copy()
        self.nld.to_MeV()  # Make sure E is in right scale
        nld = self.nld.copy()

        self.norm_pars = self.self_if_none(norm_pars)
        self.norm_pars.is_changed(include=["D0", "Sn", "spincutModel",
                                           "spincutPars"])  # check that set

        # ensure that it's updated if running again
        self.res = ResultsNormalized(name="Results NLD")

        self.LOG.info(f"\n\n---------\nNormalizing nld #{num}")
        nld = nld.copy()
        self.LOG.debug("Setting NLD, convert to MeV")
        nld.to_MeV()

        # Get the model we will use
        model = None
        if self.model.upper() == 'CT':
            self.model_range = np.linspace(limit_high[0],
                                           self.norm_pars.Sn[0], 101)
            model = self.setup_CT_model(nld=nld,
                                        limit_low=limit_low,
                                        limit_high=limit_high,
                                        model_range=self.model_range)
        elif self.model.upper() == 'BSFG':
            self.model_range = np.linspace(limit_high[0],
                                           self.norm_pars.Sn[0], 101)
            model = self.setup_BSFG_model(nld=nld, gsf=gsf,
                                          limit_low=limit_low,
                                          limit_high=limit_high,
                                          model_range=self.model_range)
            with model:
                trace = pm.sample(**kwargs)
                self.trace = trace
            return trace, E_pred
        else:
            raise NotImplementedError(
                'Only CT or BSFG model have been implemented.')

        # Make sure we also have the gSF fitted
        mode_init = copy.deepcopy(model)
        with mode_init:
            trace = pm.sample()
        self.trace = trace
        # model = self.setup_gsf_model(pymc_model=model, nld_trace=trace,
        #                             gsf=gsf, E_pred=E_pred)

        model = self.setup_gsf_model_ub_onl(pymc_model=model,
                                            alpha=trace['α'].mean(),
                                            T=trace['T'].mean(),
                                            sm=sm, E_pred=E_pred,
                                            gsf=gsf)
        #return trace, model
        # Now we can infere the NLD
        print(model)
        with model:
            trace = pm.sample(**kwargs)
        self.trace = trace

        def make_tbl(trace, names):
            def fmt_tbl(trace):
                i = max(0, int(-np.floor(np.log10(trace.std()))+1))
                fmt = ''
                if i > 4:
                    i = 4
                fmt = '%%.%df' % i
                fmts = '\t'.join([fmt + " ± " + fmt])
                return fmts % (trace.mean(), trace.std())
            strs = [fmt_tbl(trace[name]) for name in names]
            return strs

        def tbl_str(trace, hdrs):
            tbl = ''
            for i, hdr in enumerate(hdrs):
                if i == 0:
                    tbl = tt.to_string([make_tbl(trace, hdr)], header=hdr)
                else:
                    tbl += "\n%s" % (tt.to_string([make_tbl(trace, hdr)], header=hdr))
            return tbl

        hdr = ['A', 'B', 'α', 'T', 'Eshift', 'ϵ', 'ν']
        hdr2 = ['B', 'α', 'gsf_ϵ', 'gsf_ν']
        #hdr3 = ['gdr_mean', 'gdr_width', 'gdr_size', 'sf_mean', 'sf_width', 'sf_size', 'beta', 'gamma']
        hdr3 = ['gdr_mean', 'gdr_width', 'gdr_size', 'sf_mean', 'sf_width',
                'sf_size', 'sm_renorm']

        #self.LOG.info("pyMC3 results:\n%s", tbl_str(trace, [hdr, hdr2, hdr3]))

        return trace, E_pred

    def E1SF_model(self, E, T):
        """ The E1 and SF model with 'default' parameters
        """
        gsf = SLMO_model(E, self.norm_pars.GSFmodelPars['GDR']['E'],
                         self.norm_pars.GSFmodelPars['GDR']['G'],
                         self.norm_pars.GSFmodelPars['GDR']['S'], T)
        gsf += SLO_model(E, self.norm_pars.GSFmodelPars['SF']['E'],
                         self.norm_pars.GSFmodelPars['SF']['G'],
                         self.norm_pars.GSFmodelPars['SF']['S'])
        return gsf

    def setup_gsf_model_stepwise(self, nld_trace,
                                 gsf: Optional[Vector] = None,
                                 nld: Optional[Vector] = None) -> Tuple[pm.Model, ndarray]:
        """
        """

        def from_posterior(param, samples):
            smin, smax = np.min(samples), np.max(samples)
            width = smax - smin
            x = np.linspace(smin, smax, 100)
            y = stats.gaussian_kde(samples)(x)

            # what was never sampled should have a small probability but not 0,
            # so we'll extend the domain and use linear approximation of density on it
            x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
            y = np.concatenate([[0], y, [0]])
            return pm.Interpolated(param, x, y)

        def from_posterior_no_ext(param, samples):
            smin, smax = np.min(samples), np.max(samples)
            width = smax - smin
            x = np.linspace(smin, smax, 100)
            y = stats.gaussian_kde(samples)(x)
            return pm.Interpolated(param, x, y)

        E_pred_np = np.linspace(0, 20, 1001)
        gsf = self.self_if_none(gsf).copy()
        nld = self.self_if_none(nld).copy()
        gsf.to_MeV()
        nld.to_MeV()

        # We need an initial guess for the B value. This is done by estimating
        # Best = SMLO(E[-1])/gsf(E[-1])*exp(-α*E[-1])
        # B ~ N(Best, 10*Best)

        gsf_model = lambda E: self.E1SF_model(E, nld_trace['T'].mean())
        Best, Best_err = estimate_B(gsf.cut(3.5, 1e9, inplace=False).transform(alpha=nld_trace['α'].mean(), inplace=False),
                                    gsf_model)
        ub_slope, ub_size = estimate_upbend(gsf.cut(0, 2.5, inplace=False).transform(Best, nld_trace['α'].mean(), inplace=False),
                                            gsf_model)

        self.LOG.info("Estimated GSF parameters\n%s", tt.to_string([[Best, ub_slope, ub_size]],
                      header=['Best', 'UB slope [1/MeV]', 'UB strength [mb/MeV]']))

        gInt = GamGam(nld, gsf, self.norm_pars, [1.7, 2.8], [4, 5])

        with pm.Model() as gsf_model:

            # Data
            E_pred = pm.Data('E_pred', E_pred_np)
            E_gsf = pm.Data('E_gsf', gsf.E)
            gsf_v = pm.Data('gsf_v', gsf.values)
            gsf_l = pm.Data('gsf_l', np.log(gsf.values))

            # Priors taken from the NLD posterior
            α = from_posterior('α', nld_trace['α'])
            A = from_posterior('A', nld_trace['A'])
            T = from_posterior('T', nld_trace['T'])


            # Priors
            B = pm.Bound(pm.Normal, lower=0)('B', mu=Best, sigma=4*Best)
            D = pm.Deterministic('D', pm.math.log(B))

            gdr_mean = pm.Normal("gdr_mean", mu=self.norm_pars.GSFmodelPars['GDR']['E'],
                                 sigma=self.norm_pars.GSFmodelPars['GDR']['E_err'])
            gdr_width = pm.Normal("gdr_width", mu=self.norm_pars.GSFmodelPars['GDR']['G'],
                                 sigma=self.norm_pars.GSFmodelPars['GDR']['G_err'])
            gdr_size = pm.Normal("gdr_size", mu=self.norm_pars.GSFmodelPars['GDR']['S'],
                                 sigma=self.norm_pars.GSFmodelPars['GDR']['S_err'])
            sf_mean = pm.Normal("sf_mean", mu=self.norm_pars.GSFmodelPars['SF']['E'],
                                sigma=self.norm_pars.GSFmodelPars['SF']['E_err'])
            sf_width = pm.Normal("sf_width", mu=self.norm_pars.GSFmodelPars['SF']['G'],
                                sigma=self.norm_pars.GSFmodelPars['SF']['G_err'])
            sf_size = pm.Normal("sf_size", mu=self.norm_pars.GSFmodelPars['SF']['S'],
                                sigma=self.norm_pars.GSFmodelPars['SF']['S_err'])

            beta = pm.Normal('beta', mu=ub_slope, sigma=ub_slope)
            gamma = pm.Normal('gamma', mu=ub_size, sigma=ub_size)

            # Model functions
            GDR_model = lambda E: SLMO_model(E, gdr_mean, gdr_width, gdr_size, T)
            SF_model = lambda E: SLO_model(E, sf_mean, sf_width, sf_size)
            ub_model = lambda E: UB_model(E, beta, gamma)

            # Deterministic results from the inference
            gsf_norm = pm.Deterministic('gsf_norm', B*gsf_v*pm.math.exp(α*E_gsf))
            gsf_pred = pm.Deterministic('gsf_pred', GDR_model(E_pred)+SF_model(E_pred)+ub_model(E_pred))
            gsf_mod = pm.Deterministic('gsf_mod', GDR_model(E_gsf)+SF_model(E_gsf)+ub_model(E_gsf))
            
            # Likelihood!
            #gsf_obs = pm.StudentT('gsf_obs', mu=α*E_gsf + D, sd=ϵ, nu=ν, observed=pm.math.log(gsf_mod/gsf_v))
            if gsf.std is None:
                ϵ = pm.HalfNormal('ϵ', 5)
                ν_ = pm.Exponential('ν_', 1/30)
                ν = pm.Deterministic('ν', ν_ + 1)
                gsf_obs = pm.StudentT('gsf_obs', mu=gsf_mod*pm.math.exp(-α*E_gsf)/B, sd=ϵ, nu=ν, observed=gsf_v)
            else:
                gsf_sigma = pm.Data('gsf_sigma', gsf.std/gsf.values)
                #gsf_obs = pm.Normal('gsf_obs', mu=gsf_mod*pm.math.exp(-α*E_gsf)/B, sigma=gsf_sigma, observed=gsf_v)
                gsf_obs = pm.Normal('gsf_obs', mu=α*E_gsf + D, sigma=gsf_sigma, observed=pm.math.log(gsf_mod/gsf_v))

        return gsf_model, E_pred_np

    def setup_gsf_model_ub_onl(self, pymc_model: pm.Model,
                               alpha: float, T: float, sm: Vector,
                               E_pred: ndarray = np.linspace(0, 20., 1001),
                               gsf: Optional[Vector] = None):
        """
        """

        gsf = self.self_if_none(gsf).copy()
        gsf.to_MeV()

        # First we need to create an interpolation object
        # to generate the SM data points used in the fit
        sm = sm.copy()
        sm.to_MeV()

        def make_extra(vec, max_E):
            Norm = vec.values[-1]
            de = vec.E[1]-vec.E[0]
            new_Es = np.arange(vec.E[-1], max_E, de)
            new_Es += de
            new_Vs = Norm*np.exp(-(new_Es - vec.E[-1])**2/(2))
            return Vector(E=np.append(vec.E, new_Es),
                          values=np.append(vec.values, new_Vs))

        # Need to make sure that if the last point is not zero, the
        # we need to artificially exponentially go to zero.
        if sm.values[-1] != 0:
            sm = make_extra(sm, max(E_pred))

        sm_interp = log_interp1d(sm.E, sm.values, bounds_error=False,
                                 fill_value=(-np.inf, -np.inf))

        sm.cut(min(gsf.E), max(gsf.E), inplace=True)

        # Estimate B to get a well behaved prior
        gsf_fit = gsf.cut(min(gsf.E), 2, inplace=False)
        sm_fit = sm.cut(min(gsf.E), 2, inplace=False)
        gsf_tmp = gsf.cut(3.5, 1e9, inplace=False)
        gsf_tmp.transform(alpha=alpha, inplace=True)
        Best, Best_err = estimate_B(gsf_tmp, lambda E: self.E1SF_model(E, T))
        # return Best, Best_err

        gsf_fit.E = np.append(gsf_fit.E, gsf.cut(3.5, np.inf, inplace=False).E)
        gsf_fit.values = np.append(gsf_fit.values,
                                   gsf.cut(3.5, np.inf, inplace=False).values)
        sm_fit.E = np.append(sm_fit.E, sm.cut(3.5, np.inf, inplace=False).E)
        sm_fit.values = np.append(sm_fit.values,
                                  sm.cut(3.5, np.inf, inplace=False).values)

        gsf_pars = self.norm_pars.GSFmodelPars

        with pymc_model:

            # Add data
            E_gsf_fit = pm.Data('E_gsf_fit', gsf_fit.E)
            v_gsf_fit = pm.Data('v_gsf_fit', gsf_fit.values)
            v_sm_fit = pm.Data('v_sm_fit', sm_fit.values)
            E_gsf = pm.Data('E_gsf', gsf.E)
            v_gsf = pm.Data('v_gsf', gsf.values)
            v_sm = pm.Data('v_sm', sm.values)

            E_prd = pm.Data('E_prd', E_pred)
            sm_prd = pm.Data('sm_prd', sm_interp(E_pred))

            # Priors
            B = pm.TruncatedNormal('B', mu=Best, sigma=Best, testval=Best, lower=0.0)
            D = pm.Deterministic('D', pm.math.log(B))
            D_e = D
            #D = pm.Uniform('D', np.log10(Best)-, np.log10(Best)+4, testval=np.log10(Best))
            #B = pm.Deterministic('B', 10**D)
            #D_e = pm.Deterministic('D_e', pmm.log(B))

            gdr_mean = pm.TruncatedNormal("gdr_mean", mu=gsf_pars['GDR']['E'],
                                          sigma=gsf_pars['GDR']['E_err'],
                                          testval=gsf_pars['GDR']['E'],
                                          lower=0.0)
            gdr_width = pm.TruncatedNormal("gdr_width",
                                           mu=gsf_pars['GDR']['G'],
                                           sigma=gsf_pars['GDR']['G_err'],
                                           testval=gsf_pars['GDR']['G'],
                                           lower=0.0)
            gdr_size = pm.TruncatedNormal("gdr_size", mu=gsf_pars['GDR']['S'],
                                          sigma=gsf_pars['GDR']['S_err'],
                                          testval=gsf_pars['GDR']['S'],
                                          lower=0.0)
            sf_mean = pm.TruncatedNormal("sf_mean", mu=gsf_pars['SF']['E'],
                                         sigma=gsf_pars['SF']['E_err'],
                                         testval=gsf_pars['SF']['E'],
                                         lower=0.0)
            sf_width = pm.TruncatedNormal("sf_width", mu=gsf_pars['SF']['G'],
                                          sigma=gsf_pars['SF']['G_err'],
                                          testval=gsf_pars['SF']['G'],
                                          lower=0.0)
            sf_size = pm.TruncatedNormal("sf_size", mu=gsf_pars['SF']['S'],
                                         sigma=gsf_pars['SF']['S_err'],
                                         testval=gsf_pars['SF']['S'],
                                         lower=0.0)

            #ub_scale = pm.Uniform('ub_scale', 1., 4.)

            def GDR_model(E):
                return SLMO_model(E, gdr_mean, gdr_width, gdr_size,
                                  pymc_model.T)

            def SF_model(E):
                return SLO_model(E, sf_mean, sf_width, sf_size)

            gsf_norm = pm.Deterministic('gsf_norm',
                                        B*v_gsf*pmm.exp(pymc_model.α*E_gsf))
            GDR_pred = pm.Deterministic('GDR_pred', GDR_model(E_prd))
            SF_pred = pm.Deterministic('SF_pred', SF_model(E_prd))
            UB_pred = pm.Deterministic('UB_pred', sm_prd)#*ub_scale)
            gsf_pred = pm.Deterministic('gsf_pred', GDR_pred+SF_pred+UB_pred)

            gsf_ϵ = pm.HalfNormal('gsf_ϵ', 5)
            gsf_ν_ = pm.Exponential('gsf_ν_', 1/29)
            gsf_ν = pm.Deterministic('gsf_ν', gsf_ν_ + 1)

            #gsf_mod = GDR_model(E_gsf_fit)+SF_model(E_gsf_fit)+v_sm_fit

            #mu_obs = gsf_mod*pmm.exp(-pymc_model.α*E_gsf_fit)/B
            mu_obs = pymc_model.α*E_gsf_fit + D
            gsf_obs = pm.StudentT('gsf_obs', mu=mu_obs,
                                  sd=gsf_ϵ, nu=gsf_ν,
                                  observed=pmm.log(v_sm_fit/v_gsf_fit))
        return pymc_model

    def setup_gsf_model_exp_ub(self, pymc_model: pm.Model,
                               alpha: float, T: float, sm: Vector,
                               E_pred: ndarray = np.linspace(0, 20., 1001),
                               gsf: Optional[Vector] = None):
        """
        """

        gsf = self.self_if_none(gsf).copy()
        gsf.to_MeV()

        # First we need to create an interpolation object
        # to generate the SM data points used in the fit
        sm = sm.copy()
        sm.to_MeV()
        sm_interp = log_interp1d(sm.E, sm.values, bounds_error=False,
                                 fill_value=(-np.inf, -np.inf))


        # Estimate B to get a well behaved prior
        gsf_tmp = gsf.cut(3.5, 1e9, inplace=False)
        gsf_tmp.transform(alpha=alpha, inplace=True)
        Best, Best_err = estimate_B(gsf_tmp, lambda E: self.E1SF_model(E, T))

        gsf_pars = self.norm_pars.GSFmodelPars

        with pymc_model:

            # Add data
            E_gsf = pm.Data('E_gsf', gsf.E)
            v_gsf = pm.Data('v_gsf', gsf.values)
            v_sm = pm.Data('v_sm', sm_interp(gsf.E))

            E_prd = pm.Data('E_prd', E_pred)
            sm_prd = pm.Data('sm_prd', sm_interp(E_pred))

            # Priors
            B = pm.Bound(pm.Normal, lower=0)('B', mu=Best, sigma=Best,
                                             testval=Best)
            D = pm.Deterministic('D', pm.math.log(B))

            gdr_mean = pm.Normal("gdr_mean", mu=gsf_pars['GDR']['E'],
                                 sigma=gsf_pars['GDR']['E_err'],
                                 testval=gsf_pars['GDR']['E'])
            gdr_width = pm.Normal("gdr_width", mu=gsf_pars['GDR']['G'],
                                  sigma=gsf_pars['GDR']['G_err'],
                                  testval=gsf_pars['GDR']['G'])
            gdr_size = pm.Normal("gdr_size", mu=gsf_pars['GDR']['S'],
                                 sigma=gsf_pars['GDR']['S_err'],
                                 testval=gsf_pars['GDR']['S'])
            sf_mean = pm.Normal("sf_mean", mu=gsf_pars['SF']['E'],
                                sigma=gsf_pars['SF']['E_err'],
                                testval=gsf_pars['SF']['E'],)
            sf_width = pm.Normal("sf_width", mu=gsf_pars['SF']['G'],
                                 sigma=gsf_pars['SF']['G_err'],
                                 testval=gsf_pars['SF']['G'])
            sf_size = pm.Normal("sf_size", mu=gsf_pars['SF']['S'],
                                sigma=gsf_pars['SF']['S_err'],
                                testval=gsf_pars['SF']['S'])

            # sm_renorm_log = pm.Uniform('sm_renorm_log', 0, 1)
            sm_renorm = pm.Uniform('sm_renorm', 1, 4)
            # sm_renorm = pm.Deterministic('sm_renorm', 10**sm_renorm_log)

            def GDR_model(E):
                return SLMO_model(E, gdr_mean, gdr_width, gdr_size,
                                  pymc_model.T)

            def SF_model(E):
                return SLO_model(E, sf_mean, sf_width, sf_size)

            gsf_norm = pm.Deterministic('gsf_norm',
                                        B*v_gsf*pmm.exp(pymc_model.α*E_gsf))
            gsf_mod = pm.Deterministic('gsf_mod',
                                       GDR_model(E_gsf) +
                                       SF_model(E_gsf) +
                                       sm_renorm*v_sm)
            GDR_pred = pm.Deterministic('GDR_pred', GDR_model(E_prd))
            SF_pred = pm.Deterministic('SF_pred', SF_model(E_prd))
            UB_pred = pm.Deterministic('UB_pred', sm_renorm*sm_prd)
            gsf_pred = pm.Deterministic('gsf_pred', GDR_pred+SF_pred+UB_pred)

            gsf_ϵ = pm.HalfNormal('gsf_ϵ', 5)
            gsf_ν_ = pm.Exponential('gsf_ν_', 1/29)
            gsf_ν = pm.Deterministic('gsf_ν', gsf_ν_ + 1)

            mu_obs = gsf_mod*pmm.exp(-pymc_model.α*E_gsf)/B
            gsf_obs = pm.StudentT('gsf_obs', mu=mu_obs,
                                  sd=gsf_ϵ, nu=gsf_ν,
                                  observed=v_gsf)
        return pymc_model

    def setup_gsf_model(self, pymc_model: pm.Model,
                        nld_trace: any,
                        E_pred: ndarray = np.linspace(0, 20., 1001),
                        gsf: Optional[Vector] = None) -> pm.Model:
        """ Method for extending a NLD CT fit with fit of gSF to model.
        Currently the model that will be fitted is:
            f(Eg) = fSLMO(Eg) + fSLO(Eg) + strength*exp(-slope*Eg)/(3*pi^2*hbar^2*c^2)
        Args:
            pymc_model: Model to be extended
            nld_trace: Results of an initial sample of the model. Needed
                to estimate the priors of the B normalization factor and
                the upbend parameters.
            E_pred: Energies to sample the gSF model.
            gsf: Un-normalized gsf to normalize.
        Returns:
            model: the pymc3 model.
        """

        gsf = self.self_if_none(gsf).copy()
        gsf.to_MeV()

        def gsf_model(E):
            return self.E1SF_model(E, nld_trace['T'].mean())

        gsf_tmp = gsf.cut(3.5, 1e9, inplace=False)
        gsf_tmp.transform(alpha=nld_trace['α'].mean(), inplace=True)
        Best, Best_err = estimate_B(gsf_tmp, gsf_model)
        gsf_tmp = gsf.cut(0, 2.5, inplace=False)
        gsf_tmp.transform(Best, nld_trace['α'].mean(), inplace=True)
        ub_slope, ub_size = estimate_upbend(gsf_tmp, gsf_model)

        self.LOG.info("Estimated GSF parameters\n%s",
                      tt.to_string([[Best, ub_slope, ub_size]],
                                   header=['Best',
                                           'UB slope [1/MeV]',
                                           'UB strength [mb/MeV]']))

        # At the moment, we are unable to fit with std of the gSF
        if gsf.std is not None:
            gsf.std = None

        gsf_pars = self.norm_pars.GSFmodelPars

        with pymc_model:

            # Data
            E_prd = pm.Data('E_prd', E_pred)
            E_gsf = pm.Data('E_gsf', gsf.E)
            gsf_v = pm.Data('gsf_v', gsf.values)
            gsf_l = pm.Data('gsf_l', np.log(gsf.values))

            # Priors
            B = pm.Bound(pm.Normal, lower=0)('B', mu=Best, sigma=Best)
            D = pm.Deterministic('D', pm.math.log(B))

            gdr_mean = pm.Normal("gdr_mean", mu=gsf_pars['GDR']['E'],
                                 sigma=gsf_pars['GDR']['E_err'])
            gdr_width = pm.Normal("gdr_width", mu=gsf_pars['GDR']['G'],
                                  sigma=gsf_pars['GDR']['G_err'])
            gdr_size = pm.Normal("gdr_size", mu=gsf_pars['GDR']['S'],
                                 sigma=gsf_pars['GDR']['S_err'])
            sf_mean = pm.Normal("sf_mean", mu=gsf_pars['SF']['E'],
                                sigma=gsf_pars['SF']['E_err'])
            sf_width = pm.Normal("sf_width", mu=gsf_pars['SF']['G'],
                                 sigma=gsf_pars['SF']['G_err'])
            sf_size = pm.Normal("sf_size", mu=gsf_pars['SF']['S'],
                                sigma=gsf_pars['SF']['S_err'])

            beta = pm.Normal('beta', mu=ub_slope, sigma=ub_slope)
            gamma = pm.Normal('gamma', mu=ub_size, sigma=ub_size)

            def GDR_model(E):
                return SLMO_model(E, gdr_mean, gdr_width, gdr_size,
                                  pymc_model.T)

            def SF_model(E):
                return SLO_model(E, sf_mean, sf_width, sf_size)

            def ub_model(E):
                return UB_model(E, beta, gamma)
            # For now, we will skip this...
            # ub_model = lambda E: UB_model(E, beta, gamma)

            gsf_norm = pm.Deterministic('gsf_norm',
                                        B*gsf_v*pmm.exp(pymc_model.α*E_gsf))
            gsf_mod = pm.Deterministic('gsf_mod',
                                       GDR_model(E_gsf) +
                                       SF_model(E_gsf) +
                                       ub_model(E_gsf))

            gsf_pred = pm.Deterministic('gsf_pred',
                                        GDR_model(E_prd) +
                                        SF_model(E_prd) +
                                        ub_model(E_prd))

            if gsf.std is None:
                gsf_ϵ = pm.HalfNormal('gsf_ϵ', 5)
                gsf_ν_ = pm.Exponential('gsf_ν_', 1/29)
                gsf_ν = pm.Deterministic('gsf_ν', gsf_ν_ + 1)
                gsf_obs = pm.StudentT('gsf_obs',
                                      mu=gsf_mod*pmm.exp(-pymc_model.α*E_gsf)/B,
                                      sd=gsf_ϵ, nu=gsf_ν, observed=gsf_v)
            else:
                gsf_sigma = pm.Data('gsf_sigma', gsf.std)
                gsf_obs = pm.Normal('gsf_obs', mu=pymc_model.α*E_gsf + D,
                                    sigma=gsf_sigma,
                                    observed=pm.math.log(gsf_mod/gsf_v))
        return pymc_model

    def setup_CT_model(self, nld: Optional[Vector] = None,
                       limit_low: Optional[Tuple[float, float]] = None,
                       limit_high: Optional[Tuple[float, float]] = None,
                       model_range: Optional[np.ndarray] = None,
                       pymc_model=None) -> pm.Model:
        """ Method responsible for declaring our pyMC model.
        """

        # We set our data
        nld = self.self_if_none(nld).copy()
        nld.to_MeV()

        limit_low = self.self_if_none(limit_low)
        self.limit_low = limit_low

        nld_low = nld.cut(*limit_low, inplace=False)
        discrete_low = self.discrete.cut(*limit_low, inplace=False)

        limit_high = self.self_if_none(limit_high)
        self.limit_high = limit_high

        nld_high = nld.cut(*limit_high, inplace=False)

        model_range = self.self_if_none(model_range)

        # Standarizing
        q_tmp = np.log(discrete_low.values/nld_low.values)
        data_low = {'E': nld_low.E,
                    'E_mean': nld_low.E.mean(),
                    'E_std': nld_low.E.std(),
                    'q': q_tmp,
                    'q_mean': q_tmp.mean(),
                    'q_std': q_tmp.std()}
        data_low['E_st'] = (data_low['E'] - data_low['E_mean'])/data_low['E_std']
        data_low['q_st'] = (data_low['q'] - data_low['q_mean'])/data_low['q_std']
        q_tmp = np.log(nld_high.values)
        data_high = {'E': nld_high.E,
                     'E_mean': nld_high.E.mean(),
                     'E_std': nld_high.E.std(),
                     'q': q_tmp,
                     'q_mean': q_tmp.mean(),
                     'q_std': q_tmp.std()}
        data_high['E_st'] = (data_high['E'] - data_high['E_mean'])/data_high['E_std']
        data_high['q_st'] = (data_high['q'] - data_high['q_mean'])/data_high['q_std']

        if pymc_model is None:
            pymc_model = pm.Model()

        with pymc_model:

            # Set the data from the lower region fitting the discrete states
            E_low = pm.Data('E_low', data_low['E_st'])
            q_low = pm.Data('q_low', data_low['q_st'])
            qerr_low = None

            # Set the data from the higher region fitting the CT model
            E_high = pm.Data('E_high', data_high['E_st'])
            q_high = pm.Data('q_high', data_high['q_st'])
            qerr_high = None

            # Set data for infering the NLD and gSF after normalization.
            E_rho = pm.Data('E_rho', nld.E)
            rho = pm.Data('rho', nld.values)

            ϵ = None
            ν_ = None
            ν = None

            if nld.std is None:
                ϵ = pm.HalfNormal('ϵ', 5)
                ν_ = pm.Exponential('ν_', 1/29)
                ν = pm.Deterministic('ν', ν_ + 1)
            else:
                qerr_low = pm.Data('qerr_low', (nld_low.std/nld_low.values)/data_low['q_std'])
                qerr_high = pm.Data('qerr_high', (nld_high.std/nld_high.values)/data_high['q_std'])

            # Set dummy data for infering the CT model after sampling
            E_model = pm.Data('E_model', model_range)
            Sn = self.norm_pars.Sn[0]

            # Setup of the prior
            b0 = pm.Normal('b0', mu=0, sigma=100)
            b1 = pm.Normal('b1', mu=0, sigma=100)

            c0 = pm.Normal('c0', mu=0, sigma=100)
            T = pm.Uniform('T', 0, 3, testval=0.6)
            #c1 = pm.Normal('c1', mu=0, sigma=100)

            # The spin_dist
            sigmaD = pm.Normal("sigmaD", mu=self.norm_pars.spincutPars['sigma2_disc'][1],
                               sigma=0.1*self.norm_pars.spincutPars['sigma2_disc'][1])
            sigmaSn = pm.Normal("sigmaSn", mu=self.norm_pars.spincutPars['sigma2_Sn'][1],
                                sigma=0.1*self.norm_pars.spincutPars['sigma2_Sn'][1])

            α = pm.Deterministic('α', data_low['q_std']/data_low['E_std']*b1)
            c1 = pm.Deterministic('c1', data_high['E_std']*(1/T - α)/data_high['q_std'])
            C = pm.Deterministic('C', data_low['q_mean'] + b0*data_low['q_std'] - α*data_low['E_mean'])
            A = pm.Deterministic('A', pm.math.exp(C))

            c1st = data_high['q_std']*c1/data_high['E_std']
            c0st = data_high['q_mean'] + c0*data_high['q_std'] - c1st*data_high['E_mean']

            b1st = data_low['q_std']*b1/data_low['E_std']
            b0st = data_low['q_mean'] + b0*data_low['q_std'] - b1st*data_low['E_mean']

            #T = pm.Deterministic('T', 1./(c1st + b1st))
            Eshift = pm.Deterministic('Eshift', pm.math.log(c1st + b1st)*T - (b0st + c0st)*T)
            
            Ed, Sn = self.norm_pars.spincutPars['sigma2_disc'][0], self.norm_pars.Sn[0]
            sigma2 = lambda Ex: pm.math.where(pm.math.lt(Ex, Ed), sigmaD**2, sigmaD**2 + (Ex - Ed)*(sigmaSn**2 - sigmaD**2)/(Sn - Ed))
            spin_dist = lambda Ex: pm.math.exp(-0.5/sigma2(Ex))/sigma2(Ex)

            # Some deterministic functions that are useful later.
            rho_norm = pm.Deterministic('rho_norm', rho*pm.math.exp(α*E_rho + C))
            model = pm.Deterministic('model', pm.math.exp((E_model - Eshift)/T)/T)
            rhoSn = pm.Deterministic('rhoSn', pm.math.exp((Sn - Eshift)/T)/T)
            D0 = pm.Deterministic("D0", 2.*1e6/(spin_dist(Sn)*rhoSn)) # eV

            # Our likelihoods
            y_low = None
            y_high = None


            if nld.std is None:
                y_low = pm.StudentT('y_low', mu=b0 + b1*E_low, sd=ϵ, nu=ν, observed=q_low)
                y_high = pm.StudentT('y_high', mu=c0 + c1*E_high, sd=ϵ, nu=ν, observed=q_high)
            else:
                y_low = pm.Normal('y_low', mu=b0 + b1*E_low, sigma=qerr_low, observed=q_low)
                y_high = pm.Normal('y_high', mu=c0 + c1*E_high - α*E_high - C, sigma=qerr_high, observed=q_high)


        # Return the model
        return pymc_model

    def setup_BSFG_model(self, nld: Optional[Vector] = None,
                         gsf: Optional[Vector] = None,
                         limit_low: Optional[Tuple[float,float]] = None,
                         limit_high: Optional[Tuple[float,float]] = None,
                         model_range: Optional[np.ndarray] = None) -> pm.Model:
        """ Method responsible for declaring our pyMC model.
        """

        # We set our data
        nld = self.self_if_none(nld).copy()
        gsf = self.self_if_none(gsf).copy()
        nld.to_MeV()
        gsf.to_MeV()

        limit_low = self.self_if_none(limit_low)
        self.limit_low = limit_low

        nld_low = nld.cut(*limit_low, inplace=False)
        discrete_low = self.discrete.cut(*limit_low, inplace=False)

        limit_high = self.self_if_none(limit_high)
        self.limit_high = limit_high

        nld_high = nld.cut(*limit_high, inplace=False)

        # Standarizing
        q_tmp = np.log(discrete_low.values/nld_low.values)
        data_low = {'E': nld_low.E,
                    'E_mean': nld_low.E.mean(),
                    'E_std': nld_low.E.std(),
                    'q': q_tmp,
                    'q_mean': q_tmp.mean(),
                    'q_std': q_tmp.std()}
        data_low['E_st'] = (data_low['E'] - data_low['E_mean'])/data_low['E_std']
        data_low['q_st'] = (data_low['q'] - data_low['q_mean'])/data_low['q_std']
        q_tmp = nld_high.values # Unfortunatly the BSFG model isn't linear in log space.
        data_high = {'E': nld_high.E,
                     'E_mean': nld_high.E.mean(),
                     'E_std': nld_high.E.std(),
                     'q': q_tmp,
                     'q_mean': q_tmp.mean(),
                     'q_std': q_tmp.std()}
        data_high['E_st'] = (data_high['E'] - data_high['E_mean'])/data_high['E_std']
        data_high['q_st'] = (data_high['q'] - data_high['q_mean'])/data_high['q_std']


        with pm.Model() as pymc_model:

            # Set the data from the lower region fitting the discrete states
            E_low = pm.Data('E_low', data_low['E_st'])
            q_low = pm.Data('q_low', data_low['q_st'])
            qerr_low = None

            # Set the data from the higher region fitting the BSFG model
            E_high = pm.Data('E_high', data_high['E'])
            q_high = pm.Data('q_high', data_high['q'])
            qerr_high = None

            # Set data for infering the NLD after normalization.
            E_rho = pm.Data('E_rho', nld.E)
            rho = pm.Data('rho', nld.values)
            E_gsf = pm.Data('E_gsf', gsf.E)
            gSF = pm.Data('gSF', gsf.values)

            E_disc = pm.Data('E_disc', self.discrete.E)
            rho_disc = pm.Data('rho_disc', self.discrete.values)

            ϵ = None
            ν_ = None
            ν = None

            if nld.std is None:
                ϵ = pm.HalfNormal('ϵ', 5)
                ν_ = pm.Exponential('ν_', 1/29)
                ν = pm.Deterministic('ν', ν_ + 1)
            else:
                qerr_low = pm.Data('qerr_low', nld_low.std/nld_low.values)
                qerr_high = pm.Data('qerr_high', nld_high.std/nld_high.values)

            # Set dummy data for infering the BSFG model after sampling
            E_model = pm.Data('E_model', model_range)
            Sn = self.norm_pars.Sn[0]

            # Setup of the posterior
            b0 = pm.Normal('b0', mu=0, sigma=100)
            b1 = pm.Normal('b1', mu=0, sigma=100)

            α = pm.Deterministic('α', data_low['q_std']/data_low['E_std']*b1)
            C = pm.Deterministic('C', data_low['q_mean'] + b0*data_low['q_std'] - α*data_low['E_mean'])
            A = pm.Deterministic('A', pm.math.exp(C))

            # BSFG model
            a = pm.Normal('a', mu=8.498, sigma=10*8.498)
            Eshift = pm.Bound(pm.Normal, upper=min(data_high['E']))('Eshift', mu=-0.067, sigma=5)

            # Some deterministic functions that are useful later.
            rho_norm = pm.Deterministic('rho_norm', rho*pm.math.exp(α*E_rho + C))
            gsf_norm = pm.Deterministic('gsf_norm', gSF*pm.math.exp(α*E_gsf))
            model = pm.Deterministic('model', self.rhoBSFG(E_model, a, Eshift))
            rhoSn = pm.Deterministic('rhoSn', self.rhoBSFG(Sn, a, Eshift))

            spin_dist = lambda Ex: pm.math.exp(-0.5/self.sigma(Ex, a, Eshift)**2)/self.sigma(Ex, a, Eshift)
            D0 = pm.Deterministic("D0", 2.*1e6/(spin_dist(Sn)*rhoSn)) # eV

            # Our likelihoods
            y_low = None
            y_high = None

            if nld.std is None:
                y_low = pm.StudentT('y_low', mu=b0 + b1*E_low, sd=ϵ, nu=ν, observed=q_low)
                y_high = pm.StudentT('y_high', mu=pm.math.log(self.rhoBSFG(E_high, a, Eshift)), sd=ϵ, nu=ν, observed=pm.math.log(q_high) + α*E_high + C)
            else:
                y_low = pm.Normal('y_low', mu=b0 + b1*E_low, sigma=qerr_low, observed=q_low)
                y_high = pm.Normal('y_high', mu=self.rhoBSFG(E_high, a, Eshift), sigma=A*qerr_high*pm.math.exp(α*E_high), observed=A*q_high*pm.math.exp(α*E_high))

        return pymc_model

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

        if fig is not None and add_figlegend:
            fig.legend(loc=9, ncol=3, frameon=False)

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

