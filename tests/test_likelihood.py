import pytest
import ompy as om

import numpy as np
from numpy.testing import assert_equal, assert_allclose


def test_likelihood():

    x = np.linspace(0, 10, 11)
    y = 5. * x + 2. + np.random.rand(len(x))*0.4

    def model(x, a, b):
        return x*a + b

    lh = om.Likelihood(x=x, y=y, model=model)

    with pytest.raises(NotImplementedError):
        lh(np.zeros(2))

    with pytest.raises(NotImplementedError):
        lh.logp(np.zeros(5))


def test_NormalLikelihood():
    x = np.linspace(0, 10, 11)
    y = 5. * x + 2. + np.random.rand(len(x))*0.4

    yerr = 0.4 * np.ones(len(x))

    def model(x, a, b):
        return a*x + b

    lh = om.NormalLikelihood(x=x, y=y, yerr=yerr,
                             model=lambda x, param: model(x, *param))

    logp_ompy = lh([3.7, 2.4])

    logp_real = np.sum(np.log(1/(np.sqrt(2*np.pi)*yerr)))
    logp_real -= 0.5*np.sum(((y - model(x, 3.7, 2.4))/yerr)**2)

    assert_allclose(logp_ompy, logp_real)


def test_OsloNormalLikelihood():
    x = np.linspace(0, 10, 11)
    y = 5. * x + 2. + np.random.rand(len(x))*0.4

    yerr = 0.4 * np.ones(len(x))

    def model(x, a, b):
        return a*x + b

    lh = om.OsloNormalLikelihood(x=x, y=y, yerr=yerr,
                                 model=lambda x, param: model(x, *param))

    logp_ompy = lh(2.0, 0.5, [3.7, 2.4])

    y = 2.0*y*np.exp(0.5*x)
    yerr = 2.0*yerr*np.exp(0.5*x)

    logp_real = np.sum(np.log(1/(np.sqrt(2*np.pi)*yerr)))
    logp_real -= 0.5*np.sum(((y - model(x, 3.7, 2.4))/yerr)**2)

    assert_allclose(logp_ompy, logp_real)

