import pytest
import ompy as om
import numpy as np


def test_pepare_data():
    Ex = np.linspace(0, 4., 11)
    rho = np.exp((Ex - 0.3)/0.8)/0.8
    Eg = Ex + 1.5
    N = 10

    def gsf_vals(Eg):
        ub = om.UB_model(Eg, 1.61, 1.78)
        slmo = om.SLMO_model(Eg, 17.6, 7.0, 999., 0.8)
        return slmo+ub

    nlds = [om.Vector(E=Ex, values=rho+np.random.normal(size=rho.size)*0.2) for i in range(N)]
    gsfs = [om.Vector(E=Eg, values=gsf_vals(Eg)++np.random.normal(size=rho.size)*0.2) for i in range(N)]

    disc = om.Vector(E=Ex, values=rho)

    data = om.prepare_data(nlds, gsfs, (1.2, 2.8),
                           (3.0, 3.5), (1.7, 2.4), (0, 1000.), nld_ref_low=disc)


if __name__ == '__main__':
    print("running")
    test_pepare_data()