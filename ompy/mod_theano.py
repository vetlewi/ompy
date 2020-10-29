import theano
import theano.tensor as tt
import numpy as np
import numba as nb

from scipy.interpolate import interp1d

pi = float(np.pi)


class SMLO(theano.Op):
    itypes = [tt.dvector,
              tt.dscalar,
              tt.dscalar,
              tt.dscalar,
              tt.dscalar]

    otypes = [tt.dvector]

    def perform(self, node, inputs_storage, output_storage):
        E, mean, width, size, T = inputs_storage

        width_ = width*(E/mean + (2*pi*T/mean)**2)
        smlo = 8.6737e-8*size/(1-tt.exp(-E/T))
        smlo *= (2./pi)*E*width_
        smlo /= ((E**2 - mean**2)**2 + (E*width_)**2)
        output_storage = smlo


class model_matrix(theano.Op):

    # Input is energies matrix, 
    itypes = [tt.dmatrix,  # Ex
              tt.dmatrix,  # Eg
              # tt.dscalar,  # Sigma conv. of discrete
              tt.dscalar,  # Temperature
              tt.dscalar,  # Eshift,
              tt.dscalar,  # GDR mean
              tt.dscalar,  # GDR width
              tt.dscalar,  # GDR size
              tt.dscalar,  # PDR mean
              tt.dscalar,  # PDR width
              tt.dscalar,  # PDR size
              tt.dscalar,  # SF mean
              tt.dscalar,  # SF width
              tt.dscalar,  # SF size
              tt.dscalar]  # SM scaling

    otypes = [tt.dmatrix]

    def __init__(self, discrete, shell_model, disc_Emax=3.0):

        de = np.diff(discrete.E)[0]

        self.discrete = interp1d(discrete.E-de/2., discrete.values,
                                 kind='previous', bounds_error=False,
                                 fill_value=(0., 0.))

        self.shell_model = interp1d(shell_model.E-de/2., shell_model.values,
                                    kind='previous', bounds_error=False,
                                    fill_value=(0., 0.))

        self.disc_Emax = float(disc_Emax)

    def calculate_NLD(self, E, T, Eshift):
        return tt.where(E < self.disc_Emax, self.discrete(E),
                        tt.exp((E-Eshift)/T)/T)

    def perform(self, node, inputs_storage, output_storage):
        Ex, Eg, T, Eshift, gdr_mean, gdr_width, gdr_size, \
            pdr_mean, pdr_width, pdr_size, sf_mean, \
            sf_width, sf_size, sm_scale, = inputs_storage

        # First we will get the NLD
        rho = self.calculate_NLD(Ex-Eg, T, Eshift)
        strength = SMLO(Eg, gdr_mean, gdr_width, gdr_size, T)
        strength += SLO(Eg, pdr_mean, pdr_width, pdr_size)
        strength += SLO(Eg, sf_mean, sf_width, sf_size)
        strength += self.shell_model(Eg)*sm_scale

        tau = 2*pi*strength
        fg = rho * tau
        print(tt.sum(fg, axis=1).shape)
        fg = fg / tt.sum(fg, axis=1).reshape(fg.shape[0], 1)

        output_storage = rho*tau  #  Aaaand normalize!


@nb.jit(nopython=True)
def SMLO(E, mean, width,
         size, T):
    """ The simple modified Lorentzian model.

    Args:
        E: Gamma ray energy in [MeV]
        mean: Mean position of the resonance in [MeV]
        width: Width of the resonance in [MeV]
        size: Total cross-section in [mb MeV]
        T: Temperature of the final level in [MeV]
    Returns: Gamma ray strength [MeV^(-3)]
    """

    width_ = width*(E/mean + (2*np.pi*T/mean)**2)
    smlo = 8.6737e-8*size/(1-np.exp(-E/T))
    smlo *= (2./np.pi)*E*width_
    smlo /= ((E**2 - mean**2)**2 + (E*width_)**2)
    return smlo


@nb.jit(nopython=True)
def GLO(E, mean, width, size, T):
    """ The Generalized Lorentzian model.

    Args:
        E: Gamma ray energy in [MeV]
        mean: Mean position of the resonance in [MeV]
        width: Width of the resonance in [MeV]
        size: Total cross-section in [mb]
        T: Temperature of the final level in [MeV]
    Returns: Gamma ray strength [MeV^(-3)]
    """

    width_ = width*(E**2 + (2*np.pi*T)**2)/mean**2
    glo = E*width_/((E**2 - mean**2)**2+(E*width_)**2)
    glo += 0.7*width*(2*np.pi*T)**2/mean**5
    glo *= 8.6737e-8*size*width
    return glo

@nb.jit(nopython=True)
def SLO(E, mean, width, size):
    """ The standard Lorentzian.

    Args:
        E: Gamma ray energy in [MeV]
        mean: Mean position of the resonance in [MeV]
        width: Width of the resonance in [MeV]
        size: Total cross-section in [mb]
        T: Temperature of the final level in [MeV]
    Returns: Gamma ray strength [MeV^(-3)]
    """

    slo = 8.6737e-8*size*E*width**2
    slo /= ((E**2 - mean**2)**2 + (E*width)**2)
    return slo
