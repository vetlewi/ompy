import numpy as np
import pymc3 as pm
from typing import Optional, Sequence, Tuple, Any, Union, Dict

gsf_factor = 8.6737e-8

def SLMO_model(Eg, mean, width, size, T):
    """ Calculate the gSF from the SLMO model.

    Args:
        Eg: (float or Sequence) Energies to calculate the strength for (MeV).
        mean: Mean of the resonance (MeV).
        width: Width of the resonance (MeV).
        size: Integrated area of the resonance (mb MeV).
    Returns: The SLMO model at energ(y/ies) Eg.
    """
    width_ = width*(Eg/mean + (2*np.pi*T/mean)**2)
    slmo = gsf_factor*size/(1-np.exp(-Eg/T))
    slmo *= (2./np.pi)*Eg*width_/((Eg**2 - mean**2)**2 + (Eg*width_)**2)
    return slmo

def SLO_model(Eg, mean, width, size):
    """ The standard lortentzian.
    Args:
        Eg: (float or Sequence) Energies to evaluate.
        mean: Mean of the SLO (MeV).
        width: Width at half-maximum (MeV).
        size: Size of the resonance (mb).
    Returns: SLO at the input energies.
    """
    
    slo = gsf_factor*size*Eg*width**2/((Eg**2 - mean**2)**2 + (Eg*width)**2)
    return slo

def UB_model(Eg, slope, size):
    """ An exponential model to describe the upbend.
    Args:
        Eg: (float or Sequence) energy to evaluate at.
        slope: Slope of the upbend (1/MeV)
        size: Absolute value of the upbend (mb/MeV)
    """
    return size*gsf_factor*np.exp(-slope*Eg)