import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence


class Model:
    """ Parent class for all derived models.

    Attributes:
        name: Name of the model
        N_free: Number of free parameters in the model.

    """
    def __init__(self, name: str, N_free: int):
        """ Setup name
        Args:
            name: Name of the model (str)
        """
        self.name = name
        self.N_free = N_free

    def prior(self, param: ndarray) -> ndarray:
        """ Convert from unit cube to parameter cube.
        Args:
            param: (Listlike, ndarray) Unit cube.
        Returns: Parameter cube.
        """
        raise NotImplementedError

    def __call__(self, *args) -> ndarray:
        """ Evaluate the model. To be implemented
        in the derived classes.
        """
        raise NotImplementedError

    def prior_predicative(self, E: ndarray) -> ndarray:
        """ Evaluate the prior predicative. To be implemented in
        the derived classes.
        """
        raise NotImplementedError
