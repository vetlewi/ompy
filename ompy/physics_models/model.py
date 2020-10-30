import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Any, Union, Callable, Dict, List, Sequence


class model:
    """ Parent class for all derived models.

    Attributes:
        name: Name of the model

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
        """
        """
        raise NotImplementedError

    def mean(self, E: ndarray) -> ndarray:
        raise NotImplementedError
