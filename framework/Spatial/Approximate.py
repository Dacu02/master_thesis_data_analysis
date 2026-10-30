from abc import ABC, abstractmethod
from . import Curve


class Approximate(ABC):
    """An abstract base class for approximation methods."""

    def __init__(self, *args, **kwargs):
        """Initialize the Approximate class."""
        super().__init__()
  
    @abstractmethod
    def approximate(self, curve: Curve) -> tuple[dict, Curve] | tuple[list[dict], Curve]:
        """Approximate the given curve points using the specific approximation method.

        Parameters
        ----------
        curve : Curve
            The curve to approximate.

        Returns
        -------
        dict
            A dictionary containing the parameters of the approximation.
        Curve
            The approximated curve.
        """
        raise NotImplementedError("The 'approximate' method must be implemented in subclasses.")