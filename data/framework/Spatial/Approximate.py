from abc import ABC, abstractmethod
from . import Curve
class Approximate(ABC):
    """An abstract base class for approximation methods."""

    def __init__(self, *args, **kwargs):
        """Initialize the Approximate class."""
        super().__init__()
  
    @abstractmethod
    def approximate(self, curve: Curve) -> tuple[dict, Curve]:
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
        pass

    @abstractmethod
    def get_approximation_paramters(self) -> dict:
        """Return the parameters of the approximation as a dictionary."""
        pass

    @abstractmethod
    def generate_curve(self, **kwargs) -> Curve:
        """Generate a curve based on the approximation parameters.

        Parameters
        ----------
        kwargs : dict
            The parameters for generating the curve.

        Returns
        -------
        Curve
            The generated curve.
        """
        pass