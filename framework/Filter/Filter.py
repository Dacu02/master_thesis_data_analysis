from abc import ABC, abstractmethod
from framework import Trajectory, Curve
class Filter(ABC):

    @abstractmethod
    def apply(self, data: Trajectory | Curve) -> Trajectory | Curve:
        """Apply the filter to the given data."""
        pass