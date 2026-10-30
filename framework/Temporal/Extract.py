from . import StrokeSequence
from abc import ABC, abstractmethod
class Extract(ABC):
  
    @abstractmethod
    def approximate(self, recording) -> StrokeSequence:
        """Approximate the given curve points using the specific approximation method.

        Parameters
        ----------
        recording : Recording
            The recording to approximate.

        Returns
        -------
        StrokeSequence
            The approximated stroke sequence.
        """
        pass