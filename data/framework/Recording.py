from . import Trajectory, Curve
import numpy as np
class Recording():

    def __init__(self, trajectory: Trajectory, curve: Curve, t: np.ndarray|None = None):
        """Initialize the Recording with a trajectory and a curve.
        
        Parameters
        ----------
        trajectory : Trajectory
            The trajectory associated with the recording.
        curve : Curve
            The curve associated with the recording.
        t : np.ndarray, optional
            The time array associated with the recording, in seconds. If None, it will be generated from 0 to 1.
        """
        self.trajectory = trajectory
        self.curve = curve

        if len(trajectory) != len(curve):
            raise ValueError("Trajectory and Curve must have the same length.")
        
        if t is not None:
            if len(t) != len(trajectory):
                raise ValueError("Time array must have the same length as trajectory and curve.")
            
            self.trajectory.t = t
            self.curve.t = t

    @property
    def t(self) -> np.ndarray:
        """Return the time array associated with the recording."""
        return self.trajectory.t