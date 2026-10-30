from .Filter import Filter
from scipy.signal import savgol_filter, sosfiltfilt
from framework import Trajectory, Curve
import numpy as np

class SavGol(Filter):
  """Savitzky-Golay digital filter."""

  def __init__(self, window_length:int, polyorder:int):
    self.window_length = window_length
    self.polyorder = polyorder

  def apply(self, data: Trajectory | Curve) -> Trajectory | Curve:
    """Apply the Savitzky-Golay filter to the input data."""
    if isinstance(data, Curve):
        return Curve(
            x = savgol_filter(data.x, self.window_length, self.polyorder),
            y = savgol_filter(data.y, self.window_length, self.polyorder),
            z = savgol_filter(data.z, self.window_length, self.polyorder),
            t = data.t
        )
    else:
        return Trajectory(
            vx = savgol_filter(data.vx, self.window_length, self.polyorder),
            vy = savgol_filter(data.vy, self.window_length, self.polyorder),
            vz = savgol_filter(data.vz, self.window_length, self.polyorder),
            t = data.t
        )