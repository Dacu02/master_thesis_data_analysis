from .Filter import Filter
from scipy.signal import cheby2, sosfiltfilt
from framework import Trajectory, Curve
import numpy as np
class Chebyshev(Filter):
	"""Chebyshev type II digital filter."""

	def __init__(self, order:int, stopband_attenuation:float, cutoff:float, btype='lowpass'):
		self.order = order
		self.stopband_attenuation = stopband_attenuation
		self.cutoff = cutoff
		self.btype = btype

	def apply(self, data: Trajectory|Curve) -> Trajectory|Curve:
		dt = np.median(np.diff(data.t))
		nyquist_hz = 0.5 / dt
		filter = cheby2(self.order, # type: ignore
      self.stopband_attenuation,
      self.cutoff / nyquist_hz,
      btype=self.btype,
			output='sos'
	)

		if isinstance(data, Curve):
			return Curve(
				x = sosfiltfilt(filter, data.x),
				y = sosfiltfilt(filter, data.y),
				z = sosfiltfilt(filter, data.z),
				t = data.t
			)
		else:
			return Trajectory(
				vx = sosfiltfilt(filter, data.vx),
				vy = sosfiltfilt(filter, data.vy),
				vz = sosfiltfilt(filter, data.vz),
				t = data.t
			)
