from .Filter import Filter
from scipy.interpolate import PchipInterpolator, interp1d
import numpy as np
from framework import Trajectory, Curve

class Resample(Filter):
	"""Resample trajectory data on a uniform time grid."""

	def __init__(self, interpolation="pchip", target_fs=None):
		self.interpolation = interpolation
		self.target_fs = target_fs

	def apply(self, data: Trajectory|Curve, target_fs=None) -> Trajectory|Curve:
		"""Resample ``data`` and return a copy with updated trajectory fields."""
		target_fs = self.target_fs if target_fs is None else target_fs
		if target_fs is None:
			raise ValueError("target_fs must be specified")
		if target_fs <= 0:
			raise ValueError("target_fs must be positive")

		t_old = np.asarray(data.t)
		if t_old.ndim != 1 or t_old.size < 2:
			raise ValueError("data.t must contain at least two samples")

		t_new = np.arange(t_old[0], t_old[-1] + 0.5 / target_fs, 1.0 / target_fs)
		if self.interpolation == "pchip":
			interpolate = lambda values: PchipInterpolator(  # noqa: E731
				t_old, values, axis=0
			)(t_new)
		else:
			interpolate = lambda values: interp1d(  # noqa: E731
				t_old, values, kind=self.interpolation, axis=0
			)(t_new)

		if isinstance(data, Curve):
			return Curve(
				x=interpolate(data.x),
				y=interpolate(data.y),
				z=interpolate(data.z),
				t=t_new
			)
		else:
			return Trajectory(
				vx=interpolate(data.vx),
				vy=interpolate(data.vy),
				vz=interpolate(data.vz),
				t=t_new
			)