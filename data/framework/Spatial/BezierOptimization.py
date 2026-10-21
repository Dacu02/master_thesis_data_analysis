import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.special import erf

from . import Curve, Approximate


class BezierOptimization(Approximate):
  """
  Fits a chain of cubic Bezier segments to a curve. Each segment goes from a
  local minimum to the next one, has prescribed end tangent directions, and its
  arc length is constrained to be exactly Ds[i]. The only free parameters of a
  segment are the two tangent lengths (lambda_s, lambda_e), which are chosen to
  minimize the mean squared error with respect to the original points.
  """

  # Gauss-Legendre nodes/weights on [0, 1], used to integrate the arc length.
  _GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
  _GL_S = 0.5 * (_GL_X + 1.0)
  _GL_W = 0.5 * _GL_W

  # Dense grid on [0, 1] used to invert the arc length ("arclength" parametrization).
  _GRID = np.linspace(0.0, 1.0, 4001)

  def __init__(self):
    super().__init__()

  # ------------------------------------------------------------------ #
  # Bezier primitives
  # ------------------------------------------------------------------ #
  @staticmethod
  def _basis(s, p0, p3, first_direction, last_direction):
    """
    Split the Bezier curve into B(s) = base + lambda_s * col_s + lambda_e * col_e,
    which is affine in the tangent lengths.
    """
    s = np.asarray(s, dtype=float).reshape(-1, 1)   # (N, 1)
    base = ((1 - s) ** 3 + 3 * (1 - s) ** 2 * s) * p0 \
         + (3 * (1 - s) * s ** 2 + s ** 3) * p3      # (N, 3)
    col_s = 3 * (1 - s) ** 2 * s * first_direction   # (N, 3)
    col_e = -3 * (1 - s) * s ** 2 * last_direction   # (N, 3)
    return base, col_s, col_e

  @staticmethod
  def _bezier(s, lambda_s, lambda_e, p0, p3, first_direction, last_direction):
    base, col_s, col_e = BezierOptimization._basis(s, p0, p3, first_direction, last_direction)
    return base + lambda_s * col_s + lambda_e * col_e   # -> (N, 3)

  @staticmethod
  def _bezier_derivative(s, lambda_s, lambda_e, p0, p3, first_direction, last_direction):
    s = np.asarray(s, dtype=float).reshape(-1, 1)   # (N, 1)
    p1 = p0 + first_direction * lambda_s
    p2 = p3 - last_direction * lambda_e
    return (
        3 * (1 - s) ** 2 * (p1 - p0)
        + 6 * (1 - s) * s * (p2 - p1)
        + 3 * s ** 2 * (p3 - p2)
    )   # -> (N, 3)

  @classmethod
  def _arc_length(cls, lambda_s, lambda_e, p0, p3, first_direction, last_direction):
    """Arc length of the Bezier segment, via Gauss-Legendre quadrature."""
    d = cls._bezier_derivative(cls._GL_S, lambda_s, lambda_e, p0, p3, first_direction, last_direction)
    return float(np.linalg.norm(d, axis=1) @ cls._GL_W)

  @classmethod
  def _arclength_parameters(cls, sigma, lambda_s, lambda_e, p0, p3, first_direction, last_direction):
    """
    Bezier parameters s_k such that the normalized arc length of the curve up to s_k
    is equal to sigma_k: s_k = Lambda_B^{-1}(sigma_k), with
    Lambda_B(s) = (1 / L) * integral_0^s |B'(u)| du  and  L = Lambda_B-normalization (total length).
    """
    speed = np.linalg.norm(
      cls._bezier_derivative(cls._GRID, lambda_s, lambda_e, p0, p3, first_direction, last_direction), axis=1
    )
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(cls._GRID))])
    cumulative /= cumulative[-1]
    return np.interp(sigma, cumulative, cls._GRID)

  # ------------------------------------------------------------------ #
  # Helpers
  # ------------------------------------------------------------------ #
  @staticmethod
  def lognormal_fraction(t, t0, mu, sd):
    """
    Fraction of its arc length covered by a lognormal stroke at the times `t`:
    Phi(t) = 0.5 * (1 + erf((ln(t - t0) - mu) / (sd * sqrt(2)))), with Phi = 0 for t <= t0,
    rescaled so that it goes from 0 at t[0] to 1 at t[-1]. The result can be used as an
    element of `output_sigma`.
    """
    t = np.asarray(t, dtype=float)
    after = t > t0
    phi = np.zeros_like(t)
    phi[after] = 0.5 * (1.0 + erf((np.log(t[after] - t0) - mu) / (sd * np.sqrt(2.0))))
    span = phi[-1] - phi[0]
    if span <= 0:
      raise ValueError("The stroke does not advance between the first and the last sample.")
    return (phi - phi[0]) / span

  @staticmethod
  def _check_output_sigma(values, n, i, tol=1e-6):
    sigma_out = np.asarray(values, dtype=float)
    if sigma_out.shape != (n,):
      raise ValueError(f"output_sigma[{i}] must have one value per sample of the segment ({n}), got shape {sigma_out.shape}.")
    if not np.all(np.isfinite(sigma_out)) or np.any(np.diff(sigma_out) < 0):
      raise ValueError(f"output_sigma[{i}] must be finite and non-decreasing.")
    if abs(sigma_out[0]) > tol or abs(sigma_out[-1] - 1.0) > tol:
      raise ValueError(f"output_sigma[{i}] must go from 0 to 1 over the segment.")
    return sigma_out

  @staticmethod
  def _boundary_direction(points: np.ndarray, i0: int, i1: int, min_dist: float, from_start: bool) -> np.ndarray:
    """
    Unit tangent direction at one end of the segment points[i0:i1 + 1].
    The returned vector always points along the direction of travel.

    mode="chord": the distance is the straight-line (Euclidean) distance between
        the end point and a candidate sample; the direction is the chord from the
        end point to the first sample at least `min_dist` away from it.
    mode="path": the distance is the path length accumulated sample by sample from
        the end point; the direction is that of the single step at which the
        accumulated length first reaches `min_dist` (original behaviour).
    """
    if from_start:
      anchor, indices = points[i0], range(i0 + 1, i1 + 1)
    else:
      anchor, indices = points[i1], range(i1 - 1, i0 - 1, -1)

    for j in indices:
      chord = (points[j] - anchor) if from_start else (anchor - points[j])
      norm = np.linalg.norm(chord)
      if norm >= min_dist and norm > 0:
        return chord / norm
    raise ValueError("The segment is shorter than direction_distance: cannot estimate the tangent direction.")

  def _fit_segment(self, segment_points, sigma, p0, p3, first_direction, last_direction,
                   length, lambda_min, lambda_max, plot_comparison=False):
    """
    Minimize the MSE over (lambda_s, lambda_e) subject to
        lambda_min <= lambda <= lambda_max   and   arc_length(lambda_s, lambda_e) = length.

    `sigma` is the data-side parameter of each sample. Sigma is the normalized path fraction of each sample and the Bezier
    parameter is s_k = Lambda_B^{-1}(sigma_k), which depends on the lambdas themselves.

    Returns (lambda_s, lambda_e, mse, s), with s the Bezier parameter of each sample.
    """
    chord = float(np.linalg.norm(p3 - p0))
    if length < chord * (1 - 1e-9):
      raise ValueError(f"Ds = {length:.6g} is shorter than the distance between the segment end points ({chord:.6g}): no curve can satisfy it.")

    # Work in units of `length` so that variables, objective and constraint are O(1).
    scale = length
    lo, hi = lambda_min / scale, lambda_max / scale

    base, col_s, col_e = self._basis(sigma, p0, p3, first_direction, last_direction)
    A = np.column_stack([col_s.ravel(), col_e.ravel()]) * scale
    b = (segment_points - base).ravel()

    def mse_affine(u):
      return np.mean(np.sum((A @ u - b).reshape(-1, 3) ** 2, axis=1)) / scale ** 2

    def mse_arclength(u):
      ls, le = u * scale
      s = self._arclength_parameters(sigma, ls, le, p0, p3, first_direction, last_direction)
      model = self._bezier(s, ls, le, p0, p3, first_direction, last_direction)
      return np.mean(np.sum((model - segment_points) ** 2, axis=1)) / scale ** 2

    def length_residual(u):
      return self._arc_length(u[0] * scale, u[1] * scale, p0, p3, first_direction, last_direction) / length - 1.0

    def solve(objective, x0):
      result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(lo, hi), (lo, hi)],
        constraints=[{"type": "eq", "fun": length_residual}],
        options={"ftol": 1e-12, "maxiter": 1000},
      )
      violation = abs(length_residual(result.x))
      if violation > 1e-4:
        raise RuntimeError(
          f"Could not reach the required arc length Ds = {length:.6g} within "
          f"[lambda_min, lambda_max] = [{lambda_min:.6g}, {lambda_max:.6g}] "
          f"(relative violation {violation:.2e}; solver: {result.message})."
        )
      return result.x

    # Stage 1: affine objective. The bounds-only problem is a linear least squares, solved
    # exactly and used as a warm start for the length-constrained one.
    if not self.skip_discrete_step:
      x_d = solve(mse_affine, lsq_linear(A, b, bounds=(lo, hi)).x)
    else:
      x_d = lsq_linear(A, b, bounds=(lo, hi)).x
    # Stage 2: the objective is no longer affine because s_k depends on the lambdas.
    # The length constraint leaves a single degree of freedom, so this stays cheap.
    x = solve(mse_arclength, x_d)

    if not self.skip_discrete_step and self.plot_comparison:
      import matplotlib.pyplot as plt

      lambda_s_first, lambda_e_first = x_d * scale
      s_first = self._arclength_parameters(
        sigma, lambda_s_first, lambda_e_first, p0, p3,
        first_direction, last_direction)
      first_solution = self._bezier(
        s_first, lambda_s_first, lambda_e_first, p0, p3,
        first_direction, last_direction)

      lambda_s_second, lambda_e_second = x * scale
      s_second = self._arclength_parameters(
        sigma, lambda_s_second, lambda_e_second, p0, p3,
        first_direction, last_direction)
      second_solution = self._bezier(
        s_second, lambda_s_second, lambda_e_second, p0, p3,
        first_direction, last_direction)

      figure = plt.figure()
      axes = figure.add_subplot(111, projection="3d")
      axes.plot(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2],
                "k.", label="data")
      axes.plot(first_solution[:, 0], first_solution[:, 1], first_solution[:, 2],
                "b--", label="first solver (affine)")
      axes.plot(second_solution[:, 0], second_solution[:, 1], second_solution[:, 2],
                "r-", label="second solver (arc-length)")
      axes.set_xlabel("x")
      axes.set_ylabel("y")
      axes.set_zlabel("z")
      axes.set_title("Bezier solver comparison")
      axes.legend()
      figure.tight_layout()
      plt.show()
    
    lambda_s, lambda_e = x * scale
    s = self._arclength_parameters(sigma, lambda_s, lambda_e, p0, p3, first_direction, last_direction)
    return float(lambda_s), float(lambda_e), float(mse_arclength(x) * scale ** 2), s

  # ------------------------------------------------------------------ #
  # Public API
  # ------------------------------------------------------------------ #
  def set_problem(self,
                  minima_idx: list[int],
                  Ds: list[float],
                  lambda_min: float = 1e-3,
                  lambda_max: float | None = None,
                  direction_distance: float = 0.001,
                  output_sigma: list | None = None,
                  plot_comparison: bool = False,
                  skip_discrete_step: bool = False):
    """
    Set up the optimization problem for Bezier curve fitting.

    Parameters
    ----------
    minima_idx : list[int]
        Indices of the local minima in the curve (segment boundaries).
    Ds : list[float]
        Arc length of each Bezier segment. Must have len(minima_idx) - 1 entries;
        Ds[i] is enforced exactly for the segment between minima_idx[i] and minima_idx[i + 1].
    lambda_min : float, optional
        Minimum allowed tangent length (same units as the curve). Default is 1e-3.
    lambda_max : float | None, optional
        Maximum allowed tangent length. If None, it is chosen per segment as
        max(2 * chord, Ds[i]), where chord is the distance between the segment end points.
        Default is None.
    direction_distance : float, optional
        Minimum distance from a segment end point used to estimate the tangent direction
        (same units as the curve). Default is 0.001.
    output_sigma : list of np.ndarray or None, optional
        Optional timing law used to *generate* the output samples. One array per segment,
        with one value per curve sample of the segment (minima_idx[i + 1] - minima_idx[i] + 1),
        non-decreasing from 0 to 1: the fraction of the segment's arc length covered at each
        sample. The geometry (lambda_s, lambda_e) is still fitted on the data with
        `parametrization`; output sample k is then placed on the fitted Bezier at arc-length
        fraction output_sigma[i][k], so the generated speed is Ds[i] * d(output_sigma[i])/dt,
        independently of the data timing and of the Bezier parametric speed. For a lognormal
        stroke use `lognormal_fraction(t, t0, mu, sd)`. If None, each sample keeps the position
        it has in the fit. Default is None.
      plot_comparison : bool, optional
        If True, display a 3D comparison of the affine and arc-length solver
        solutions for each segment. Default is False.
      skip_discrete_step : bool, optional
        If True, skip the discrete step of the optimization (after the least-squares fit) and
        only perform the continuous step (the arc-length constrained optimization).
    """
    if lambda_min < 0:
      raise ValueError("lambda_min must be non-negative.")
    if lambda_max is not None and lambda_max < lambda_min:
      raise ValueError("lambda_max must be greater than or equal to lambda_min.")

    self.minima_idx = list(minima_idx)
    self.Ds = list(Ds)
    self.lambda_min = lambda_min
    self.lambda_max = lambda_max
    self.direction_distance = direction_distance
    self.output_sigma = output_sigma
    self.skip_discrete_step = skip_discrete_step
    self.plot_comparison = plot_comparison

  def approximate(self, curve: Curve) -> tuple[list[dict], Curve]:
    """
    Approximate the given curve points using Bezier optimization.

    Parameters
    ----------
    curve : Curve
        The curve to approximate.

    Returns
    -------
    tuple[list[dict], Curve]
        A list with, for each segment, the fitted parameters ("lambda_s", "lambda_e",
        "error" = fit MSE, "error_output" = MSE of the generated samples, "length" = Bezier
        arc length), and the approximated Bezier spline,
        sampled at the original samples between minima_idx[0] and minima_idx[-1].
    """
    curve_points = np.asarray(curve.points, dtype=float)
    n_points = len(curve_points)
    idx = self.minima_idx

    if n_points < 2:
      raise ValueError("At least two points are required to compute the first direction.")
    if len(idx) < 2:
      raise ValueError("At least two minima are required to define a segment.")
    if len(self.Ds) != len(idx) - 1:
      raise ValueError(f"Ds must have len(minima_idx) - 1 = {len(idx) - 1} entries, got {len(self.Ds)}.")
    if self.output_sigma is not None and len(self.output_sigma) != len(idx) - 1:
      raise ValueError(f"output_sigma must have len(minima_idx) - 1 = {len(idx) - 1} entries, got {len(self.output_sigma)}.")
    if idx[0] < 0 or idx[-1] >= n_points:
      raise ValueError("minima_idx contains indices outside the curve.")
    if any(a >= b for a, b in zip(idx[:-1], idx[1:])):
      raise ValueError("minima_idx must be in strictly increasing order.")
    if any(D <= 0 for D in self.Ds):
      raise ValueError("All Ds must be strictly positive.")

    spline_parameters = []
    full_approximation_points = []

    for i in range(len(idx) - 1):
      i0, i1 = idx[i], idx[i + 1]
      D = float(self.Ds[i])
      segment_points = curve_points[i0:i1 + 1]   # (n, 3)
      n = segment_points.shape[0]
      p0, p3 = segment_points[0], segment_points[-1]

      first_direction = self._boundary_direction(curve_points, i0, i1, self.direction_distance, True)
      last_direction = self._boundary_direction(curve_points, i0, i1, self.direction_distance, False)

      steps = np.linalg.norm(np.diff(segment_points, axis=0), axis=1)
      total = steps.sum()
      if total == 0:
        raise ValueError("Degenerate segment: all points coincide.")
      sigma = np.concatenate([[0.0], np.cumsum(steps)]) / total

      lambda_max = self.lambda_max if self.lambda_max is not None else 2 * D

      lambda_s, lambda_e, error, s = self._fit_segment(
        segment_points, sigma, p0, p3, first_direction, last_direction, D,
        self.lambda_min, lambda_max, self.plot_comparison)

      if self.output_sigma is not None:
        # Geometry comes from the fit; timing comes from the external law.
        sigma_out = self._check_output_sigma(self.output_sigma[i], n, i)
        s = self._arclength_parameters(sigma_out, lambda_s, lambda_e, p0, p3, first_direction, last_direction)

      seg = self._bezier(s, lambda_s, lambda_e, p0, p3, first_direction, last_direction)
      full_approximation_points.append(seg if i == 0 else seg[1:])  # Avoid duplicating the shared end point
      spline_parameters.append({
        "lambda_s": lambda_s,
        "u_s": first_direction,
        "lambda_e": lambda_e,
        "delta_p": p3 - p0,
        "u_e": last_direction,
        "error": error,
        "error_output": float(np.mean(np.sum((seg - segment_points) ** 2, axis=1))),
        "length": self._arc_length(lambda_s, lambda_e, p0, p3, first_direction, last_direction),
      })

    points = np.concatenate(full_approximation_points, axis=0)
    t = np.asarray(curve.t, dtype=float)[idx[0]:idx[-1] + 1]   # same samples as the output points
    return spline_parameters, Curve(x=points[:, 0], y=points[:, 1], z=points[:, 2], t=t)