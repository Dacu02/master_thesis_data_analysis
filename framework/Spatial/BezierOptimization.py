from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.special import erf

from . import Curve, Approximate


@dataclass
class _Segment:
  """Data of one segment of the chain, in the units of the curve."""
  points: np.ndarray   # (n, 3) samples between two consecutive minima, end points included
  sigma: np.ndarray    # (n,) normalized cumulative chord length of the samples
  p0: np.ndarray       # first sample
  p3: np.ndarray       # last sample
  length: float        # required arc length Ds[i]
  lambda_min: float    # bounds of the two tangent lengths
  lambda_max: float


class BezierOptimization(Approximate):
  """
  Fits a chain of cubic Bezier segments to a curve. Each segment goes from a local
  minimum to the next one and its arc length is constrained to be exactly Ds[i].
  The tangent direction at every node of the chain is a free parameter, initialized
  with a finite-difference estimate of the velocity: forward difference at the start of
  the chain, backward difference at its end, mean of the backward and forward
  differences at every interior junction. Each interior direction is shared by the two
  segments that meet there, so the chain is G1 and all the segments are fitted as a
  single problem, together with the two tangent lengths (lambda_s, lambda_e) of every
  segment, to minimize the mean squared error with respect to the original points.
  """

  # Gauss-Legendre nodes/weights on [0, 1], used to integrate the arc length.
  _GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
  _GL_S = 0.5 * (_GL_X + 1.0)
  _GL_W = 0.5 * _GL_W

  # Dense grid on [0, 1] used to invert the arc length ("arclength" parametrization).
  _GRID = np.linspace(0.0, 1.0, 4001)

  # Largest accepted relative violation of an arc-length constraint.
  _LENGTH_TOL = 1e-4

  # Relative step of the forward differences used for gradients and Jacobians.
  _FD_STEP = np.sqrt(np.finfo(float).eps)

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
    Bezier parameters s_k = Lambda_B^{-1}(sigma_k), where
    Lambda_B(s) = (1 / L) * integral_0^s |B'(u)| du is the normalized curvilinear abscissa of the curve
    and L its length. B evaluated at the returned parameters is the curve reparametrized by its
    normalized curvilinear abscissa, evaluated at sigma_k.
    """
    speed = np.linalg.norm(
      cls._bezier_derivative(cls._GRID, lambda_s, lambda_e, p0, p3, first_direction, last_direction), axis=1
    )
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(cls._GRID))])
    cumulative /= cumulative[-1]
    return np.interp(sigma, cumulative, cls._GRID)

  # ------------------------------------------------------------------ #
  # Directions
  # ------------------------------------------------------------------ #
  @staticmethod
  def _tangent_basis(direction):
    """Two unit vectors orthogonal to `direction` and to each other."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    return e1, e2

  @staticmethod
  def _rotate_direction(direction, e1, e2, a, b):
    """
    Unit vector at angular distance sqrt(a^2 + b^2) [rad] from `direction`, rotated towards a * e1 + b * e2
    (exponential map of the unit sphere at `direction`). (a, b) = (0, 0) gives `direction` itself.
    """
    angle = np.hypot(a, b)
    if angle < 1e-12:
      rotated = direction + a * e1 + b * e2
      return rotated / np.linalg.norm(rotated)
    return np.cos(angle) * direction + np.sin(angle) * (a * e1 + b * e2) / angle

  @staticmethod
  def _one_sided_velocity(points: np.ndarray, t: np.ndarray, k: int, indices: range) -> np.ndarray | None:
    """
    Difference quotient (points[j] - points[k]) / (t[j] - t[k]), with j the first sample of
    `indices` that does not coincide with points[k], so that repeated samples (a still pen) are
    skipped. The same expression is a forward difference when `indices` follows k and a backward
    difference when it precedes k. Returns None if every sample of `indices` coincides with points[k].
    """
    for j in indices:
      if np.any(points[j] != points[k]):
        return (points[j] - points[k]) / (t[j] - t[k])
    return None

  @classmethod
  def _finite_difference_directions(cls, points: np.ndarray, t: np.ndarray, idx: list[int]) -> list[np.ndarray]:
    """
    Unit tangent directions at the samples idx: forward difference at the start of the chain,
    backward difference at its end, mean of the backward and forward differences at every interior
    junction. Each difference only uses samples of the segment adjacent to the node on that side.
    The returned vectors point along the direction of travel.
    """
    lo, hi = idx[0], idx[-1]
    if np.any(np.diff(t[lo:hi + 1]) <= 0):
      raise ValueError("The time stamps between minima_idx[0] and minima_idx[-1] must be strictly increasing.")

    velocities = [cls._one_sided_velocity(points, t, idx[0], range(idx[0] + 1, idx[1] + 1))]
    for m in range(1, len(idx) - 1):
      k = idx[m]
      backward = cls._one_sided_velocity(points, t, k, range(k - 1, idx[m - 1] - 1, -1))
      forward = cls._one_sided_velocity(points, t, k, range(k + 1, idx[m + 1] + 1))
      velocities.append(None if backward is None or forward is None else 0.5 * (backward + forward))
    velocities.append(cls._one_sided_velocity(points, t, idx[-1], range(idx[-1] - 1, idx[-2] - 1, -1)))

    directions = []
    for k, velocity in zip(idx, velocities):
      speed = 0.0 if velocity is None else np.linalg.norm(velocity)
      if not speed > 0:
        raise ValueError(
          f"Cannot estimate the tangent direction at sample {k}: the finite-difference velocity "
          f"vanishes there (the backward and forward differences cancel out).")
      directions.append(velocity / speed)
    return directions

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

  def _make_segment(self, points: np.ndarray, i0: int, i1: int, length: float) -> _Segment:
    """Samples, normalized chord-length abscissa and tangent-length bounds of points[i0:i1 + 1]."""
    segment_points = points[i0:i1 + 1]
    steps = np.linalg.norm(np.diff(segment_points, axis=0), axis=1)
    total = steps.sum()
    if total == 0:
      raise ValueError(f"Degenerate segment [{i0}, {i1}]: all points coincide.")

    p0, p3 = segment_points[0], segment_points[-1]
    chord = float(np.linalg.norm(p3 - p0))
    if length < chord * (1 - 1e-9):
      raise ValueError(
        f"Ds = {length:.6g} for segment [{i0}, {i1}] is shorter than the distance between its end "
        f"points ({chord:.6g}): no curve can satisfy it.")

    return _Segment(
      points=segment_points,
      sigma=np.concatenate([[0.0], np.cumsum(steps)]) / total,
      p0=p0,
      p3=p3,
      length=length,
      lambda_min=self.lambda_min,
      lambda_max=self.lambda_max if self.lambda_max is not None else 2 * length,
    )

  # ------------------------------------------------------------------ #
  # Optimization
  # ------------------------------------------------------------------ #
  def _affine_warm_start(self, i: int, segment: _Segment, first_direction, last_direction) -> np.ndarray:
    """
    Tangent lengths of segment i, divided by its length, with the directions kept at their initial
    estimates. Sample k is compared with B(sigma_k), i.e. the Bezier parameter is taken equal to the
    normalized abscissa of the data, which makes the MSE affine in the tangent lengths. The
    bounds-only problem is a linear least squares, solved exactly; unless skip_discrete_step is set,
    its solution is the starting point of the same problem with the arc-length constraint.
    """
    scale = segment.length
    lo, hi = segment.lambda_min / scale, segment.lambda_max / scale

    base, col_s, col_e = self._basis(segment.sigma, segment.p0, segment.p3, first_direction, last_direction)
    A = np.column_stack([col_s.ravel(), col_e.ravel()]) * scale
    b = (segment.points - base).ravel()
    x = lsq_linear(A, b, bounds=(lo, hi)).x
    if self.skip_discrete_step:
      return x

    def mse_affine(u):
      return np.mean(np.sum((A @ u - b).reshape(-1, 3) ** 2, axis=1)) / scale ** 2

    def length_residual(u):
      return self._arc_length(u[0] * scale, u[1] * scale, segment.p0, segment.p3,
                              first_direction, last_direction) / segment.length - 1.0

    result = minimize(
      mse_affine,
      x0=x,
      method="SLSQP",
      bounds=[(lo, hi), (lo, hi)],
      constraints=[{"type": "eq", "fun": length_residual}],
      options={"ftol": 1e-12, "maxiter": 1000},
    )
    violation = abs(length_residual(result.x))
    if violation > self._LENGTH_TOL:
      raise RuntimeError(
        f"Segment {i}: could not reach the required arc length Ds = {segment.length:.6g} within "
        f"[lambda_min, lambda_max] = [{segment.lambda_min:.6g}, {segment.lambda_max:.6g}] with the "
        f"initial directions (relative violation {violation:.2e}; solver: {result.message}).")
    return result.x

  def _fit_chain(self, segments: list[_Segment], directions: list[np.ndarray],
                 warm_start: np.ndarray) -> list[tuple]:
    """
    Minimize, over the tangent lengths of every segment and over every node direction, the sum of
    the segment MSEs, each normalized by the squared segment length, subject to
        lambda_min <= lambda <= lambda_max   and   arc_length_i = Ds[i] for every segment i.

    `sigma` is the normalized curvilinear abscissa of each sample of the data; sample k is compared
    with the Bezier point at the same normalized curvilinear abscissa, B(Lambda_B^{-1}(sigma_k)).

    Each node direction is parametrized by a rotation (a, b) [rad] of its finite-difference
    estimate, which is the starting point. Interior directions are shared by the two adjacent
    segments, so the fitted chain is G1. The rotation angle r = sqrt(a^2 + b^2) is limited to
    r <= self.max_direction_angle and its square is penalized with weight self.direction_weight.

    Segment i depends only on its two tangent lengths and on directions i and i + 1, so the
    gradient and the Jacobian are assembled from forward differences of each segment's own terms.

    Returns, for each segment, (lambda_s, lambda_e, first_direction, last_direction, mse).
    """
    n_segments = len(segments)
    n_lambda = 2 * n_segments
    n_nodes = n_segments + 1
    bases = [self._tangent_basis(d) for d in directions]

    # Optimization vector: [lambda_s, lambda_e] / Ds[i] of every segment, followed by (a, b) of
    # every node direction
    local = [np.array([2 * i, 2 * i + 1] + [n_lambda + 2 * i + j for j in range(4)])
             for i in range(n_segments)]

    lower = np.r_[np.concatenate([[seg.lambda_min / seg.length] * 2 for seg in segments]),
                  np.full(2 * n_nodes, -self.max_direction_angle)]
    upper = np.r_[np.concatenate([[seg.lambda_max / seg.length] * 2 for seg in segments]),
                  np.full(2 * n_nodes, self.max_direction_angle)]

    def unpack(i, v):
      """Local vector of segment i -> (lambda_s, lambda_e, first direction, last direction)."""
      segment = segments[i]
      first_direction = self._rotate_direction(directions[i], *bases[i], v[2], v[3])
      last_direction = self._rotate_direction(directions[i + 1], *bases[i + 1], v[4], v[5])
      return v[0] * segment.length, v[1] * segment.length, first_direction, last_direction

    def segment_mse(i, v):
      """MSE of segment i, normalized by its squared length."""
      segment = segments[i]
      ls, le, fd, ld = unpack(i, v)
      s = self._arclength_parameters(segment.sigma, ls, le, segment.p0, segment.p3, fd, ld)
      model = self._bezier(s, ls, le, segment.p0, segment.p3, fd, ld)
      return np.mean(np.sum((model - segment.points) ** 2, axis=1)) / segment.length ** 2

    def segment_length_residual(i, v):
      """Relative deviation of the arc length of segment i from Ds[i]."""
      segment = segments[i]
      ls, le, fd, ld = unpack(i, v)
      return self._arc_length(ls, le, segment.p0, segment.p3, fd, ld) / segment.length - 1.0

    def local_gradient(fun, i, v):
      """
      Value and forward-difference gradient of fun(i, .) at the local vector v. A step that would
      cross an upper bound is taken backwards.
      """
      f0 = fun(i, v)
      gradient = np.empty(v.size)
      upper_local = upper[local[i]]
      for j in range(v.size):
        h = self._FD_STEP * max(1.0, abs(v[j]))
        if v[j] + h > upper_local[j]:
          h = -h
        w = v.copy()
        w[j] += h
        gradient[j] = (fun(i, w) - f0) / (w[j] - v[j])
      return f0, gradient

    def objective(x):
      value, gradient = 0.0, np.zeros_like(x)
      for i in range(n_segments):
        f, g = local_gradient(segment_mse, i, x[local[i]])
        value += f
        gradient[local[i]] += g
      angles = x[n_lambda:]
      value += self.direction_weight * (angles @ angles)
      gradient[n_lambda:] += 2.0 * self.direction_weight * angles
      return value, gradient

    def length_residuals(x):
      return np.array([segment_length_residual(i, x[local[i]]) for i in range(n_segments)])

    def length_jacobian(x):
      jacobian = np.zeros((n_segments, x.size))
      for i in range(n_segments):
        _, jacobian[i, local[i]] = local_gradient(segment_length_residual, i, x[local[i]])
      return jacobian

    rows = np.arange(n_nodes)

    def angle_margins(x):
      """max_direction_angle^2 - r^2 for every node: non-negative when its rotation r is within the limit."""
      ab = x[n_lambda:].reshape(-1, 2)
      return self.max_direction_angle ** 2 - np.sum(ab ** 2, axis=1)

    def angle_margins_jacobian(x):
      ab = x[n_lambda:].reshape(-1, 2)
      jacobian = np.zeros((n_nodes, x.size))
      jacobian[rows, n_lambda + 2 * rows] = -2.0 * ab[:, 0]
      jacobian[rows, n_lambda + 2 * rows + 1] = -2.0 * ab[:, 1]
      return jacobian

    constraints = [
      {"type": "eq", "fun": length_residuals, "jac": length_jacobian},
      {"type": "ineq", "fun": angle_margins, "jac": angle_margins_jacobian},
    ]

    # The objective is not affine: s_k depends on the tangent lengths and on the directions.
    # The directions start from their initial estimates, i.e. from a zero rotation.
    result = minimize(
      objective,
      x0=np.r_[warm_start, np.zeros(2 * n_nodes)],
      jac=True,
      method="SLSQP",
      bounds=list(zip(lower, upper)),
      constraints=constraints,
      options={"ftol": 1e-12, "maxiter": 1000},
    )
    x = result.x

    violations = np.abs(length_residuals(x))
    failed = np.flatnonzero(violations > self._LENGTH_TOL)
    if failed.size:
      details = "; ".join(
        f"segment {i}: Ds = {segments[i].length:.6g}, relative violation {violations[i]:.2e}" for i in failed)
      raise RuntimeError(
        f"Could not reach the required arc lengths within [lambda_min, lambda_max] "
        f"({details}; solver: {result.message}).")

    fits = []
    for i, segment in enumerate(segments):
      v = x[local[i]]
      lambda_s, lambda_e, first_direction, last_direction = unpack(i, v)
      fits.append((float(lambda_s), float(lambda_e), first_direction, last_direction,
                   float(segment_mse(i, v) * segment.length ** 2)))
    return fits

  def _plot_comparison(self, segments, directions, warm_start, fits):
    """3D plot of the data, of the chain given by the warm start and of the fitted chain."""
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    for i, (segment, fit) in enumerate(zip(segments, fits)):
      lambda_s0, lambda_e0 = warm_start[2 * i:2 * i + 2] * segment.length
      s0 = self._arclength_parameters(
        segment.sigma, lambda_s0, lambda_e0, segment.p0, segment.p3, directions[i], directions[i + 1])
      first_solution = self._bezier(
        s0, lambda_s0, lambda_e0, segment.p0, segment.p3, directions[i], directions[i + 1])

      lambda_s, lambda_e, first_direction, last_direction, _ = fit
      s = self._arclength_parameters(
        segment.sigma, lambda_s, lambda_e, segment.p0, segment.p3, first_direction, last_direction)
      second_solution = self._bezier(s, lambda_s, lambda_e, segment.p0, segment.p3, first_direction, last_direction)

      # Labels only on the first segment, so that the legend has one entry per curve type
      labels = ("data", "first solver (affine)", "second solver (arc-length)") if i == 0 else (None,) * 3
      axes.plot(*segment.points.T, "k.", label=labels[0])
      axes.plot(*first_solution.T, "b--", label=labels[1])
      axes.plot(*second_solution.T, "r-", label=labels[2])

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title("Bezier solver comparison")
    axes.legend()
    figure.tight_layout()
    plt.show()

  # ------------------------------------------------------------------ #
  # Public API
  # ------------------------------------------------------------------ #
  def set_problem(self,
                  minima_idx: list[int],
                  Ds: list[float],
                  lambda_min: float = 1e-3,
                  lambda_max: float | None = None,
                  output_sigma: list | None = None,
                  plot_comparison: bool = False,
                  skip_discrete_step: bool = False,
                  max_direction_angle: float = np.pi / 2,
                  direction_weight: float = 0.0):
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
        Maximum allowed tangent length. If None, it is 2 * Ds[i] for segment i. Default is None.
    output_sigma : list of np.ndarray or None, optional
        Optional timing law used to *generate* the output samples. One array per segment,
        with one value per curve sample of the segment (minima_idx[i + 1] - minima_idx[i] + 1),
        non-decreasing from 0 to 1: the fraction of the segment's arc length covered at each
        sample. The geometry is still fitted on the data; output sample k is then placed on the
        fitted Bezier at arc-length fraction output_sigma[i][k]. If None, each sample keeps the
        position it has in the fit. Default is None.
    plot_comparison : bool, optional
        If True, display a 3D comparison of the chain obtained with the affine solver (initial
        directions, Bezier parameter equal to the data abscissa) and of the fitted chain.
        Default is False.
    skip_discrete_step : bool, optional
        If True, the length-constrained affine step is skipped and the warm start of the
        arc-length step is the bounds-only least-squares solution. Default is False.
    max_direction_angle : float, optional
        Largest angle [rad] between a fitted direction and its finite-difference estimate.
        Must be smaller than pi. Default is pi / 2.
    direction_weight : float, optional
        Weight of the penalty on the squared rotation angle [rad^2] of each direction with respect
        to its finite-difference estimate, added to the sum of the segment MSEs normalized by the
        squared segment lengths. 0 leaves the directions unpenalized. Default is 0.
    """
    if lambda_min < 0:
      raise ValueError("lambda_min must be non-negative.")
    if lambda_max is not None and lambda_max < lambda_min:
      raise ValueError("lambda_max must be greater than or equal to lambda_min.")
    if not 0 < max_direction_angle < np.pi:
      raise ValueError("max_direction_angle must be in (0, pi).")
    if direction_weight < 0:
      raise ValueError("direction_weight must be non-negative.")

    self.minima_idx = list(minima_idx)
    self.Ds = list(Ds)
    self.lambda_min = lambda_min
    self.lambda_max = lambda_max
    self.output_sigma = output_sigma
    self.skip_discrete_step = skip_discrete_step
    self.plot_comparison = plot_comparison
    self.max_direction_angle = float(max_direction_angle)
    self.direction_weight = float(direction_weight)

  def approximate(self, curve: Curve) -> tuple[list[dict], Curve]:
    """
    Approximate the given curve points using Bezier optimization.

    Parameters
    ----------
    curve : Curve
        The curve to approximate. Its time stamps must be strictly increasing between
        minima_idx[0] and minima_idx[-1].

    Returns
    -------
    tuple[list[dict], Curve]
        A list with, for each segment, the fitted parameters ("lambda_s", "u_s", "lambda_e",
        "delta_p", "u_e", "error" = fit MSE, "error_output" = MSE of the generated samples,
        "length" = Bezier arc length), and the approximated Bezier spline,
        sampled at the original samples between minima_idx[0] and minima_idx[-1].
    """
    curve_points = np.asarray(curve.points, dtype=float)
    t = np.asarray(curve.t, dtype=float)
    n_points = len(curve_points)
    idx = self.minima_idx

    if n_points < 2:
      raise ValueError("At least two points are required to compute the first direction.")
    if len(t) != n_points:
      raise ValueError(f"The curve has {n_points} points but {len(t)} time stamps.")
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

    segments = [self._make_segment(curve_points, idx[i], idx[i + 1], float(self.Ds[i]))
                for i in range(len(idx) - 1)]
    sigmas_out = None
    if self.output_sigma is not None:
      sigmas_out = [self._check_output_sigma(values, segment.points.shape[0], i)
                    for i, (values, segment) in enumerate(zip(self.output_sigma, segments))]

    # Initial tangent directions at the two ends of the chain and at every interior junction;
    # each interior direction is shared by the two segments that meet there (G1 continuity)
    directions = self._finite_difference_directions(curve_points, t, idx)

    warm_start = np.concatenate([
      self._affine_warm_start(i, segment, directions[i], directions[i + 1])
      for i, segment in enumerate(segments)
    ])
    fits = self._fit_chain(segments, directions, warm_start)

    if self.plot_comparison:
      self._plot_comparison(segments, directions, warm_start, fits)

    spline_parameters = []
    full_approximation_points = []
    for i, (segment, fit) in enumerate(zip(segments, fits)):
      lambda_s, lambda_e, first_direction, last_direction, error = fit

      # Geometry comes from the fit; timing comes from the data or from the external law
      sigma = segment.sigma if sigmas_out is None else sigmas_out[i]
      s = self._arclength_parameters(sigma, lambda_s, lambda_e, segment.p0, segment.p3,
                                     first_direction, last_direction)
      seg = self._bezier(s, lambda_s, lambda_e, segment.p0, segment.p3, first_direction, last_direction)

      full_approximation_points.append(seg if i == 0 else seg[1:])  # Avoid duplicating the shared end point
      spline_parameters.append({
        "lambda_s": lambda_s,
        "u_s": first_direction.copy(),
        "lambda_e": lambda_e,
        "delta_p": segment.p3 - segment.p0,
        "u_e": last_direction.copy(),
        "error": error,
        "error_output": float(np.mean(np.sum((seg - segment.points) ** 2, axis=1))),
        "length": self._arc_length(lambda_s, lambda_e, segment.p0, segment.p3, first_direction, last_direction),
      })

    points = np.concatenate(full_approximation_points, axis=0)
    t_out = t[idx[0]:idx[-1] + 1]   # same samples as the output points
    return spline_parameters, Curve(x=points[:, 0], y=points[:, 1], z=points[:, 2], t=t_out)