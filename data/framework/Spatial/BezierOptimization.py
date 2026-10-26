import numpy as np
from scipy.optimize import minimize, lsq_linear
from scipy.special import erf

from . import Curve, Approximate


class BezierOptimization(Approximate):
  """
  Fits a chain of cubic Bezier segments to a curve. Each segment goes from a
  local minimum to the next one, has prescribed end tangent directions, and its
  arc length is constrained to be exactly Ds[i]. The free parameters of a segment
  are the two tangent lengths (lambda_s, lambda_e), chosen to minimize the mean
  squared error with respect to the original points. Optionally, the start direction
  of the first segment and the end direction of the last one (the two ends of the
  chain, which are not shared with another segment) are free parameters as well.
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
  def _point_at_distance(points: np.ndarray, anchor: int, indices: range,
                         radius: float | None) -> np.ndarray | None:
    """
    Point where the polyline that starts at points[anchor] and goes through
    points[indices[0]], points[indices[1]], ... first reaches the Euclidean distance
    `radius` from points[anchor]. The crossing between the last sample inside the sphere
    and the first one outside it is found by linear interpolation. If radius is None or
    not positive, the first sample distinct from points[anchor] is returned.
    Returns None if the polyline never reaches the required distance.
    """
    if radius is not None and radius <= 0:
      radius = None

    center = points[anchor]
    previous = center
    for j in indices:
      current = points[j]
      distance = np.linalg.norm(current - center)
      if distance > 0 and (radius is None or distance >= radius):
        if radius is None:
          return current
        # Solve |previous + t * (current - previous) - center| = radius for t in (0, 1]
        a = current - previous
        b = previous - center
        aa, ab, c = a @ a, a @ b, b @ b - radius ** 2
        t = (-ab + np.sqrt(ab ** 2 - aa * c)) / aa
        return previous + t * a
      previous = current
    return None

  @classmethod
  def _boundary_direction(cls, points: np.ndarray, i0: int, i1: int,
                          min_dist: float | None, from_start: bool) -> np.ndarray:
    """
    Unit tangent direction at one end of the segment points[i0:i1 + 1], estimated with
    the chord between that end point and the point where the segment crosses the sphere
    of radius min_dist centered at it. The returned vector always points along the
    direction of travel.
    """
    if from_start:
      anchor, indices = i0, range(i0 + 1, i1 + 1)
    else:
      anchor, indices = i1, range(i1 - 1, i0 - 1, -1)

    crossing = cls._point_at_distance(points, anchor, indices, min_dist)
    if crossing is None:
      raise ValueError(
        f"Cannot estimate the tangent direction at end point {anchor}: the segment never "
        f"gets farther than direction_distance from it.")

    chord = (crossing - points[anchor]) if from_start else (points[anchor] - crossing)
    return chord / np.linalg.norm(chord)

  @classmethod
  def _junction_direction(cls, points: np.ndarray, k: int, lo: int, hi: int,
                          min_dist: float | None) -> np.ndarray:
    """
    Unit tangent direction at the interior junction points[k], estimated with the chord
    between the two points where the curve crosses the sphere of radius min_dist centered
    at points[k]: one before the junction (samples in [lo, k)) and one after it (samples
    in (k, hi]). Both chord ends are exactly min_dist away from the junction, so the
    neighbourhood is symmetric in space. The returned vector points along the direction of travel.
    """
    behind = cls._point_at_distance(points, k, range(k - 1, lo - 1, -1), min_dist)
    ahead = cls._point_at_distance(points, k, range(k + 1, hi + 1), min_dist)
    if behind is None or ahead is None:
      raise ValueError(
        f"Cannot estimate the tangent direction at junction {k}: a neighbouring segment "
        f"never gets farther than direction_distance from the junction.")

    chord = ahead - behind
    norm = np.linalg.norm(chord)
    if norm == 0:
      raise ValueError(f"Degenerate chord at junction {k}: the two crossing points coincide.")
    return chord / norm

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

  def _fit_segment(self, segment_points, sigma, p0, p3, first_direction, last_direction,
                   length, lambda_min, lambda_max, free_start=False, free_end=False):
    """
    Minimize the MSE over (lambda_s, lambda_e) subject to
        lambda_min <= lambda <= lambda_max   and   arc_length(lambda_s, lambda_e) = length.

    `sigma` is the normalized curvilinear abscissa of each sample of the data; sample k is compared
    with the Bezier point at the same normalized curvilinear abscissa, B(Lambda_B^{-1}(sigma_k)).

    If `free_start` (`free_end`) is True, the start (end) direction is also optimized. It is
    parametrized by a rotation (a, b) [rad] of the given direction, which is the initial guess.
    The rotation angle r = sqrt(a^2 + b^2), i.e. the angle between the optimized and the given
    direction, is limited to r <= self.max_direction_angle (the same limit in every direction) and
    its square is penalized with weight self.direction_weight.

    Returns (lambda_s, lambda_e, mse, s, first_direction, last_direction), with s the Bezier
    parameter of each sample and the two directions actually used.
    """
    chord = float(np.linalg.norm(p3 - p0))
    if length < chord * (1 - 1e-9):
      raise ValueError(f"Ds = {length:.6g} is shorter than the distance between the segment end points ({chord:.6g}): no curve can satisfy it.")

    # Work in units of `length` so that variables, objective and constraint are O(1).
    scale = length
    lo, hi = lambda_min / scale, lambda_max / scale

    # Optimization variables: [lambda_s, lambda_e] / scale, then (a, b) of each free direction
    start_basis = self._tangent_basis(first_direction) if free_start else None
    end_basis = self._tangent_basis(last_direction) if free_end else None
    n_angles = 2 * (int(free_start) + int(free_end))

    def unpack(u):
      """Optimization vector -> (lambda_s, lambda_e, first direction, last direction, rotation angles)."""
      ls, le = u[0] * scale, u[1] * scale
      fd, ld, angles, j = first_direction, last_direction, [], 2
      if free_start:
        fd = self._rotate_direction(first_direction, *start_basis, u[j], u[j + 1])
        angles.append(np.hypot(u[j], u[j + 1]))
        j += 2
      if free_end:
        ld = self._rotate_direction(last_direction, *end_basis, u[j], u[j + 1])
        angles.append(np.hypot(u[j], u[j + 1]))
      return ls, le, fd, ld, angles

    base, col_s, col_e = self._basis(sigma, p0, p3, first_direction, last_direction)
    A = np.column_stack([col_s.ravel(), col_e.ravel()]) * scale
    b = (segment_points - base).ravel()

    def mse_affine(u):
      return np.mean(np.sum((A @ u - b).reshape(-1, 3) ** 2, axis=1)) / scale ** 2

    def mse_arclength(u):
      ls, le, fd, ld, _ = unpack(u)
      s = self._arclength_parameters(sigma, ls, le, p0, p3, fd, ld)
      model = self._bezier(s, ls, le, p0, p3, fd, ld)
      return np.mean(np.sum((model - segment_points) ** 2, axis=1)) / scale ** 2

    def objective(u):
      _, _, _, _, angles = unpack(u)
      return mse_arclength(u) + self.direction_weight * sum(angle ** 2 for angle in angles)

    def length_residual(u):
      ls, le, fd, ld, _ = unpack(u)
      return self._arc_length(ls, le, p0, p3, fd, ld) / length - 1.0

    def length_residual_initial(u):
      return length_residual(np.r_[u, np.zeros(n_angles)])

    def angle_margins(u):
      """max_direction_angle^2 - r^2 for every free direction: non-negative when its rotation r is within the limit."""
      _, _, _, _, angles = unpack(u)
      return np.array([self.max_direction_angle ** 2 - angle ** 2 for angle in angles])

    def solve(fun, x0, bounds, residual, limit_angles=False):
      constraints = [{"type": "eq", "fun": residual}]
      if limit_angles:
        constraints.append({"type": "ineq", "fun": angle_margins})
      result = minimize(
        fun,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
      )
      violation = abs(residual(result.x))
      if violation > 1e-4:
        raise RuntimeError(
          f"Could not reach the required arc length Ds = {length:.6g} within "
          f"[lambda_min, lambda_max] = [{lambda_min:.6g}, {lambda_max:.6g}] "
          f"(relative violation {violation:.2e}; solver: {result.message})."
        )
      return result.x

    lambda_bounds = [(lo, hi), (lo, hi)]
    angle_bounds = [(-self.max_direction_angle, self.max_direction_angle)] * n_angles

    # Stage 1: affine objective with the initial directions. The bounds-only problem is a linear
    # least squares, solved exactly and used as a warm start for the length-constrained one.
    x_lsq = lsq_linear(A, b, bounds=(lo, hi)).x
    if not self.skip_discrete_step:
      x_d = solve(mse_affine, x_lsq, lambda_bounds, length_residual_initial)
    else:
      x_d = x_lsq
    # Stage 2: the objective is no longer affine because s_k depends on the lambdas (and on the free
    # directions). The free directions start from the initial ones, i.e. from a zero rotation.
    x = solve(objective, np.r_[x_d, np.zeros(n_angles)], lambda_bounds + angle_bounds, length_residual,
              limit_angles=n_angles > 0)

    lambda_s, lambda_e, first_direction_fit, last_direction_fit, _ = unpack(x)
    s = self._arclength_parameters(sigma, lambda_s, lambda_e, p0, p3, first_direction_fit, last_direction_fit)

    if not self.skip_discrete_step and self.plot_comparison:
      import matplotlib.pyplot as plt

      lambda_s_first, lambda_e_first = x_d * scale
      s_first = self._arclength_parameters(
        sigma, lambda_s_first, lambda_e_first, p0, p3, first_direction, last_direction)
      first_solution = self._bezier(
        s_first, lambda_s_first, lambda_e_first, p0, p3, first_direction, last_direction)
      second_solution = self._bezier(
        s, lambda_s, lambda_e, p0, p3, first_direction_fit, last_direction_fit)

      figure = plt.figure()
      axes = figure.add_subplot(111, projection="3d")
      axes.plot(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2], "k.", label="data")
      axes.plot(first_solution[:, 0], first_solution[:, 1], first_solution[:, 2], "b--", label="first solver (affine)")
      axes.plot(second_solution[:, 0], second_solution[:, 1], second_solution[:, 2], "r-", label="second solver (arc-length)")
      axes.set_xlabel("x")
      axes.set_ylabel("y")
      axes.set_zlabel("z")
      axes.set_title("Bezier solver comparison")
      axes.legend()
      figure.tight_layout()
      plt.show()

    return (float(lambda_s), float(lambda_e), float(mse_arclength(x) * scale ** 2), s,
            first_direction_fit, last_direction_fit)

  # ------------------------------------------------------------------ #
  # Public API
  # ------------------------------------------------------------------ #
  def set_problem(self,
                  minima_idx: list[int],
                  Ds: list[float],
                  lambda_min: float = 1e-3,
                  lambda_max: float | None = None,
                  direction_distance: float | None = None,
                  output_sigma: list | None = None,
                  plot_comparison: bool = False,
                  skip_discrete_step: bool = False,
                  optimize_end_directions: bool = False,
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
    direction_distance : float | None, optional
        Radius (same units as the curve) of the neighbourhood used to estimate the tangent
        directions: at the two ends of the chain the chord reaches the point where the curve crosses
        the sphere of this radius; at interior junctions the chord joins the two crossing points
        before and after the junction. If None, adjacent samples are used.
    output_sigma : list of np.ndarray or None, optional
        Optional timing law used to *generate* the output samples. One array per segment,
        with one value per curve sample of the segment (minima_idx[i + 1] - minima_idx[i] + 1),
        non-decreasing from 0 to 1: the fraction of the segment's arc length covered at each
        sample. The geometry is still fitted on the data; output sample k is then placed on the
        fitted Bezier at arc-length fraction output_sigma[i][k]. If None, each sample keeps the
        position it has in the fit. Default is None.
    plot_comparison : bool, optional
        If True, display a 3D comparison of the affine and arc-length solver
        solutions for each segment. Default is False.
    skip_discrete_step : bool, optional
        If True, the length-constrained affine step is skipped and the warm start of the
        arc-length step is the bounds-only least-squares solution. Default is False.
    optimize_end_directions : bool, optional
        If True, the start direction of the first segment and the end direction of the last
        segment (the two ends of the chain) are optimized together with the tangent lengths,
        starting from the directions estimated with `direction_distance`. Interior junction
        directions stay fixed and shared by the adjacent segments. Default is False.
    max_direction_angle : float, optional
        Largest angle [rad] between a free direction and its initial estimate (the direction
        obtained with `direction_distance`). Must be smaller than pi. Default is pi / 2.
    direction_weight : float, optional
        Weight of the penalty on the squared rotation angle [rad^2] of the free directions, added to
        the MSE normalized by the squared segment length. 0 leaves the directions unpenalized.
        Default is 0.
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
    self.direction_distance = direction_distance
    self.output_sigma = output_sigma
    self.skip_discrete_step = skip_discrete_step
    self.plot_comparison = plot_comparison
    self.optimize_end_directions = optimize_end_directions
    self.max_direction_angle = float(max_direction_angle)
    self.direction_weight = float(direction_weight)

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
        A list with, for each segment, the fitted parameters ("lambda_s", "u_s", "lambda_e",
        "delta_p", "u_e", "error" = fit MSE, "error_output" = MSE of the generated samples,
        "length" = Bezier arc length), and the approximated Bezier spline,
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

    # Tangent directions at the two ends of the chain and at every interior junction;
    # each interior direction is shared by the two segments that meet there (G1 continuity)
    directions = [None] * len(idx)
    directions[0] = self._boundary_direction(curve_points, idx[0], idx[1], self.direction_distance, True)
    directions[-1] = self._boundary_direction(curve_points, idx[-2], idx[-1], self.direction_distance, False)
    for k in range(1, len(idx) - 1):
      directions[k] = self._junction_direction(curve_points, idx[k], idx[k - 1], idx[k + 1], self.direction_distance)

    spline_parameters = []
    full_approximation_points = []
    last = len(idx) - 2

    for i in range(len(idx) - 1):
      i0, i1 = idx[i], idx[i + 1]
      D = float(self.Ds[i])
      segment_points = curve_points[i0:i1 + 1]   # (n, 3)
      n = segment_points.shape[0]
      p0, p3 = segment_points[0], segment_points[-1]

      steps = np.linalg.norm(np.diff(segment_points, axis=0), axis=1)
      total = steps.sum()
      if total == 0:
        raise ValueError("Degenerate segment: all points coincide.")
      sigma = np.concatenate([[0.0], np.cumsum(steps)]) / total

      lambda_max = self.lambda_max if self.lambda_max is not None else 2 * D

      # Only the two ends of the chain may have a free direction
      lambda_s, lambda_e, error, s, first_direction, last_direction = self._fit_segment(
        segment_points, sigma, p0, p3, directions[i], directions[i + 1], D,
        self.lambda_min, lambda_max,
        free_start=self.optimize_end_directions and i == 0,
        free_end=self.optimize_end_directions and i == last)

      if self.output_sigma is not None:
        # Geometry comes from the fit; timing comes from the external law.
        sigma_out = self._check_output_sigma(self.output_sigma[i], n, i)
        s = self._arclength_parameters(sigma_out, lambda_s, lambda_e, p0, p3, first_direction, last_direction)

      seg = self._bezier(s, lambda_s, lambda_e, p0, p3, first_direction, last_direction)
      full_approximation_points.append(seg if i == 0 else seg[1:])  # Avoid duplicating the shared end point
      spline_parameters.append({
        "lambda_s": lambda_s,
        "u_s": first_direction.copy(),
        "lambda_e": lambda_e,
        "delta_p": p3 - p0,
        "u_e": last_direction.copy(),
        "error": error,
        "error_output": float(np.mean(np.sum((seg - segment_points) ** 2, axis=1))),
        "length": self._arc_length(lambda_s, lambda_e, p0, p3, first_direction, last_direction),
      })

    points = np.concatenate(full_approximation_points, axis=0)
    t = np.asarray(curve.t, dtype=float)[idx[0]:idx[-1] + 1]   # same samples as the output points
    return spline_parameters, Curve(x=points[:, 0], y=points[:, 1], z=points[:, 2], t=t)