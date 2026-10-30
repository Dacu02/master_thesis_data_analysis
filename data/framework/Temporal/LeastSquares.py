import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

from . import Extract, Stroke, StrokeSequence

SQRT_2PI = np.sqrt(2.0 * np.pi)


# ============================================================================
# Lognormal stroke model
# ============================================================================

def _check_sigma(sigma):
    """Ensure the lognormal scale parameter is strictly positive."""
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")


def _sum_lognormals(t, theta):
    """Sum the speed profiles of all strokes in theta (rows: [D, t0, mu, sigma])."""
    return sum(Stroke.lognormal_speed(t, *row) for row in theta)


def initial_guesses(t, target, window, t0_min,
                    sigmas=(0.15, 0.3, 0.5), tau_factors=(1.0, 1.5, 2.5)):
    """Generate [D, t0, mu, sigma] starting points from the target peak inside the window."""
    t, target = np.asarray(t, dtype=float), np.asarray(target, dtype=float)
    dt = float(np.median(np.diff(t)))
    t_start = t[window.start]

    peak = window.start + int(np.argmax(target[window]))
    t_peak, v_peak = t[peak], max(float(target[peak]), 1e-9)

    guesses = []
    for sigma in sigmas:
        for factor in tau_factors:
            t0 = max(t_peak - max(factor * (t_peak - t_start), 3.0 * dt), t0_min)
            mu = np.log(max(t_peak - t0, 3.0 * dt)) + sigma**2
            D = v_peak * sigma * SQRT_2PI * np.exp(mu - sigma**2 / 2.0)
            guesses.append([D, t0, mu, sigma])

    return np.asarray(guesses, dtype=float)


# ============================================================================
# Least Squares extractor
# ============================================================================

class LeastSquares(Extract):

    def __init__(
        self,
        *,
        boundaries=None,
        lock_D=True,
        min_distance=None,
        prominence=None,
        sigma_bounds=(0.05, 1.5),
        max_lead=1.0,
        refine=True,
        reg_weight=0.0,
        t0_shift_frac=0.5,
        mu_shift=0.5,
        D_max_factor=2.0,
    ):
        """
        Least-squares extraction of a sequence of lognormal strokes.

        Args
        ----
        boundaries : array-like, optional
            Indices of speed minima. If None, minima are detected automatically.
        lock_D : bool
            If True, D is fixed to the geometric arc length between consecutive boundaries.
        min_distance : int, optional
            Minimum distance between detected minima.
        prominence : float, optional
            Prominence used for speed-minimum detection.
        sigma_bounds : tuple(float, float)
            Bounds for sigma.
        max_lead : float
            Maximum amount by which t0 can precede the fitting window.
        refine : bool
            If True, perform joint optimization after sequential fitting.
        reg_weight : float
            Regularization weight for the joint optimization.
        t0_shift_frac : float
            Maximum relative shift applied to t0 during refinement.
        mu_shift : float
            Maximum variation of mu during refinement.
        D_max_factor : float
            Maximum multiplicative change of D when D is free.
        """
        if len(sigma_bounds) != 2:
            raise ValueError("sigma_bounds must contain (sigma_min, sigma_max).")
        sigma_min, sigma_max = sigma_bounds

        # Parameter validation: each entry is (violation condition, error message).
        checks = [
            (sigma_min <= 0, "sigma_min must be greater than zero."),
            (sigma_max <= sigma_min, "sigma_max must be greater than sigma_min."),
            (max_lead < 0, "max_lead must be non-negative."),
            (t0_shift_frac < 0, "t0_shift_frac must be non-negative."),
            (mu_shift < 0, "mu_shift must be non-negative."),
            (D_max_factor < 1, "D_max_factor must be >= 1."),
        ]
        for violated, message in checks:
            if violated:
                raise ValueError(message)

        self.boundaries = None if boundaries is None else np.asarray(boundaries, dtype=int)
        self.lock_D, self.refine = bool(lock_D), bool(refine)
        self.min_distance, self.prominence = min_distance, prominence
        self.sigma_bounds = (float(sigma_min), float(sigma_max))
        self.max_lead, self.reg_weight = float(max_lead), float(reg_weight)
        self.t0_shift_frac, self.mu_shift = float(t0_shift_frac), float(mu_shift)
        self.D_max_factor = float(D_max_factor)
        self._stroke_sequence = None

    def set_boundaries(self, boundaries):
        """Set the speed-minima indices used for stroke extraction."""
        self.boundaries = np.asarray(boundaries, dtype=int)

    # ========================================================================
    # Boundary detection
    # ========================================================================

    def _extract_boundaries(self, speed):
        """Detect local minima of the speed profile; first and last samples are always included."""
        speed = np.asarray(speed, dtype=float)
        if speed.ndim != 1:
            raise ValueError("speed must be a 1D array.")
        if len(speed) < 2:
            raise ValueError("speed must contain at least two samples.")

        minima, _ = find_peaks(-speed, distance=self.min_distance, prominence=self.prominence)
        return np.unique(np.concatenate([[0], minima, [len(speed) - 1]]).astype(int))

    # ========================================================================
    # Geometric D
    # ========================================================================

    def _compute_geometric_D(self, curve, boundaries):
        """Measured 3D arc length of the trajectory between consecutive boundaries."""
        if not hasattr(curve, "points"):
            raise AttributeError("Curve must expose a 'points' property.")

        position = np.asarray(curve.points, dtype=float)
        D = np.zeros(len(boundaries) - 1)

        for k, (i0, i1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            segment = position[i0:i1 + 1]
            if len(segment) < 2:
                raise ValueError(f"Stroke {k} contains fewer than two position samples.")
            D[k] = np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()

        return D

    # ========================================================================
    # Single-stroke fitting
    # ========================================================================

    def _fit_stroke(self, t, target, window, D_fixed=None):
        """
        Fit one lognormal stroke inside `window`, starting from several initial guesses.

        Returns
        -------
        theta : ndarray
            [D, t0, mu, sigma]
        info : dict
            Fit diagnostics.
        """
        t, target = np.asarray(t, dtype=float), np.asarray(target, dtype=float)

        if t.ndim != 1:
            raise ValueError("t must be a 1D array.")
        if target.ndim != 1:
            raise ValueError("target must be a 1D array.")
        if len(t) != len(target):
            raise ValueError("t and target must have equal lengths.")
        if len(t) < 2:
            raise ValueError("At least two samples are required.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing.")

        dt = float(np.median(np.diff(t)))
        t_start, t_end = float(t[window.start]), float(t[window.stop - 1])
        if t_end <= t_start:
            raise ValueError("Invalid fitting window.")

        # Area of the positive part of the target bounds the admissible D.
        area = max(
            float(np.trapezoid(np.clip(target[window], 0.0, None), t[window])),  # type: ignore
            1e-9,
        )
        t0_min = t_start - self.max_lead * (t_end - t_start)

        # Full parameter bounds, ordered as [D, t0, mu, sigma].
        lower = np.array([1e-9, t0_min, -6.0, self.sigma_bounds[0]])
        upper = np.array([max(50.0 * area, 1e-8), t_end - 3.0 * dt, 3.0, self.sigma_bounds[1]])

        if D_fixed is not None:
            D_fixed = float(D_fixed)
            if D_fixed <= 0:
                raise ValueError("D_fixed must be greater than zero.")

        # With a fixed D the optimizer only sees [t0, mu, sigma].
        free = np.array([D_fixed is None, True, True, True])

        def to_theta(x):
            """Rebuild the full [D, t0, mu, sigma] vector from the free variables."""
            return np.asarray(x, dtype=float) if D_fixed is None else np.array([D_fixed, *x])

        def residuals(x):
            return Stroke.lognormal_speed(t[window], *to_theta(x)) - target[window]

        lo, hi = lower[free], upper[free]
        best = None

        # Multi-start: keep the solution with the lowest cost.
        for x0_full in initial_guesses(t, target, window, t0_min):
            x0 = np.clip(x0_full[free], lo + 1e-12, hi - 1e-12)
            try:
                solution = least_squares(
                    residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=400  # type: ignore
                )
            except ValueError:
                continue
            if best is None or solution.cost < best.cost:
                best = solution

        if best is None:
            raise RuntimeError("Could not fit lognormal stroke.")

        return to_theta(best.x), {
            "rmse": float(np.sqrt(np.mean(residuals(best.x) ** 2))),
            "cost": float(best.cost),
            "success": bool(best.success),
            "optimality": float(best.optimality),
            "nfev": int(best.nfev),
        }

    # ========================================================================
    # Sequential decomposition
    # ========================================================================

    def _fit_sequential(self, t, speed, boundaries, D):
        """Extract strokes one at a time, subtracting each fit from the running residual."""
        t, speed = np.asarray(t, dtype=float), np.asarray(speed, dtype=float)
        residual, steps = speed.copy(), []

        for k, (i0, i1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            window = slice(int(i0), int(i1) + 1)
            theta, info = self._fit_stroke(
                t, residual, window, D_fixed=float(D[k]) if self.lock_D else None
            )
            fitted = Stroke.lognormal_speed(t, *theta)
            residual_after = residual - fitted

            steps.append({
                "k": k,
                "window": window,
                "theta": theta,
                "fitted": fitted,
                "residual_before": residual,
                "residual_after": residual_after,
                **info,
            })
            residual = residual_after

        return steps, residual

    # ========================================================================
    # Joint refinement
    # ========================================================================

    def _refine_jointly(self, t, speed, steps):
        """
        Jointly refine all strokes around their sequential solution.

        If lock_D=True, D stays fixed; otherwise it can vary within D_max_factor.
        """
        t, speed = np.asarray(t, dtype=float), np.asarray(speed, dtype=float)
        if len(steps) == 0:
            raise ValueError("No strokes available for refinement.")

        dt = float(np.median(np.diff(t)))
        n_strokes = len(steps)
        theta0 = np.asarray([step["theta"] for step in steps])
        if theta0.shape != (n_strokes, 4):
            raise ValueError("Invalid stroke parameter shape.")

        # Per-stroke bounds and regularization scales, columns: [D, t0, mu, sigma].
        lower, upper, scale = (np.empty_like(theta0) for _ in range(3))

        for k, step in enumerate(steps):
            D0, t00, mu0, _ = theta0[k]
            t_start, t_end = float(t[step["window"].start]), float(t[step["window"].stop - 1])
            duration = max(t_end - t_start, dt)
            shift = self.t0_shift_frac * duration

            lower[k] = [
                max(D0 / self.D_max_factor, 1e-12),
                t00 - shift,
                mu0 - self.mu_shift,
                self.sigma_bounds[0],
            ]
            upper[k] = [
                max(D0 * self.D_max_factor, 1e-12),
                min(t00 + shift, t_end - 3.0 * dt),
                mu0 + self.mu_shift,
                self.sigma_bounds[1],
            ]
            scale[k] = [max(D0, 1e-9), max(duration, 1e-9), 1.0, 1.0]

            # Guard against a degenerate t0 interval.
            if upper[k, 1] <= lower[k, 1]:
                upper[k, 1] = lower[k, 1] + 1e-9

        # Mask of optimized parameters (D is excluded when locked).
        free = np.ones((n_strokes, 4), dtype=bool)
        free[:, 0] = not self.lock_D

        lo, hi = lower[free], upper[free]
        x0 = np.clip(theta0[free], lo + 1e-12, hi - 1e-12)
        v_scale = max(float(np.max(np.abs(speed))), 1e-9)

        def unpack(x_free):
            """Insert the free variables into the full parameter matrix."""
            theta = theta0.copy()
            theta[free] = x_free
            return theta

        def residuals(x_free):
            """Reconstruction error, optionally augmented with a penalty on parameter drift."""
            theta = unpack(x_free)
            error = _sum_lognormals(t, theta) - speed
            if self.reg_weight > 0:
                penalty = self.reg_weight * v_scale * (theta - theta0) / scale
                error = np.concatenate([error, penalty.ravel()])
            return error

        solution = least_squares(
            residuals, x0, bounds=(lo, hi), x_scale="jac", max_nfev=500  # type: ignore
        )

        theta = unpack(solution.x)
        refined = [
            {"k": k, "window": step["window"], "theta": theta[k],
             "fitted": Stroke.lognormal_speed(t, *theta[k])}
            for k, step in enumerate(steps)
        ]
        residual = speed - sum(item["fitted"] for item in refined)

        info = {
            "success": bool(solution.success),
            "nfev": int(solution.nfev),
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
        }
        return refined, residual, info

    # ========================================================================
    # Convert parameters to StrokeSequence
    # ========================================================================

    def _to_stroke_sequence(self, refined):
        """Convert optimized parameter arrays into a StrokeSequence."""
        return StrokeSequence([
            Stroke(D=float(D), mu=float(mu), sigma=float(sigma), to=float(t0))
            for D, t0, mu, sigma in (item["theta"] for item in refined)
        ])

    # ========================================================================
    # Public API
    # ========================================================================

    def approximate(self, recording):
        """Extract a StrokeSequence from a Recording."""
        if recording.trajectory.t is None:
            raise ValueError("LeastSquares requires trajectory.t.")

        t = np.asarray(recording.trajectory.t, dtype=float)
        speed = np.asarray(recording.trajectory.speed, dtype=float)

        if len(t) != len(speed):
            raise ValueError("Trajectory time and speed arrays must have equal length.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Trajectory time must be strictly increasing.")

        # 1. Speed minima: detected automatically or supplied by the user.
        if self.boundaries is None:
            boundaries = self._extract_boundaries(speed)
        else:
            boundaries = np.asarray(self.boundaries, dtype=int)
            if len(boundaries) < 2:
                raise ValueError("At least two boundaries are required.")
            if np.any(np.diff(boundaries) <= 0):
                raise ValueError("boundaries must be strictly increasing.")
            if boundaries[0] < 0:
                raise ValueError("boundaries contain a negative index.")
            if boundaries[-1] >= len(speed):
                raise ValueError("boundaries contain an out-of-range index.")

        # 2. Geometric stroke amplitudes from the measured path length.
        D_geom = self._compute_geometric_D(recording.curve, boundaries)

        # 3. Sequential decomposition, optionally followed by joint refinement.
        steps, _ = self._fit_sequential(t, speed, boundaries, D_geom)
        refined = (
            self._refine_jointly(t, speed, steps)[0]
            if self.refine
            else [{key: step[key] for key in ("k", "window", "theta", "fitted")} for step in steps]
        )

        # 4. Build and store the resulting StrokeSequence.
        self._stroke_sequence = self._to_stroke_sequence(refined)
        return self._stroke_sequence