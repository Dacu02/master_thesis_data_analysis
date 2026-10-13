import numpy as np

from scipy.optimize import least_squares
from scipy.signal import find_peaks

from . import Extract, Stroke, StrokeSequence

SQRT_2PI = np.sqrt(2.0 * np.pi)


# ============================================================================
# Lognormal stroke model
# ============================================================================

def lognormal_speed(
    t,
    D,
    t0,
    mu,
    sigma,
):
    """
    Speed profile of a single lognormal stroke.

    Parameters
    ----------
    t : array-like
        Global time.
    D : float
        Integrated distance / stroke amplitude.
    t0 : float
        Stroke onset.
    mu : float
        Lognormal location parameter.
    sigma : float
        Lognormal scale parameter.
    """
    t = np.asarray(t, dtype=float)

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    tau = t - t0

    out = np.zeros_like(t)

    positive = tau > 0.0

    if np.any(positive):
        tau_pos = tau[positive]

        out[positive] = (
            D
            / (
                sigma
                * SQRT_2PI
                * tau_pos
            )
            * np.exp(
                -(
                    (np.log(tau_pos) - mu) ** 2
                    / (2.0 * sigma**2)
                )
            )
        )

    return out


def lognormal_peak(
    D,
    t0,
    mu,
    sigma,
):
    """
    Return the time and value of the maximum speed.
    """
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    tau_peak = np.exp(
        mu - sigma**2
    )

    v_peak = (
        D
        / (sigma * SQRT_2PI)
        * np.exp(
            -mu
            + sigma**2 / 2.0
        )
    )

    return (
        t0 + tau_peak,
        v_peak,
    )


def initial_guesses(
    t,
    target,
    window,
    t0_min,
    sigmas=(0.15, 0.3, 0.5),
    tau_factors=(1.0, 1.5, 2.5),
):
    """
    Generate initial guesses for:

        [D, t0, mu, sigma]

    using the peak of the target inside the fitting window.
    """
    t = np.asarray(t, dtype=float)
    target = np.asarray(target, dtype=float)

    dt = float(
        np.median(np.diff(t))
    )

    t_start = t[window.start]

    target_window = target[window]

    peak_local = int(
        np.argmax(target_window)
    )

    peak_index = (
        window.start
        + peak_local
    )

    t_peak = t[peak_index]

    v_peak = max(
        float(target[peak_index]),
        1e-9,
    )

    guesses = []

    for sigma in sigmas:

        for factor in tau_factors:

            t0 = max(
                t_peak
                - max(
                    factor
                    * (t_peak - t_start),
                    3.0 * dt,
                ),
                t0_min,
            )

            tau_peak = max(
                t_peak - t0,
                3.0 * dt,
            )

            mu = (
                np.log(tau_peak)
                + sigma**2
            )

            D = (
                v_peak
                * sigma
                * SQRT_2PI
                * np.exp(
                    mu
                    - sigma**2 / 2.0
                )
            )

            guesses.append([
                D,
                t0,
                mu,
                sigma,
            ])

    return np.asarray(
        guesses,
        dtype=float,
    )


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
        ----------
        boundaries : array-like, optional
            Indices of speed minima. If None, minima are detected
            automatically.

        lock_D : bool
            If True, D is fixed to the geometric arc length between
            consecutive boundaries.

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
            raise ValueError(
                "sigma_bounds must contain "
                "(sigma_min, sigma_max)."
            )

        sigma_min, sigma_max = sigma_bounds

        if sigma_min <= 0:
            raise ValueError(
                "sigma_min must be greater than zero."
            )

        if sigma_max <= sigma_min:
            raise ValueError(
                "sigma_max must be greater than sigma_min."
            )

        if max_lead < 0:
            raise ValueError(
                "max_lead must be non-negative."
            )

        if t0_shift_frac < 0:
            raise ValueError(
                "t0_shift_frac must be non-negative."
            )

        if mu_shift < 0:
            raise ValueError(
                "mu_shift must be non-negative."
            )

        if D_max_factor < 1:
            raise ValueError(
                "D_max_factor must be >= 1."
            )

        self.boundaries = (
            None
            if boundaries is None
            else np.asarray(
                boundaries,
                dtype=int,
            )
        )

        self.lock_D = bool(lock_D)

        self.min_distance = min_distance
        self.prominence = prominence

        self.sigma_bounds = (
            float(sigma_min),
            float(sigma_max),
        )

        self.max_lead = float(
            max_lead
        )

        self.refine = bool(refine)

        self.reg_weight = float(
            reg_weight
        )

        self.t0_shift_frac = float(
            t0_shift_frac
        )

        self.mu_shift = float(
            mu_shift
        )

        self.D_max_factor = float(
            D_max_factor
        )

        self._stroke_sequence = None

    def set_boundaries(
        self,
        boundaries,
    ):
        """
        Set the speed minima used for stroke extraction.

        Parameters
        ----------
        boundaries : array-like
            Indices of speed minima.
        """
        self.boundaries = np.asarray(
            boundaries,
            dtype=int,
        )

    # ========================================================================
    # Boundary detection
    # ========================================================================

    def _extract_boundaries(
        self,
        speed,
    ):
        """
        Detect local minima of the speed profile.

        The first and last samples are always included.
        """
        speed = np.asarray(
            speed,
            dtype=float,
        )

        if speed.ndim != 1:
            raise ValueError(
                "speed must be a 1D array."
            )

        if len(speed) < 2:
            raise ValueError(
                "speed must contain at least two samples."
            )

        minima, _ = find_peaks(
            -speed,
            distance=self.min_distance,
            prominence=self.prominence,
        )

        boundaries = np.concatenate([
            np.array([0], dtype=int),
            minima.astype(int),
            np.array([len(speed) - 1], dtype=int),
        ])

        return np.unique(
            boundaries
        )

    # ========================================================================
    # Geometric D
    # ========================================================================
    
    def _compute_geometric_D(self, curve, boundaries: np.ndarray):
        """
        Compute the geometric path length between consecutive boundaries.

        D[k] is the measured arc length of the original 3D trajectory
        between boundaries[k] and boundaries[k + 1].
        """
        if not hasattr(
            curve,
            "points",
        ):
            raise AttributeError(
                "Curve must expose a 'points' property."
            )

        position = np.asarray(
            curve.points,
            dtype=float,
        )

        D = np.zeros(
            len(boundaries) - 1,
            dtype=float,
        )

        for k, (i0, i1) in enumerate(
            zip(
                boundaries[:-1],
                boundaries[1:],
            )
        ):
            segment = position[
                i0:i1 + 1
            ]

            if len(segment) < 2:
                raise ValueError(
                    f"Stroke {k} contains fewer than "
                    "two position samples."
                )

            D[k] = np.sum(
                np.linalg.norm(
                    np.diff(
                        segment,
                        axis=0,
                    ),
                    axis=1,
                )
            )

        return D

    # ========================================================================
    # Single-stroke fitting
    # ========================================================================

    def _fit_stroke(
        self,
        t,
        target,
        window,
        D_fixed=None,
    ):
        """
        Fit one lognormal stroke.

        Returns
        -------
        theta : ndarray
            [D, t0, mu, sigma]

        info : dict
            Fit diagnostics.
        """
        t = np.asarray(
            t,
            dtype=float,
        )

        target = np.asarray(
            target,
            dtype=float,
        )

        if t.ndim != 1:
            raise ValueError(
                "t must be a 1D array."
            )

        if target.ndim != 1:
            raise ValueError(
                "target must be a 1D array."
            )

        if len(t) != len(target):
            raise ValueError(
                "t and target must have equal lengths."
            )

        if len(t) < 2:
            raise ValueError(
                "At least two samples are required."
            )

        if np.any(
            np.diff(t) <= 0
        ):
            raise ValueError(
                "t must be strictly increasing."
            )

        dt = float(
            np.median(
                np.diff(t)
            )
        )

        t_start = float(
            t[window.start]
        )

        t_end = float(
            t[window.stop - 1]
        )

        if t_end <= t_start:
            raise ValueError(
                "Invalid fitting window."
            )

        positive_area = np.clip(
            target[window],
            0.0,
            None,
        )

        area = max(
            float(
                np.trapezoid( # type: ignore
                    positive_area,
                    t[window],
                )
            ),
            1e-9,
        )

        t0_min = (
            t_start
            - self.max_lead
            * (t_end - t_start)
        )

        D_upper = 50.0 * area

        # Full parameter bounds:
        #
        # D, t0, mu, sigma
        lower = np.array([
            1e-9,
            t0_min,
            -6.0,
            self.sigma_bounds[0],
        ])

        upper = np.array([
            max(D_upper, 1e-8),
            t_end - 3.0 * dt,
            3.0,
            self.sigma_bounds[1],
        ])

        # In the fixed-D case D is not an optimization variable.
        if D_fixed is None:

            free = np.array([
                True,
                True,
                True,
                True,
            ])

        else:

            D_fixed = float(D_fixed)

            if D_fixed <= 0:
                raise ValueError(
                    "D_fixed must be greater than zero."
                )

            free = np.array([
                False,
                True,
                True,
                True,
            ])

        def to_theta(x):
            if D_fixed is None:
                return np.asarray(
                    x,
                    dtype=float,
                )

            return np.array([
                D_fixed,
                x[0],
                x[1],
                x[2],
            ])

        def residuals(x):
            theta = to_theta(x)

            return (
                lognormal_speed(
                    t[window],
                    *theta,
                )
                - target[window]
            )

        starts = initial_guesses(
            t,
            target,
            window,
            t0_min,
        )

        best = None

        lower_free = lower[free]
        upper_free = upper[free]

        for x0_full in starts:

            x0 = x0_full[free]

            # If D is free, use the full initialization.
            # If D is fixed, only [t0, mu, sigma] are used.
            x0 = np.clip(
                x0,
                lower_free + 1e-12,
                upper_free - 1e-12,
            )

            try:
                solution = least_squares(
                    residuals,
                    x0,
                    bounds=(
                        lower_free,
                        upper_free,
                    ),
                    x_scale="jac", # type: ignore
                    max_nfev=400,
                )
            except ValueError:
                continue

            if (
                best is None
                or solution.cost < best.cost
            ):
                best = solution

        if best is None:
            raise RuntimeError(
                "Could not fit lognormal stroke."
            )

        theta = to_theta(
            best.x
        )

        residual = residuals(
            best.x
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    residual**2
                )
            )
        )

        return theta, {
            "rmse": rmse,
            "cost": float(
                best.cost
            ),
            "success": bool(
                best.success
            ),
            "optimality": float(
                best.optimality
            ),
            "nfev": int(
                best.nfev
            ),
        }

    # ========================================================================
    # Sequential decomposition
    # ========================================================================

    def _fit_sequential(
        self,
        t,
        speed,
        boundaries,
        D,
    ):
        """
        Sequentially extract strokes and subtract them from the residual.
        """
        t = np.asarray(
            t,
            dtype=float,
        )

        speed = np.asarray(
            speed,
            dtype=float,
        )

        residual = speed.copy()

        steps = []

        for k in range(
            len(boundaries) - 1
        ):
            i0 = int(
                boundaries[k]
            )

            i1 = int(
                boundaries[k + 1]
            )

            window = slice(
                i0,
                i1 + 1,
            )

            D_k = (
                float(D[k])
                if self.lock_D
                else None
            )

            theta, info = (
                self._fit_stroke(
                    t,
                    residual,
                    window,
                    D_fixed=D_k,
                )
            )

            fitted = lognormal_speed(
                t,
                *theta,
            )

            residual_before = (
                residual.copy()
            )

            residual_after = (
                residual_before
                - fitted
            )

            steps.append({
                "k": k,
                "window": window,
                "theta": theta,
                "fitted": fitted,
                "residual_before": residual_before,
                "residual_after": residual_after,
                **info,
            })

            residual = residual_after

        return (
            steps,
            residual,
        )

    # ========================================================================
    # Joint refinement
    # ========================================================================

    def _refine_jointly(
        self,
        t,
        speed,
        steps,
    ):
        """
        Jointly refine all lognormal strokes.

        If lock_D=True, D remains fixed.
        Otherwise D is allowed to vary around its sequential value.
        """
        t = np.asarray(
            t,
            dtype=float,
        )

        speed = np.asarray(
            speed,
            dtype=float,
        )

        if len(steps) == 0:
            raise ValueError(
                "No strokes available for refinement."
            )

        dt = float(
            np.median(
                np.diff(t)
            )
        )

        n_strokes = len(
            steps
        )

        theta0 = np.asarray([
            step["theta"]
            for step in steps
        ])

        if theta0.shape != (
            n_strokes,
            4,
        ):
            raise ValueError(
                "Invalid stroke parameter shape."
            )

        lower = np.empty_like(
            theta0
        )

        upper = np.empty_like(
            theta0
        )

        scale = np.empty_like(
            theta0
        )

        for k, step in enumerate(
            steps
        ):
            D0, t00, mu0, sigma0 = (
                theta0[k]
            )

            t_start = float(
                t[
                    step["window"].start
                ]
            )

            t_end = float(
                t[
                    step["window"].stop - 1
                ]
            )

            duration = max(
                t_end - t_start,
                dt,
            )

            shift = (
                self.t0_shift_frac
                * duration
            )

            lower[k] = [
                max(D0 / self.D_max_factor, 1e-12),
                t00 - shift,
                mu0 - self.mu_shift,
                self.sigma_bounds[0],
            ]

            upper[k] = [
                max(
                    D0 * self.D_max_factor,
                    1e-12,
                ),
                min(
                    t00 + shift,
                    t_end - 3.0 * dt,
                ),
                mu0 + self.mu_shift,
                self.sigma_bounds[1],
            ]

            # Keep the original D as regularization scale.
            scale[k] = [
                max(D0, 1e-9),
                max(duration, 1e-9),
                1.0,
                1.0,
            ]

            # In case numerical bounds became invalid.
            if upper[k, 1] <= lower[k, 1]:
                upper[k, 1] = (
                    lower[k, 1] + 1e-9
                )

        # --------------------------------------------------------------
        # Which parameters are free?
        #
        # theta = [D, t0, mu, sigma]
        #
        # With lock_D=True:
        #       [t0, mu, sigma]
        #
        # Otherwise:
        #       [D, t0, mu, sigma]
        # --------------------------------------------------------------

        free = np.ones(
            (n_strokes, 4),
            dtype=bool,
        )

        if self.lock_D:
            free[:, 0] = False

        x0 = theta0[
            free
        ]

        lower_free = lower[
            free
        ]

        upper_free = upper[
            free
        ]

        v_scale = max(
            float(
                np.max(
                    np.abs(speed)
                )
            ),
            1e-9,
        )

        def unpack(
            x_free,
        ):
            theta = theta0.copy()

            theta[free] = x_free

            return theta

        def residuals(
            x_free,
        ):
            theta = unpack(
                x_free
            )

            reconstruction = np.zeros_like(
                speed
            )

            for stroke_theta in theta:
                reconstruction += (
                    lognormal_speed(
                        t,
                        *stroke_theta,
                    )
                )

            residual = (
                reconstruction
                - speed
            )

            if self.reg_weight > 0:

                regularization = (
                    self.reg_weight
                    * v_scale
                    * (
                        theta
                        - theta0
                    )
                    / scale
                )

                residual = np.concatenate([
                    residual,
                    regularization.ravel(),
                ])

            return residual

        x0 = np.clip(
            x0,
            lower_free + 1e-12,
            upper_free - 1e-12,
        )

        solution = least_squares(
            residuals,
            x0,
            bounds=(
                lower_free,
                upper_free,
            ),
            x_scale="jac", # type: ignore
            max_nfev=500,
        )

        theta = unpack(
            solution.x
        )

        refined = []

        for k, step in enumerate(
            steps
        ):
            stroke_theta = theta[k]

            fitted = lognormal_speed(
                t,
                *stroke_theta,
            )

            refined.append({
                "k": k,
                "window": step["window"],
                "theta": stroke_theta,
                "fitted": fitted,
            })

        reconstruction = np.zeros_like(
            speed
        )

        for item in refined:
            reconstruction += item[
                "fitted"
            ]

        residual = (
            speed
            - reconstruction
        )

        info = {
            "success": bool(
                solution.success
            ),
            "nfev": int(
                solution.nfev
            ),
            "cost": float(
                solution.cost
            ),
            "optimality": float(
                solution.optimality
            ),
        }

        return (
            refined,
            residual,
            info,
        )

    # ========================================================================
    # Convert parameters to StrokeSequence
    # ========================================================================

    def _to_stroke_sequence(
        self,
        refined,
    ):
        """
        Convert optimized parameter arrays into Stroke objects.
        """
        strokes = []

        for item in refined:

            D, t0, mu, sigma = (
                item["theta"]
            )

            strokes.append(
                Stroke(
                    D=float(D),
                    mu=float(mu),
                    sigma=float(sigma),
                    to=float(t0),
                )
            )

        return StrokeSequence(
            strokes
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def approximate(
        self,
        recording,
    ):
        """
        Extract a StrokeSequence from a Recording.
        """
        if recording.trajectory.t is None:
            raise ValueError(
                "LeastSquares requires trajectory.t."
            )

        t = np.asarray(
            recording.trajectory.t,
            dtype=float,
        )

        speed = np.asarray(
            recording.trajectory.speed,
            dtype=float,
        )

        if len(t) != len(speed):
            raise ValueError(
                "Trajectory time and speed arrays "
                "must have equal length."
            )

        if np.any(
            np.diff(t) <= 0
        ):
            raise ValueError(
                "Trajectory time must be strictly increasing."
            )

        # --------------------------------------------------------------
        # 1. Speed minima
        # --------------------------------------------------------------

        if self.boundaries is None:

            boundaries = (
                self._extract_boundaries(
                    speed
                )
            )

        else:

            boundaries = np.asarray(
                self.boundaries,
                dtype=int,
            )

            if len(boundaries) < 2:
                raise ValueError(
                    "At least two boundaries are required."
                )

            if np.any(
                np.diff(boundaries) <= 0
            ):
                raise ValueError(
                    "boundaries must be strictly increasing."
                )

            if boundaries[0] < 0:
                raise ValueError(
                    "boundaries contain a negative index."
                )

            if boundaries[-1] >= len(speed):
                raise ValueError(
                    "boundaries contain an out-of-range index."
                )

        # --------------------------------------------------------------
        # 2. Geometric D
        # --------------------------------------------------------------

        D_geom = (
            self._compute_geometric_D(
                recording.curve,
                boundaries,
            )
        )

        # --------------------------------------------------------------
        # 3. Sequential decomposition
        # --------------------------------------------------------------

        steps, residual_sequential = (
            self._fit_sequential(
                t,
                speed,
                boundaries,
                D_geom,
            )
        )

        # --------------------------------------------------------------
        # 4. Joint refinement
        # --------------------------------------------------------------

        if self.refine:

            (
                refined,
                residual,
                refinement_info,
            ) = self._refine_jointly(
                t,
                speed,
                steps,
            )

        else:

            refined = []

            for step in steps:

                refined.append({
                    "k": step["k"],
                    "window": step["window"],
                    "theta": step["theta"],
                    "fitted": step["fitted"],
                })

            residual = (
                residual_sequential
            )

            refinement_info = {
                "success": True,
                "nfev": 0,
                "cost": float(
                    0.5
                    * np.sum(
                        residual**2
                    )
                ),
                "optimality": 0.0,
            }

        # --------------------------------------------------------------
        # 5. StrokeSequence
        # --------------------------------------------------------------

        stroke_sequence = (
            self._to_stroke_sequence(
                refined
            )
        )

        # --------------------------------------------------------------
        # 6. Save parameters
        # --------------------------------------------------------------

        self._stroke_sequence = (
            stroke_sequence
        )

        return stroke_sequence