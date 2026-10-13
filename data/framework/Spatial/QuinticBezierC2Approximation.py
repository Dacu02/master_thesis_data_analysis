from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.optimize import brentq

from . import Curve, Approximate
from framework import Trajectory


class QuinticBezierC2Approximation(Approximate):
    """
    C2 quintic Bézier approximation of a 3D curve.

    The input path is split into strokes using `boundaries`. The velocity
    trajectory provides the endpoint velocity vectors at every knot. If
    `accelerations` is not provided, they are estimated from the velocities
    via central finite differences in time.

    For each stroke with duration dt and endpoint data
    (p0, v0, a0) and (p1, v1, a1), the six control points are:

        P0 = p0
        P1 = p0 + (dt/5)  * v0
        P2 = p0 + (2dt/5) * v0 + (dt^2/20) * a0
        P3 = p1 - (2dt/5) * v1 + (dt^2/20) * a1
        P4 = p1 - (dt/5)  * v1
        P5 = p1

    Therefore:

        dB/dt(0)  = v0,    d^2B/dt^2(0)  = a0
        dB/dt(dt) = v1,    d^2B/dt^2(dt) = a1

    and C0, C1, C2 continuity hold exactly at shared knots.
    """

    def __init__(
        self,
        boundaries: np.ndarray,
        trajectory: Trajectory,
        accelerations: np.ndarray | None = None,
        D: np.ndarray | None = None,
        dt_source: str = "timestamps",
        n_length_nodes: int = 64,
        dt_max_factor: float = 1000.0,
        length_tolerance: float = 1e-8,
    ):
        """
        Parameters
        ----------
        boundaries : np.ndarray
            1D array of indices that split the input curve into strokes.
        trajectory : Trajectory
            The velocity trajectory providing the endpoint velocities.
        accelerations : np.ndarray | None
            (N, 3) array of accelerations at every sample. If None, they are
            estimated from `trajectory.vectors` using central differences
            with respect to `trajectory.t`.
        D : np.ndarray | None
            1D array of target lengths for each stroke.
            If None, the measured polyline length of each stroke is used.
        dt_source : {"timestamps", "solve_D"}
            How the stroke duration is determined.
            - "timestamps": dt = t[i1] - t[i0]. D is used only as a
              consistency check (a warning is raised on mismatch).
            - "solve_D": dt is solved so that the Bézier arc length equals
              D[k]. `trajectory.t` is ignored for dt.
        n_length_nodes : int
            Number of Gauss-Legendre nodes for arc-length quadrature.
        dt_max_factor : float
            Cap on the bracket expansion when solving dt from D.
        length_tolerance : float
            Tolerance for the length-consistency warning.
        """
        self.boundaries = np.asarray(boundaries, dtype=int)

        if self.boundaries.ndim != 1:
            raise ValueError("boundaries must be a 1D array.")
        if len(self.boundaries) < 2:
            raise ValueError("At least two boundaries are required.")
        if np.any(np.diff(self.boundaries) <= 0):
            raise ValueError("boundaries must be strictly increasing.")

        self.trajectory = trajectory

        if dt_source not in ("timestamps", "solve_D"):
            raise ValueError("dt_source must be 'timestamps' or 'solve_D'.")
        self.dt_source = dt_source

        n_strokes = len(self.boundaries) - 1

        if D is not None:
            D = np.asarray(D, dtype=float)
            if D.ndim != 1:
                raise ValueError("D must be a 1D array.")
            if len(D) != n_strokes:
                raise ValueError("D must contain one value for every stroke.")
            if np.any(D <= 0):
                raise ValueError("All D values must be positive.")
        self.D = D

        # Accelerations: provided or estimated.
        if accelerations is not None:
            accelerations = np.asarray(accelerations, dtype=float)
            if accelerations.shape != trajectory.vectors.shape:
                raise ValueError(
                    "accelerations must have shape (N, 3) matching the "
                    "trajectory."
                )
        self.accelerations = accelerations

        self.n_length_nodes = n_length_nodes
        self.dt_max_factor = dt_max_factor
        self.length_tolerance = length_tolerance

        self._parameters = None
        self._curve = None
        self._sample_counts = None

    # ------------------------------------------------------------------
    # Bézier (quintic)
    # ------------------------------------------------------------------

    @staticmethod
    def _control_points(
        p0: np.ndarray,
        p1: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        a0: np.ndarray,
        a1: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Construct quintic Bézier control points from endpoint data.

        dt is the physical duration of the stroke (scalar).
        """
        dt2 = dt * dt
        return np.vstack((
            p0,
            p0 + (dt / 5.0) * v0,
            p0 + (2.0 * dt / 5.0) * v0 + (dt2 / 20.0) * a0,
            p1 - (2.0 * dt / 5.0) * v1 + (dt2 / 20.0) * a1,
            p1 - (dt / 5.0) * v1,
            p1,
        ))

    @staticmethod
    def _evaluate(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Evaluate quintic Bézier."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        w = 1.0 - u
        return (
            w[:, None] ** 5 * ctrl[0]
            + 5.0 * w[:, None] ** 4 * u[:, None] * ctrl[1]
            + 10.0 * w[:, None] ** 3 * u[:, None] ** 2 * ctrl[2]
            + 10.0 * w[:, None] ** 2 * u[:, None] ** 3 * ctrl[3]
            + 5.0 * w[:, None] * u[:, None] ** 4 * ctrl[4]
            + u[:, None] ** 5 * ctrl[5]
        )

    @staticmethod
    def _derivative_u(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
        """First derivative with respect to normalized parameter u."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        w = 1.0 - u
        return (
            5.0 * w[:, None] ** 4 * (ctrl[1] - ctrl[0])
            + 20.0 * w[:, None] ** 3 * u[:, None] * (ctrl[2] - ctrl[1])
            + 30.0 * w[:, None] ** 2 * u[:, None] ** 2 * (ctrl[3] - ctrl[2])
            + 20.0 * w[:, None] * u[:, None] ** 3 * (ctrl[4] - ctrl[3])
            + 5.0 * u[:, None] ** 4 * (ctrl[5] - ctrl[4])
        )

    @staticmethod
    def _second_derivative_u(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Second derivative with respect to normalized parameter u."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        w = 1.0 - u
        return (
            20.0 * w[:, None] ** 3 * (ctrl[0] - 2.0 * ctrl[1] + ctrl[2])
            + 60.0 * w[:, None] ** 2 * u[:, None] * (ctrl[1] - 2.0 * ctrl[2] + ctrl[3])
            + 60.0 * w[:, None] * u[:, None] ** 2 * (ctrl[2] - 2.0 * ctrl[3] + ctrl[4])
            + 20.0 * u[:, None] ** 3 * (ctrl[3] - 2.0 * ctrl[4] + ctrl[5])
        )

    def _arc_length(self, ctrl: np.ndarray) -> float:
        """Arc length using Gauss-Legendre quadrature."""
        x, w = np.polynomial.legendre.leggauss(self.n_length_nodes)
        u = 0.5 * (x + 1.0)
        weights = 0.5 * w
        derivative = self._derivative_u(ctrl, u)
        speed_u = np.linalg.norm(derivative, axis=1)
        return float(np.sum(weights * speed_u))

    # ------------------------------------------------------------------
    # Solve dt from target length
    # ------------------------------------------------------------------

    def _length_for_dt(self, p0, p1, v0, v1, a0, a1, dt) -> float:
        ctrl = self._control_points(p0, p1, v0, v1, a0, a1, dt)
        return self._arc_length(ctrl)

    def _solve_duration(self, p0, p1, v0, v1, a0, a1, D_target) -> float:
        """
        Find dt such that the quintic Bézier has length D_target.

        As dt -> 0 the arc length -> chord ||p1 - p0||.
        """
        chord = np.linalg.norm(p1 - p0)

        if D_target < chord - self.length_tolerance:
            raise ValueError(
                f"Requested D={D_target:.6g} is smaller than "
                f"the chord={chord:.6g}."
            )
        if abs(D_target - chord) <= self.length_tolerance:
            return 0.0

        speed_scale = max(
            np.linalg.norm(v0), np.linalg.norm(v1), 1e-9
        )
        dt0 = max(D_target / speed_scale, 1e-6)

        def f(dt):
            return self._length_for_dt(p0, p1, v0, v1, a0, a1, dt) - D_target

        dt_left = 1e-10
        f_left = f(dt_left)
        dt_right = dt0

        for _ in range(100):
            f_right = f(dt_right)
            if f_left * f_right <= 0:
                break
            dt_right *= 2.0
            if dt_right > dt0 * self.dt_max_factor:
                raise RuntimeError(
                    f"Could not find a duration dt satisfying "
                    f"D={D_target:.6g}."
                )
        else:
            raise RuntimeError("Could not bracket the duration solution.")

        dt = brentq(
            f, dt_left, dt_right,
            xtol=1e-12, rtol=1e-10,  # type: ignore
            maxiter=500,
        )
        return float(dt)  # type: ignore

    # ------------------------------------------------------------------
    # Approximate interface
    # ------------------------------------------------------------------

    def approximate(self, curve: Curve) -> tuple[dict, Curve]:
        p = curve.points
        n_points = len(p)

        if len(self.trajectory.vx) != n_points:
            raise ValueError(
                "Curve and Trajectory must contain the same number "
                "of samples."
            )
        if self.boundaries[-1] >= n_points:
            raise IndexError("A boundary lies outside the curve.")

        K = len(self.boundaries) - 1
        velocities = self.trajectory.vectors

        # Accelerations: provided or estimated.
        if self.accelerations is not None:
            accelerations = self.accelerations
        else:
            t = np.asarray(self.trajectory.t, dtype=float)
            accelerations = np.gradient(velocities, t, axis=0)

        # Target lengths.
        if self.D is None:
            D = np.zeros(K)
            for k in range(K):
                i0 = self.boundaries[k]
                i1 = self.boundaries[k + 1]
                segment = p[i0:i1 + 1]
                D[k] = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
        else:
            D = self.D.copy()

        control_points = []
        reconstructed_parts = []
        stroke_info = []
        self._sample_counts = []

        for k in range(K):
            i0 = self.boundaries[k]
            i1 = self.boundaries[k + 1]

            p0, p1 = p[i0], p[i1]
            v0, v1 = velocities[i0], velocities[i1]
            a0, a1 = accelerations[i0], accelerations[i1]

            n_samples = i1 - i0 + 1
            self._sample_counts.append(n_samples)

            # ----------------------------------------------------------
            # Duration: from timestamps or solved from D.
            # ----------------------------------------------------------
            if self.dt_source == "timestamps":
                dt = float(self.trajectory.t[i1] - self.trajectory.t[i0])
                if dt <= 0:
                    raise ValueError(
                        f"Stroke {k}: non-positive duration from timestamps."
                    )
                ctrl = self._control_points(p0, p1, v0, v1, a0, a1, dt)
                actual_length = self._arc_length(ctrl)
                length_error = actual_length - D[k]

                if abs(length_error) > self.length_tolerance:
                    warnings.warn(
                        f"Stroke {k}: Bézier length "
                        f"{actual_length:.6g} differs from target "
                        f"D={D[k]:.6g} by {length_error:.3e}."
                    )
            else:  # solve_D
                dt = self._solve_duration(p0, p1, v0, v1, a0, a1, D[k])
                ctrl = self._control_points(p0, p1, v0, v1, a0, a1, dt)
                actual_length = self._arc_length(ctrl)
                length_error = actual_length - D[k]

            # ----------------------------------------------------------
            # Sample reconstructed stroke.
            # ----------------------------------------------------------
            u = np.linspace(0.0, 1.0, n_samples)
            fitted = self._evaluate(ctrl, u)

            if k == 0:
                reconstructed_parts.append(fitted)
            else:
                reconstructed_parts.append(fitted[1:])

            control_points.append(ctrl)

            stroke_rmse = np.sqrt(
                np.mean(np.sum((p[i0:i1 + 1] - fitted) ** 2, axis=1))
            )

            stroke_info.append({
                "stroke": k,
                "boundary_start": i0,
                "boundary_end": i1,

                "D_target_m": float(D[k]),
                "D_bezier_m": float(actual_length),
                "D_error_m": float(length_error),

                "duration": float(dt),

                "speed_start_m_s": float(np.linalg.norm(v0)),
                "speed_end_m_s": float(np.linalg.norm(v1)),
                "accel_start_m_s2": float(np.linalg.norm(a0)),
                "accel_end_m_s2": float(np.linalg.norm(a1)),

                "rmse_m": float(stroke_rmse),

                "P0": ctrl[0].copy(),
                "P1": ctrl[1].copy(),
                "P2": ctrl[2].copy(),
                "P3": ctrl[3].copy(),
                "P4": ctrl[4].copy(),
                "P5": ctrl[5].copy(),
            })

        reconstructed = np.vstack(reconstructed_parts)

        approximated_curve = Curve(
            reconstructed[:, 0],
            reconstructed[:, 1],
            reconstructed[:, 2],
        )

        self._curve = approximated_curve
        self._parameters = {
            "degree": 5,
            "continuity": "C2",
            "boundaries": self.boundaries.copy(),
            "D": D.copy(),
            "strokes": stroke_info,
            "control_points": [cp.copy() for cp in control_points],
        }

        return self.get_approximation_paramters(), approximated_curve

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_approximation_paramters(self) -> dict:
        if self._parameters is None:
            raise RuntimeError("approximate() must be called first.")
        return self._parameters

    # ------------------------------------------------------------------
    # Generate curve
    # ------------------------------------------------------------------

    def generate_curve(self, **kwargs) -> Curve:
        if self._parameters is None:
            raise RuntimeError("No approximation parameters available.")

        strokes = self._parameters["strokes"]
        reconstructed_parts = []

        for k, stroke in enumerate(strokes):
            ctrl = np.array([
                stroke["P0"], stroke["P1"], stroke["P2"],
                stroke["P3"], stroke["P4"], stroke["P5"],
            ])
            n_samples = self._sample_counts[k]  # type: ignore
            u = np.linspace(0.0, 1.0, n_samples)
            fitted = self._evaluate(ctrl, u)
            if k == 0:
                reconstructed_parts.append(fitted)
            else:
                reconstructed_parts.append(fitted[1:])

        reconstructed = np.vstack(reconstructed_parts)
        return Curve(
            reconstructed[:, 0],
            reconstructed[:, 1],
            reconstructed[:, 2],
        )

    # ------------------------------------------------------------------
    # C0/C1/C2 diagnostics
    # ------------------------------------------------------------------

    def continuity_table(self):
        """
        Return C0/C1/C2 continuity diagnostics at every internal knot.
        """
        if self._parameters is None:
            raise RuntimeError("approximate() must be called first.")

        rows = []
        strokes = self._parameters["strokes"]

        for j in range(1, len(strokes)):
            left = strokes[j - 1]
            right = strokes[j]

            ctrl_l = np.array([
                left["P0"], left["P1"], left["P2"],
                left["P3"], left["P4"], left["P5"],
            ])
            ctrl_r = np.array([
                right["P0"], right["P1"], right["P2"],
                right["P3"], right["P4"], right["P5"],
            ])

            dt_l = left["duration"]
            dt_r = right["duration"]

            # Physical velocity at end of left / start of right.
            v_left = 5.0 * (ctrl_l[5] - ctrl_l[4]) / dt_l
            v_right = 5.0 * (ctrl_r[1] - ctrl_r[0]) / dt_r

            # Physical acceleration at end of left / start of right.
            a_left = 20.0 * (ctrl_l[5] - 2.0 * ctrl_l[4] + ctrl_l[3]) / (dt_l * dt_l)
            a_right = 20.0 * (ctrl_r[0] - 2.0 * ctrl_r[1] + ctrl_r[2]) / (dt_r * dt_r)

            target_idx = self.boundaries[j]
            v_target = self.trajectory.vectors[target_idx]

            rows.append({
                "knot": j,
                "index": int(target_idx),

                "C0_gap_m": float(np.linalg.norm(ctrl_l[5] - ctrl_r[0])),
                "C1_gap_m_s": float(np.linalg.norm(v_left - v_right)),
                "C2_gap_m_s2": float(np.linalg.norm(a_left - a_right)),

                "velocity_error_left_m_s": float(np.linalg.norm(v_left - v_target)),
                "velocity_error_right_m_s": float(np.linalg.norm(v_right - v_target)),
            })

        return rows