from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.optimize import brentq

from . import Curve, Approximate
from framework import Trajectory


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Unit vector of `vector`, or `fallback` if the vector is (almost) null."""

    norm = np.linalg.norm(vector)

    if norm > 1e-12:
        return vector / norm

    return np.asarray(fallback, dtype=float)


class CubicBezierC1Approximation(Approximate):
    """
    C1 cubic Bézier approximation of a 3D curve.

    The input path is split into strokes using `boundaries`.
    The velocity trajectory provides the endpoint velocity vectors
    at every knot.

    For each stroke:

        P0 = p0
        P1 = p0 + dt/3 * v0
        P2 = p1 - dt/3 * v1
        P3 = p1

    Therefore:

        dB/dt(0) = v0
        dB/dt(1) = v1

    and C1 continuity is imposed exactly at shared knots.

    `dt` is the parameter interval of the stroke. With time_scale="time" it is the
    physical duration of the stroke. With time_scale="chord" it is proportional to the
    chord of the stroke (same total duration), so that the C1 condition between two
    consecutive strokes only involves quantities that are known when the strokes are
    generated one after the other.

    The parameters that a generative model has to predict are returned in
    `parameters["by_stroke"]` (see `approximate`).
    """

    def __init__(
        self,
        boundaries: np.ndarray|None = None,
        trajectory: Trajectory|None = None,
        D: np.ndarray | None = None,
        n_length_nodes: int = 64,
        dt_max_factor: float = 1000.0,
        length_tolerance: float = 0.1,
        time_scale: str = "time",
    ):
        """
        Parameters
        ----------
        boundaries : np.ndarray | None
            1D array of indices that split the input curve into strokes. If None, the curve is treated as a single stroke.
        trajectory : Trajectory | None
            The velocity trajectory providing the endpoint velocity vectors. If None, the velocity vectors are not used.
        D : np.ndarray | None, optional
            1D array of target lengths for each stroke. If None, the measured
            polyline length of each stroke is used. Default is None.
        n_length_nodes : int, optional
            Number of nodes for Gauss-Legendre quadrature to compute Bézier arc length.
            Default is 64.
        dt_max_factor : float, optional
            Maximum factor to increase the initial guess for dt when solving for the target length.
            Default is 1000.0.
        length_tolerance : float, optional
            Tolerance for length consistency checks. Default is 0.02.
        time_scale : {"time", "chord"}, optional
            Parameter interval of every stroke: the physical duration ("time") or a
            duration proportional to the chord ("chord"). Default is "time".
        """
        if boundaries is None:
            self.boundaries = None
        else:
            self.boundaries = np.asarray(boundaries, dtype=int)

            if self.boundaries.ndim != 1:
                raise ValueError(
                    "boundaries must be a 1D array."
                )

            if len(self.boundaries) < 2:
                raise ValueError(
                    "At least two boundaries are required."
                )

            if np.any(np.diff(self.boundaries) <= 0):
                raise ValueError(
                    "boundaries must be strictly increasing."
                )

        if time_scale not in ("time", "chord"):
            raise ValueError(
                "time_scale must be 'time' or 'chord'."
            )

        self.time_scale = time_scale

        self.trajectory = trajectory

        if self.boundaries is None:
            n_strokes = 1
        else:
            n_strokes = len(self.boundaries) - 1

        if D is not None:
            D = np.asarray(D, dtype=float)

            if D.ndim != 1:
                raise ValueError(
                    "D must be a 1D array."
                )

            if len(D) != n_strokes:
                raise ValueError(
                    "D must contain one value for every stroke."
                )

            if np.any(D <= 0):
                raise ValueError(
                    "All D values must be positive."
                )

        self.D = D

        self.n_length_nodes = n_length_nodes
        self.dt_max_factor = dt_max_factor
        self.length_tolerance = length_tolerance

        self._parameters = None
        self._curve = None
        self._sample_counts = None

    # ------------------------------------------------------------------
    # Bézier
    # ------------------------------------------------------------------

    def set_problem(self, trajectory: Trajectory, boundaries: np.ndarray, D: np.ndarray | None = None):
        """
        Set the trajectory and boundaries for the approximation problem.

        This method allows to reuse the same CubicBezierC1Approximation instance
        for different trajectories and boundaries.
        """
        if D is not None:
            D = np.asarray(D, dtype=float)

        self.trajectory = trajectory
        self.boundaries = np.asarray(boundaries, dtype=int)
        self.D = D

    @staticmethod
    def _control_points(
        p0: np.ndarray,
        p1: np.ndarray,
        v0: np.ndarray,
        v1: np.ndarray,
        dt: np.ndarray,
    ) -> np.ndarray:
        """
        Construct cubic Bézier control points.

        dt is the parameter interval of the stroke.
        """

        return np.vstack((
            p0,
            p0 + (dt / 3.0) * v0,
            p1 - (dt / 3.0) * v1,
            p1,
        ))

    @staticmethod
    def _evaluate(
        ctrl: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Evaluate cubic Bézier."""

        u = np.asarray(u, dtype=float)
        u = np.atleast_1d(u)

        w = 1.0 - u

        return (
            w[:, None] ** 3 * ctrl[0]
            + 3.0 * w[:, None] ** 2 * u[:, None] * ctrl[1]
            + 3.0 * w[:, None] * u[:, None] ** 2 * ctrl[2]
            + u[:, None] ** 3 * ctrl[3]
        )

    @staticmethod
    def _derivative_u(
        ctrl: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """First derivative with respect to normalized parameter u."""

        u = np.asarray(u, dtype=float)
        u = np.atleast_1d(u)

        w = 1.0 - u

        return (
            3.0 * w[:, None] ** 2 * (ctrl[1] - ctrl[0])
            + 6.0 * w[:, None] * u[:, None] * (ctrl[2] - ctrl[1])
            + 3.0 * u[:, None] ** 2 * (ctrl[3] - ctrl[2])
        )

    def _arc_length(
        self,
        ctrl: np.ndarray,
    ) -> float:
        """
        Arc length using Gauss-Legendre quadrature.
        """

        x, w = np.polynomial.legendre.leggauss(
            self.n_length_nodes
        )

        u = 0.5 * (x + 1.0)
        weights = 0.5 * w

        derivative = self._derivative_u(
            ctrl,
            u,
        )

        speed_u = np.linalg.norm(
            derivative,
            axis=1,
        )

        return float(
            np.sum(weights * speed_u)
        )

    # ------------------------------------------------------------------
    # Stroke durations (parameter intervals)
    # ------------------------------------------------------------------

    def _stroke_durations(
        self,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Parameter interval of every stroke.

        "time":  physical duration between consecutive boundaries.
        "chord": proportional to the chord of the stroke, with the same total duration.
        """

        t = self.trajectory.t

        if t is None:
            raise ValueError(
                "trajectory.t is required to define the stroke durations."
            )

        durations = np.diff(
            np.asarray(t, dtype=float)[self.boundaries]
        )

        if self.time_scale == "time":
            return durations

        chords = np.linalg.norm(
            np.diff(p[self.boundaries], axis=0),
            axis=1,
        )

        return durations.sum() * chords / chords.sum()

    # ------------------------------------------------------------------
    # Approximate interface
    # ------------------------------------------------------------------

    def approximate(
        self,
        curve: Curve,
    ) -> tuple[dict, Curve]:
        """
        Approximate the given Curve with C1 cubic Bézier strokes.

        The returned parameters contain, besides the diagnostics of every stroke
        ("strokes") and the control points, the parameters that a generative model has
        to predict ("by_stroke"). For every stroke:

            delta_p   : P3 - P0, arrival point relative to the start of the stroke (3)
            u_e       : unit tangent at the arrival point (3)
            lambda_e  : |P3 - P2| / chord (1)

        and, only for the first stroke (all the other strokes inherit them):

            u_s       : unit tangent at the start point (3)
            lambda_s  : |P1 - P0| / chord (1)

        With time_scale="time" the physical duration of every stroke ("duration") is also
        needed to inherit lambda_s; with time_scale="chord" the inheritance is simply
        u_s' = u_e and lambda_s' = lambda_e. The length D of every stroke is derived from
        the curve and is not a parameter. "start_point" is the first point of the curve.
        """

        p = curve.points
        n_points = len(p)

        if len(self.trajectory.vx) != n_points:
            raise ValueError(
                "Curve and Trajectory must contain the same number "
                "of samples."
            )

        if self.boundaries[-1] >= n_points:
            raise IndexError(
                "A boundary lies outside the curve."
            )

        K = len(self.boundaries) - 1

        velocities = self.trajectory.vectors

        durations = self._stroke_durations(p)

        # --------------------------------------------------------------
        # Determine target lengths.
        #
        # If D is not passed, use the measured polyline length of each
        # stroke.
        # --------------------------------------------------------------

        if self.D is None:

            D = np.zeros(K)

            for k in range(K):

                i0 = self.boundaries[k]
                i1 = self.boundaries[k + 1]

                segment = p[i0:i1 + 1]

                D[k] = np.sum(
                    np.linalg.norm(
                        np.diff(
                            segment,
                            axis=0,
                        ),
                        axis=1,
                    )
                )

        else:
            D = self.D.copy()

        control_points = []
        reconstructed_parts = []
        stroke_info = []
        by_stroke = []

        self._sample_counts = []

        # --------------------------------------------------------------
        # Fit each stroke
        # --------------------------------------------------------------

        for k in range(K):

            i0 = self.boundaries[k]
            i1 = self.boundaries[k + 1]

            p0 = p[i0]
            p1 = p[i1]

            v0 = velocities[i0]
            v1 = velocities[i1]

            n_samples = i1 - i0 + 1

            self._sample_counts.append(
                n_samples
            )

            dt = durations[k]

            ctrl = self._control_points(
                p0,
                p1,
                v0,
                v1,
                dt,
            )

            actual_length = self._arc_length(
                ctrl
            )

            length_error = (
                actual_length - D[k]
            )

            # D is a consistency check.
            if abs(length_error) > self.length_tolerance:

                warnings.warn(
                    f"Stroke {k}: Bézier length "
                    f"{actual_length:.6g} differs from target "
                    f"D={D[k]:.6g} by {length_error:.3e}."
                )

            # ----------------------------------------------------------
            # Parameters to be predicted
            # ----------------------------------------------------------

            delta_p = ctrl[3] - ctrl[0]
            chord = float(np.linalg.norm(delta_p))
            chord_dir = delta_p / max(chord, 1e-12)

            handle_start = ctrl[1] - ctrl[0]
            handle_end = ctrl[3] - ctrl[2]

            entry = {
                "stroke": k,
                "delta_p": delta_p.copy(),
                "u_e": _unit(handle_end, chord_dir),
                "lambda_e": float(
                    np.linalg.norm(handle_end) / max(chord, 1e-12)
                ),
            }

            if k == 0:
                entry["u_s"] = _unit(handle_start, chord_dir)
                entry["lambda_s"] = float(
                    np.linalg.norm(handle_start) / max(chord, 1e-12)
                )
            else:
                entry["u_s"] = by_stroke[k - 1]["u_e"]
                entry["lambda_s"] = by_stroke[k - 1]["lambda_e"]

            if self.time_scale == "time":
                entry["duration"] = float(dt)

            by_stroke.append(entry)

            # ----------------------------------------------------------
            # Sample reconstructed stroke
            # ----------------------------------------------------------

            u = np.linspace(
                0.0,
                1.0,
                n_samples,
            )

            fitted = self._evaluate(
                ctrl,
                u,
            )

            if k == 0:
                reconstructed_parts.append(
                    fitted
                )
            else:
                # Shared knot already inserted by previous stroke.
                reconstructed_parts.append(
                    fitted[1:]
                )

            control_points.append(ctrl)

            stroke_rmse = np.sqrt(
                np.mean(
                    np.sum(
                        (
                            p[i0:i1 + 1]
                            - fitted
                        ) ** 2,
                        axis=1,
                    )
                )
            )

            stroke_info.append({
                "stroke": k,
                "boundary_start": i0,
                "boundary_end": i1,

                "D_target_m": float(D[k]),
                "D_bezier_m": float(actual_length),
                "D_error_m": float(length_error),

                "duration": float(dt),

                "speed_start_m_s": float(
                    np.linalg.norm(v0)
                ),
                "speed_end_m_s": float(
                    np.linalg.norm(v1)
                ),

                "rmse_m": float(
                    stroke_rmse
                ),

                "P0": ctrl[0].copy(),
                "P1": ctrl[1].copy(),
                "P2": ctrl[2].copy(),
                "P3": ctrl[3].copy(),
            })

        reconstructed = np.vstack(
            reconstructed_parts
        )

        approximated_curve = Curve(
            reconstructed[:, 0],
            reconstructed[:, 1],
            reconstructed[:, 2],
        )

        self._curve = approximated_curve

        self._parameters = {
            "degree": 3,
            "continuity": "C1",
            "time_scale": self.time_scale,

            "boundaries": self.boundaries.copy(),
            "D": D.copy(),

            "start_point": np.asarray(p[self.boundaries[0]], dtype=float).copy(),
            "by_stroke": by_stroke,

            "strokes": stroke_info,

            "control_points": [
                cp.copy()
                for cp in control_points
            ],
        }

        return (
            self.get_approximation_paramters(),
            approximated_curve,
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def get_approximation_paramters(self) -> dict:
        """Return the fitted approximation parameters."""

        if self._parameters is None:
            raise RuntimeError(
                "approximate() must be called first."
            )

        return self._parameters

    # ------------------------------------------------------------------
    # Rebuild the curve from the parameters to be predicted
    # ------------------------------------------------------------------

    @classmethod
    def control_points_from_by_stroke(
        cls,
        by_stroke: list[dict],
        start_point: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Build the control points of every stroke from the by_stroke parameters.

        The first stroke uses u_s and lambda_s. Every other stroke inherits
        u_s' = u_e and lambda_s' = lambda_e * (T'/T) * (c/c'), where T and c are the
        duration and the chord of the previous stroke (T'/T = c'/c with
        time_scale="chord", so lambda_s' = lambda_e). The durations are used only if
        every entry contains "duration".
        """

        control_points = []

        P0 = np.asarray(start_point, dtype=float)
        previous = None

        for k, entry in enumerate(by_stroke):

            delta_p = np.asarray(entry["delta_p"], dtype=float)
            P3 = P0 + delta_p

            chord = float(np.linalg.norm(delta_p))
            chord_dir = delta_p / max(chord, 1e-12)

            u_e = _unit(
                np.asarray(entry["u_e"], dtype=float),
                chord_dir,
            )
            lambda_e = float(entry["lambda_e"])

            if previous is None:

                u_s = _unit(
                    np.asarray(entry["u_s"], dtype=float),
                    chord_dir,
                )
                lambda_s = float(entry["lambda_s"])

            else:

                u_s = previous["u_e"]

                ratio = 1.0

                if "duration" in entry and "duration" in previous["entry"]:
                    ratio = (
                        entry["duration"] / previous["entry"]["duration"]
                    ) * (
                        previous["chord"] / max(chord, 1e-12)
                    )

                lambda_s = previous["lambda_e"] * ratio

            control_points.append(np.vstack((
                P0,
                P0 + lambda_s * chord * u_s,
                P3 - lambda_e * chord * u_e,
                P3,
            )))

            previous = {
                "u_e": u_e,
                "lambda_e": lambda_e,
                "chord": chord,
                "entry": entry,
            }

            P0 = P3

        return control_points

    @classmethod
    def curve_from_by_stroke(
        cls,
        by_stroke: list[dict],
        start_point: np.ndarray,
        sample_counts: list[int],
    ) -> Curve:
        """
        Generate a Curve from by_stroke parameters (for example predicted by a network).

        sample_counts : number of samples of every stroke (the shared knot is counted once).
        """

        parts = []

        control_points = cls.control_points_from_by_stroke(
            by_stroke,
            start_point,
        )

        for k, (ctrl, n_samples) in enumerate(zip(control_points, sample_counts)):

            fitted = cls._evaluate(
                ctrl,
                np.linspace(0.0, 1.0, n_samples),
            )

            parts.append(fitted if k == 0 else fitted[1:])

        points = np.vstack(parts)

        return Curve(
            points[:, 0],
            points[:, 1],
            points[:, 2],
        )

    def by_stroke_roundtrip_error(self) -> float:
        """
        Largest distance (m) between the control points fitted on the data and the control
        points rebuilt from by_stroke alone. It is ~0 if the parameters are sufficient.
        """

        parameters = self.get_approximation_paramters()

        rebuilt = self.control_points_from_by_stroke(
            parameters["by_stroke"],
            parameters["start_point"],
        )

        return max(
            float(np.max(np.abs(r - np.asarray(c))))
            for r, c in zip(rebuilt, parameters["control_points"])
        )

    # ------------------------------------------------------------------
    # Generate curve
    # ------------------------------------------------------------------

    def generate_curve(
        self,
        **kwargs,
    ) -> Curve:
        """
        Generate the approximated curve from the stored parameters.
        """

        if self._parameters is None:
            raise RuntimeError(
                "No approximation parameters available."
            )

        strokes = self._parameters["strokes"]

        reconstructed_parts = []

        for k, stroke in enumerate(strokes):

            ctrl = np.array([
                stroke["P0"],
                stroke["P1"],
                stroke["P2"],
                stroke["P3"],
            ])

            n_samples = self._sample_counts[k] # type: ignore

            u = np.linspace(
                0.0,
                1.0,
                n_samples,
            )

            fitted = self._evaluate(
                ctrl,
                u,
            )

            if k == 0:
                reconstructed_parts.append(
                    fitted
                )
            else:
                reconstructed_parts.append(
                    fitted[1:]
                )

        reconstructed = np.vstack(
            reconstructed_parts
        )

        return Curve(
            reconstructed[:, 0],
            reconstructed[:, 1],
            reconstructed[:, 2],
        )

    # ------------------------------------------------------------------
    # C1 diagnostics
    # ------------------------------------------------------------------

    def continuity_table(self):
        """
        Return C1 continuity diagnostics at every internal knot.
        """

        if self._parameters is None:
            raise RuntimeError(
                "approximate() must be called first."
            )

        rows = []

        strokes = self._parameters["strokes"]

        for j in range(1, len(strokes)):

            left = strokes[j - 1]
            right = strokes[j]

            ctrl_l = np.array([
                left["P0"],
                left["P1"],
                left["P2"],
                left["P3"],
            ])

            ctrl_r = np.array([
                right["P0"],
                right["P1"],
                right["P2"],
                right["P3"],
            ])

            dt_l = left["duration"]
            dt_r = right["duration"]

            # Velocity at end of left stroke (in the chosen parameter).
            v_left = (
                3.0
                * (ctrl_l[3] - ctrl_l[2])
                / dt_l
            )

            # Velocity at beginning of right stroke (in the chosen parameter).
            v_right = (
                3.0
                * (ctrl_r[1] - ctrl_r[0])
                / dt_r
            )

            target_idx = self.boundaries[j]

            v_target = self.trajectory.vectors[
                target_idx
            ]

            rows.append({
                "knot": j,
                "index": int(target_idx),

                "C0_gap_m": float(
                    np.linalg.norm(
                        ctrl_l[3] - ctrl_r[0]
                    )
                ),

                "C1_gap_m_s": float(
                    np.linalg.norm(
                        v_left - v_right
                    )
                ),

                "velocity_error_left_m_s": float(
                    np.linalg.norm(
                        v_left - v_target
                    )
                ),

                "velocity_error_right_m_s": float(
                    np.linalg.norm(
                        v_right - v_target
                    )
                ),
            })

        return rows