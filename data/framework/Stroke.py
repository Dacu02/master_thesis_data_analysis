from __future__ import annotations

import numpy as np
from scipy.special import erf, erfinv
from framework import Movement
from framework.literals import Point, DEFAULT_COMPLETION, _COLLINEAR_TOL, _EPS, _LINE_ANGLE_TOL
# =============================================================================
# Stroke
# =============================================================================
class Stroke(Movement):
    """
    Sigma-lognormal stroke whose trajectory is a circular arc in 3D.

    Angles describe the unit tangent at the start and at the end of the arc:
        theta = zenith (angle from +Z, in [0, pi]),  psi = azimuth (atan2(y, x)).
    The arc is planar with constant curvature, and the total swept angle is the angle
    between the two tangents (in [0, pi]). D is the arc length (lognormal amplitude).
    """

    def __init__(self, mu: float, sigma: float, t0: float,
                 theta_s: float, psi_s: float, theta_e: float, psi_e: float,
                 D: float = 1.0, start_point: Point = (0.0, 0.0, 0.0)):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu, self.sigma, self.t0, self.D = float(mu), float(sigma), float(t0), float(D)
        self.theta_s, self.psi_s = float(theta_s), float(psi_s)
        self.theta_e, self.psi_e = float(theta_e), float(psi_e)

        self._origin = np.asarray(start_point, dtype=float)
        self._t_s, self._m, self._total = Stroke._frame(self.theta_s, self.psi_s,
                                                        self.theta_e, self.psi_e)
        mid_off, end_off = Stroke._arc_offset([0.5, 1.0], self.D, self._t_s, self._m, self._total)
        self._start_point = Stroke._as_point(self._origin)
        self._middle_point = Stroke._as_point(self._origin + mid_off)
        self._end_point = Stroke._as_point(self._origin + end_off)

    @classmethod
    def from_points(cls, mu: float, sigma: float, t0: float, start_point: Point,
                    middle_point: Point, end_point: Point, D: float | None = None) -> Stroke:
        """Build a stroke from three points of its arc (middle = point at half of the swept angle)."""
        t_s, t_e, length = cls._arc_from_points(start_point, middle_point, end_point)
        return cls(mu, sigma, t0, *cls._angles_from_direction(t_s), *cls._angles_from_direction(t_e),
                   D=length if D is None else D, start_point=start_point)

    # ---- typical parameters ----------------------------------------------------
    def get_parameters(self) -> dict[str, float]:
        """Typical Sigma-Lognormal parameters of the stroke."""
        return {"D": self.D, "mu": self.mu, "sigma": self.sigma, "t0": self.t0,
                "theta_s": self.theta_s, "psi_s": self.psi_s,
                "theta_e": self.theta_e, "psi_e": self.psi_e}

    def get_middle_point(self) -> Point:
        return self._middle_point

    def get_swept_angle(self) -> float:
        """Total angle swept by the tangent, in [0, pi]."""
        return self._total

    def with_start_point(self, start_point: Point) -> Stroke:
        """Copy of the stroke (same timing, angles and length) starting at start_point."""
        return Stroke(self.mu, self.sigma, self.t0, self.theta_s, self.psi_s,
                      self.theta_e, self.psi_e, D=self.D, start_point=start_point)

    def __repr__(self) -> str:
        p = self.get_parameters()
        return "Stroke(" + ", ".join(f"{k}={v:.4g}" for k, v in p.items()) + ")"

    # ---- Movement interface ----------------------------------------------------
    def get_start_point(self) -> Point:
        return self._start_point

    def get_end_point(self) -> Point:
        return self._end_point

    def get_onset(self) -> float:
        return self.t0

    def end_time(self, completion: float = DEFAULT_COMPLETION) -> float:
        if not 0.0 < completion < 1.0:
            raise ValueError("completion must be in (0, 1)")
        tau = np.exp(self.mu + self.sigma * np.sqrt(2.0) * erfinv(2.0 * completion - 1.0))
        return float(self.t0 + tau)

    def progress(self, t) -> np.ndarray:
        """Lognormal CDF: fraction of the arc covered at time t, shape (N,)."""
        tau = np.atleast_1d(np.asarray(t, dtype=float)) - self.t0
        valid = tau > 0
        tau_s = np.where(valid, tau, 1.0)
        delta = 0.5 * (1.0 + erf((np.log(tau_s) - self.mu) / (self.sigma * np.sqrt(2.0))))
        return np.where(valid, delta, 0.0)

    def speed(self, t) -> np.ndarray:
        """Scalar speed D * Lambda(t; t0, mu, sigma). Zero for t <= t0."""
        tau = np.atleast_1d(np.asarray(t, dtype=float)) - self.t0
        valid = tau > 0
        tau_s = np.where(valid, tau, 1.0)
        speed = (self.D / (self.sigma * np.sqrt(2.0 * np.pi) * tau_s)
                 * np.exp(-((np.log(tau_s) - self.mu) ** 2) / (2.0 * self.sigma ** 2)))
        return np.where(valid, speed, 0.0)

    def position(self, t) -> np.ndarray:
        """Position along the stroke: start point before t0, end point after the stroke."""
        offsets = Stroke._arc_offset(self.progress(t), self.D, self._t_s, self._m, self._total)
        return self._origin + offsets

    def velocity(self, t) -> np.ndarray:
        """Vector velocity: speed times the unit tangent at the current progress."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        speed = self.speed(t)
        if self._total == 0.0:
            tangent = np.tile(self._t_s, (t.size, 1))
        else:
            phi = self._total * self.progress(t)
            # d/dphi of R*(sin(phi) t_s + (1 - cos(phi)) m) is R*(cos(phi) t_s + sin(phi) m)
            tangent = np.cos(phi)[:, None] * self._t_s + np.sin(phi)[:, None] * self._m
        return speed[:, None] * tangent

    def arc_points(self, n: int = 200) -> np.ndarray:
        """Geometric arc sampled uniformly in progress, shape (n, 3)."""
        return self._origin + Stroke._arc_offset(np.linspace(0.0, 1.0, n), self.D,
                                                 self._t_s, self._m, self._total)

    def _waypoints(self) -> list[tuple[str, str, np.ndarray]]:
        return [("Start", "green", np.array([self._start_point])),
                ("Middle", "orange", np.array([self._middle_point])),
                ("End", "red", np.array([self._end_point]))]

    # ---- geometry helpers ------------------------------------------------------
    @staticmethod
    def _as_point(v) -> Point:
        return (float(v[0]), float(v[1]), float(v[2]))

    @staticmethod
    def _direction_from_angles(theta: float, psi: float) -> np.ndarray:
        """Unit vector from zenith theta (from +Z) and azimuth psi (atan2(y, x))."""
        return np.array([np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)])

    @staticmethod
    def _angles_from_direction(d) -> tuple[float, float]:
        """(zenith, azimuth) of a direction."""
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        return float(np.arccos(np.clip(d[2], -1.0, 1.0))), float(np.arctan2(d[1], d[0]))

    @staticmethod
    def _perpendicular(t: np.ndarray) -> np.ndarray:
        """Any unit vector orthogonal to t."""
        helper = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        m = np.cross(t, helper)
        return m / np.linalg.norm(m)

    @staticmethod
    def _arc_from_points(start_point, middle_point, end_point) -> tuple[np.ndarray, np.ndarray, float]:
        """Tangents at start and end, and arc length, of the circle through the 3 points."""
        s, mid, e = (np.asarray(p, dtype=float) for p in (start_point, middle_point, end_point))
        u, v = mid - s, e - s
        w = np.cross(u, v)
        nu, nv, nw = np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(w)

        if nu < _EPS or nv < _EPS or nw <= _COLLINEAR_TOL * nu * nv:   # straight segment
            d = v / max(nv, _EPS)
            return d, d, float(nv)

        center = s + (np.dot(v, v) * np.cross(w, u) + np.dot(u, u) * np.cross(v, w)) / (2.0 * nw ** 2)
        r_s, r_m, r_e = s - center, mid - center, e - center
        cross = np.cross(r_s, r_m)
        normal = cross / np.linalg.norm(cross)          # rotation start -> mid is positive about it
        radius = np.linalg.norm(r_s)
        half = np.arctan2(np.linalg.norm(cross), np.dot(r_s, r_m))   # angle(start -> mid)

        t_s = np.cross(normal, r_s) / radius
        t_e = np.cross(normal, r_e) / np.linalg.norm(r_e)
        return t_s, t_e, float(radius * 2.0 * half)

    @staticmethod
    def angles_from_arc(start_point: Point, middle_point: Point, end_point: Point
                        ) -> tuple[float, float, float, float]:
        """(theta_s, psi_s, theta_e, psi_e) of the tangents at the two ends of the arc."""
        t_s, t_e, _ = Stroke._arc_from_points(start_point, middle_point, end_point)
        return (*Stroke._angles_from_direction(t_s), *Stroke._angles_from_direction(t_e))

    @staticmethod
    def _frame(theta_s, psi_s, theta_e, psi_e) -> tuple[np.ndarray, np.ndarray, float]:
        """(t_s, m, total): start tangent, in-plane unit vector pointing to the center of
        curvature, and total swept angle in [0, pi] (m = 0 for a straight line)."""
        t_s = Stroke._direction_from_angles(theta_s, psi_s)
        t_e = Stroke._direction_from_angles(theta_e, psi_e)
        cos_t = float(np.clip(np.dot(t_s, t_e), -1.0, 1.0))
        perp = t_e - cos_t * t_s
        sin_t = float(np.linalg.norm(perp))
        total = float(np.arctan2(sin_t, cos_t))
        if total < _LINE_ANGLE_TOL:
            return t_s, np.zeros(3), 0.0
        m = perp / sin_t if sin_t > _EPS else Stroke._perpendicular(t_s)   # antiparallel: any plane
        return t_s, m, total

    @staticmethod
    def _arc_offset(delta, D, t_s, m, total) -> np.ndarray:
        """Displacement from the start point at arc progress delta in [0, 1], shape (N, 3)."""
        delta = np.atleast_1d(np.asarray(delta, dtype=float))
        if total == 0.0:
            return (D * delta)[:, None] * t_s
        phi = total * delta
        radius = D / total
        # R*(sin(phi) t_s + (1 - cos(phi)) m); 1 - cos(phi) = 2 sin^2(phi/2) avoids cancellation
        return radius * (np.sin(phi)[:, None] * t_s + (2.0 * np.sin(phi / 2.0) ** 2)[:, None] * m)