import math
import matplotlib.pyplot as plt
import numpy as np
from sympy import erfinv


class Stroke:
    """
    Single lognormal velocity stroke.

    Parameters
    ----------
    D : float
        Stroke amplitude / travelled distance.
        The integral of v(t) is D.
    mu : float
        Lognormal parameter in local time coordinates.
    sigma : float
        Lognormal shape parameter.
    to : float
        Global start time of the stroke.
    """

    def __init__(
        self,
        D: float,
        mu: float,
        sigma: float,
        to: float,
    ):
        if D <= 0:
            raise ValueError("D must be greater than zero")

        if sigma <= 0:
            raise ValueError("sigma must be greater than zero")

        self.D = float(D)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.to = float(to)

    def v(self, t: float) -> float:
        """
        Return the global velocity contribution of this stroke.
        """

        # Convert global time to local stroke time.
        tau = t - self.to

        if tau <= 0:
            return 0.0

        pdf = (
            math.exp(
                -(
                    (math.log(tau) - self.mu) ** 2
                    / (2 * self.sigma**2)
                )
            )
            / (
                tau
                * self.sigma
                * math.sqrt(2 * math.pi)
            )
        )

        return self.D * pdf

    def plot(
        self,
        ax=None,
        points=1000,
    ) -> tuple[plt.Figure, plt.Axes]: # type: ignore
        """Plot the velocity profile of the stroke."""

        if points < 2:
            raise ValueError("points must be at least 2")

        if ax is None:
            _, ax = plt.subplots()

        t_local = np.linspace(
            math.exp(self.mu - 4 * self.sigma),
            math.exp(self.mu + 4 * self.sigma),
            points,
        )

        velocity = np.array([
            self.v(self.to + t)
            for t in t_local
        ])

        t_global = self.to + t_local

        ax.plot(
            t_global,
            velocity,
            label=(
                f"Stroke "
                f"(D={self.D:.3f}, "
                f"μ={self.mu:.3f}, "
                f"σ={self.sigma:.3f})"
            ),
        )

        ax.set_xlabel("t [s]")
        ax.set_ylabel("v(t) [m/s]")
        ax.set_title("Stroke velocity")
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax.figure, ax

    def copy(self) -> "Stroke":
        """Return a copy of the stroke."""
        return Stroke(
            D=self.D,
            mu=self.mu,
            sigma=self.sigma,
            to=self.to,
        )
    
    def lognormal_peak(self):
        """Time and value of the speed maximum of a single lognormal stroke."""
        t0 = float(self.to)
        mu = float(self.mu)
        sigma = float(self.sigma)
        D = abs(float(self.D))

        t_peak = t0 + np.exp(mu - sigma**2)
        v_peak = D / (sigma * np.sqrt(2 * np.pi)) * np.exp(-mu + sigma**2 / 2)
        return t_peak, v_peak



    def lognormal_time_at_fraction(self, fraction):
        """Return the time at which a stroke reaches a fraction of its distance."""
        return float(self.to) + np.exp(
            float(self.mu)
            + float(self.sigma) * np.sqrt(2.0) * erfinv(2.0 * fraction - 1.0)
        )