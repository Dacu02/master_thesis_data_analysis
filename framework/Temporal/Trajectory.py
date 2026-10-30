import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Trajectory():

    def  __init__(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, t: np.ndarray | None = None):
        """Initialize the Trajectory with vx, vy, vz coordinates.
        Parameters
        ----------
        vx : np.ndarray
            The x-components of the velocity.
        vy : np.ndarray
            The y-components of the velocity.
        vz : np.ndarray
            The z-components of the velocity.
        t : np.ndarray, optional
            The time array, in seconds and starts at any value.
        """
        if not (len(vx) == len(vy) == len(vz)):
            raise ValueError("vx, vy, vz arrays must have the same length.")
        self.vx = vx
        self.vy = vy
        self.vz = vz
        if t is not None and len(t) != len(vx):
            raise ValueError("t array must have the same length as vx, vy, vz arrays.")
        if t is not None:
            self.t = t
        else:
            self.t = np.linspace(0, 1, len(vx))

    def plot(
        self,
        ax: plt.Axes | None = None, # type: ignore
        points: int = 1000,
        *,
        show: bool = False,
        **plot_kwargs,
    ) -> tuple[plt.Figure, plt.Axes]: # type: ignore
        """Plot the speed magnitude over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing 2D axes to draw on. If ``None``, a new figure and axes
            are created.
        points : int, default 1000
            Number of samples along the trajectory. Retained for API
            compatibility; the available samples are plotted directly.
        show : bool, default False
            If ``True``, calls ``plt.show()`` before returning.
        **plot_kwargs
            Extra keyword arguments forwarded to ``Axes.plot``
            (e.g. ``color``, ``lw``, ``label``).

        Returns
        -------
        (figure, axes) : tuple[plt.Figure, Axes3D]
        """
        # --- input validation ---
        if points < 2:
            raise ValueError(f"`points` must be >= 2, got {points}.")

        # --- axes/figure setup ---
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            if getattr(ax, "name", None) == "3d":
                raise TypeError(
                    "`ax` must be a 2D axes, got a 3D axes."
                )
            fig = ax.get_figure()

        # --- plot ---
        ax.plot(self.t, self.speed, **plot_kwargs)

        # --- labels & auto-scaling ---
        ax.set_xlabel("time")
        ax.set_ylabel("speed magnitude")

        if show:
            plt.show()

        return fig, ax
        
    @property
    def vectors(self) -> np.ndarray:
        """Velocity vectors, shape (N, 3)."""
        return np.column_stack((
            self.vx,
            self.vy,
            self.vz,
        ))

    @property
    def speed(self) -> np.ndarray:
        """Velocity magnitude."""
        return np.linalg.norm(self.vectors, axis=1)

    def __len__(self) -> int:
        """Number of points in the trajectory."""
        return len(self.vx)