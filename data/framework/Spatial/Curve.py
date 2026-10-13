import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

class Curve():

    def  __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray | None = None):
        """Initialize the Curve with x, y, z coordinates.
        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the curve.
        y : np.ndarray
            The y-coordinates of the curve.
        z : np.ndarray
            The z-coordinates of the curve.
        t : np.ndarray, optional
            The time array, in seconds and starts at any value.
        """
        if not (len(x) == len(y) == len(z)):
            raise ValueError("x, y, z arrays must have the same length.")
        self.x = x
        self.y = y
        self.z = z
        if t is not None and len(t) != len(x):
            raise ValueError("t array must have the same length as x, y, z arrays.")
        self.t = t if t is not None else np.linspace(0, 1, len(x))
        

    def plot(
        self,
        ax: Axes3D | None = None, # type: ignore
        points: int = 1000,
        *,
        show: bool = False,
        **plot_kwargs,
    ) -> tuple[plt.Figure, Axes3D]: # type: ignore
        """Plot the 3D curve.

        Parameters
        ----------
        ax : Axes3D, optional
            Existing 3D axes to draw on. If ``None``, a new figure and axes
            are created.
        points : int, default 1000
            Number of samples along the curve.
        show : bool, default False
            If ``True``, calls ``plt.show()`` before returning.
        **plot_kwargs
            Extra keyword arguments forwarded to ``Axes3D.plot``
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
            ax = fig.add_subplot(111, projection="3d")
        else:
            if not isinstance(ax, Axes3D): # type: ignore
                raise TypeError(
                    f"`ax` must be a 3D axes (Axes3D), got {type(ax).__name__}."
                )
            fig = ax.get_figure() # type: ignore

        # --- plot ---
        (line,) = ax.plot(self.x, self.y, self.z, **plot_kwargs) # type: ignore

        # --- labels & auto-scaling ---
        ax.set_xlabel("x") # type: ignore
        ax.set_ylabel("y") # type: ignore
        ax.set_zlabel("z") # type: ignore

        # Equal aspect ratio when possible (best-effort, mplot3d is quirky)
        try:
            ax.set_box_aspect( # type: ignore
                (np.ptp(self.x), np.ptp(self.y), np.ptp(self.z))  # ptp = peak-to-peak range
            )
        except Exception:
            pass  # not all matplotlib versions / backends support this

        # --- optional display ---
        if show:
            plt.show()

        return fig, ax

    @property
    def points(self) -> np.ndarray:
        return np.column_stack((self.x, self.y, self.z))

    def snr(self, other: "Curve") -> float:
        if not isinstance(other, Curve):
            raise TypeError("other must be a Curve instance.")

        ref = np.asarray(self.points, dtype=float)
        est = np.asarray(other.points, dtype=float)

        if ref.shape != est.shape:
            raise ValueError("Curves must have the same shape.")

        signal_power = np.mean(ref ** 2)
        noise_power = np.mean((ref - est) ** 2)

        if noise_power == 0:
            return float("inf") if signal_power > 0 else float("nan")
        if signal_power == 0:
            return float("-inf")

        return float(10 * np.log10(signal_power / noise_power))

    def mse(self, other: "Curve") -> tuple[float, float]:
        """Calculate the mean squared error (MSE) between this curve and another curve, and outputs the maximum error."""
        if not isinstance(other, Curve):
            raise TypeError("other must be a Curve instance.")

        ref = np.asarray(self.points, dtype=float)
        est = np.asarray(other.points, dtype=float)

        if ref.shape != est.shape:
            raise ValueError("Curves must have the same shape.")

        return (float(np.mean((ref - est) ** 2)), float(np.max((ref - est) ** 2)))

    def __len__(self) -> int:
        return len(self.x)