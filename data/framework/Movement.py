from __future__ import annotations

from abc import ABC, abstractmethod
from framework.literals import Point, DEFAULT_COMPLETION
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class Movement(ABC):
    """
    Common interface of Stroke and Trajectory.

    Subclasses implement the kinematics (position, velocity, timing); this class provides
    speed, duration and the plots (3D path colored by speed, speed profile over time).
    All time-dependent methods take a scalar or an array of times and return arrays:
    position/velocity -> (N, 3), speed -> (N,).
    """

    # ---- to be implemented by subclasses -------------------------------------
    @abstractmethod
    def get_start_point(self) -> Point: ...

    @abstractmethod
    def get_end_point(self) -> Point: ...

    @abstractmethod
    def get_onset(self) -> float:
        """Time at which the movement starts."""

    @abstractmethod
    def end_time(self, completion: float = DEFAULT_COMPLETION) -> float:
        """Time at which the given fraction of the movement has been completed."""

    @abstractmethod
    def position(self, t) -> np.ndarray: ...

    @abstractmethod
    def velocity(self, t) -> np.ndarray: ...

    # ---- optional hooks for the plots ----------------------------------------
    def _components(self) -> list[Movement]:
        """Sub-movements drawn as dashed lines (empty for a single stroke)."""
        return []

    def _waypoints(self) -> list[tuple[str, str, np.ndarray]]:
        """(label, color, points of shape (k, 3)) markers drawn on the path."""
        return [("Start", "green", np.array([self.get_start_point()])),
                ("End", "red", np.array([self.get_end_point()]))]

    # ---- timing ----------------------------------------------------------------
    def duration(self, completion: float = DEFAULT_COMPLETION) -> float:
        """Time needed to reach the end point, measured from the onset."""
        return self.end_time(completion) - self.get_onset()

    def speed(self, t) -> np.ndarray:
        """Scalar speed |v(t)|, shape (N,)."""
        return np.linalg.norm(self.velocity(t), axis=1)

    def _time_grid(self, n: int, t_range: tuple[float, float] | None = None) -> np.ndarray:
        lo, hi = t_range if t_range is not None else (self.get_onset(), self.end_time())
        return np.linspace(lo, hi, n)

    # ---- plots -----------------------------------------------------------------
    @staticmethod
    def _set_true_scale_axes(ax, points, pad: float = 0.05, min_rel_span: float = 1e-3) -> None:
        """Tight limits and box aspect equal to the data ranges (same length per unit on x, y, z)."""
        pts = np.vstack(points)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = (hi + lo) / 2.0
        span = np.maximum((hi - lo) * (1.0 + 2.0 * pad),
                          min_rel_span * max(float((hi - lo).max()), 1e-12))
        ax.set_xlim(center[0] - span[0] / 2.0, center[0] + span[0] / 2.0)
        ax.set_ylim(center[1] - span[1] / 2.0, center[1] + span[1] / 2.0)
        ax.set_zlim(center[2] - span[2] / 2.0, center[2] + span[2] / 2.0)
        ax.set_box_aspect(tuple(span))

    def plot_path(self, ax=None, n: int = 600, show_waypoints: bool = True,
                  show_components: bool = True, title: str | None = None):
        """3D path colored by the real speed; components are drawn as dashed arcs."""
        if ax is None:
            ax = plt.figure(figsize=(8, 6)).add_subplot(111, projection="3d")
        fig = ax.figure
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        t = self._time_grid(n)
        pos, speed = self.position(t), self.speed(t)
        segments = np.stack([pos[:-1], pos[1:]], axis=1)
        collection = Line3DCollection(segments, cmap="viridis",
                                      norm=plt.Normalize(speed.min(), speed.max()), linewidth=3) # type: ignore
        collection.set_array(0.5 * (speed[:-1] + speed[1:]))
        ax.add_collection3d(collection)
        fig.colorbar(collection, ax=ax, shrink=0.6, pad=0.1, label="speed (data units / time unit)")

        extent = [pos]
        if show_components:
            for i, component in enumerate(self._components()):
                arc = component.position(component._time_grid(200))
                ax.plot(*arc.T, "--", color=palette[i % len(palette)], linewidth=1,
                        label=f"{type(component).__name__} {i + 1}")
                extent.append(arc)
        if show_waypoints:
            for label, color, pts in self._waypoints():
                if len(pts) == 0:
                    continue
                ax.scatter(*pts.T, c=color, s=45, depthshade=False, label=label)
                extent.append(pts)

        self._set_true_scale_axes(ax, extent)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(title or type(self).__name__)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        return ax

    def plot_speed(self, ax=None, n: int = 600, t_range: tuple[float, float] | None = None,
                   show_components: bool = True):
        """Speed over time; components are dashed, the resultant is black."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3.5))
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        t = self._time_grid(n, t_range)

        if show_components:
            for i, component in enumerate(self._components()):
                color = palette[i % len(palette)]
                ax.plot(t, component.speed(t), "--", color=color,
                        label=f"{type(component).__name__} {i + 1}")
                ax.axvline(component.get_onset(), color=color, linestyle=":", linewidth=0.8)
        ax.plot(t, self.speed(t), color="black", linewidth=2, label="Resultant |v|")
        ax.set_xlabel("t")
        ax.set_ylabel("speed (data units / time unit)")
        ax.set_title("Speed profile")
        ax.legend()
        return ax

    def plot(self, figsize: tuple[float, float] = (14, 5.5), **path_kwargs):
        """Path in space (left) and speed over time (right)."""
        fig = plt.figure(figsize=figsize)
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        self.plot_path(ax=ax3, **path_kwargs)
        self.plot_speed(ax=ax2)
        fig.tight_layout()
        return fig, (ax3, ax2)
