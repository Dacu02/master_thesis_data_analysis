from framework import Movement, Stroke
from framework.literals import Point, DEFAULT_COMPLETION
from typing import Sequence, Iterator
import numpy as np
class Trajectory(Movement):
    """
    Ordered sequence of strokes. Each stroke is translated to start where the previous one
    ends (the first one keeps its own start point unless start_point is given), and the
    velocity vectors of all strokes are summed.
    """

    def __init__(self, strokes: Sequence[Stroke], start_point: Point | None = None):
        strokes = list(strokes)
        if not strokes:
            raise ValueError("A Trajectory needs at least one Stroke")
        self._start = np.asarray(strokes[0].get_start_point() if start_point is None else start_point,
                                 dtype=float)
        self._strokes: list[Stroke] = []
        for stroke in strokes:
            self.append(stroke)

    # ---- sequence interface ----------------------------------------------------
    def append(self, stroke: Stroke) -> None:
        """Add a stroke at the tail. A relocated copy is stored, starting at the current end point."""
        anchor = self._strokes[-1].get_end_point() if self._strokes else self._start
        self._strokes.append(stroke.with_start_point(anchor)) # type: ignore

    def get_stroke(self, i: int) -> Stroke:
        return self._strokes[i]

    def get_num_strokes(self) -> int:
        return len(self._strokes)

    def get_strokes_parameters(self) -> list[dict[str, float]]:
        return [s.get_parameters() for s in self._strokes]

    def __len__(self) -> int:
        return self.get_num_strokes()

    def __getitem__(self, i: int) -> Stroke:
        return self.get_stroke(i)

    def __iter__(self) -> Iterator[Stroke]:
        return iter(self._strokes)

    def __repr__(self) -> str:
        return f"Trajectory({len(self)} strokes, onset={self.get_onset():.4g}, end={self.end_time():.4g})"

    # ---- Movement interface ----------------------------------------------------
    def get_start_point(self) -> Point:
        return self._strokes[0].get_start_point()

    def get_end_point(self) -> Point:
        return self._strokes[-1].get_end_point()

    def get_onset(self) -> float:
        return min(s.get_onset() for s in self._strokes)

    def end_time(self, completion: float = DEFAULT_COMPLETION) -> float:
        return max(s.end_time(completion) for s in self._strokes)

    def position(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        pos = np.tile(np.asarray(self.get_start_point()), (t.size, 1))
        for s in self._strokes:
            pos += s.position(t) - np.asarray(s.get_start_point())
        return pos

    def velocity(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return sum(s.velocity(t) for s in self._strokes) # type: ignore

    def _components(self) -> list[Movement]:
        return list(self._strokes)

    def _waypoints(self) -> list[tuple[str, str, np.ndarray]]:
        ends = np.array([s.get_end_point() for s in self._strokes[:-1]]).reshape(-1, 3)
        return [("Start", "green", np.array([self.get_start_point()])),
                ("Stroke ends", "black", ends),
                ("End", "red", np.array([self.get_end_point()]))]
