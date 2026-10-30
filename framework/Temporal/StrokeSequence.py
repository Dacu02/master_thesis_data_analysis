from . import Stroke
import math
import numpy as np
from collections.abc import Collection

class StrokeSequence:
    """
    Ordered sequence of lognormal velocity strokes.
    """

    def __init__(self, strokes: Collection[Stroke]):
        strokes = list(strokes)

        if not strokes:
            raise ValueError(
                "StrokeSequence requires at least one Stroke."
            )

        if not all(
            isinstance(stroke, Stroke)
            for stroke in strokes
        ):
            raise TypeError(
                "All elements must be instances of Stroke."
            )

        self._strokes = strokes

    @property
    def strokes(self) -> list[Stroke]:
        """Return the list of strokes."""
        return self._strokes

    def get(self, i: int) -> Stroke:
        """Return stroke i."""
        return self._strokes[i]

    def __getitem__(self, i: int) -> Stroke:
        """Allow sequence[i]."""
        return self._strokes[i]

    def __len__(self) -> int:
        """Number of strokes."""
        return len(self._strokes)

    def __iter__(self):
        """Iterate over the strokes."""
        return iter(self._strokes)

    def index(self, stroke: Stroke) -> int:
        """Return the index of a stroke."""
        return self._strokes.index(stroke)

    def v(self, t: float) -> float:
        """
        Return the total velocity at global time t.

        Stroke contributions are summed.
        """
        return sum(
            stroke.v(t)
            for stroke in self._strokes
        )

    @property
    def start_time(self) -> float:
        """Global start time of the sequence."""
        return self._strokes[0].to

    @property
    def end_time(self) -> float:
        """
        Approximate global end time.

        Since a lognormal distribution has infinite support,
        this is not mathematically finite. This property therefore
        uses a practical cutoff of +4 sigma in log-time.
        """
        return max(
            stroke.to
            + math.exp(
                stroke.mu + 4 * stroke.sigma
            )
            for stroke in self._strokes
        )

    def v_array(self, t):
        t = np.asarray(t, dtype=float)
        return np.array([self.v(float(ti)) for ti in t])

    def copy(self) -> "StrokeSequence":
        """Return a deep copy of the StrokeSequence."""
        return StrokeSequence([stroke.copy() for stroke in self._strokes])

    def filter_strokes_by_peak(self, velocity_threshold):
            """Skip the weak strokes at the beginning, keep the strong ones, stop at the first weak one after them.
    
            Returns the kept strokes and the first weak stroke found after them (corrective last stroke), or None.
            """
            filtered, corrective = [], None
            for stroke in self._strokes:
                _, v_peak = stroke.lognormal_peak()
                if v_peak >= velocity_threshold:
                    filtered.append(stroke)
                elif filtered:
                    corrective = stroke
                    break
            return filtered, corrective
    