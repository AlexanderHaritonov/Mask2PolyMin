from collections.abc import Hashable
from typing import NamedTuple

import numpy as np

from mask2polymin import energy
from mask2polymin.sequence_moments import subsequence
from mask2polymin.sequence_segment import SequenceSegment


class FitResult(NamedTuple):
    """Result of FitterToPointsSequence.fit() and refit(); unpacks as `polygon, segments`."""
    polygon: np.ndarray
    segments: list[SequenceSegment]

    @property
    def frozen(self) -> dict[Hashable, SequenceSegment]:
        """The frozen segments that have a key, by key: see SequenceSegment.freeze()."""
        return {segment.key: segment for segment in self.segments if segment.frozen and segment.key is not None}

    def orphans_count(self) -> int:
        """Number of contour points assigned to no segment."""
        return energy.orphans_count(self.segments)

    def sse(self) -> float:
        """Sum of squared distances from each segment's points to its current line.
        Orphans are not included."""
        total = 0.0
        for segment in self.segments:
            points = subsequence(segment.whole_sequence, segment.first_index, segment.last_index)
            squared_distances = segment.line_segment_params.squared_distances_to_line(points)
            total += float(squared_distances.sum())
        return total

    def energy(self, tolerance: float, segment_cost: float) -> float:
        """Energy of this result. Pass the values from FitterConfig."""
        return energy.energy(self.sse(), self.orphans_count(), len(self.segments), tolerance, segment_cost)
