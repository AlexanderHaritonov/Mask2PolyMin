from typing import NamedTuple

import numpy as np

from mask2polymin.sequence_moments import subsequence
from mask2polymin.sequence_segment import SequenceSegment


class FitResult(NamedTuple):
    """Result of FitterToPointsSequence.fit() and refit(); unpacks as `polygon, segments`."""
    polygon: np.ndarray
    segments: list[SequenceSegment]

    def orphans_count(self) -> int:
        """Number of contour points assigned to no segment."""
        return len(self.segments[0].whole_sequence) - sum(s.points_count() for s in self.segments)

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
        """Energy = sse + tolerance² · orphans_count + segment_cost · tolerance² · segments_count.
        An orphan costs as much as a point at distance tolerance; a segment as much as segment_cost such points.
        Pass the values from FitterConfig."""
        tolerance_sq = tolerance * tolerance
        return self.sse() + tolerance_sq * self.orphans_count() + segment_cost * tolerance_sq * len(self.segments)
