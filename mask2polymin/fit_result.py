from collections.abc import Hashable
from typing import NamedTuple

import numpy as np

from mask2polymin.energy import orphans_count as segments_orphans_count
from mask2polymin.sequence_moments import subsequence
from mask2polymin.sequence_segment import SequenceSegment


def _current_loss(segment: SequenceSegment) -> float:
    """Sum of squared distances from the segment's points to its current line."""
    points = subsequence(segment.whole_sequence, segment.first_index, segment.last_index)
    return float(segment.line_segment_params.squared_distances_to_line(points).sum())


class FitResult(NamedTuple):
    """Result of FitterToPointsSequence.fit() and refit(); unpacks as `polygon, segments, energy`.
    energy: the Energy of segments when fit() or refit() returned them; freeze() does not update it."""
    polygon: np.ndarray
    segments: list[SequenceSegment]
    energy: float

    @property
    def frozen(self) -> dict[Hashable, SequenceSegment]:
        """The frozen segments that have a key, by key: see SequenceSegment.freeze()."""
        return {segment.key: segment for segment in self.segments if segment.frozen and segment.key is not None}

    def orphans_count(self) -> int:
        """Number of contour points assigned to no segment."""
        return segments_orphans_count(self.segments)

    def sse(self) -> float:
        """Sum of squared distances from each segment's points to its current line.
        Orphans are not included."""
        return sum((_current_loss(segment) for segment in self.segments), 0.0)
