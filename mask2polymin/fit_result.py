import dataclasses
from collections.abc import Hashable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from mask2polymin import energy
from mask2polymin.sequence_moments import subsequence
from mask2polymin.sequence_segment import SequenceSegment

if TYPE_CHECKING:
    from mask2polymin.fit_to_points_sequence import FitterConfig


def _current_loss(segment: SequenceSegment) -> float:
    """Sum of squared distances from the segment's points to its current line."""
    points = subsequence(segment.whole_sequence, segment.first_index, segment.last_index)
    return float(segment.line_segment_params.squared_distances_to_line(points).sum())


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
        return sum((_current_loss(segment) for segment in self.segments), 0.0)

    def energy(self, config: "FitterConfig") -> float:
        """Energy of this result: config.energy if given, otherwise the built-in one."""
        if config.energy is None:
            return energy.energy(self.sse(), self.orphans_count(), len(self.segments), config.tolerance, config.segment_cost)
        # freeze() leaves the loss stale, so config.energy gets clones with the loss of their current lines
        clones = [segment.clone() for segment in self.segments]
        for clone in clones:
            clone.line_segment_params = dataclasses.replace(clone.line_segment_params, loss=_current_loss(clone))
        return config.energy(clones)
