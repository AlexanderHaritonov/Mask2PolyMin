import numpy as np
from dataclasses import dataclass
from typing import Optional

from mask2polymin.line_segment_params import LineSegmentParams


def points_count(sequence_length: int, first_index: int, last_index: int) -> int:
    # Range convention: first <= last is the linear range [first..last] (equal indices = a single point),
    # first > last wraps past the end; the full circle is [0..n-1].
    if last_index >= first_index:
        return last_index - first_index + 1
    else:
        return sequence_length - first_index + last_index + 1 # for closed polygon / circular case


def subsequence(sequence: np.ndarray, left, right) -> np.ndarray:
    # Range convention: left <= right is the linear range [left..right] (equal indices = a single point),
    # left > right wraps past the end; the full circle is [0..n-1].
    if left <= right:
        return sequence[left:right+1]
    else:
        return np.vstack([ sequence[left:], sequence[:right+1] ])


@dataclass
class SequenceSegment:
    """ whole_sequence: numpy array of shape (N, 2) """
    whole_sequence: np.ndarray
    first_index: int
    last_index: int
    line_segment_params: Optional[LineSegmentParams] = None

    ''' Set by the caller via freeze(), together with the line in line_segment_params. '''
    frozen: bool = False

    def points_count(self) -> int:
        return points_count(len(self.whole_sequence), self.first_index, self.last_index)

    def points(self) -> np.ndarray:
        """The segment's points, (points_count, 2), in contour order."""
        return subsequence(self.whole_sequence, self.first_index, self.last_index)

    def freeze(self, point: np.ndarray, direction: np.ndarray) -> None:
        """Pin this segment to the line through `point` along `direction`; refit() will not refit, split or merge it.
        refit() fills in loss and the segment's start and end points."""
        direction = np.asarray(direction, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)
        self.line_segment_params = LineSegmentParams(point, point, direction / np.linalg.norm(direction))
        self.frozen = True

    def clone(self) -> 'SequenceSegment':
        """Create a copy of this segment (whole_sequence is shared, line_segment_params is immutable)"""
        return SequenceSegment(
            whole_sequence=self.whole_sequence,  # shared reference, not copied
            first_index=self.first_index,
            last_index=self.last_index,
            line_segment_params=self.line_segment_params,  # immutable, safe to share
            frozen=self.frozen
        )

def print_segments_info(segments):
    print("\n" + "=" * 60)
    print("Segment Details:")
    print("=" * 60)
    for i, segment in enumerate(segments):
        params = segment.line_segment_params
        print(f"\nSegment {i+1}:")
        print(f"  Points: {segment.first_index} to {segment.last_index} ({segment.points_count()} points)")
        print(f"  Loss: {params.loss:.4f}")
        print(f"  Start point: [{params.start_point[0]:.2f}, {params.start_point[1]:.2f}]")
        print(f"  End point: [{params.end_point[0]:.2f}, {params.end_point[1]:.2f}]")
        print(f"  Direction: [{params.direction[0]:.4f}, {params.direction[1]:.4f}]")

    # Show fitted polygon
    print("\n" + "=" * 60)
    print("=" * 60)

