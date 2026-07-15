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


@dataclass
class SequenceSegment:
    """ whole_sequence: numpy array of shape (N, 2) """
    whole_sequence: np.ndarray
    first_index: int
    last_index: int
    line_segment_params: Optional[LineSegmentParams] = None

    def points_count(self) -> int:
        return points_count(len(self.whole_sequence), self.first_index, self.last_index)

    def clone(self) -> 'SequenceSegment':
        """Create a copy of this segment (whole_sequence is shared, line_segment_params is immutable)"""
        return SequenceSegment(
            whole_sequence=self.whole_sequence,  # shared reference, not copied
            first_index=self.first_index,
            last_index=self.last_index,
            line_segment_params=self.line_segment_params  # immutable, safe to share
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

