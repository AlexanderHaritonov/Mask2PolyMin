"""Energy of a segmentation"""
from mask2polymin.sequence_segment import SequenceSegment


def orphans_count(segments: list[SequenceSegment]) -> int:
    """Number of contour points assigned to no segment."""
    return len(segments[0].whole_sequence) - sum(s.points_count() for s in segments)


def segment_sse(segment: SequenceSegment) -> float:
    """Sum of squared distances from the segment's points to its line: its loss, for free and frozen segments."""
    return segment.line_segment_params.loss


def energy(sse: float, orphans_count: int, segments_count: int, tolerance: float, segment_cost: float) -> float:
    """Energy = sse + tolerance² · orphans_count + segment_cost · tolerance² · segments_count.
    An orphan costs as much as a point at distance tolerance; a segment as much as segment_cost such points."""
    tolerance_sq = tolerance * tolerance
    return sse + tolerance_sq * orphans_count + segment_cost * tolerance_sq * segments_count
