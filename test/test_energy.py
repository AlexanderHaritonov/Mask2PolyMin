"""energy.py: orphans_count() and energy()."""
import numpy as np
import pytest

from mask2polymin.energy import energy, orphans_count
from mask2polymin.sequence_moments import SequenceMoments, fit_range
from mask2polymin.sequence_segment import SequenceSegment


def segments_over(points, ranges):
    moments = SequenceMoments(points)
    return [SequenceSegment(points, first, last, fit_range(moments, first, last, with_endpoints=False)) for first, last in ranges]


def test_orphans_count():
    points = np.stack([np.arange(20.0), np.zeros(20)], axis=1)
    # orphans: 5, 6 between the segments, and 18, 19 at the end
    assert orphans_count(segments_over(points, [(0, 4), (7, 17)])) == 4


def test_orphans_count_with_a_wrapping_segment():
    points = np.stack([np.arange(20.0), np.zeros(20)], axis=1)
    # 15..2 wraps past the end; orphans: 3, 4 and 13, 14
    assert orphans_count(segments_over(points, [(5, 12), (15, 2)])) == 4


def test_energy_adds_sse_orphans_and_segments():
    # tolerance 1.5, so tolerance and tolerance² differ
    assert energy(sse=2.0, orphans_count=3, segments_count=2, tolerance=1.5, segment_cost=15.0) == pytest.approx(
        2.0 + 1.5 ** 2 * 3 + 15.0 * 1.5 ** 2 * 2)
