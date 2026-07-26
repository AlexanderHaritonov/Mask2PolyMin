import numpy as np
import pytest
from mask2polymin.sequence_segment import SequenceSegment
from fit_line_segment_reference import fit_line_segment
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence

def make_segment(seq, first, last):
    if last > first:
        params = fit_line_segment(seq[first:last+1])
    else:
        params = fit_line_segment(seq[first:] + seq[:last+1])
    return SequenceSegment(seq, first, last, params)

def test_split_segment_regular_case():
    # 10 points along a diagonal line, one unambiguous outlier at index 6.
    # Offset verified empirically: a large enough outlier rotates the fitted direction
    # toward itself, which can shift the argmax to a *different* point than the outlier
    # (e.g. +50 here flips the argmax to index 9) -- +10 keeps a clear ~3x margin instead.
    seq = np.array([[i, i] for i in range(10)], dtype=float)
    seq[6] = [6, 16]
    fitter = FitterToPointsSequence()
    fitter._prepare(seq)
    segment = make_segment(seq, 0, 9)
    segments = [segment]

    result = fitter._split_segment(segments, 0)
    # Pivot lands exactly on the outlier (index 6), well clear of the min-2-points clamp
    assert len(result) == 2
    assert result[0].first_index == 0
    assert result[0].last_index == 6
    assert result[1].first_index == 7
    assert result[1].last_index == 9

def test_split_segment_segment_ends_on_last_index():
    seq = np.array([[i, i] for i in range(8)], dtype=float)
    seq[5] = [5, 8]  # outlier at local offset 3 within segment [2..7]
    fitter = FitterToPointsSequence()
    fitter._prepare(seq)
    segment = make_segment(seq, 2, 7)  # ends at last index
    segments = [segment]

    result = fitter._split_segment(segments, 0)
    # Pivot lands on the outlier (index 5)
    assert result[0].first_index == 2
    assert result[0].last_index == 5
    assert result[1].first_index == 6
    assert result[1].last_index == 7

def test_split_segment_first_half_ends_on_last_index():
    seq = np.array([[i, i] for i in range(6)], dtype=float)
    seq[2] = [2, 7]  # outlier at local offset 2
    fitter = FitterToPointsSequence()
    fitter._prepare(seq)
    segment = make_segment(seq, 0, 5)
    segments = [segment]

    result = fitter._split_segment(segments, 0)
    # Pivot lands on the outlier (index 2)
    assert result[0].first_index == 0
    assert result[0].last_index == 2
    assert result[1].first_index == 3
    assert result[1].last_index == 5

def test_split_segment_clamps_pivot_near_segment_start():
    # Outlier at the very first point: raw argmax would leave segment1 with a single
    # point, so MIN_SEGMENT_POINTS clamps the pivot forward instead.
    seq = np.array([[i, i] for i in range(10)], dtype=float)
    seq[0] = [0, 5]
    fitter = FitterToPointsSequence()
    fitter._prepare(seq)
    segment = make_segment(seq, 0, 9)
    segments = [segment]

    result = fitter._split_segment(segments, 0)
    assert result[0].first_index == 0
    assert result[0].last_index == 1
    assert result[1].first_index == 2
    assert result[1].last_index == 9

def test_split_segment_clamps_pivot_near_segment_end():
    # Outlier at the very last point: raw argmax would leave segment2 with a single
    # point, so MIN_SEGMENT_POINTS clamps the pivot backward instead.
    seq = np.array([[i, i] for i in range(10)], dtype=float)
    seq[9] = [9, 12]
    fitter = FitterToPointsSequence()
    fitter._prepare(seq)
    segment = make_segment(seq, 0, 9)
    segments = [segment]

    result = fitter._split_segment(segments, 0)
    assert result[0].first_index == 0
    assert result[0].last_index == 7
    assert result[1].first_index == 8
    assert result[1].last_index == 9

def test_split_segment_second_half_starts_at_index_0():
    seq = np.array([[i, i] for i in range(6)])
    fitter = FitterToPointsSequence()
    fitter._prepare(seq, is_closed=True)
    # Closed segment: first_index=4, last_index=1
    segment = make_segment(seq, 4, 1)
    segments = [segment]

    result = fitter._split_segment(segments, 0)
    # Ensure one of the new segments starts at 0
    assert any(seg.first_index == 0 for seg in result)

def test_split_segment_starts_on_last_index():
    seq = np.array([[i, i] for i in range(6)])
    fitter = FitterToPointsSequence()
    fitter._prepare(seq, is_closed=True)
    # Closed segment starting at last index
    segment = make_segment(seq, 5, 2)
    segments = [segment]

    result = fitter._split_segment(segments, 0)
    # Ensure first segment starts at 5
    assert result[0].first_index == 5

