import numpy as np
import pytest
from mask2polymin.sequence_segment import SequenceSegment
from mask2polymin.fit_line_segment import fit_line_segment
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence

def make_segment(seq, first, last):
    if last > first:
        params = fit_line_segment(seq[first:last+1])
    else:
        params = fit_line_segment(seq[first:] + seq[:last+1])
    return SequenceSegment(seq, first, last, params)

def test_split_segment_regular_case():
    # 10 points along a diagonal line
    seq = np.array([[i, i] for i in range(10)])
    fitter = FitterToPointsSequence(seq)
    segment = make_segment(seq, 0, 9)
    segments = [segment]

    result = fitter.split_segment(segments, 0)
    # Expect two segments: [0..4], [5..9]
    assert len(result) == 2
    assert result[0].first_index == 0
    assert result[0].last_index == 4
    assert result[1].first_index == 5
    assert result[1].last_index == 9

def test_split_segment_segment_ends_on_last_index():
    seq = np.array([[i, i] for i in range(8)])
    fitter = FitterToPointsSequence(seq)
    segment = make_segment(seq, 2, 7)  # ends at last index
    segments = [segment]

    result = fitter.split_segment(segments, 0)
    # Pivot = (2+7)//2 = 4
    assert result[0].first_index == 2
    assert result[0].last_index == 4
    assert result[1].first_index == 5
    assert result[1].last_index == 7

def test_split_segment_first_half_ends_on_last_index():
    seq = np.array([[i, i] for i in range(6)])
    fitter = FitterToPointsSequence(seq)
    segment = make_segment(seq, 0, 5)
    segments = [segment]

    result = fitter.split_segment(segments, 0)
    # Pivot = (0+5)//2 = 2
    assert result[0].first_index == 0
    assert result[0].last_index == 2
    assert result[1].first_index == 3
    assert result[1].last_index == 5

def test_split_segment_second_half_starts_at_index_0():
    seq = np.array([[i, i] for i in range(6)])
    fitter = FitterToPointsSequence(seq, is_closed=True)
    # Closed segment: first_index=4, last_index=1
    segment = make_segment(seq, 4, 1)
    segments = [segment]

    result = fitter.split_segment(segments, 0)
    # Ensure one of the new segments starts at 0
    assert any(seg.first_index == 0 for seg in result)

def test_split_segment_starts_on_last_index():
    seq = np.array([[i, i] for i in range(6)])
    fitter = FitterToPointsSequence(seq, is_closed=True)
    # Closed segment starting at last index
    segment = make_segment(seq, 5, 2)
    segments = [segment]

    result = fitter.split_segment(segments, 0)
    # Ensure first segment starts at 5
    assert result[0].first_index == 5

