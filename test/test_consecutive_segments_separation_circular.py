import numpy as np
import pytest
from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.sequence_segment import SequenceSegment
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence

def make_line_segment(whole_sequence: np.ndarray, i_start: int, i_end: int) -> LineSegmentParams:
    start = whole_sequence[i_start].astype(np.float64)
    end = whole_sequence[i_end].astype(np.float64)
    direction = (end - start).astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction = (direction / norm).astype(np.float64)
    return LineSegmentParams(start_point=start, end_point=end, direction=direction, loss=0.0)

@pytest.fixture
def fitter():
    # 8 points: rectangle perimeter
    points = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],   # indices 0..3
        [3, 1], [2, 1], [1, 1], [0, 1]    # indices 4..7
    ], dtype=np.float64)
    return FitterToPointsSequence(points, is_closed=True)

def build_segments(fitter, seg1_first, seg1_last, seg2_first, seg2_last):
    seq = fitter.whole_sequence
    seg1 = SequenceSegment(seq, seg1_first, seg1_last,
                           line_segment_params=make_line_segment(seq, seg1_first, seg1_last))
    seg2 = SequenceSegment(seq, seg2_first, seg2_last,
                           line_segment_params=make_line_segment(seq, seg2_first, seg2_last))
    N = len(seq)
    # adjacency constraints
    assert (seg1.last_index + 1) % N == seg2.first_index
    assert (seg2.last_index + 1) % N == seg1.first_index
    return seg1, seg2

def assert_valid_separation(fitter, last1, first2):
    assert isinstance(last1, int)
    assert isinstance(first2, int)
    gap = (first2 - last1 - 1) % len(fitter.whole_sequence)
    assert gap <= fitter.config.max_orphans_per_junction

# ---- Hardcoded testcases ----

def test_seg1_end_in_first_half(fitter):
    # seg1 covers indices 4..6, seg2 covers 7..3
    seg1, seg2 = build_segments(fitter, 4, 6, 7, 3)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)

def test_seg1_end_on_left_limit(fitter):
    # seg1 covers indices 4..7, seg2 covers 0..3
    seg1, seg2 = build_segments(fitter, 4, 7, 0, 3)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)

def test_seg1_end_in_second_half(fitter):
    # seg1 covers indices 5..7, seg2 covers 0..4
    seg1, seg2 = build_segments(fitter, 5, 7, 0, 4)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)

def test_seg1_end_exactly_at_end(fitter):
    # seg1 covers indices 6..7, seg2 covers 0..5
    seg1, seg2 = build_segments(fitter, 6, 7, 0, 5)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)

def test_seg2_end_in_first_half(fitter):
    # seg2 covers indices 0..2, seg1 covers 3..7
    seg1, seg2 = build_segments(fitter, 3, 7, 0, 2)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)

def test_seg2_end_on_left_limit(fitter):
    # seg2 covers indices 0..3, seg1 covers 4..7
    seg1, seg2 = build_segments(fitter, 4, 7, 0, 3)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)

def test_seg2_end_in_second_half(fitter):
    # seg2 covers indices 0..4, seg1 covers 5..7
    seg1, seg2 = build_segments(fitter, 5, 7, 0, 4)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)

def test_seg2_end_exactly_at_end(fitter):
    # seg2 covers indices 0..5, seg1 covers 6..7
    seg1, seg2 = build_segments(fitter, 6, 7, 0, 5)
    last1, first2 = fitter.best_consecutive_segments_separation(seg1, seg2)
    assert_valid_separation(fitter, last1, first2)