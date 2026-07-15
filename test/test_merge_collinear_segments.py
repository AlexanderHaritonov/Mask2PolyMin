import numpy as np
from mask2polymin.sequence_segment import SequenceSegment
from mask2polymin.sequence_moments import fit_range
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence


def make_segment(fitter, first, last):
    # Fit via the fitter's own moments so params match what _merge_collinear_segments recomputes
    return SequenceSegment(fitter.whole_sequence, first, last,
                           fit_range(fitter._moments, first, last))


def make_segments(fitter, index_ranges):
    return [make_segment(fitter, first, last) for first, last in index_ranges]


def test_merges_two_collinear_segments():
    seq = np.array([[i, 0] for i in range(10)], dtype=float)
    fitter = FitterToPointsSequence(seq)
    segments = make_segments(fitter, [(0, 4), (5, 9)])

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 1
    assert result[0].first_index == 0
    assert result[0].last_index == 9
    assert result[0].line_segment_params.loss == 0.0


def test_merges_chain_of_three_collinear_segments():
    seq = np.array([[i, 0] for i in range(12)], dtype=float)
    fitter = FitterToPointsSequence(seq)
    segments = make_segments(fitter, [(0, 3), (4, 7), (8, 11)])

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 1
    assert result[0].first_index == 0
    assert result[0].last_index == 11


def test_corner_segments_stay_separate():
    # L-shape: the direction pre-check (~11 degrees) rejects the perpendicular pair outright
    horizontal = [[x, 0] for x in range(5)]
    vertical = [[4, y] for y in range(1, 6)]
    seq = np.array(horizontal + vertical, dtype=float)
    fitter = FitterToPointsSequence(seq)
    segments = make_segments(fitter, [(0, 4), (5, 9)])

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 2
    assert (result[0].first_index, result[0].last_index) == (0, 4)
    assert (result[1].first_index, result[1].last_index) == (5, 9)


def test_parallel_but_offset_segments_stay_separate():
    # Same direction (passes the pre-check) but offset by 5 pixels: the combined fit exceeds tolerance
    lower = [[x, 0] for x in range(4)]
    upper = [[x, 5] for x in range(4, 8)]
    seq = np.array(lower + upper, dtype=float)
    fitter = FitterToPointsSequence(seq)
    segments = make_segments(fitter, [(0, 3), (4, 7)])

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 2
    assert (result[0].first_index, result[0].last_index) == (0, 3)
    assert (result[1].first_index, result[1].last_index) == (4, 7)


def test_closed_wraparound_merge():
    # Closed square whose bottom side spans the sequence-start boundary, so the two bottom
    # halves are the last and first segments and can only merge through the wrap-around branch.
    bottom_right = [[x, 0] for x in range(3, 7)]   # indices 0..3
    right = [[6, y] for y in range(1, 7)]          # indices 4..9
    top = [[x, 6] for x in range(5, -1, -1)]       # indices 10..15
    left = [[0, y] for y in range(5, 0, -1)]       # indices 16..20
    bottom_left = [[x, 0] for x in range(0, 3)]    # indices 21..23
    seq = np.array(bottom_right + right + top + left + bottom_left, dtype=float)
    fitter = FitterToPointsSequence(seq, is_closed=True)
    segments = make_segments(fitter, [(0, 3), (4, 9), (10, 15), (16, 20), (21, 23)])

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 4
    # The merged bottom side comes first and wraps past the end of the sequence
    merged = result[0]
    assert merged.first_index == 21
    assert merged.last_index == 3
    assert merged.points_count() == 7
    assert merged.line_segment_params.loss == 0.0
    # The three untouched sides keep their order after the merged one
    assert [(s.first_index, s.last_index) for s in result[1:]] == [(4, 9), (10, 15), (16, 20)]


def test_open_sequence_never_merges_across_the_ends():
    # Open zig-zag: first and last segments are parallel but not adjacent, and an open
    # sequence must not consider the wrap pair at all (limit is n - 1, not n)
    first = [[x, 0] for x in range(4)]
    middle = [[3, y] for y in range(1, 5)]
    last = [[x, 4] for x in range(4, 8)]
    seq = np.array(first + middle + last, dtype=float)
    fitter = FitterToPointsSequence(seq)
    segments = make_segments(fitter, [(0, 3), (4, 7), (8, 11)])

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 3
