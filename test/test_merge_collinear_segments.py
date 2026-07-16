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
    # The merged bottom side wraps past the end of the sequence; a wrap merge appends it at the end of the ring
    merged = result[-1]
    assert merged.first_index == 21
    assert merged.last_index == 3
    assert merged.points_count() == 7
    assert merged.line_segment_params.loss == 0.0
    # The three untouched sides keep their order before the merged one
    assert [(s.first_index, s.last_index) for s in result[:-1]] == [(4, 9), (10, 15), (16, 20)]


def test_wraparound_merge_enables_forward_neighbour_merge():
    # Circular order D -> A -> B -> C: the run D+A+B belongs on one line, but B starts with a leverage
    # point 0.9 px high whose distance to the short A∪B combined fit exceeds tolerance, so (A, B) alone
    # must not merge; only after the wrap-around merge (D, A) does the long flat combined fit bring that
    # point back within tolerance. The walk must therefore re-examine (merged, B) after a wrap merge.
    a = [[0, -0.9], [1, -0.9]]                       # indices 0..1
    b = [[2, 0.9]] + [[x, 0] for x in range(3, 13)]  # indices 2..12, leverage point first
    c = [[13, y] for y in range(1, 6)]               # indices 13..17, vertical: never merges
    d = [[x, 0] for x in range(-10, 0)]              # indices 18..27, wraps around to a
    seq = np.array(a + b + c + d, dtype=float)
    fitter = FitterToPointsSequence(seq, is_closed=True)
    segments = make_segments(fitter, [(0, 1), (2, 12), (13, 17), (18, 27)])

    # fixture premise: the A∪B combined fit alone leaves the leverage point beyond tolerance
    ab_fit = fit_range(fitter._moments, 0, 12)
    assert ab_fit.squared_distances_to_line(seq[0:13]).max() > 1.0

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 2
    merged = result[-1]
    assert (merged.first_index, merged.last_index) == (18, 12)  # D + A + B, wrapping past the end
    assert (result[0].first_index, result[0].last_index) == (13, 17)


def test_short_stub_with_noisy_direction_still_merges():
    # A 2-point stub fitted through jittered points gets a direction ~22 degrees off the line it belongs to.
    # The direction pre-check must be bypassed for such short segments so the combined fit (well within tolerance here) can approve the merge.
    seq = np.array([[0, 0], [1, 0], [2, 0], [3, 0],
                    [4, 0.2], [5, -0.2],
                    [6, 0], [7, 0], [8, 0], [9, 0]], dtype=float)
    fitter = FitterToPointsSequence(seq)
    segments = make_segments(fitter, [(0, 3), (4, 5), (6, 9)])
    stub_direction = segments[1].line_segment_params.direction
    assert abs(float(np.dot(stub_direction, np.array([1.0, 0.0])))) < 0.98  # pre-check would reject

    result = fitter._merge_collinear_segments(segments)

    assert len(result) == 1
    assert result[0].first_index == 0
    assert result[0].last_index == 9


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
