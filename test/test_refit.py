"""refit(): fit again, starting from the segmentation of a previous FitResult."""
import numpy as np
import pytest

from mask2polymin import FitResult
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig
from mask2polymin.sequence_moments import SequenceMoments, fit_range, refit_segment, subsequence
from mask2polymin.sequence_segment import SequenceSegment


def square_contour(side):
    """closed 1-px-step square contour; corner k sits at index k * side"""
    pts = []
    for x in range(0, side): pts.append([x, 0])
    for y in range(0, side): pts.append([side, y])
    for x in range(side, 0, -1): pts.append([x, side])
    for y in range(side, 0, -1): pts.append([0, y])
    return np.array(pts, dtype=np.float64)


def noisy_rectangle_contour(rng, sigma):
    """closed 1-px-step rectangle contour of random size and rotation, with gaussian jitter"""
    width, height = int(rng.integers(8, 25)), int(rng.integers(8, 25))
    pts = []
    for x in range(0, width): pts.append([x, 0])
    for y in range(0, height): pts.append([width, y])
    for x in range(width, 0, -1): pts.append([x, height])
    for y in range(height, 0, -1): pts.append([0, y])
    pts = np.array(pts, dtype=np.float64)
    angle = rng.uniform(0, np.pi / 2)
    c, s = np.cos(angle), np.sin(angle)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    return pts + rng.normal(0, sigma, pts.shape)


def noisy_l_shape(rng, sigma):
    """open contour: two perpendicular 40-point arms, with gaussian jitter"""
    n = 40
    horizontal = np.stack([np.arange(n), np.zeros(n)], axis=1)
    vertical = np.stack([np.full(n, n - 1.0), np.arange(1, n + 1)], axis=1)
    pts = np.vstack([horizontal, vertical])
    return pts + rng.normal(0, sigma, pts.shape)


def misplaced_junctions(points, side, shift):
    """the 4 sides of square_contour(side) as fitted segments, every junction `shift` points past its corner"""
    n = len(points)
    moments = SequenceMoments(points)
    segments = []
    for k in range(4):
        first, last = (k * side + shift) % n, ((k + 1) * side + shift - 1) % n
        segments.append(SequenceSegment(points, first, last, fit_range(moments, first, last, with_endpoints=True)))
    return segments


def five_segments_with_gaps(points, frozen_indices):
    """5 segments of 7 points over the 40 points of square_contour(10), one orphan after each: 7, 15, 23, 31, 39"""
    moments = SequenceMoments(points)
    segments = []
    for k in range(5):
        first, last = 8 * k, 8 * k + 6
        segments.append(SequenceSegment(points, first, last, fit_range(moments, first, last, with_endpoints=False),
                                        frozen=k in frozen_indices))
    return segments


def segments_from_ranges(points, ranges, frozen_indices=()):
    """fitted segments over the given index ranges; those in frozen_indices are marked frozen, with their fitted line"""
    moments = SequenceMoments(points)
    return [SequenceSegment(points, first, last, fit_range(moments, first, last, with_endpoints=False), frozen=k in frozen_indices)
            for k, (first, last) in enumerate(ranges)]


def index_ranges(segments):
    return [(s.first_index, s.last_index) for s in segments]


# ---- length check ----

def test_refit_contour_of_different_length_raises():
    fitter = FitterToPointsSequence()
    previous = fitter.fit(square_contour(8), is_closed=True)
    with pytest.raises(ValueError, match="got 40 points .* previous has 32"):
        fitter.refit(square_contour(10), previous, is_closed=True)


def test_refit_length_check_ignores_closing_duplicate():
    points = square_contour(8)
    fitter = FitterToPointsSequence()
    previous = fitter.fit(points, is_closed=True)

    result = fitter.refit(np.vstack([points, points[:1]]), previous, is_closed=True)
    assert len(result.segments[0].whole_sequence) == len(points)


# ---- previous stays intact ----

def test_refit_leaves_previous_intact():
    points = square_contour(8)
    previous = FitResult(np.empty((0, 2)), misplaced_junctions(points, side=8, shift=3))
    snapshot = [(s.first_index, s.last_index, s.line_segment_params, s.whole_sequence) for s in previous.segments]

    same_points = points.copy()
    result = FitterToPointsSequence(config=FitterConfig(tolerance=1.0)).refit(same_points, previous, is_closed=True)

    # refit() did move the junctions, on its own segments only
    assert index_ranges(result.segments) != index_ranges(previous.segments)
    assert not any(r is p for r in result.segments for p in previous.segments)
    assert all(s.whole_sequence is same_points for s in result.segments)

    for segment, (first_index, last_index, params, whole_sequence) in zip(previous.segments, snapshot):
        assert (segment.first_index, segment.last_index) == (first_index, last_index)
        assert segment.line_segment_params is params
        assert segment.whole_sequence is whole_sequence


# ---- nothing frozen ----

@pytest.mark.parametrize("is_closed", [True, False])
@pytest.mark.parametrize("seed", range(20))
def test_refit_of_fit_result_with_nothing_frozen_is_not_worse_than_fit(seed, is_closed):
    # Mild noise only: from sigma ~0.35 on, refit() sometimes splits fit()'s merged result again
    # and ends with a higher sse() (a few percent of 200 seeds per shape at 0.35, up to ~10% at 0.7).
    rng = np.random.default_rng(seed)
    points = noisy_rectangle_contour(rng, sigma=0.2) if is_closed else noisy_l_shape(rng, sigma=0.2)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))

    fitted = fitter.fit(points, is_closed=is_closed)
    refitted = fitter.refit(points, fitted, is_closed=is_closed)
    assert refitted.sse() <= fitted.sse() + 1e-9


@pytest.mark.parametrize("side", [8, 10, 12])
def test_refit_from_misplaced_junctions_is_not_worse_than_fit(side):
    # shift 3 puts the start out of tolerance, so the split loop runs.
    # With shift 2 the start is within tolerance and comes back unchanged: junctions move only after a split.
    points = square_contour(side)
    previous = FitResult(np.empty((0, 2)), misplaced_junctions(points, side, shift=3))
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))

    fitted = fitter.fit(points, is_closed=True)
    refitted = fitter.refit(points, previous, is_closed=True)
    assert refitted.sse() <= fitted.sse() + 1e-9
    assert len(refitted.segments) == len(fitted.segments) == 4


# ---- frozen segments ----

def test_refit_keeps_a_frozen_line_and_computes_loss_and_endpoints_against_it():
    points = square_contour(10)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    previous = fitter.fit(points, is_closed=True)

    # close to the fitted line, but not the TLS fit of any points: shifted by 0.3 and turned by 2°
    fitted_line = previous.segments[1].line_segment_params
    c, s = np.cos(np.radians(2)), np.sin(np.radians(2))
    direction = np.array([c * fitted_line.direction[0] - s * fitted_line.direction[1],
                          s * fitted_line.direction[0] + c * fitted_line.direction[1]])
    point = fitted_line.start_point + 0.3 * np.array([-fitted_line.direction[1], fitted_line.direction[0]])
    previous.segments[1].freeze(point, direction)
    frozen_line = previous.segments[1].line_segment_params

    result = fitter.refit(points, previous, is_closed=True)

    [segment] = [s for s in result.segments if s.frozen]
    params = segment.line_segment_params
    np.testing.assert_array_equal(params.direction, frozen_line.direction)
    assert params.squared_distances_to_line(point[None])[0] == pytest.approx(0.0, abs=1e-12)

    segment_points = subsequence(points, segment.first_index, segment.last_index)
    assert params.loss == pytest.approx(frozen_line.squared_distances_to_line(segment_points).sum())
    projections = (segment_points - point) @ frozen_line.direction
    np.testing.assert_allclose(params.start_point, point + projections.min() * frozen_line.direction, atol=1e-9)
    np.testing.assert_allclose(params.end_point, point + projections.max() * frozen_line.direction, atol=1e-9)


def test_fit_computes_the_loss_of_frozen_segments_against_their_line():
    points = square_contour(10)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    segments = fitter.fit(points, is_closed=True).segments

    # shifted by 0.3: within tolerance, so _fit() returns without splitting
    fitted_line = segments[1].line_segment_params
    point = fitted_line.start_point + 0.3 * np.array([-fitted_line.direction[1], fitted_line.direction[0]])
    segments[1].freeze(point, fitted_line.direction)
    frozen_line = segments[1].line_segment_params
    assert frozen_line.loss == 0.0  # freeze() sets only the line

    fitter._prepare(points, is_closed=True)
    result = fitter._fit(segments)

    [segment] = [s for s in result if s.frozen]
    params = segment.line_segment_params
    np.testing.assert_array_equal(params.direction, frozen_line.direction)
    assert params.squared_distances_to_line(point[None])[0] == pytest.approx(0.0, abs=1e-12)
    segment_points = subsequence(points, segment.first_index, segment.last_index)
    assert params.loss == pytest.approx(frozen_line.squared_distances_to_line(segment_points).sum())


@pytest.mark.parametrize("is_closed, orphans_counted", [(True, 3), (False, 4)])
def test_sse_per_segment_counts_free_segments_and_orphans_next_to_them_only(is_closed, orphans_counted):
    # Segments 0, 1 and 4 are frozen. Orphan 7 lies between the frozen 0 and 1, so it never counts.
    # Orphan 39 lies between the frozen 4 and 0 when closed, and after the last segment when open.
    points = square_contour(10)
    segments = five_segments_with_gaps(points, frozen_indices={0, 1, 4})
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitter._prepare(points, is_closed=is_closed)

    free = segments[2:4]
    expected = (sum(s.line_segment_params.loss * s.points_count() for s in free) / sum(s.points_count() for s in free)
                + orphans_counted * 1.0 / len(free))
    assert fitter._sse_per_segment(segments) == pytest.approx(expected)


def test_sse_per_segment_with_everything_frozen_is_zero():
    points = square_contour(10)
    segments = five_segments_with_gaps(points, frozen_indices=set(range(5)))
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitter._prepare(points, is_closed=True)
    assert fitter._sse_per_segment(segments) == 0.0


def test_fit_does_not_split_when_only_a_frozen_segment_is_out_of_tolerance():
    # Side 1 is frozen 2 px off: out of tolerance. Sides 0 and 2 are frozen to their own lines,
    # so the junctions of side 1 do not move. The free side 3 fits exactly.
    points = square_contour(10)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    segments = fitter.fit(points, is_closed=True).segments
    freeze_shifted(segments[0], 0.0)
    freeze_shifted(segments[1], 2.0)
    freeze_shifted(segments[2], 0.0)
    frozen_line = segments[1].line_segment_params
    range_before = (segments[1].first_index, segments[1].last_index)

    fitter._prepare(points, is_closed=True)
    result = fitter._fit(segments)

    assert len(result) == 4
    assert (result[1].first_index, result[1].last_index) == range_before
    np.testing.assert_array_equal(result[1].line_segment_params.direction, frozen_line.direction)
    assert result[1].line_segment_params.loss / result[1].points_count() > 1.0


def test_adjust_segmentation_keeps_a_frozen_line_when_its_junctions_move():
    points = square_contour(10)
    segments = misplaced_junctions(points, side=10, shift=3)
    # Segment 1 covers 13..22: the side x = 10 from y = 3 on, and 3 points past the corner.
    # It is frozen to a line near that side (shifted by 0.3, turned by 2°), not the TLS fit of any of its points.
    point = np.array([10.3, 5.0])
    segments[1].freeze(point, [-np.sin(np.radians(2)), np.cos(np.radians(2))])
    frozen_line = segments[1].line_segment_params

    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitter._prepare(points, is_closed=True)
    refit_segment(fitter._moments, segments[1], with_endpoints=False)  # as _fit() does first
    range_before = (segments[1].first_index, segments[1].last_index)

    fitter._adjust_segmentation(segments, 0)

    segment = segments[1]
    assert (segment.first_index, segment.last_index) != range_before
    params = segment.line_segment_params
    np.testing.assert_array_equal(params.direction, frozen_line.direction)
    assert params.squared_distances_to_line(point[None])[0] == pytest.approx(0.0, abs=1e-12)
    segment_points = subsequence(points, segment.first_index, segment.last_index)
    assert params.loss == pytest.approx(frozen_line.squared_distances_to_line(segment_points).sum())


@pytest.mark.parametrize("frozen_is_first, other_range, direction, expected_frozen_range", [
    (True, (20, 29), [0.8 / 4.5, 1.0], (10, 18)),   # x = 11.1 at y = 9: the frozen side gives up (10, 9)
    (False, (0, 9), [-0.8 / 3.5, 1.0], (12, 19)),   # x = 11.1 at y = 1: the frozen side gives up (10, 1) and the corner (10, 0)
])
def test_separation_scores_a_frozen_segment_against_its_own_line(frozen_is_first, other_range, direction, expected_frozen_range):
    # Side 1 (x = 10, indices 10..19) is frozen to a tilted line, within tolerance on average.
    # One point near the junction is ~1.08 px from that line and 1 px from the other line, so it leaves the frozen segment.
    # Scored against the core half (on x = 10), it would stay.
    # Only the frozen end is checked: the point may join the other segment or become an orphan.
    points = square_contour(10)
    moments = SequenceMoments(points)
    frozen = SequenceSegment(points, 10, 19, fit_range(moments, 10, 19, with_endpoints=False))
    frozen.freeze([10.3, 4.5], direction)
    other = SequenceSegment(points, *other_range, fit_range(moments, *other_range, with_endpoints=False))

    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitter._prepare(points, is_closed=True)
    refit_segment(fitter._moments, frozen, with_endpoints=False)  # as _fit() does first

    if frozen_is_first:
        new_last, _, _ = fitter._best_consecutive_segments_separation(frozen, other)
        assert (frozen.first_index, new_last) == expected_frozen_range
    else:
        _, new_first, _ = fitter._best_consecutive_segments_separation(other, frozen)
        assert (new_first, frozen.last_index) == expected_frozen_range


def test_separation_allows_orphans_next_to_a_frozen_line_out_of_tolerance():
    # Side 1 (x = 10, indices 10..19) is frozen to x = 11.5, out of tolerance on average.
    # Side 2 (y = 10, indices 20..29) is free and fits exactly.
    # Orphans are still allowed: (10, 7) and (10, 8) become orphans, (10, 9) goes to side 2.
    # If the frozen line counted, there would be no orphans: (18, 19).
    points = square_contour(10)
    moments = SequenceMoments(points)
    frozen = SequenceSegment(points, 10, 19, fit_range(moments, 10, 19, with_endpoints=False))
    frozen.freeze([11.5, 0.0], [0.0, 1.0])
    free = SequenceSegment(points, 20, 29, fit_range(moments, 20, 29, with_endpoints=False))

    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitter._prepare(points, is_closed=True)
    refit_segment(fitter._moments, frozen, with_endpoints=False)  # as _fit() does first
    assert frozen.line_segment_params.loss / frozen.points_count() > 1.0

    new_last, new_first, _ = fitter._best_consecutive_segments_separation(frozen, free)
    assert (new_last, new_first) == (16, 19)


# Segment 0 covers the corner at index 10: side 0 and the first 5 points of side 1.
CORNER_INSIDE_FIRST_SEGMENT = [(0, 14), (15, 19), (20, 29), (30, 39)]


@pytest.mark.parametrize("frozen_indices, expected", [((), 0), ((0,), None)])
def test_choose_segment_index_for_split_skips_frozen_segments(frozen_indices, expected):
    points = square_contour(10)
    segments = segments_from_ranges(points, CORNER_INSIDE_FIRST_SEGMENT, frozen_indices)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitter._prepare(points, is_closed=True)
    assert fitter._choose_segment_index_for_split(segments) == expected


@pytest.mark.parametrize("frozen_indices, expected_count", [((), 4), ((0,), 5), ((1,), 5), ((0, 1), 5)])
def test_merge_collinear_segments_skips_frozen_segments(frozen_indices, expected_count):
    # side 0 is cut into two collinear halves: 0..4 and 5..9
    points = square_contour(10)
    segments = segments_from_ranges(points, [(0, 4), (5, 9), (10, 19), (20, 29), (30, 39)], frozen_indices)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitter._prepare(points, is_closed=True)

    merged = fitter._merge_collinear_segments(segments)
    assert len(merged) == expected_count
    assert sum(s.frozen for s in merged) == len(frozen_indices)


def freeze_shifted(segment, shift):
    """freeze the segment to its current line, moved sideways by `shift`"""
    line = segment.line_segment_params
    segment.freeze(line.start_point + shift * np.array([-line.direction[1], line.direction[0]]), line.direction)


def matching_line_index(params, lines):
    """index of the line in `lines` that `params` lies on: same direction, and the line's point lies on it"""
    [k] = [k for k, line in enumerate(lines)
           if np.array_equal(params.direction, line.direction)
           and params.squared_distances_to_line(line.start_point[None])[0] == pytest.approx(0.0, abs=1e-12)]
    return k


def test_refit_with_everything_frozen_keeps_segments_and_lines():
    # Side 0 is cut into two collinear halves. All 5 segments are frozen to their lines, moved by 0.1 to 0.5,
    # so no two frozen lines are the same.
    points = square_contour(10)
    segments = segments_from_ranges(points, [(0, 4), (5, 9), (10, 19), (20, 29), (30, 39)])
    for k, segment in enumerate(segments):
        freeze_shifted(segment, 0.1 * (k + 1))
    frozen_lines = [s.line_segment_params for s in segments]

    result = FitterToPointsSequence(config=FitterConfig(tolerance=1.0)).refit(
        points, FitResult(np.empty((0, 2)), segments), is_closed=True)

    assert all(s.frozen for s in result.segments)
    assert [matching_line_index(s.line_segment_params, frozen_lines) for s in result.segments] == [0, 1, 2, 3, 4]


def test_refit_never_splits_or_merges_frozen_segments_and_keeps_their_order():
    # 0..4 (side 0) frozen to y = 0; the free 5..9 is collinear with it.
    # 10..24 (side 1 and 5 points of side 2) frozen to x = 10.
    # The free 25..32 covers the corner at index 30.
    # 33..39 (side 3) frozen to x = 0.
    points = square_contour(10)
    segments = segments_from_ranges(points, [(0, 4), (5, 9), (10, 24), (25, 32), (33, 39)])
    segments[0].freeze([0.0, 0.0], [1.0, 0.0])
    segments[2].freeze([10.0, 0.0], [0.0, 1.0])
    segments[4].freeze([0.0, 0.0], [0.0, 1.0])
    frozen_lines = [s.line_segment_params for s in segments if s.frozen]

    result = FitterToPointsSequence(config=FitterConfig(tolerance=1.0)).refit(
        points, FitResult(np.empty((0, 2)), segments), is_closed=True)

    order = [matching_line_index(s.line_segment_params, frozen_lines) for s in result.segments if s.frozen]
    assert order in ([0, 1, 2], [1, 2, 0], [2, 0, 1])  # the same cyclic order


def test_refit_moves_the_junctions_next_to_a_moved_frozen_line():
    # Side 1 (x = 10, indices 10..19) is frozen to a tilted line, within tolerance on average, but 1.1 px off at y = 1.
    # The free segments fit exactly, so no split happens: only the adjust pass moves the junction.
    points = square_contour(10)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    previous = fitter.fit(points, is_closed=True)
    previous.segments[1].freeze([10.3, 4.5], [-0.8 / 3.5, 1.0])

    result = fitter.refit(points, previous, is_closed=True)

    assert len(result.segments) == 4
    [segment] = [s for s in result.segments if s.frozen]
    assert segment.first_index > previous.segments[1].first_index


@pytest.mark.parametrize("second_point, second_direction", [
    ([0.0, 0.0], [1.0, 0.0]),                                     # the same line
    ([0.0, 0.1], [np.cos(np.radians(0.5)), np.sin(np.radians(0.5))]),   # almost the same line
])
def test_junction_between_two_frozen_segments_does_not_move(second_point, second_direction):
    # Side 0 of a 20-square is cut at 9|10, and both halves are frozen to (almost) the same line.
    # The free 40..79 covers two corners, so _fit() splits and adjusts every junction.
    points = square_contour(20)
    segments = segments_from_ranges(points, [(0, 9), (10, 19), (20, 39), (40, 79)])
    segments[0].freeze([0.0, 0.0], [1.0, 0.0])
    segments[1].freeze(second_point, second_direction)

    result = FitterToPointsSequence(config=FitterConfig(tolerance=1.0)).refit(
        points, FitResult(np.empty((0, 2)), segments), is_closed=True)

    [first, second] = [s for s in result.segments if s.frozen]
    assert (first.last_index, second.first_index) == (9, 10)


# ---- keys of frozen segments ----

@pytest.mark.parametrize("ranges, frozen_index, expected_index", [
    ([(0, 19), (20, 29), (30, 39)], 1, 2),                       # the free 0..19 covers a corner: split before the frozen segment
    ([(0, 4), (5, 9), (10, 19), (20, 29), (30, 39)], 3, 2),      # side 0 cut in two: merge before the frozen segment
    ([(5, 9), (10, 19), (20, 29), (30, 39), (0, 4)], 2, 1),      # merge of the last and the first segment
])
def test_refit_keeps_the_key_of_a_frozen_segment_wherever_it_moves(ranges, frozen_index, expected_index):
    # side 2 (indices 20..29) frozen to its own line y = 10
    points = square_contour(10)
    segments = segments_from_ranges(points, ranges)
    segments[frozen_index].freeze([0.0, 10.0], [1.0, 0.0], key="edge")
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))

    result = fitter.refit(points, FitResult(np.empty((0, 2)), segments), is_closed=True)
    edge = result.frozen["edge"]
    assert edge is result.segments[expected_index]
    assert edge.line_segment_params.start_point[1] == pytest.approx(10.0)  # still on y = 10; its junctions may move

    # and in the next round
    assert fitter.refit(points, result, is_closed=True).frozen["edge"].key == "edge"


def test_frozen_holds_only_the_frozen_segments_with_a_key():
    points = square_contour(10)
    segments = segments_from_ranges(points, [(0, 9), (10, 19), (20, 29), (30, 39)], frozen_indices=(0,))
    segments[2].freeze([0.0, 10.0], [1.0, 0.0], key=("left", "right"))
    segments[3].key = "free"

    assert FitResult(np.empty((0, 2)), segments).frozen == {("left", "right"): segments[2]}


def test_freeze_again_without_key_keeps_the_key():
    points = square_contour(10)
    [segment] = segments_from_ranges(points, [(20, 29)])
    segment.freeze([0.0, 10.0], [1.0, 0.0], key="edge")
    segment.freeze([0.0, 10.5], [1.0, 0.0])
    assert segment.key == "edge"
    assert segment.clone().key == "edge"


def test_refit_with_two_frozen_segments_of_the_same_key_raises():
    points = square_contour(10)
    segments = segments_from_ranges(points, [(0, 9), (10, 19), (20, 29), (30, 39)])
    segments[0].freeze([0.0, 0.0], [1.0, 0.0], key="edge")
    segments[2].freeze([0.0, 10.0], [1.0, 0.0], key="edge")
    with pytest.raises(ValueError, match="unique"):
        FitterToPointsSequence().refit(points, FitResult(np.empty((0, 2)), segments), is_closed=True)


def test_guard_keeps_the_key_when_it_returns_the_start(monkeypatch):
    # _fit() returns the 4 sides, free, with every junction 3 points past its corner: a higher Energy
    points = square_contour(10)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    fitted = fitter.fit(points, is_closed=True)
    fitted.segments[2].freeze([0.0, 10.0], [1.0, 0.0], key="edge")
    monkeypatch.setattr(fitter, "_fit", lambda segments: misplaced_junctions(points, side=10, shift=3))

    result = fitter.refit(points, fitted, is_closed=True)
    assert index_ranges([result.frozen["edge"]]) == index_ranges([fitted.segments[2]])


# ---- guard ----

def energy(result, config):
    return result.energy(config.tolerance, config.segment_cost)


@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("frozen_shift", [None, 0.5])
@pytest.mark.parametrize("is_closed", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_refit_never_raises_the_energy(seed, is_closed, frozen_shift, guard):
    # sigma 0.7. With frozen_shift, every other segment is frozen to its line, moved by that much.
    # Without the guard too: E decides every split, merge and junction move.
    rng = np.random.default_rng(seed)
    points = noisy_rectangle_contour(rng, sigma=0.7) if is_closed else noisy_l_shape(rng, sigma=0.7)
    config = FitterConfig(tolerance=1.0)
    fitter = FitterToPointsSequence(config=config)
    previous = fitter.fit(points, is_closed=is_closed)
    if frozen_shift is not None:
        for segment in previous.segments[::2]:
            freeze_shifted(segment, frozen_shift)

    result = fitter.refit(points, previous, is_closed=is_closed, guard=guard)
    assert energy(result, config) <= energy(previous, config) + 1e-9


def test_guard_returns_the_start_when_refit_raises_the_energy(monkeypatch):
    # _fit() returns the 4 sides with every junction 3 points past its corner.
    points = square_contour(10)
    config = FitterConfig(tolerance=1.0)
    fitter = FitterToPointsSequence(config=config)
    fitted = fitter.fit(points, is_closed=True)
    monkeypatch.setattr(fitter, "_fit", lambda segments: misplaced_junctions(points, side=10, shift=3))

    unguarded = fitter.refit(points, fitted, is_closed=True, guard=False)
    guarded = fitter.refit(points, fitted, is_closed=True)

    # without the guard: the patched result, with the higher Energy
    assert energy(unguarded, config) > energy(fitted, config)
    assert index_ranges(unguarded.segments) == index_ranges(misplaced_junctions(points, side=10, shift=3))
    # with the guard: the start, as new segments
    assert index_ranges(guarded.segments) == index_ranges(fitted.segments)
    assert energy(guarded, config) == pytest.approx(energy(fitted, config))
    assert not any(g is f for g in guarded.segments for f in fitted.segments)


# ---- E decides ----

def noisy_fit_with_frozen_lines(seed, is_closed, frozen_shift):
    """fit() of a noisy contour (sigma 0.7); with frozen_shift, every other segment is frozen to its line, moved by that much"""
    rng = np.random.default_rng(seed)
    points = noisy_rectangle_contour(rng, sigma=0.7) if is_closed else noisy_l_shape(rng, sigma=0.7)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    segments = fitter.fit(points, is_closed=is_closed).segments
    if frozen_shift is not None:
        for segment in segments[::2]:
            freeze_shifted(segment, frozen_shift)
            refit_segment(fitter._moments, segment, with_endpoints=False)  # as _fit() does first
    return fitter, segments


def segments_energy(segments, config):
    return energy(FitResult(np.empty((0, 2)), segments), config)


@pytest.mark.parametrize("frozen_shift", [None, 0.5])
@pytest.mark.parametrize("is_closed", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_adjust_segmentation_never_raises_the_energy(seed, is_closed, frozen_shift):
    fitter, segments = noisy_fit_with_frozen_lines(seed, is_closed, frozen_shift)
    start = segments_energy(segments, fitter.config)

    fitter._adjust_segmentation(segments, 0)
    assert segments_energy(segments, fitter.config) <= start + 1e-9


@pytest.mark.parametrize("frozen_shift", [None, 0.5])
@pytest.mark.parametrize("is_closed", [True, False])
@pytest.mark.parametrize("seed", range(40))
def test_split_loop_never_raises_the_energy(seed, is_closed, frozen_shift):
    fitter, segments = noisy_fit_with_frozen_lines(seed, is_closed, frozen_shift)
    start = segments_energy(segments, fitter.config)

    result = fitter._fit(segments)
    assert segments_energy(result, fitter.config) <= start + 1e-9
