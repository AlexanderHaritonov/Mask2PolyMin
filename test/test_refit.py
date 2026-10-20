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
    points = square_contour(10)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    segments = fitter.fit(points, is_closed=True).segments

    # shifted by 2: out of tolerance, while the free segments fit exactly
    fitted_line = segments[1].line_segment_params
    segments[1].freeze(fitted_line.start_point + 2.0 * np.array([-fitted_line.direction[1], fitted_line.direction[0]]),
                       fitted_line.direction)
    frozen_line = segments[1].line_segment_params

    fitter._prepare(points, is_closed=True)
    result = fitter._fit(segments)

    assert len(result) == 4
    [segment] = [s for s in result if s.frozen]
    np.testing.assert_array_equal(segment.line_segment_params.direction, frozen_line.direction)
    assert segment.line_segment_params.loss / segment.points_count() > 1.0


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
