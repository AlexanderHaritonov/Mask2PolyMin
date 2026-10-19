"""refit(): fit again, starting from the segmentation of a previous FitResult."""
import numpy as np
import pytest

from mask2polymin import FitResult
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig
from mask2polymin.sequence_moments import SequenceMoments, fit_range, subsequence
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
