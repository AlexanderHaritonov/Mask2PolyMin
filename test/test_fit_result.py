import numpy as np
import pytest

from mask2polymin import FitResult
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig
from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.sequence_moments import SequenceMoments, fit_range
from mask2polymin.sequence_segment import SequenceSegment


def make_params(start, end, loss=0.0):
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)
    direction = (end - start) / np.linalg.norm(end - start)
    return LineSegmentParams(start_point=start, end_point=end, direction=direction, loss=loss)


def zigzag(count):
    """points along x, alternating between y=0 and y=1"""
    return np.array([[x, x % 2] for x in range(count)], dtype=np.float64)


def square_contour(side=6, corner_displacement=0.0):
    """closed 1-px-step square contour; corners optionally displaced diagonally outward"""
    pts = []
    for x in range(0, side): pts.append([x, 0])
    for y in range(0, side): pts.append([side, y])
    for x in range(side, 0, -1): pts.append([x, side])
    for y in range(side, 0, -1): pts.append([0, y])
    pts = np.array(pts, dtype=np.float64)
    d = corner_displacement / np.sqrt(2.0)
    for i, s in [(0, [-d, -d]), (side, [d, -d]), (2 * side, [d, d]), (3 * side, [-d, d])]:
        pts[i] += np.array(s)
    return pts


def junction_gaps(segments, n):
    return [(segments[(i + 1) % len(segments)].first_index - segments[i].last_index - 1) % n
            for i in range(len(segments))]


# ---- unpacking ----

def test_fit_returns_fit_result_that_unpacks_as_polygon_segments_energy():
    result = FitterToPointsSequence().fit(square_contour(), is_closed=True)
    assert isinstance(result, FitResult)

    polygon, segments, energy = result
    assert polygon is result.polygon
    assert segments is result.segments
    assert energy == result.energy
    assert len(result) == 3


# ---- sse ----

def test_sse_against_given_line():
    points = zigzag(6)
    segment = SequenceSegment(points, 0, 5, make_params([0, 0.5], [5, 0.5]))
    result = FitResult(np.empty((0, 2)), [segment], np.nan)

    # every point is 0.5 off the line y=0.5
    assert result.sse() == pytest.approx(6 * 0.25)


def test_sse_of_frozen_segment_uses_frozen_line():
    # pairs at y=0 and y=1 for each x: the TLS line is exactly y=0.5
    points = np.array([[x, y] for x in range(3) for y in (0, 1)], dtype=np.float64)
    fitted = SequenceSegment(points, 0, 5, fit_range(SequenceMoments(points), 0, 5, with_endpoints=False))
    frozen = SequenceSegment(points, 0, 5, fit_range(SequenceMoments(points), 0, 5, with_endpoints=False))
    frozen.freeze([0, 0], [1, 0])

    # frozen line y=0: the three points at y=1 count 1 each, not the fitted line's 0.25
    assert FitResult(np.empty((0, 2)), [frozen], np.nan).sse() == pytest.approx(3.0)
    assert FitResult(np.empty((0, 2)), [fitted], np.nan).sse() == pytest.approx(6 * 0.25)


def test_sse_covers_wrapping_segment():
    points = zigzag(8)
    # wraps: indices 6, 7, 0, 1
    wrapping = SequenceSegment(points, 6, 1, make_params([0, 0], [7, 0]))
    inner = SequenceSegment(points, 2, 5, make_params([0, 0], [7, 0]))
    result = FitResult(np.empty((0, 2)), [inner, wrapping], np.nan)

    # odd indices sit at y=1: 3 and 5 in inner, 7 and 1 in wrapping
    assert result.sse() == pytest.approx(4.0)


def test_sse_after_fit_matches_segment_losses():
    pts = square_contour(corner_displacement=1.5)
    result = FitterToPointsSequence(config=FitterConfig(tolerance=1.0)).fit(pts, is_closed=True)

    # loss is n * eig_min, the SSE to the TLS line
    expected = sum(s.line_segment_params.loss for s in result.segments)
    assert result.sse() == pytest.approx(expected)


# ---- orphans_count ----

def test_orphans_count_of_given_segments():
    points = zigzag(10)
    segments = [
        SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0])),
        SequenceSegment(points, 5, 7, make_params([5, 0], [7, 0])),
    ]
    # index 4 and indices 8, 9 belong to no segment
    assert FitResult(np.empty((0, 2)), segments, np.nan).orphans_count() == 3


def test_orphans_count_after_fit():
    pts = square_contour(corner_displacement=1.5)
    fitter = FitterToPointsSequence(config=FitterConfig(tolerance=1.0))
    result = fitter.fit(pts, is_closed=True)

    gaps = junction_gaps(result.segments, len(fitter.whole_sequence))
    assert result.orphans_count() == sum(gaps)
    assert result.orphans_count() >= 1


def test_clean_square_has_no_orphans():
    result = FitterToPointsSequence(config=FitterConfig(tolerance=1.0)).fit(square_contour(), is_closed=True)
    assert result.orphans_count() == 0


# ---- energy ----

def test_energy_adds_sse_orphans_and_segments():
    # tolerance 1.5, so tolerance and tolerance² differ
    config = FitterConfig(tolerance=1.5, segment_cost=2.0)
    result = FitterToPointsSequence(config).fit(square_contour(corner_displacement=2.5), is_closed=True)
    assert result.orphans_count() >= 1

    expected = result.sse() + 1.5 ** 2 * result.orphans_count() + 2.0 * 1.5 ** 2 * len(result.segments)
    assert result.energy == pytest.approx(expected)


# ---- SequenceSegment.frozen ----

def test_frozen_defaults_to_false_and_clone_copies_it():
    points = zigzag(6)
    segment = SequenceSegment(points, 0, 5, make_params([0, 0], [5, 0]))
    assert segment.frozen is False

    segment.frozen = True
    assert segment.clone().frozen is True


def test_freeze_sets_line_and_flag():
    points = zigzag(6)
    segment = SequenceSegment(points, 0, 5, make_params([0, 0], [5, 0]))
    segment.freeze([0, 1], [3, 0])  # direction need not be unit length

    assert segment.frozen is True
    np.testing.assert_allclose(segment.line_segment_params.start_point, [0, 1])
    np.testing.assert_allclose(segment.line_segment_params.direction, [1, 0])
    assert (segment.first_index, segment.last_index) == (0, 5)


def test_fit_returns_no_frozen_segments():
    result = FitterToPointsSequence().fit(square_contour(), is_closed=True)
    assert not any(s.frozen for s in result.segments)
