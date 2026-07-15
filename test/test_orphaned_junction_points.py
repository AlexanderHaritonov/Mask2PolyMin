import numpy as np
import pytest

from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.sequence_segment import SequenceSegment, points_count
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig, MIN_SEGMENT_POINTS
from mask2polymin.polyline import segments_to_polyline


def make_params(start, end, loss=0.0):
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)
    direction = (end - start) / np.linalg.norm(end - start)
    return LineSegmentParams(start_point=start, end_point=end, direction=direction, loss=loss)


def junction_gaps(segments, n):
    return [(segments[(i + 1) % len(segments)].first_index - segments[i].last_index - 1) % n
            for i in range(len(segments))]


# ---- unit: the separation search ----

def test_corner_straddling_point_is_orphaned():
    # horizontal y=0 (idx 0-3), corner point 1.5px off both lines (idx 4), vertical x=6 (idx 5-8)
    points = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [4.5, 1.5],
        [6, 3], [6, 4], [6, 5], [6, 6]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0]))
    seg2 = SequenceSegment(points, 4, 8, make_params([6, 3], [6, 6]))

    fitter = FitterToPointsSequence(points)
    last1, first2, _ = fitter.best_consecutive_segments_separation(seg1, seg2)

    # orphaning saves min(2.25, 2.25) = 2.25 at penalty 1.0 => point 4 orphaned
    assert (last1, first2) == (3, 5)


def test_subtolerance_corner_point_is_not_orphaned():
    # like a marching-squares half-pixel chamfer: corner point only 0.5px off seg1's line
    points = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [4, 0.5],
        [5, 2], [5, 3], [5, 4], [5, 5]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0]))
    seg2 = SequenceSegment(points, 4, 8, make_params([5, 2], [5, 5]))

    fitter = FitterToPointsSequence(points)
    last1, first2, _ = fitter.best_consecutive_segments_separation(seg1, seg2)

    # adopting into seg1 costs 0.25 < penalty 1.0 => no orphan
    assert first2 == last1 + 1
    assert last1 == 4


def test_min_segment_guard_blocks_orphaning():
    # tiny 2-point segment whose second point is hostile (far from both lines):
    # orphaning it would be cheapest but must be vetoed by the min-segment guard
    points = np.array([
        [0, 0], [1, 1.5],
        [3, 3], [3, 4], [3, 5]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 1, make_params([0, 0], [1, 0]))   # line y=0
    seg2 = SequenceSegment(points, 2, 4, make_params([3, 3], [3, 5]))   # line x=3

    fitter = FitterToPointsSequence(points)
    last1, first2, _ = fitter.best_consecutive_segments_separation(seg1, seg2)

    assert first2 == last1 + 1  # no orphan despite point 1 being >tolerance from both lines
    n_seg1 = points_count(len(points), seg1.first_index, last1)
    n_seg2 = points_count(len(points), first2, seg2.last_index)
    assert n_seg1 >= MIN_SEGMENT_POINTS
    assert n_seg2 >= MIN_SEGMENT_POINTS


def test_two_point_segment_corner_point_is_orphaned():
    # seg1 has only 2 points, so its half of the contested window is the single point [first..first];
    # the core trim must read those equal indices as one point and fall back to seg1's own line.
    # Under the old full-wrap reading of equal indices the "core" spanned nearly the whole sequence
    # and the corner point (farther than tolerance from both lines) was wrongly adopted into seg1.
    points = np.array([
        [0, 0], [1, 0],
        [1.5, 1.2],
        [3, 2], [3, 3], [3, 4], [3, 5]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 1, make_params([0, 0], [1, 0]))
    seg2 = SequenceSegment(points, 2, 6, make_params([3, 2], [3, 5]))

    fitter = FitterToPointsSequence(points)
    last1, first2, _ = fitter.best_consecutive_segments_separation(seg1, seg2)

    # orphaning saves min(1.2², 1.5²) = 1.44 at penalty 1.0 => point 2 orphaned
    assert (last1, first2) == (1, 3)


# ---- end to end ----

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


TRUE_CORNERS = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=np.float64)


def vertex_errors(segments):
    poly = segments_to_polyline(segments, is_closed=True)
    vertices = poly[:-1]
    return [np.linalg.norm(vertices - c, axis=1).min() for c in TRUE_CORNERS]


def test_displaced_corners_end_to_end():
    pts = square_contour(corner_displacement=1.5)
    fitter = FitterToPointsSequence(pts, is_closed=True, config=FitterConfig(tolerance=1.0))
    _, segments = fitter.fit()

    assert len(segments) == 4
    gaps = junction_gaps(segments, len(fitter.whole_sequence))
    assert all(g <= fitter.config.max_orphans_per_junction for g in gaps)
    assert sum(gaps) >= 1  # orphaning fires at least at one displaced corner
    assert all(err <= 1.5 for err in vertex_errors(segments))


def test_displaced_corners_all_orphaned_strict():
    pts = square_contour(corner_displacement=1.5)
    fitter = FitterToPointsSequence(pts, is_closed=True, config=FitterConfig(tolerance=1.0))
    _, segments = fitter.fit()

    assert len(segments) == 4
    gaps = junction_gaps(segments, len(fitter.whole_sequence))
    assert all(g >= 1 for g in gaps)  # every displaced corner orphaned
    for seg in segments:  # every fitted line axis-aligned like its side
        assert np.abs(seg.line_segment_params.direction).max() >= 0.999
    assert all(err <= 0.3 for err in vertex_errors(segments))


def test_clean_square_has_no_orphans():
    pts = square_contour(corner_displacement=0.0)
    fitter = FitterToPointsSequence(pts, is_closed=True, config=FitterConfig(tolerance=1.0))
    _, segments = fitter.fit()

    assert len(segments) == 4
    assert all(g == 0 for g in junction_gaps(segments, len(fitter.whole_sequence)))
    assert all(err <= 1e-6 for err in vertex_errors(segments))
