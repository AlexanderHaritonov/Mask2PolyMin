"""Direct unit tests for segments_to_polyline: endpoint projection, corner reconstruction,
the near-parallel / implausible-intersection fallbacks, and the degenerate cases."""
import numpy as np
import pytest

from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.sequence_segment import SequenceSegment
from mask2polymin.polyline import segments_to_polyline


def make_params(start, end, loss=0.0):
    start = np.array(start, dtype=np.float64)
    end = np.array(end, dtype=np.float64)
    direction = (end - start) / np.linalg.norm(end - start)
    return LineSegmentParams(start_point=start, end_point=end, direction=direction, loss=loss)


def test_empty_segments_raises():
    with pytest.raises(ValueError):
        segments_to_polyline([], is_closed=False)


def test_single_segment_open_endpoints_are_projections():
    # first and last input points lie off the fitted line; the polyline ends at their projections
    points = np.array([[0, 0.4], [1, 0], [2, 0], [3, -0.4]], dtype=np.float64)
    seg = SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0]))
    poly = segments_to_polyline([seg], is_closed=False)
    np.testing.assert_allclose(poly, [[0, 0], [3, 0]], atol=1e-9)


def test_single_segment_closed_degenerate():
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    seg = SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0]))
    poly = segments_to_polyline([seg], is_closed=True)
    # degenerate single-line "polygon", still closed per the contract (first point equals last)
    np.testing.assert_allclose(poly, [[0, 0], [3, 0], [0, 0]], atol=1e-9)


def test_open_two_segments_corner_is_line_intersection():
    points = np.array([
        [0, 0.4], [1, 0], [2, 0], [3, 0],
        [4, 1], [4, 2], [4, 3], [4, 4]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0]))   # line y=0
    seg2 = SequenceSegment(points, 4, 7, make_params([4, 1], [4, 4]))   # line x=4
    poly = segments_to_polyline([seg1, seg2], is_closed=False)
    # the corner (4, 0) is the line intersection — no input point lies there (sub-pixel reconstruction)
    np.testing.assert_allclose(poly, [[0, 0], [4, 0], [4, 4]], atol=1e-9)


def test_exactly_parallel_lines_fall_back_to_projection_midpoint():
    points = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [4, 0.5], [5, 0.5], [6, 0.5], [7, 0.5]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0]))       # y = 0
    seg2 = SequenceSegment(points, 4, 7, make_params([4, 0.5], [7, 0.5]))   # y = 0.5, exactly parallel
    poly = segments_to_polyline([seg1, seg2], is_closed=False)
    # no intersection exists; the vertex is the midpoint of the adjacent input points' projections
    np.testing.assert_allclose(poly[1], [3.5, 0.25], atol=1e-9)


def test_implausibly_far_intersection_falls_back_to_anchor():
    # nearly parallel lines: they do intersect, but ~160 px away from the junction —
    # a genuine corner cannot be that far, so the vertex falls back to the anchor
    points = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [4, 0.5], [5, 0.503], [6, 0.506], [7, 0.509]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 3, make_params([0, 0], [3, 0]))
    seg2 = SequenceSegment(points, 4, 7, make_params([4, 0.5], [7, 0.509]))
    poly = segments_to_polyline([seg1, seg2], is_closed=False, tolerance=1.0)
    np.testing.assert_allclose(poly[1], [3.5, 0.25], atol=1e-9)


def test_orphan_mean_anchors_the_junction():
    # a gap between the segments: the anchor is the orphaned points' mean,
    # and with parallel lines it becomes the vertex itself
    points = np.array([
        [0, 0], [1, 0], [2, 0],
        [3.5, 1.7],
        [5, 3], [6, 3], [7, 3]
    ], dtype=np.float64)
    seg1 = SequenceSegment(points, 0, 2, make_params([0, 0], [2, 0]))   # y = 0
    seg2 = SequenceSegment(points, 4, 6, make_params([5, 3], [7, 3]))   # y = 3, parallel
    poly = segments_to_polyline([seg1, seg2], is_closed=False)
    np.testing.assert_allclose(poly[1], [3.5, 1.7], atol=1e-9)


def test_closed_square_vertices_and_ordering():
    pts = []
    for x in range(0, 3): pts.append([x, 0])
    for y in range(0, 3): pts.append([3, y])
    for x in range(3, 0, -1): pts.append([x, 3])
    for y in range(3, 0, -1): pts.append([0, y])
    points = np.array(pts, dtype=np.float64)
    segments = [
        SequenceSegment(points, 0, 2, make_params([0, 0], [2, 0])),    # bottom, y=0
        SequenceSegment(points, 3, 5, make_params([3, 0], [3, 2])),    # right,  x=3
        SequenceSegment(points, 6, 8, make_params([3, 3], [1, 3])),    # top,    y=3
        SequenceSegment(points, 9, 11, make_params([0, 3], [0, 1])),   # left,   x=0
    ]
    poly = segments_to_polyline(segments, is_closed=True)
    # closed: first == last, and the walk starts at the wrap-around corner before segment 0
    np.testing.assert_allclose(poly, [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]], atol=1e-9)
