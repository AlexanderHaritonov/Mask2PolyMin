# test_best_consecutive_segments_separation.py

import numpy as np
import pytest
from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.sequence_segment import SequenceSegment
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence


def test_horizontal_then_vertical():
    points = np.array([
        [0,0], [1,0], [2,0], [3,0],   # horizontal
        [3,1], [3,2], [3,3], [3,4]    # vertical
    ], dtype=np.float64)

    seg1_params = LineSegmentParams(
        start_point=np.array([0.0, 0.0], dtype=np.float64),
        end_point=np.array([3.0, 0.0], dtype=np.float64),
        direction=np.array([1.0, 0.0], dtype=np.float64),
        loss=0.0
    )
    seg1 = SequenceSegment(points, 0, 3, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([3.0, 0.0], dtype=np.float64),
        end_point=np.array([3.0, 4.0], dtype=np.float64),
        direction=np.array([0.0, 1.0], dtype=np.float64),
        loss=0.0
    )
    seg2 = SequenceSegment(points, 4, 7, seg2_params)

    fitter = FitterToPointsSequence(points)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    # Expect separation near the end of horizontal (index 3)
    assert idx in (2, 3)

def test_horizontal_then_vertical_with_step():
    points = np.array([
        [0,0], [1,0], [2,0], [3,0],   # horizontal
        [4,1], [4,2], [4,3], [4,4]    # vertical
    ], dtype=np.float64)

    seg1_params = LineSegmentParams(
        start_point=np.array([0.0, 0.0], dtype=np.float64),
        end_point=np.array([3.0, 0.0], dtype=np.float64),
        direction=np.array([1.0, 0.0], dtype=np.float64),
        loss=0.0
    )
    seg1 = SequenceSegment(points, 0, 3, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([4.0, 0.0], dtype=np.float64),
        end_point=np.array([4.0, 4.0], dtype=np.float64),
        direction=np.array([0.0, 1.0], dtype=np.float64),
        loss=0.0
    )
    seg2 = SequenceSegment(points, 4, 7, seg2_params)

    fitter = FitterToPointsSequence(points)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    # Expect separation near the end of horizontal (index 3)
    assert idx == 3


def test_diagonal_then_horizontal():
    points = np.array([
        [0,0], [1,1], [2,2], [3,3],   # diagonal
        [4,3], [5,3], [6,3], [7,3]    # horizontal
    ], dtype=np.float64)

    seg1_params = LineSegmentParams(
        start_point=np.array([0.0, 0.0], dtype=np.float64),
        end_point=np.array([3.0, 3.0], dtype=np.float64),
        direction=np.array([1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)], dtype=np.float64),
        loss=0.0
    )
    seg1 = SequenceSegment(points, 0, 3, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([3.0, 3.0], dtype=np.float64),
        end_point=np.array([7.0, 3.0], dtype=np.float64),
        direction=np.array([1.0, 0.0], dtype=np.float64),
        loss=0.0
    )
    seg2 = SequenceSegment(points, 4, 7, seg2_params)

    fitter = FitterToPointsSequence(points)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    assert idx in (2, 3)


def test_noisy_horizontal_then_vertical():
    points = np.array([
        [0,0], [1,0.1], [2,-0.1], [3,0.05],   # noisy horizontal
        [3.1,1], [2.9,2], [3.05,3], [3,4.1]   # noisy vertical
    ], dtype=np.float64)

    seg1_params = LineSegmentParams(
        start_point=np.array([0.0, 0.0], dtype=np.float64),
        end_point=np.array([3.0, 0.0], dtype=np.float64),
        direction=np.array([1.0, 0.0], dtype=np.float64),
        loss=0.1
    )
    seg1 = SequenceSegment(points, 0, 3, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([3.0, 0.0], dtype=np.float64),
        end_point=np.array([3.0, 4.0], dtype=np.float64),
        direction=np.array([0.0, 1.0], dtype=np.float64),
        loss=0.1
    )
    seg2 = SequenceSegment(points, 4, 7, seg2_params)

    fitter = FitterToPointsSequence(points)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    # With noise, still expect separation near index 3
    assert idx in (2, 3, 4)


def test_closed_polygon_square():
    points = np.array([
        [0,0], [1,0], [2,0], [2,1], [2,2], [1,2], [0,2], [0,1], [0,0]
    ], dtype=np.float64)

    seg1_params = LineSegmentParams(
        start_point=np.array([0.0, 0.0], dtype=np.float64),
        end_point=np.array([2.0, 0.0], dtype=np.float64),
        direction=np.array([1.0, 0.0], dtype=np.float64),
        loss=0.0
    )
    seg1 = SequenceSegment(points, 0, 2, seg1_params)

    seg2_params = LineSegmentParams(
        start_point=np.array([2.0, 0.0], dtype=np.float64),
        end_point=np.array([2.0, 2.0], dtype=np.float64),
        direction=np.array([0.0, 1.0], dtype=np.float64),
        loss=0.0
    )
    seg2 = SequenceSegment(points, 3, 5, seg2_params)

    fitter = FitterToPointsSequence(points, is_closed=True)
    idx = fitter.best_consecutive_segments_separation(seg1, seg2)

    assert idx in  (1, 2)