"""fit() input validation: friendly errors for bad shapes, coercion for lists and integer arrays."""
import numpy as np
import pytest

from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig


L_SHAPE = [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3]]


def test_plain_list_input_is_coerced():
    polygon, segments, _ = FitterToPointsSequence().fit(L_SHAPE)
    assert polygon.dtype == np.float64
    assert len(segments) == 2


def test_integer_array_input_is_coerced():
    polygon, _, _ = FitterToPointsSequence().fit(np.array(L_SHAPE, dtype=np.int32))
    assert polygon.dtype == np.float64


def test_cv2_findcontours_shape_is_accepted():
    # cv2.findContours returns (N, 1, 2) int32 arrays; must fit identically to the reshaped (N, 2) equivalent
    cv2_shaped = np.array(L_SHAPE, dtype=np.int32).reshape(-1, 1, 2)
    polygon_cv2, segments_cv2, _ = FitterToPointsSequence().fit(cv2_shaped)
    polygon_plain, segments_plain, _ = FitterToPointsSequence().fit(np.array(L_SHAPE, dtype=np.int32))

    assert polygon_cv2.dtype == np.float64
    assert np.array_equal(polygon_cv2, polygon_plain)
    assert len(segments_cv2) == len(segments_plain)


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        FitterToPointsSequence().fit(np.zeros(5))            # 1D
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        FitterToPointsSequence().fit(np.zeros((5, 3)))       # 3 columns


def test_non_finite_input_raises():
    bad = np.array(L_SHAPE, dtype=np.float64)
    bad[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        FitterToPointsSequence().fit(bad)
    bad[2, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        FitterToPointsSequence().fit(bad)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 2 points"):
        FitterToPointsSequence().fit(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="at least 2 points"):
        FitterToPointsSequence().fit(np.zeros((0, 2)))
    # closed contour whose only 2 points are the closing duplicate collapses to a single point
    with pytest.raises(ValueError, match="at least 2 points"):
        FitterToPointsSequence().fit(np.array([[1.0, 1.0], [1.0, 1.0]]), is_closed=True)


def test_negative_segment_cost_raises():
    with pytest.raises(ValueError, match="segment_cost must be >= 0"):
        FitterConfig(segment_cost=-1.0)
