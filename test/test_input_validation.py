"""Constructor input validation: friendly errors for bad shapes, coercion for lists and integer arrays."""
import numpy as np
import pytest

from mask2polymin.fit_to_points_sequence import FitterToPointsSequence


L_SHAPE = [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3]]


def test_plain_list_input_is_coerced():
    polygon, segments = FitterToPointsSequence(L_SHAPE).fit()
    assert polygon.dtype == np.float64
    assert len(segments) == 2


def test_integer_array_input_is_coerced():
    polygon, _ = FitterToPointsSequence(np.array(L_SHAPE, dtype=np.int32)).fit()
    assert polygon.dtype == np.float64


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        FitterToPointsSequence(np.zeros(5))            # 1D
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        FitterToPointsSequence(np.zeros((5, 3)))       # 3 columns


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 2 points"):
        FitterToPointsSequence(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="at least 2 points"):
        FitterToPointsSequence(np.zeros((0, 2)))
    # closed contour whose only 2 points are the closing duplicate collapses to a single point
    with pytest.raises(ValueError, match="at least 2 points"):
        FitterToPointsSequence(np.array([[1.0, 1.0], [1.0, 1.0]]), is_closed=True)
