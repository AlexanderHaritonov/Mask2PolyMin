import numpy as np
import pytest
from mask2polymin.line_segment_params import LineSegmentParams

class TestSquaredDistancesToLine:

    def test_points_on_line(self):
        line = LineSegmentParams(
            start_point=np.array([0., 0.]),
            end_point=np.array([1., 0.]),
            direction=np.array([1., 0.]),  # x-axis
            loss=0.0
        )
        points = np.array([[0., 0.], [2., 0.], [-3., 0.]])
        expected = np.array([0., 0., 0.])
        result = line.squared_distances_to_line(points)
        assert np.allclose(result, expected)

    def test_points_above_below_line(self):
        line = LineSegmentParams(
            start_point=np.array([0., 0.]),
            end_point=np.array([1., 0.]),
            direction=np.array([1., 0.]),  # x-axis
            loss=0.0
        )
        points = np.array([[0., 1.], [2., -2.], [3., 4.]])
        expected = np.array([1., 4., 16.])  # y^2
        result = line.squared_distances_to_line(points)
        assert np.allclose(result, expected)

    def test_diagonal_line(self):
        line = LineSegmentParams(
            start_point=np.array([0., 0.]),
            end_point=np.array([1., 1.]),
            direction=np.array([np.sqrt(2)/2, np.sqrt(2)/2]),  # unit vector along y=x
            loss=0.0
        )
        points = np.array([[1., 0.], [0., 1.], [2., 0.]])
        expected = np.array([0.5, 0.5, 2.0])
        result = line.squared_distances_to_line(points)
        assert np.allclose(result, expected)
