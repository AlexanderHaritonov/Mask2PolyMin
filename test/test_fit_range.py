"""Cross-checks: the moment-based fit_range must agree with fit_line_segment."""
import numpy as np
import pytest

from mask2polymin.sequence_moments import principal_axis, subsequence, fit_range
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence
from fit_line_segment_reference import fit_line_segment


def fitter_fit_range(fitter, first, last):
    return fit_range(fitter._moments, first, last, with_endpoints=True)


def assert_equivalent_fits(fitter, first, last):
    expected = fit_line_segment(subsequence(fitter.whole_sequence, first, last))
    actual = fitter_fit_range(fitter, first, last)

    # direction: same line; sign may differ (eigenvector sign is arbitrary)
    assert abs(float(np.dot(expected.direction, actual.direction))) == pytest.approx(1.0, abs=1e-9)
    assert actual.loss == pytest.approx(expected.loss, rel=1e-6, abs=1e-9)
    assert actual.straightness == pytest.approx(expected.straightness, rel=1e-6, abs=1e-9)
    # endpoints: a direction sign flip swaps start/end, so compare as a set
    expected_ends = sorted(map(tuple, np.round([expected.start_point, expected.end_point], 9)))
    actual_ends = sorted(map(tuple, np.round([actual.start_point, actual.end_point], 9)))
    np.testing.assert_allclose(actual_ends, expected_ends, atol=1e-6)


def test_open_polyline_random_ranges():
    rng = np.random.default_rng(7)
    points = np.cumsum(rng.normal(0.5, 0.4, size=(60, 2)), axis=0) + 100
    fitter = FitterToPointsSequence(points)
    for _ in range(50):
        first = int(rng.integers(0, len(points) - 1))
        last = int(rng.integers(first + 1, len(points)))
        assert_equivalent_fits(fitter, first, last)


def test_closed_contour_wrapping_ranges():
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rng = np.random.default_rng(11)
    points = np.stack([50 + 20 * np.cos(theta), 50 + 20 * np.sin(theta)], axis=1)
    points += rng.normal(0, 0.2, points.shape)
    fitter = FitterToPointsSequence(points, is_closed=True)
    n = len(fitter.whole_sequence)
    for first, last in [(35, 5), (n - 1, 0), (30, 29 - 1), (20, 3)]:
        assert_equivalent_fits(fitter, first, last)


def test_exact_line_zero_loss():
    points = np.array([[x, 2.0 * x + 1] for x in range(10)], dtype=np.float64)
    fitter = FitterToPointsSequence(points)
    assert_equivalent_fits(fitter, 2, 8)
    assert fitter_fit_range(fitter, 2, 8).loss == pytest.approx(0.0, abs=1e-9)


def test_two_point_range():
    points = np.array([[0, 0], [1, 0], [2, 1], [3, 3]], dtype=np.float64)
    fitter = FitterToPointsSequence(points)
    assert_equivalent_fits(fitter, 1, 2)


def test_single_point_range_raises():
    points = np.array([[0, 0], [1, 0], [2, 1]], dtype=np.float64)
    fitter = FitterToPointsSequence(points)
    with pytest.raises(ValueError):
        fitter_fit_range(fitter, 1, 1)


def test_identical_points_degenerate():
    points = np.array([[5, 5]] * 4 + [[6, 7], [7, 9]], dtype=np.float64)
    fitter = FitterToPointsSequence(points)
    expected = fit_line_segment(subsequence(fitter.whole_sequence, 0, 3))
    actual = fitter_fit_range(fitter, 0, 3)
    np.testing.assert_allclose(actual.start_point, expected.start_point, atol=1e-9)
    np.testing.assert_allclose(actual.end_point, expected.end_point, atol=1e-9)
    assert actual.loss == 0.0


def test_without_endpoints_same_line_cheaper_params():
    rng = np.random.default_rng(5)
    points = np.cumsum(rng.normal(0.5, 0.4, size=(40, 2)), axis=0)
    fitter = FitterToPointsSequence(points)
    full = fit_range(fitter._moments, 4, 30, with_endpoints=True)
    slim = fit_range(fitter._moments, 4, 30, with_endpoints=False)

    np.testing.assert_allclose(slim.direction, full.direction)
    assert slim.loss == full.loss
    assert slim.straightness == full.straightness
    # endpoints collapse to the centroid — still a point on the fitted line
    np.testing.assert_array_equal(slim.start_point, slim.end_point)
    np.testing.assert_allclose(
        full.squared_distances_to_line(points[4:31]),
        slim.squared_distances_to_line(points[4:31]), atol=1e-12)


def test_principal_axis_matches_eigh():
    rng = np.random.default_rng(3)
    for _ in range(50):
        pts = rng.normal(0, 3, size=(20, 2))
        cov = np.cov(pts, rowvar=False, ddof=0)
        direction, eig_max, eig_min = principal_axis(cov[0, 0], cov[1, 1], cov[0, 1])
        eigvals, eigvecs = np.linalg.eigh(cov)
        assert eig_max == pytest.approx(eigvals[-1], rel=1e-9, abs=1e-12)
        assert eig_min == pytest.approx(eigvals[0], rel=1e-9, abs=1e-12)
        assert abs(float(np.dot(direction, eigvecs[:, -1]))) == pytest.approx(1.0, abs=1e-9)
