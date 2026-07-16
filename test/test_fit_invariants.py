"""Invariant: every segment returned by fit() carries line params that match a
fresh fit of its own index range.

Regression guard: adjust_segmentation used to refit only the right-hand segment
of a moved junction, so the left-hand segment could keep params fitted to its
old extent — corrupting corner reconstruction and the SSE gates downstream.
"""
import numpy as np
import pytest

from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig
from mask2polymin.sequence_moments import fit_range


def assert_segments_freshly_fitted(fitter, segments):
    for s in segments:
        fresh = fit_range(fitter._moments, s.first_index, s.last_index, with_endpoints=True)
        stored = s.line_segment_params
        # direction: same line; sign may differ (eigenvector sign is arbitrary)
        assert abs(float(np.dot(fresh.direction, stored.direction))) == pytest.approx(1.0, abs=1e-12), \
            f"stale direction on segment [{s.first_index}..{s.last_index}]"
        assert stored.loss == pytest.approx(fresh.loss, rel=1e-9, abs=1e-12), \
            f"stale loss on segment [{s.first_index}..{s.last_index}]"
        # endpoints: a direction sign flip swaps start/end, so compare as a set
        stored_ends = sorted(map(tuple, [stored.start_point, stored.end_point]))
        fresh_ends = sorted(map(tuple, [fresh.start_point, fresh.end_point]))
        np.testing.assert_allclose(stored_ends, fresh_ends, atol=1e-9)


def noisy_rectangle(rng, width, height, sigma=0.35):
    """closed 1-px-step rectangle contour with gaussian jitter"""
    pts = []
    for x in range(0, width): pts.append([x, 0])
    for y in range(0, height): pts.append([width, y])
    for x in range(width, 0, -1): pts.append([x, height])
    for y in range(height, 0, -1): pts.append([0, y])
    pts = np.array(pts, dtype=np.float64)
    return pts + rng.normal(0, sigma, pts.shape)


def test_closed_fit_returns_freshly_fitted_segments():
    rng = np.random.default_rng(0)
    for _ in range(15):
        width, height = int(rng.integers(8, 25)), int(rng.integers(8, 25))
        pts = noisy_rectangle(rng, width, height)
        fitter = FitterToPointsSequence(pts, is_closed=True, config=FitterConfig(tolerance=1.0))
        _, segments = fitter.fit()
        assert_segments_freshly_fitted(fitter, segments)


def test_open_fit_returns_freshly_fitted_segments():
    rng = np.random.default_rng(1)
    for _ in range(15):
        n = 40
        horizontal = np.stack([np.arange(n), np.zeros(n)], axis=1)
        vertical = np.stack([np.full(n, n - 1.0), np.arange(1, n + 1)], axis=1)
        pts = np.vstack([horizontal, vertical]) + rng.normal(0, 0.3, (2 * n, 2))
        fitter = FitterToPointsSequence(pts, is_closed=False, config=FitterConfig(tolerance=1.0))
        _, segments = fitter.fit()
        assert_segments_freshly_fitted(fitter, segments)
