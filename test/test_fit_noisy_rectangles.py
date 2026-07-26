"""Randomized property test: a noisy rectangle contour fits to exactly 4 segments
whose polyline vertices land within tolerance of the true corners.

Generalizes the single-square end-to-end tests in test_orphaned_junction_points.py
to random sizes, rotations and translations, with deterministic per-seed rectangles.

The noise level is deliberately mild (sigma 0.1, verified 0 failures over 1000 seeds).
From sigma ~0.15 upward the fitter occasionally leaves an extra segment or misses a corner
by more than tolerance: the adjust phase can settle on a junction placement from which the
collinear-merge gates rightly refuse to recombine the pieces.
"""
import numpy as np
import pytest

from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig


def noisy_rectangle_contour(rng, width, height, angle, center, sigma):
    """closed 1-px-step rectangle contour, rotated and translated, with gaussian jitter;
    :returns (points, true corner positions)"""
    pts = []
    for x in range(0, width): pts.append([x, 0])
    for y in range(0, height): pts.append([width, y])
    for x in range(width, 0, -1): pts.append([x, height])
    for y in range(height, 0, -1): pts.append([0, y])
    pts = np.array(pts, dtype=np.float64)
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    pts = pts @ rotation.T + center
    corners = corners @ rotation.T + center
    pts += rng.normal(0, sigma, pts.shape)
    return pts, corners


@pytest.mark.parametrize("seed", range(40))
def test_noisy_rectangle_fits_to_its_four_corners(seed):
    rng = np.random.default_rng(seed)
    width, height = int(rng.integers(8, 25)), int(rng.integers(8, 25))
    angle = rng.uniform(0, np.pi / 2)
    center = rng.uniform(-30, 30, 2)
    pts, true_corners = noisy_rectangle_contour(rng, width, height, angle, center, sigma=0.1)

    config = FitterConfig(tolerance=1.0)
    fitter = FitterToPointsSequence(config=config)
    polygon, segments = fitter.fit(pts, is_closed=True)

    assert len(segments) == 4, f"w={width} h={height} angle={np.degrees(angle):.1f}: expected 4 segments, got {len(segments)}"
    vertices = polygon[:-1]
    assert len(vertices) == 4
    for corner in true_corners:
        error = np.linalg.norm(vertices - corner, axis=1).min()
        assert error <= config.tolerance, \
            f"w={width} h={height} angle={np.degrees(angle):.1f}: vertex {error:.3f}px from corner {corner}"
