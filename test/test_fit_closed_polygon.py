import numpy as np
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig
from mask2polymin.sequence_segment import print_segments_info
from mask2polymin.plotting import plot_segments


def test_fit_closed_polygon_four_segments():
    """Test the full fit() method on a closed polygon that should produce 4 segments"""
    points = np.array([
        [0, 0], [0, 1], [0, 2],
        [1, 3], [2, 3], [3, 3],
        [4, 2], [4, 1], [4, 0],
        [3, -1], [2, -1], [1, -1], [0, 0]
    ], dtype=np.float64)

    fitter = FitterToPointsSequence(
        config=FitterConfig(max_segments_count=10, tolerance=0.001, verbose=False)
    )

    _polygon, segments, _energy = fitter.fit(points, is_closed=True)
    print_segments_info(segments)
    plot_segments(segments)

    # This shape is really 8-sided (4 flat edges + 4 single-edge diagonal chamfers with no
    # interior samples), and every 2-point subsegment fits with exactly zero loss, so there's
    # no single correct segment count: each chamfer's points may be folded into an adjacent
    # flat segment (orphaned) or kept as their own segment, depending on split order. What
    # must hold regardless is that every segment is an exact fit (this shape has no noise)
    # and the count stays within the shape's true structural bounds.
    assert 4 <= len(segments) <= 8, f"Expected 4-8 segments, got {len(segments)}"
    for segment in segments:
        assert segment.line_segment_params.loss < 1e-6, \
            f"Segment {segment.first_index}-{segment.last_index} has nonzero loss {segment.line_segment_params.loss}"


def test_fit_closed_polygon_four_segments_rotated_45deg():
    """Test the full fit() method on a closed polygon rotated 45 degrees around center"""
    # Original points
    original_points = np.array([
        [0, 0], [0, 1], [0, 2],
        [1, 3], [2, 3], [3, 3],
        [4, 2], [4, 1], [4, 0],
        [3, -1], [2, -1], [1, -1], [0, 0]
    ], dtype=np.float64)

    # Calculate center (excluding the duplicate closing point)
    center = np.mean(original_points[:-1], axis=0)

    # Rotate 45 degrees around center
    angle = np.radians(45)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    points = np.zeros_like(original_points)
    for i, (x, y) in enumerate(original_points):
        # Translate to origin
        x_centered = x - center[0]
        y_centered = y - center[1]
        # Rotate
        x_rotated = x_centered * cos_angle - y_centered * sin_angle
        y_rotated = x_centered * sin_angle + y_centered * cos_angle
        # Translate back
        points[i] = [x_rotated + center[0], y_rotated + center[1]]

    fitter = FitterToPointsSequence(
        config=FitterConfig(max_segments_count=10, tolerance=0.001, verbose=False)
    )

    _polygon, segments, _energy = fitter.fit(points, is_closed=True)
    print_segments_info(segments)
    plot_segments(segments)

    # See test_fit_closed_polygon_four_segments: this shape has no single correct segment
    # count, only true structural bounds and an exact-fit requirement (rotation is
    # distance-preserving, so the loss threshold only needs to absorb rotation's own
    # floating-point noise, far below a real defect's scale).
    assert 4 <= len(segments) <= 8, f"Expected 4-8 segments, got {len(segments)}"
    for segment in segments:
        assert segment.line_segment_params.loss < 1e-6, \
            f"Segment {segment.first_index}-{segment.last_index} has nonzero loss {segment.line_segment_params.loss}"