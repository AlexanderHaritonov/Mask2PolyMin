import numpy as np

# Below this convergence angle between two consecutive fitted lines, their intersection is numerically unstable:
# a small direction error in either TLS fit gets amplified into a large positional error along the nearly-parallel lines.
MIN_CORNER_ANGLE_DEG = 20.0
# |cross| = sin(angle between the lines) (see _corner), so this is that angle's cross-product equivalent, computed once.
_MIN_CROSS_FOR_CORNER = float(np.sin(np.deg2rad(MIN_CORNER_ANGLE_DEG)))


def segments_to_polyline(segments, is_closed: bool, tolerance: float = 1.0) -> np.ndarray:
    """Convert fitted segments to a polyline, (M, 2) float array.

    `tolerance` is the fitter's tolerance (FitterConfig.tolerance), used to scale the corner plausibility radius.

    Interior vertices are intersections of consecutive fitted infinite
    lines, which reconstructs corners with sub-pixel accuracy even when no
    input point lies at the true corner.

    Closed: every vertex is an intersection (including the wrap-around
    corner between the last and first segments); first point equals last.
    Open: the two endpoints are the projections of the first and last input
    points onto their segments' lines.

    A corner is anchored at the junction between the two segments — the  mean of the orphaned points if the junction has any,
    else the input point where the segments meet.
    If the lines converge at too shallow an angle (MIN_CORNER_ANGLE_DEG), or that intersection lands implausibly far from the anchor,
    the vertex  falls back to the anchor.
    (This also keeps closed 2-segment fits from collapsing: their two line-intersection corners would coincide)

    Note: LineSegmentParams.start_point/end_point are the min/max projections along the fitted eigenvector, whose sign is arbitrary — they are not ordered by traversal.
    """
    n = len(segments)
    if n == 0:
        raise ValueError("segments_to_polyline: empty segments list")
    if n == 1:
        s = segments[0]
        first = _project_input_point(s, s.first_index)
        last = _project_input_point(s, s.last_index)
        if is_closed:
            # Degenerate: single line "polygon", closed per the contract.
            return np.vstack([first, last, first])
        return np.vstack([first, last])

    n_corners = n if is_closed else n - 1
    corners = np.empty((n_corners, 2), dtype=np.float64)
    for i in range(n_corners):
        corners[i] = _corner(segments[i], segments[(i + 1) % n], tolerance)
    if is_closed:
        # Corner i sits between segment i and segment i+1, so the polyline
        # starts at corner n-1 (the corner before segment 0) and walks forward.
        return np.vstack([corners[-1], corners[:-1], corners[-1]])
    first = _project_input_point(segments[0], segments[0].first_index)
    last = _project_input_point(segments[-1], segments[-1].last_index)
    return np.vstack([first, corners, last])


def _corner(seg_a, seg_b, tolerance: float) -> np.ndarray:
    """Vertex between two consecutive segments.

    Line intersection when the two lines converge at more than MIN_CORNER_ANGLE_DEG and the intersection lands near the junction anchor;
    otherwise the anchor itself.
    The anchor is the mean of the orphaned points between the segments if  the junction has any,
    else the midpoint of the adjacent input points' projections onto the two lines.
    """
    a = seg_a.line_segment_params
    b = seg_b.line_segment_params
    anchor = _junction_anchor(seg_a, seg_b)

    cross = a.direction[0] * b.direction[1] - a.direction[1] * b.direction[0]
    if abs(cross) >= _MIN_CROSS_FOR_CORNER:
        dp = b.start_point - a.start_point
        t = (dp[0] * b.direction[1] - dp[1] * b.direction[0]) / cross
        intersection = a.start_point + t * a.direction
        # A genuine corner lies within roughly a segment length of the meeting point;
        # beyond that the lines are effectively parallel.
        max_offset = max(3.0 * tolerance, min(_length(a), _length(b)))
        if np.linalg.norm(intersection - anchor) <= max_offset:
            return intersection
    return anchor


def _junction_anchor(seg_a, seg_b) -> np.ndarray:
    """Anchor point for the junction between two consecutive segments:
    the mean of the orphaned points between them if the junction has any,
    else the midpoint of the adjacent input points' projections onto the two segments' lines."""
    n = len(seg_a.whole_sequence)
    gap = (seg_b.first_index - seg_a.last_index - 1) % n
    if gap > 0:
        orphan_indices = [(seg_a.last_index + 1 + k) % n for k in range(gap)]
        return seg_a.whole_sequence[orphan_indices].astype(np.float64).mean(axis=0)
    return 0.5 * (_project_input_point(seg_a, seg_a.last_index)
                  + _project_input_point(seg_b, seg_b.first_index))


def _project_input_point(segment, index: int) -> np.ndarray:
    """Project the input point at `index` onto the segment's fitted line."""
    s = segment.line_segment_params
    p = segment.whole_sequence[index].astype(np.float64)
    return s.start_point + ((p - s.start_point) @ s.direction) * s.direction


def _length(params) -> float:
    return float(np.linalg.norm(params.end_point - params.start_point))
