import numpy as np


def segments_to_polyline(segments, is_closed: bool) -> np.ndarray:
    """Convert fitted segments to a polyline, (M, 2) float array.

    Interior vertices are intersections of consecutive fitted infinite
    lines, which reconstructs corners with sub-pixel accuracy even when no
    input point lies at the true corner.

    Closed: every vertex is an intersection (including the wrap-around
    corner between the last and first segments); first point equals last.
    Open: the two endpoints are the projections of the first and last input
    points onto their segments' lines.

    A corner is anchored at the input point where the two segments meet:
    if the lines are nearly parallel, or their intersection lands
    implausibly far from that meeting point (near-parallel lines intersect
    arbitrarily far away), the vertex falls back to the meeting point's
    projections onto the two lines. This also keeps closed 2-segment fits
    from collapsing: their two line-intersection corners would coincide.

    Note: LineSegmentParams.start_point/end_point are the min/max
    projections along the fitted eigenvector, whose sign is arbitrary — they
    are not ordered by traversal.
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
        corners[i] = _corner(segments[i], segments[(i + 1) % n])
    if is_closed:
        # Corner i sits between segment i and segment i+1, so the polyline
        # starts at corner n-1 (the corner before segment 0) and walks forward.
        return np.vstack([corners[-1], corners[:-1], corners[-1]])
    first = _project_input_point(segments[0], segments[0].first_index)
    last = _project_input_point(segments[-1], segments[-1].last_index)
    return np.vstack([first, corners, last])


def _corner(seg_a, seg_b) -> np.ndarray:
    """Vertex between two consecutive segments.

    Line intersection when it lands near the segments' meeting point;
    otherwise the midpoint of the meeting point's projections onto the two
    lines (near-parallel lines, or their intersection is far away).
    """
    a = seg_a.line_segment_params
    b = seg_b.line_segment_params
    anchor = 0.5 * (_project_input_point(seg_a, seg_a.last_index)
                    + _project_input_point(seg_b, seg_b.first_index))

    cross = a.direction[0] * b.direction[1] - a.direction[1] * b.direction[0]
    if abs(cross) > 1e-9:
        dp = b.start_point - a.start_point
        t = (dp[0] * b.direction[1] - dp[1] * b.direction[0]) / cross
        intersection = a.start_point + t * a.direction
        # A genuine corner lies within roughly a segment length of the
        # meeting point; beyond that the lines are effectively parallel.
        max_offset = max(3.0, min(_length(a), _length(b)))
        if np.linalg.norm(intersection - anchor) <= max_offset:
            return intersection
    return anchor


def _project_input_point(segment, index: int) -> np.ndarray:
    """Project the input point at `index` onto the segment's fitted line."""
    s = segment.line_segment_params
    p = segment.whole_sequence[index].astype(np.float64)
    return s.start_point + ((p - s.start_point) @ s.direction) * s.direction


def _length(params) -> float:
    return float(np.linalg.norm(params.end_point - params.start_point))
