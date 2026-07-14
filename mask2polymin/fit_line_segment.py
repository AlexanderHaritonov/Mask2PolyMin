import math

import numpy as np

from mask2polymin.line_segment_params import LineSegmentParams


def principal_axis(cov_xx: float, cov_yy: float, cov_xy: float) -> tuple[np.ndarray, float, float]:
    """Closed-form eigendecomposition of the symmetric 2x2 covariance matrix [[cov_xx, cov_xy], [cov_xy, cov_yy]].
    :returns (unit direction of the largest-eigenvalue axis, eigval_max, eigval_min)."""
    half_trace = 0.5 * (cov_xx + cov_yy)
    radius = math.hypot(0.5 * (cov_xx - cov_yy), cov_xy)
    angle = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    direction = np.array([math.cos(angle), math.sin(angle)])
    return direction, half_trace + radius, half_trace - radius


def subsequence(sequence: np.ndarray, left, right) -> np.ndarray:
    if left < right:
        return sequence[left:right+1]
    else:
        return np.vstack([ sequence[left:], sequence[:right+1] ])


def range_moments(stat_moments: np.ndarray, sequence_length: int, first_index: int, last_index: int) -> tuple[np.ndarray, int]:
    M = stat_moments
    if first_index <= last_index:
        return M[last_index + 1] - M[first_index], last_index - first_index + 1
    else:  # circular wrap
        return M[-1] - M[first_index] + M[last_index + 1], sequence_length - first_index + last_index + 1


def fit_range(sequence: np.ndarray, stat_moments: np.ndarray, sequence_center: np.ndarray,
              first_index: int, last_index: int) -> LineSegmentParams:
    # TLS line fit through the points of a contiguous (possibly wrapping) index range, from the prefix moments.
    (sx, sy, sxx, syy, sxy), count = range_moments(stat_moments, len(sequence), first_index, last_index)
    if count < 2:
        raise ValueError("Need at least 2 points to fit a line.")
    mean_x, mean_y = sx / count, sy / count
    cov_xx = sxx / count - mean_x * mean_x
    cov_yy = syy / count - mean_y * mean_y
    cov_xy = sxy / count - mean_x * mean_y
    direction, eig_max, eig_min = principal_axis(cov_xx, cov_yy, cov_xy)
    centroid = np.array([mean_x, mean_y]) + sequence_center

    if eig_max <= 1e-8:  # degenerate: all points identical (same threshold scale as fit_line_segment's allclose)
        return LineSegmentParams(
            start_point=centroid,
            end_point=centroid,
            direction=np.array([1.0, 0.0], dtype=np.float64),
            loss=0.0)

    projections = (subsequence(sequence, first_index, last_index) - centroid) @ direction
    # principal_axis eigenvalues come from the population covariance (divide by count);
    # fit_line_segment's come from np.cov's sample covariance (ddof=1, divide by count-1).
    # Scale to keep loss/straightness conventions identical.
    loss = count * max(eig_min, 0.0) * count / (count - 1)
    straightness = float(eig_min / eig_max) if eig_max > 0 else 0.0
    return LineSegmentParams(
        start_point=centroid + projections.min() * direction,
        end_point=centroid + projections.max() * direction,
        direction=direction,
        loss=loss,
        straightness=straightness)
