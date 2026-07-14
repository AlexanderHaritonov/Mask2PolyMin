import math

import numpy as np

from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.sequence_segment import SequenceSegment

# Sentinel straightness of a degenerate fit (identical points): the direction is meaningless.
DEGENERATE_STRAIGHTNESS = 1.0


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


class SequenceMoments:
    """Precomputed statistical moments of a point sequence, enabling O(1) TLS line fit of any contiguous (possibly wrapping) index range."""
    def __init__(self, whole_sequence: np.ndarray):
        self.whole_sequence = whole_sequence
        self.sequence_center = whole_sequence.mean(axis=0).astype(np.float64)

        # Cumulative sums of statistical moments [Σx, Σy, Σx², Σy², Σxy] for sequence prefixes over globally centered coordinates.
        # Centering keeps the moment differences numerically stable.
        x, y = (whole_sequence - self.sequence_center).astype(np.float64).T
        self.stat_moments = np.zeros((len(x) + 1, 5))
        np.cumsum(np.stack([x, y, x * x, y * y, x * y], axis=1), axis=0, out=self.stat_moments[1:])


def range_moments(moments: SequenceMoments, first_index: int, last_index: int) -> tuple[np.ndarray, int]:
    M = moments.stat_moments
    if first_index <= last_index:
        return M[last_index + 1] - M[first_index], last_index - first_index + 1
    else:  # circular wrap
        return M[-1] - M[first_index] + M[last_index + 1], len(moments.whole_sequence) - first_index + last_index + 1


def fit_range(moments: SequenceMoments, first_index: int, last_index: int,
              with_endpoints: bool = True) -> LineSegmentParams:
    """TLS line fit through the points of a contiguous (possibly wrapping) index range, from the prefix moments.
    with_endpoints=False skips the extreme-projections pass — the only O(n) part — and sets
    start_point == end_point == centroid: still a valid point on the line for distance computations, but not the segment's extent.
    Use it where only line geometry and loss matter."""
    (sx, sy, sxx, syy, sxy), count = range_moments(moments, first_index, last_index)
    if count < 2:
        raise ValueError("Need at least 2 points to fit a line.")
    mean_x, mean_y = sx / count, sy / count
    cov_xx = sxx / count - mean_x * mean_x
    cov_yy = syy / count - mean_y * mean_y
    cov_xy = sxy / count - mean_x * mean_y
    direction, eig_max, eig_min = principal_axis(cov_xx, cov_yy, cov_xy)
    centroid = np.array([mean_x, mean_y]) + moments.sequence_center

    if eig_max <= 1e-8:  # degenerate: all points identical (same threshold scale as fit_line_segment's allclose)
        return LineSegmentParams(
            start_point=centroid,
            end_point=centroid,
            direction=np.array([1.0, 0.0], dtype=np.float64),  # arbitrary
            loss=0.0,
            straightness=DEGENERATE_STRAIGHTNESS)

    # principal_axis eigenvalues come from the population covariance (divide by count);
    # fit_line_segment's come from np.cov's sample covariance (ddof=1, divide by count-1).
    # Scale to keep loss/straightness conventions identical.
    loss = count * max(eig_min, 0.0) * count / (count - 1)
    straightness = float(eig_min / eig_max)

    if not with_endpoints:
        return LineSegmentParams(
            start_point=centroid,
            end_point=centroid,
            direction=direction,
            loss=loss,
            straightness=straightness)

    projections = (subsequence(moments.whole_sequence, first_index, last_index) - centroid) @ direction
    return LineSegmentParams(
        start_point=centroid + projections.min() * direction,
        end_point=centroid + projections.max() * direction,
        direction=direction,
        loss=loss,
        straightness=straightness)


def refit_segment(moments: SequenceMoments, segment: SequenceSegment) -> None:
    segment.line_segment_params = fit_range(moments, segment.first_index, segment.last_index)
