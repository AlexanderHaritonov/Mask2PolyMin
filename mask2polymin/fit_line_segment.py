import math

import numpy as np


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
