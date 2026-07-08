import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass(frozen=True)
class LineSegmentParams:
    start_point: NDArray[np.float64]
    end_point: NDArray[np.float64]
    direction: NDArray[np.float64]  # unit (length 1) direction vector
    loss: float
    straightness: float = 0.0  # min_eigenvalue / max_eigenvalue: 0 = perfectly straight, 1 = circular

    def squared_distances_to_line(self, points: np.ndarray) -> np.ndarray:
        # 2D shortcut: perpendicular to a unit direction (dx, dy) is (-dy, dx),
        # so the signed perpendicular distance is a single dot product.
        perp_dir = np.array([-self.direction[1], self.direction[0]])
        d_perp = (points - self.start_point) @ perp_dir  # one gemv, shape (N,)
        return d_perp * d_perp

    def squared_distances_to_line_general(self, points: np.ndarray) -> np.ndarray:
        # Gram-Schmidt decomposition
        # v = (v·d)d + perp
        v = points - self.start_point
        # Scalar projection lengths (dot product with direction)
        scalar_proj = np.dot(v, self.direction)  # shape (N,)
        # Vector projections
        proj = np.outer(scalar_proj, self.direction)  # shape (N, D)
        # Perpendicular component
        perp = v - proj
        # Squared distances = squared norm of perpendicular component
        return np.sum(perp ** 2, axis=1)