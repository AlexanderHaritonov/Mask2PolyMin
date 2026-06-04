"""
Primary quality metrics for the PolyFit2D benchmark.

All polylines are (N, 2) float arrays in (x, y) pixel space.
"""
import numpy as np
import cv2


def hausdorff(poly_a: np.ndarray, poly_b: np.ndarray, sample_step: float = 1.0) -> float:
    """Symmetric polyline-Hausdorff distance in pixels.

    Treats both inputs as closed polylines (sequences of edges), not point
    clouds: distance is point-to-edge, not point-to-point. The point-cloud
    version (`scipy.spatial.distance.directed_hausdorff`) overestimates
    badly when one polyline is sparse and the other dense, because dense-
    contour points sitting between sparse vertices are far from any vertex
    yet close to the edge between them.

    Both polylines are densified to ≤ `sample_step` px between samples so
    that the B→A direction sees edge midpoints of B, not only its vertices.
    """
    a_dense = _densify(poly_a, sample_step)
    b_dense = _densify(poly_b, sample_step)
    h_ab = float(_point_to_polyline_distances(a_dense, poly_b).max())
    h_ba = float(_point_to_polyline_distances(b_dense, poly_a).max())
    return max(h_ab, h_ba)


def _densify(polyline: np.ndarray, max_step: float) -> np.ndarray:
    """Insert points so adjacent samples are at most `max_step` px apart."""
    out = [polyline[0]]
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        d = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(d / max_step)))
        for k in range(1, n + 1):
            out.append(a + (k / n) * (b - a))
    return np.asarray(out, dtype=np.float64)


def _point_to_polyline_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Min distance from each point to the closed polyline's edges, vectorized.

    points:   (Q, 2)
    polyline: (M, 2) closed (first point equals last)
    returns:  (Q,)
    """
    a = polyline[:-1]                                  # (E, 2)
    ab = polyline[1:] - a                              # (E, 2)
    ab_sq = np.maximum((ab * ab).sum(axis=1), 1e-12)   # (E,)
    qa = points[:, None, :] - a[None, :, :]            # (Q, E, 2)
    t = np.clip((qa * ab[None, :, :]).sum(axis=2) / ab_sq[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    diffs = points[:, None, :] - closest
    return np.sqrt((diffs * diffs).sum(axis=2)).min(axis=1)


def iou_rasterized(
    fitted: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    IoU between a filled fitted polygon and the original binary mask.

    Parameters
    ----------
    fitted : (M, 2) float array in (x, y)
    mask   : (H, W) uint8 binary mask (the ground-truth rasterization)
    """
    h, w = mask.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = fitted.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 1)

    inter = np.logical_and(canvas, mask).sum()
    union = np.logical_or(canvas, mask).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def mean_distance(contour: np.ndarray, poly: np.ndarray) -> float:
    """
    Mean of per-point minimum distances from contour points to the closed
    simplified polygon's boundary.

    contour : (N, 2) dense input contour in (x, y)
    poly    : (M, 2) closed polyline in (x, y); first point must equal last
    """
    poly_cv = poly.reshape(-1, 1, 2).astype(np.float32)
    dists = [
        abs(cv2.pointPolygonTest(poly_cv, (float(p[0]), float(p[1])), True))
        for p in contour
    ]
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Self-test: run with  python metrics.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    # --- synthetic shapes ---
    # 1. Perfect square: 100×100, with vertices at corners
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]], dtype=float)

    # Dense noisy circle, radius 50, centre (150, 150)
    theta = np.linspace(0, 2 * math.pi, 500, endpoint=False)
    cx, cy, r = 150.0, 150.0, 50.0
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.3, size=500)
    circle_pts = np.stack(
        [cx + (r + noise) * np.cos(theta), cy + (r + noise) * np.sin(theta)], axis=1
    )
    circle_closed = np.vstack([circle_pts, circle_pts[0]])

    # --- 1. Hausdorff: identical polylines ---
    h = hausdorff(sq, sq)
    assert h == 0.0, f"identical hausdorff should be 0, got {h}"
    print(f"hausdorff(sq, sq) = {h:.4f}  ✓")

    # shift sq by (1,0); max distance should be ~1
    sq_shifted = sq + np.array([1.0, 0.0])
    h2 = hausdorff(sq, sq_shifted)
    print(f"hausdorff(sq, sq+1) = {h2:.4f}  (expect ≈ 1.0)")
    assert 0.9 < h2 < 1.1, f"expected ≈1.0, got {h2}"
    print("  ✓")

    # --- 2. IoU: square polygon vs its own filled mask ---
    mask_sq = np.zeros((200, 200), dtype=np.uint8)
    cv2.fillPoly(mask_sq, [sq.reshape(-1, 1, 2).astype(np.int32)], 1)
    iou_exact = iou_rasterized(sq, mask_sq)
    print(f"iou_rasterized(sq exact) = {iou_exact:.4f}  (expect ≈ 1.0)")
    assert iou_exact > 0.999, f"expected > 0.999, got {iou_exact}"
    print("  ✓")

    # Shrink square by 2 px on each side → IoU should drop noticeably
    sq_shrunk = sq + np.array([[2, 2], [-2, 2], [-2, -2], [2, -2], [2, 2]])
    iou_shrunk = iou_rasterized(sq_shrunk, mask_sq)
    print(f"iou_rasterized(sq shrunk 2px) = {iou_shrunk:.4f}  (expect < 1.0)")
    assert iou_shrunk < 0.98, f"expected < 0.98, got {iou_shrunk}"
    print("  ✓")

    # --- 3. Mean distance: contour vs itself ---
    md0 = mean_distance(circle_closed, circle_closed)
    print(f"mean_distance(circle, circle) = {md0:.6f}  (expect 0.0)")
    assert md0 < 1e-9, f"expected 0, got {md0}"
    print("  ✓")

    # mean distance from dense circle to its 4-point bounding square
    # expect roughly r*(1 - pi/4) ≈ 10.7 px  (analytical for circle→square)
    sq_big = np.array(
        [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]], dtype=float
    )
    md_circ_sq = mean_distance(circle_closed, sq_big)
    print(f"mean_distance(circle r=50, bounding square) = {md_circ_sq:.2f}  (expect ~10–12 px)")
    assert 5 < md_circ_sq < 20, f"out of expected range: {md_circ_sq}"
    print("  ✓")

    print("\nAll metric self-tests passed.")
