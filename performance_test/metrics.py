"""
Primary quality metrics for the Mask2PolyMin benchmark.

All polylines are (N, 2) float arrays in (x, y) pixel space.
"""
import numpy as np
import cv2


def hausdorff(poly_a: np.ndarray, poly_b: np.ndarray, sample_step: float = 1.0) -> float:
    """Symmetric polyline-Hausdorff distance in pixels: max of the pooled boundary distances.

    A single outlier sample dictates the value; `hd95` is the robust companion.
    See `_boundary_distances` for the sampling scheme.
    """
    return float(_boundary_distances(poly_a, poly_b, sample_step).max())


def hd95(poly_a: np.ndarray, poly_b: np.ndarray, sample_step: float = 1.0) -> float:
    """95th percentile of the pooled symmetric boundary distances, in pixels.

    Robust Hausdorff variant, standard in the segmentation literature:
    a single noise spike in either polyline moves the max, not this.
    """
    return float(np.percentile(_boundary_distances(poly_a, poly_b, sample_step), 95.0))


def _boundary_distances(poly_a: np.ndarray, poly_b: np.ndarray, sample_step: float) -> np.ndarray:
    """Pooled two-directional point-to-edge distances between two closed polylines.

    Treats both inputs as closed polylines (sequences of edges), not point clouds:
    distance is point-to-edge.
    The point-cloud version (`scipy.spatial.distance.directed_hausdorff`) overestimates
    badly when one polyline is sparse and the other dense.

    Both polylines are densified to ≤ `sample_step` px between samples so each direction sees the other's edge midpoints, not only its vertices.
    The A→B and B→A distances are concatenated, which weights each direction by its boundary length.
    """
    a_dense = _densify(poly_a, sample_step)
    b_dense = _densify(poly_b, sample_step)
    d_ab = _point_to_polyline_distances(a_dense, poly_b)
    d_ba = _point_to_polyline_distances(b_dense, poly_a)
    return np.concatenate([d_ab, d_ba])


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
    # fixed-point subpixel rasterization: a plain int cast truncates toward zero, which
    # systematically shrinks polygons with float (subpixel) vertices; integer vertices scale losslessly
    shift = 8
    pts = np.round(fitted * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], 1, cv2.LINE_8, shift)

    inter = np.logical_and(canvas, mask).sum()
    union = np.logical_or(canvas, mask).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def rms_distance(poly_a: np.ndarray, poly_b: np.ndarray, sample_step: float = 1.0) -> float:
    """Symmetric RMS boundary distance in pixels: RMS over the pooled point-to-edge distances.

    The segmentation-evaluation standard (ASSD family).
    Sees both failure modes: features the candidate dropped and geometry it invented.

    poly_a, poly_b : (N, 2) closed polylines in (x, y); first point must equal last
    """
    dists = _boundary_distances(poly_a, poly_b, sample_step)
    return float(np.sqrt(np.mean(dists ** 2)))


def rms_directed(src: np.ndarray, dst: np.ndarray, sample_step: float = 1.0) -> float:
    """Directed RMS boundary distance in pixels: densified `src` samples → `dst` edges.

    reference→fit is the curve-simplification literature's native error (what RDP's epsilon bounds).
    It sees dropped features but is blind to invented geometry, so report it next to `rms_distance`:
    symmetric ≫ directed flags a fit that invented geometry the reference lacks (e.g. overshot corners).

    src, dst : (N, 2) closed polylines in (x, y); first point must equal last
    """
    dists = _point_to_polyline_distances(_densify(src, sample_step), dst)
    return float(np.sqrt(np.mean(dists ** 2)))


def corner_metrics(
    gt_corners: np.ndarray, poly: np.ndarray, tau: float = 2.0
) -> tuple[float, float, float]:
    """Corner recall, precision, and localization error vs ground-truth corners.

    recall    — fraction of GT corners with a fitted vertex within `tau` px.
    precision — fraction of fitted vertices within `tau` px of a GT corner;
                penalizes spurious vertices (e.g. noise-tracking vertices
                along straight edges).
    loc_err   — mean GT-corner → nearest-vertex distance over the recalled
                corners (NaN if none recalled).

    Matching is nearest-neighbour, not one-to-one, so GT corners must be
    ≥ 2·tau apart or one vertex could recall two corners (the Tier 0 shape
    generator must enforce this).

    gt_corners : (K, 2) ground-truth corner positions in (x, y)
    poly       : (M, 2) closed fitted polyline in (x, y); first point equals last
    returns    : (recall, precision, loc_err)
    """
    verts = poly[:-1]
    dmat = np.linalg.norm(gt_corners[:, None, :] - verts[None, :, :], axis=2)
    d_corner = dmat.min(axis=1)   # per GT corner: distance to nearest vertex
    d_vertex = dmat.min(axis=0)   # per vertex: distance to nearest GT corner
    matched = d_corner <= tau
    recall = float(matched.mean())
    precision = float((d_vertex <= tau).mean())
    loc_err = float(d_corner[matched].mean()) if matched.any() else float("nan")
    return recall, precision, loc_err


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

    # --- 1b. HD95 vs max: single 10 px spike on the 400 px square perimeter ---
    # The spike dictates the max but contributes only ~2% of pooled samples, so hd95 ignores it.
    spiky = np.array(
        [[0, 0], [49, 0], [50, -10], [51, 0], [100, 0],
         [100, 100], [0, 100], [0, 0]],
        dtype=float,
    )
    h_max = hausdorff(sq, spiky)
    h95 = hd95(sq, spiky)
    print(f"hausdorff(sq, spiky) = {h_max:.2f}, hd95 = {h95:.2f}  (expect ≈ 10.0, ≈ 0.0)")
    assert 9.5 < h_max < 10.5, f"expected ≈10, got {h_max}"
    assert h95 < 1.0, f"expected hd95 ≈ 0, got {h95}"
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

    # --- 3. RMS distances: circle vs itself, then vs its bounding square ---
    rms0 = rms_distance(circle_closed, circle_closed)
    print(f"rms_distance(circle, circle) = {rms0:.6f}  (expect 0.0)")
    assert rms0 < 1e-9, f"expected 0, got {rms0}"
    print("  ✓")

    # Circle r=50 inside its bounding square: the two directions differ
    # (analytically ≈ 6.6 px for circle→square, ≈ 9.7 px for square→circle since the corners stick out),
    # and the symmetric value pools their samples, landing in between.
    sq_big = np.array(
        [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]], dtype=float
    )
    rms_c2s = rms_directed(circle_closed, sq_big)
    rms_s2c = rms_directed(sq_big, circle_closed)
    rms_sym = rms_distance(circle_closed, sq_big)
    print(f"rms_directed(circle→square) = {rms_c2s:.2f}  (expect ≈ 6.6)")
    print(f"rms_directed(square→circle) = {rms_s2c:.2f}  (expect ≈ 9.7)")
    print(f"rms_distance(symmetric)     = {rms_sym:.2f}  (expect between the two)")
    assert 5.5 < rms_c2s < 7.5, f"circle→square out of range: {rms_c2s}"
    assert 8.5 < rms_s2c < 10.5, f"square→circle out of range: {rms_s2c}"
    assert rms_c2s < rms_sym < rms_s2c, "symmetric RMS should lie between the directed values"
    print("  ✓")

    # --- 4. Corner metrics ---
    sq_corners = sq[:-1]  # the 4 GT corners of the square

    # exact square: all corners recalled at zero error, no spurious vertices
    rec, prec, err = corner_metrics(sq_corners, sq)
    print(f"corner_metrics(sq, sq) = recall {rec:.2f}, precision {prec:.2f}, loc_err {err:.4f}  (expect 1.0, 1.0, 0.0)")
    assert rec == 1.0 and prec == 1.0 and err < 1e-9
    print("  ✓")

    # vertices shifted by 1 px: still recalled (tau=2), loc_err ≈ sqrt(2)
    rec, prec, err = corner_metrics(sq_corners, sq + 1.0)
    print(f"corner_metrics(sq, sq+1) = recall {rec:.2f}, precision {prec:.2f}, loc_err {err:.4f}  (expect 1.0, 1.0, ≈1.41)")
    assert rec == 1.0 and prec == 1.0 and 1.3 < err < 1.5
    print("  ✓")

    # chamfered square (corners cut 5 px): nothing within tau=2 → recall 0
    ch = 5.0
    chamfered = np.array(
        [[ch, 0], [100 - ch, 0], [100, ch], [100, 100 - ch],
         [100 - ch, 100], [ch, 100], [0, 100 - ch], [0, ch], [ch, 0]],
        dtype=float,
    )
    rec, prec, err = corner_metrics(sq_corners, chamfered)
    print(f"corner_metrics(sq, chamfered 5px) = recall {rec:.2f}, precision {prec:.2f}  (expect 0.0, 0.0)")
    assert rec == 0.0 and prec == 0.0 and math.isnan(err)
    print("  ✓")

    # square with spurious edge-midpoint vertices: full recall, half precision
    sq_mid = np.array(
        [[0, 0], [50, 0], [100, 0], [100, 50], [100, 100],
         [50, 100], [0, 100], [0, 50], [0, 0]],
        dtype=float,
    )
    rec, prec, err = corner_metrics(sq_corners, sq_mid)
    print(f"corner_metrics(sq, sq+midpoints) = recall {rec:.2f}, precision {prec:.2f}  (expect 1.0, 0.5)")
    assert rec == 1.0 and prec == 0.5 and err < 1e-9
    print("  ✓")

    print("\nAll metric self-tests passed.")
