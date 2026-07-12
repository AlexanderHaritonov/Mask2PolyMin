"""
Algorithm wrappers for the Mask2PolyMin benchmark.

Both wrappers obey the same contract:
    contour: (N, 2) float array in (x, y), first point equals last point (closed).
    returns: (M, 2) float array in (x, y), first point equals last point (closed).

Tolerances are in linear pixels.
- rdp_opencv:  epsilon = L∞ Hausdorff bound (cv2 native).
- mask2polymin:   tolerance = RMS perpendicular distance bound (Mask2PolyMin native).
"""
import numpy as np
import cv2

from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig


def rdp_opencv(contour: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker via cv2.approxPolyDP. Returns closed polyline."""
    pts_cv = contour.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(pts_cv, epsilon=float(epsilon), closed=True)
    poly = approx.reshape(-1, 2).astype(np.float64)
    # cv2 omits the closing duplicate. restore it.
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def mask2polymin(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """Mask2PolyMin fitter. Returns closed polyline of fitted-line intersections.

    max_segments_count is set to len(contour) so the default cap of 30 cannot
    silently truncate tight-tolerance sweeps; fairness with RDP, which has no
    such cap, requires this.
    """
    config = FitterConfig(tolerance=float(tolerance), max_segments_count=len(contour))
    segments, _ = FitterToPointsSequence(contour, is_closed=True, config=config).fit()
    return segments_to_polyline(segments, is_closed=True, tolerance=config.tolerance)


# ---------------------------------------------------------------------------
# Smoke test: run with  python -m performance_test.baselines  (from repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    import time
    from performance_test.metrics import hausdorff, iou_rasterized, rms_distance

    # Noisy circle, radius 50, centre (150, 150), 500 points, sub-pixel jitter.
    N = 500
    theta = np.linspace(0, 2 * math.pi, N, endpoint=False)
    cx, cy, r = 150.0, 150.0, 50.0
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.3, size=N)
    contour = np.stack(
        [cx + (r + noise) * np.cos(theta), cy + (r + noise) * np.sin(theta)], axis=1
    )
    contour_closed = np.vstack([contour, contour[0]])

    # Reference mask for IoU: rasterize the dense contour itself.
    H, W = 300, 300
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [contour_closed.reshape(-1, 1, 2).astype(np.int32)], 1)

    print(f"input contour: {len(contour_closed)} points (closed)")
    print()

    # --- both algorithms at one tolerance ---
    eps = 1.0
    tol = 0.71  # ε / √2 per the plan

    t0 = time.perf_counter()
    rdp_poly = rdp_opencv(contour_closed, eps)
    t_rdp = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    m2p_poly = mask2polymin(contour_closed, tol)
    t_m2p = (time.perf_counter() - t0) * 1000

    print(f"{'algorithm':<12} {'tol':>5} {'#segs':>6} {'hausdorff':>10} {'IoU':>8} {'rms_d':>8} {'ms':>7}")
    for name, poly, t, used_tol in [
        ("RDP",       rdp_poly, t_rdp, eps),
        ("Mask2PolyMin", m2p_poly,  t_m2p,  tol),
    ]:
        h = hausdorff(contour_closed, poly)
        iou = iou_rasterized(poly, mask)
        md = rms_distance(contour_closed, poly)
        # poly closed → segment count is len-1
        n_seg = len(poly) - 1
        print(f"{name:<12} {used_tol:>5.2f} {n_seg:>6d} {h:>10.4f} {iou:>8.4f} {md:>8.4f} {t:>7.2f}")

    # --- tight-tolerance behaviour, informational ---
    # Per the plan, step 5 (IoU noise-floor measurement) is what calibrates
    # per-algorithm floors. The two algorithms differ qualitatively here:
    #   - RDP picks subset vertices; tight ε drives it toward N-vertex retention.
    #   - Mask2PolyMin fits least-squares lines and stops when splits no longer
    #     improve the fit by ~tolerance² of SSE per segment, so on a noisy
    #     circle it converges to a smooth fit, not the input.
    # We print both floors for visual inspection but only assert wrappers work.
    print("\nTight-tolerance behaviour (algorithm-specific floors expected):")
    rdp_tight = rdp_opencv(contour_closed, 0.1)
    m2p_tight = mask2polymin(contour_closed, 0.1)
    iou_rdp_tight = iou_rasterized(rdp_tight, mask)
    iou_m2p_tight = iou_rasterized(m2p_tight, mask)
    print(f"  RDP          eps=0.1: IoU = {iou_rdp_tight:.4f}  ({len(rdp_tight) - 1} segs)")
    print(f"  Mask2PolyMin tol=0.1: IoU = {iou_m2p_tight:.4f}  ({len(m2p_tight) - 1} segs)")

    # --- wrapper contract: closed, sane IoU at the matched-tolerance pair ---
    assert np.array_equal(rdp_poly[0], rdp_poly[-1]), "RDP polyline not closed"
    assert np.array_equal(m2p_poly[0], m2p_poly[-1]), "Mask2PolyMin polyline not closed"
    assert iou_rasterized(rdp_poly, mask) > 0.95, "RDP basic IoU too low"
    assert iou_rasterized(m2p_poly, mask) > 0.95, "Mask2PolyMin basic IoU too low"

    print("\nSmoke test passed.")
