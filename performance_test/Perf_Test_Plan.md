# Mask2PolyMin — Performance & Quality Benchmark Plan

## Goal

Produce evidence for:

1. Mask2PolyMin yields far fewer segments with comparable accuracy (IoU, RMS, Hausdorff not much worse than RDP).
2. Mask2PolyMin preserves area and corners (no shrinkage, no rounding) better, especially on noisy masks from segmentation nets.

Secondary: compare runtime against `cv2.approxPolyDP` across contour lengths.

Note: "equivalent fidelity" is not a realistic target — Mask2PolyMin least-squares-fits and smooths noise, so its Hausdorff to the *noisy input contour* reads systematically higher than RDP's. On noisy masks the reference for fidelity metrics is the **ground-truth shape**, not the input contour.

## Metrics

| Metric | Question it answers | How to compute |
|---|---|---|
| **Segment count** | Shape simplicity / regularity | `len(segments)` |
| **Hausdorff distance** (symmetric, px) | Worst-case fidelity | densify both polylines, point-to-edge distances, take max |
| **IoU** (rasterized polygon vs. reference mask) | Area preservation | rasterize via `cv2.fillPoly`, compare bitmaps |
| **RMS perpendicular distance** (point → polyline) | Typical fidelity | per-point min distance to edges, RMS |
| **Corner recall** | Did corners survive? | fraction of GT corners with a fitted vertex within τ = 2 px |
| **Corner precision** | Spurious vertices? | fraction of fitted vertices within τ of a GT corner |
| **Corner localization error** | How precisely? | mean distance GT corner → nearest fitted vertex, over matched corners |
| **Time per contour** | Speed | `time.perf_counter()` |

Corner metrics require ground-truth corners → Tier 0 only. IoU alone is not a corner metric: rounding a corner by chamfer *d* costs only ~*d*²/2 px of area.

## Baselines

**Required: `cv2.approxPolyDP` (RDP).** `epsilon` = L∞ Hausdorff tolerance in px, `closed=True`. Wrapper in [baselines.py](baselines.py).

**Optional: Imai–Iri** (exact min-# under Hausdorff). Defer until after the RDP comparison.

## Tolerance alignment

Each algorithm runs on its native tolerance (RDP: L∞, Mask2PolyMin: L2/RMS); metrics are compared post hoc in shared metric space. Starting alignment `ε_rdp ≈ √2 · tolerance`:

| RDP `epsilon` (px) | 0.5 | 1.0 | 2.0 | 5.0 | 8.0 |
|---|---|---|---|---|---|
| Mask2PolyMin `tolerance` | 0.35 | 0.71 | 1.41 | 3.54 | 5.66 |

Set `segment_tolerance = global_tolerance = min_split_improvement = tolerance`. If the metric ranges don't overlap after the first sweep, extend the schedule.

## Datasets

### Tier 0 — synthetic regular shapes + simulated segmentation noise
The core benchmark for the corner/area claims.

1. Generate GT polygons with known corners: rectangles, L-shapes, hexagons, stars; varied sizes and rotation angles.
2. Rasterize, then distort like a segmentation net: Gaussian-blur + re-threshold (rounds corners), boundary jitter, small morphological noise. Noise level is a sweep axis.
3. Extract the contour of the distorted mask, feed to both algorithms.
4. Compute all metrics against the **GT polygon / GT mask**, not the distorted one.

Cheap, controlled, and where "preserves corners, no shrinkage" is actually demonstrable: RDP must pick vertices from the noisy boundary; Mask2PolyMin averages the noise away.

### Tier 1 — real masks: COCO val2017
- Download `val2017.zip` (~1 GB) + `annotations_trainval2017.zip`; masks via `coco.annToMask`, contours via `skimage.measure.find_contours`.
- Filter: 200–2000 contour points, single component, no holes. Deterministic sample: sort by annotation ID, take first 300.
- GT here is a human-drawn polygon, so this tier mainly supports the fewer-segments claim; fidelity metrics are vs. the input contour.

### Tier 2 — Cityscapes (optional)
GT polygons in `gtFine/*_polygons.json`; enables comparison with Polygon-RNN++ / Curve-GCN numbers. Only if publishing.

## Conventions

- All polylines are `(M, 2)` float arrays in **(x, y)** pixel space.
- **Closed contract:** first point equals last point for inputs and outputs of both wrappers; `rdp_opencv` gets `closed=True`.
- **IoU canvas** = original image shape, not the contour bounding box.

## Implementation

Code in [performance_test/](.). Gitignore `data/` and `results/`.

```
metrics.py               # hausdorff, iou_rasterized, rms_distance, corner metrics   [done except corners]
baselines.py             # rdp_opencv, mask2polymin wrappers + smoke test               [done]
synth_shapes.py          # Tier 0: GT polygons + mask distortion
fetch_coco.py            # Tier 1: download + cache
extract_contours.py      # Tier 1: masks → filtered contours (.npz)
run_benchmark.py         # sweep tolerances × contours × algorithms → raw.csv
plot_results.py          # figures + summary.csv
```

One row per (contour, algorithm, tolerance); failures are logged and skipped, not fatal:

```
contour_id, tier, n_input_points, algorithm, tolerance, noise_level,
n_segments, hausdorff, iou, rms_dist, corner_recall, corner_precision, corner_loc_err, wall_time_ms
```

Aggregate median / p25 / p75 / p95 per (tier, algorithm, tolerance, noise_level) → `summary.csv`.

## Build order

1. ~~`metrics.py` core~~ + `baselines.py` + smoke test — **done**.
2. ~~Corner metrics in `metrics.py` (`corner_metrics`: recall, precision, localization error)~~ — **done**.
3. `synth_shapes.py` + Tier 0 run — the headline results.
4. `fetch_coco.py` + `extract_contours.py` + Tier 1 run.
5. `plot_results.py`.

## Plots

1. **Segments vs. RMS (to GT)** — Tier 0, per noise level. Lower-left is better.
2. **Corner recall vs. noise level** — Tier 0, fixed tolerance. Headline corner figure.
3. **Segments vs. IoU** — Tier 1. Mark each algorithm's IoU noise floor (tightest tolerance); differences within the floor band are not wins.
4. **Wall time vs. contour length** — Tier 1, single tolerance.

Plus a table at the canonical tolerance (ε = 2.0 / tol = 1.41): median #segs, IoU, Hausdorff, corner recall, ms/contour.

## Report

Append a "Benchmarks" section to [../README.md](../README.md): setup paragraph, the figures, the table, one paragraph of honest interpretation (where Mask2PolyMin wins and where it doesn't), link here for reproducibility.
