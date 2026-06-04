# PolyFit2D — Performance & Quality Benchmark Plan

## Goal

Produce evidence — not rhetoric — for the two claims currently in [../README.md](../README.md):

1. PolyFit2D yields fewer segments than common simplifiers at equivalent fidelity.
2. PolyFit2D preserves area and corners (no shrinkage, no rounding) better than common simplifiers at equivalent segment count.

Secondary goal: characterize wall-time vs. contour length against the practical baseline (`cv2.approxPolyDP`).

The output of this work is a benchmark harness, a [performance_test/](.) folder of result CSVs and plots, and a short report section appended to the README.

## Metrics

Three primary metrics, each answering a distinct question. 

| Metric | Question it answers | How to compute |
|---|---|---|
| **Segment count** | Representation cost | `len(segments)` |
| **Hausdorff distance** (symmetric, in pixels) | Worst-case fidelity | `scipy.spatial.distance.directed_hausdorff` both directions, take max |
| **IoU** (rasterized polygon vs. original mask) | Area / corner preservation | rasterize fitted polygon → bitmap, compare with input mask |
| Mean distance (point → polyline) | Average fidelity (supplementary) | per-point min distance, mean |
| Wall time per contour | Speed | `time.perf_counter()` around the fit call |

Rasterization for IoU: `cv2.fillPoly`.

## Baselines

### Required: OpenCV `cv2.approxPolyDP` (RDP)
Already installed (system Python). 

```python
import cv2
import numpy as np

# skimage gives (row, col) float; cv2 wants (x, y) int32 with shape (N, 1, 2)
pts = np.flip(contour, axis=1).astype(np.float32)
pts_cv = pts.reshape(-1, 1, 2).astype(np.int32)
approx = cv2.approxPolyDP(pts_cv, epsilon=tolerance, closed=True)
approx_xy = approx.reshape(-1, 2)  # (M, 2) in (x, y)
```

`epsilon` is the Hausdorff tolerance in pixels. `closed=True` for closed contours.

### Optional: Imai–Iri (exact min-# under Hausdorff)

No trustworthy pip package known. If we want the optimality-gap number, write it ourselves — the DAG-shortest-path version is short:

1. For each pair `(i, j)` with `i < j` (or wrapping for closed contours), check if the chord from point i to point j approximates the subarc `i..j` within ε. This is an O(n³) edge-feasibility scan; acceptable for benchmark contours of a few hundred points.
2. BFS from vertex 0 to vertex n-1 over feasible edges → fewest-segments path.
3. For closed contours: pick a starting vertex, run BFS, repeat for several start vertices and take the min, or solve the cyclic version directly.

Defer this until after the RDP comparison is done. Only invest the day if the RDP results are promising and we want a "near-optimal" claim.

## Tolerance alignment

Two algorithms, two native tolerances. We use the **native-tolerance / post-hoc-metric** convention: each algorithm runs on its own tolerance, then we measure the same downstream metrics (Hausdorff, IoU, mean distance) and plot in shared metric space.

The native tolerances are aligned via the cosine-residual model `ε_rdp ≈ √2 · segment_tolerance` to start the sweeps in roughly comparable regimes,
as RDP uses the L∞ tolerance, Polifit2D user L2 tolerance:

| RDP `epsilon` (px) | PolyFit2D `tolerance` ≈ ε / √2 (px) |
|---|---|
| 0.5 | 0.35 |
| 1.0 | 0.71 |
| 1.5 | 1.06 |
| 2.0 | 1.41 |
| 3.0 | 2.12 |
| 5.0 | 3.54 |
| 8.0 | 5.66 |

For the first pass, set `segment_tolerance = global_tolerance = min_split_improvement = tolerance`. Studying the three knobs as orthogonal axes is a separate experiment.

The √2 mapping is a starting point. After the first sweep, verify both algorithms' Hausdorff ranges overlap on the plot; if not, extend the tolerance schedule at the under-covered end and re-run.

## Datasets

Use real segmentation masks, not synthetic. Three options, in order of effort:

### Tier 1 — quick start: COCO val2017 instance masks
- Free, no registration. Download: `http://images.cocodataset.org/zips/val2017.zip` (~1 GB) and `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`.
- Pull masks: `coco.annToMask(ann)` returns a binary mask. Run `skimage.measure.find_contours` on it to get the input contour for the fitter.
- Filter: keep masks with ≥ 200 contour points (otherwise the simplification problem is trivial), single connected component, no holes for the first pass.
- Sample size: 300 masks across diverse categories gives tight enough error bars for the comparison. Bump higher only if curves overlap.

### Tier 2 — published benchmark comparability: Cityscapes
- Requires free registration at cityscapes-dataset.com.
- Instance polygons in `gtFine/*_polygons.json`. Use these directly (already polygons) as the *ground truth*, rasterize to masks, extract contour, refit, compare back to ground-truth polygons.
- Useful only if we want numbers comparable to Polygon-RNN++ / Curve-GCN papers. Skip for the first benchmark pass.

### Tier 3 — stress shapes: ADE20K
- Free download. Broader category set; useful for finding pathological contours (very thin, many concavities). Skip unless Tier 1 results raise questions.

Recommendation: start with **Tier 1 (COCO val2017, 300 masks)**. Add Tier 2 only if publishing.

## Coordinate convention

All polylines throughout the benchmark are in **(x, y) pixel space** (column, row). This is the contract enforced in `baselines.py`:

- `cv2.findContours` returns contours as `(N, 1, 2)` int32 arrays in `(x, y)` order. Reshape to `(N, 2)` with `.reshape(-1, 2)` before passing to any wrapper.
- Both wrappers (`rdp_opencv` and `polyfit2d`) receive and return `(M, 2)` float arrays in `(x, y)`.
- `iou_rasterized` receives the polyline in `(x, y)` and reshapes to `(M, 1, 2)` int32 for `cv2.fillPoly` internally.

**Closed-contour contract:** both algorithms operate on closed polygons. The input contour passed to each wrapper must have its first point equal to its last point. `cv2.findContours` does not guarantee this, so `extract_contours.py` appends `contour[0]` if needed before saving. PolyFit2D enforces this internally as well; `rdp_opencv` receives `closed=True`.

**IoU canvas size:** the canvas passed to `cv2.fillPoly` is the original image shape (height × width from the COCO annotation), not the contour bounding box. This avoids edge-clipping artefacts for masks that touch the image boundary.

## Sampling reproducibility

The 300-contour sample must be deterministic. After filtering (≥ 200 contour points, single connected component, no holes), sort by COCO annotation ID and take the first 300. This means the same set is used on every run without a random seed.

## Implementation outline

Code lives in this folder ([performance_test/](.)). Add to [../.gitignore](../.gitignore): `performance_test/data/`, `performance_test/results/`.

```
performance_test/
  Perf_Test_Plan.md        # this file
  fetch_coco.py            # download + cache COCO val2017 if missing
  extract_contours.py      # iterate masks, find_contours, filter, save .npz
  run_benchmark.py         # for each contour, sweep tolerance, run polyfit2d + cv2 + (optional) imai_iri
  metrics.py               # hausdorff, iou_rasterized, mean_distance
  baselines.py             # wrappers: rdp_opencv(contour, eps), polyfit2d(contour, tol), imai_iri(contour, eps)
  plot_results.py          # produce the figures listed below
  data/                    # (gitignored) COCO download + extracted contours
  results/
    raw.csv                # one row per (contour_id, algorithm, tolerance)
    summary.csv            # aggregated
    *.png                  # plots
```

Sweep schedule (linear pixels):
- **RDP `epsilon`**: `{0.5, 1.0, 2.0, 5.0, 8.0}`
- **PolyFit2D `tolerance`**: `{0.35, 0.71, 1.41, 3.54, 5.66}`

Each algorithm runs once per (contour, tolerance). Wall time is the single `time.perf_counter()` wrap around that call; aggregation across hundreds of contours provides the noise rejection. Output row schema:

```
contour_id, n_input_points, algorithm, tolerance,
n_segments, hausdorff, iou, mean_dist, wall_time_ms
```

Aggregate (median, p25, p75, p95) across `contour_id` for each `(algorithm, tolerance)` pair → `summary.csv`.

## Build order

1. **`metrics.py`** — `hausdorff`, `iou_rasterized`, `mean_distance`. Verify on synthetic shapes (square, noisy circle).
2. **`baselines.py`** — `rdp_opencv(contour, eps)` and `polyfit2d(contour, tol)` wrappers returning a uniform `(M, 2)` polyline.
3. **Smoke test** — one synthetic noisy-circle contour, both algorithms at one tolerance each, print the metrics. Sanity check: at tolerance → 0, IoU should be ≥ 0.99 for both.
4. **`fetch_coco.py` + `extract_contours.py`** — download once, cache contours as `.npz`. Filter: ≥ 200 and ≤ 2000 contour points, single connected component, no holes. The upper cap keeps wall-time statistics interpretable and prevents a handful of very large contours from dominating the timing results; contours above the cap are simply skipped (logged).
5. **`run_benchmark.py`** — full sweep on cached contours. Each (contour, algorithm, tolerance) call is wrapped in a `try/except`; on any exception, print `contour_id`, algorithm, tolerance, and the error to stderr and continue. Contours that fail for every algorithm at every tolerance are flagged for inspection but do not abort the run.
6. **`plot_results.py`** — three figures + summary table.

## Plots in the report

Three figures, each with PolyFit2D and OpenCV-RDP curves (and Imai–Iri if implemented):

1. **Segments vs. Hausdorff** — x: Hausdorff (px), y: median segment count. Shaded band: p25–p75. Lower-left is better.
2. **Segments vs. IoU** — x: median segment count, y: median IoU. Upper-left is better. This is the headline figure for the area-preservation claim. Draw a horizontal dashed line per algorithm at its median IoU at the tightest sweep tolerance — this is a per-algorithm noise floor (rasterization discreteness + algorithm-specific boundary-pixel behaviour mean the floor is not 1.0 and differs between methods). Any IoU difference between algorithms that falls within the noise-floor band must not be claimed as a win.
3. **Wall time vs. contour length** — x: input-contour length (binned), y: median wall time per contour. Single tolerance (e.g. ε = 2.0). Two curves.

Plus a small table of aggregate numbers at one canonical tolerance (ε = 2.0 px / `tolerance` ≈ 1.41 px):

| algorithm | median #segs | median IoU | median Hausdorff | median ms/contour |
|---|---|---|---|---|

## Report structure

Append a section to [../README.md](../README.md) titled "Benchmarks":

- One paragraph: dataset (COCO val2017, N contours), tolerance sweep, hardware.
- The three plots inline.
- The table.
- One paragraph of honest interpretation: where PolyFit2D wins, where it doesn't, what contour regimes (length, complexity) favor each method.
- Link to [performance_test/](.) so results are reproducible.

If results are good, this is also the basis for a blog post / arXiv note. If results are mixed, it tells us where to focus algorithmic work before packaging.
