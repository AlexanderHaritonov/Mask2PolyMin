# Mask2PolyMin
Turn noisy raster segmentation masks into clean polygons with a minimal number of segments.

Useful for post‑processing bitmask segmentation outputs from models such as MaskRCNN or YOLO‑Seg, especially when regular or low‑complexity shapes are required:
- obtaining simple geometric representations
- to reconstruct artificial objects that consist of straight edges, sharp corners, and regular geometric properties.

Unlike common point‑thinning algorithms (Ramer–Douglas–Peucker, Visvalingam–Whyatt, Zhang–Suen), this method:
- minimizes segment count while preserving the raw shape
- does not shrink the area or remove corners
- reconstructs corners with sub-pixel accuracy: vertices are intersections of least-squares fitted lines.

## Quick Start

```bash
git clone https://github.com/AlexanderHaritonov/Mask2PolyMin.git
cd Mask2PolyMin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-examples.txt
python example_usage.py
```

## Parameters
**max_segments_count** (default `18`): Upper limit on the number of segments in the output polygon. Keeping this bound relatively tight prevents over-fitting to noise and generally improves reconstructed shape accuracy.

**`tolerance`** (default `1.0`): the maximum perpendicular deviation, in input units (pixels), that a fitted line may have from the points it represents. Roughly, `tolerance ≈ epsilon / √2`, where `epsilon` is what you'd pass to Ramer–Douglas–Peucker — RDP's `epsilon` bounds the max (L∞) deviation, while `tolerance` bounds the L2 deviation.

**Rule of thumb:** `tolerance ≈ max(1.0, jitter_amp)`, where `jitter_amp` how noisy the segmentation is - the standard deviation of how far the mask's boundary randomly wanders from its true edge.
The `1.0` floor covers ordinary pixel-quantization jitter present even in a "clean" mask.

**rank_split_by_max_deviation** (default `False`)
Pass True to slightly improve simpler shapes (low segments count, enough space between opposite sides). Can damage complex shapes.

**apply_local_defect_margin** (default `True`)
Setting this to False speed up the algorithm by ~30% on complex shapes (~45% on simple) at the cost of an ~11% corner-recall regression on complex ones.

## Algorithm

The input is an ordered sequence of contour points, open or closed. Lines are fitted by total least squares (minimizing perpendicular distances), and the segmentation is refined top-down:

1. **Fit** a single line to the whole sequence.
2. **Split** the worst-fitting segment at its midpoint — a segment needs splitting when its mean squared deviation exceeds `tolerance²` or any single point lies farther than `tolerance` from its line.
3. **Adjust**: slide each junction between neighboring segments to the cut with the lowest total squared error, re-fitting the segments as points change sides; a point far from both lines may be left orphaned. Iterate until stable.
4. **Repeat** 2–3 until the average squared-error sum per segment is within `tolerance²`, a split no longer improves it by at least that much, or `max_segments_count` is reached.
5. **Merge** adjacent segments whose combined points still fit a single line within tolerance.
6. **Reconstruct vertices**: each corner is the intersection of the two adjacent fitted lines — sub-pixel accurate even when no input point lies at the true corner.

Thanks to precomputed cumulative moments of the sequence, fitting a line to any contiguous point range is O(1).

### Orphaned junction points
A junction point — where one fitted segment ends and the next begins — is often an outlier to one or both segments, and in a least-squares fit an outlier at the segment's end has disproportionately large influence. A single misplaced pixel can rotate the fitted line and drag the reconstructed vertex.
Mask2PolyMin therefore may leave up to 2 points at each junction *orphaned* — assigned to no segment: a point is orphaned iff it lies farther than `tolerance` from both adjacent lines, and the orphans' mean then anchors the corner reconstruction.

## Input conventions

`FitterToPointsSequence` takes a dense, ordered contour as a float `(N, 2)` array and is agnostic to what the two columns mean: it never interprets the axes, and the returned vertices are in the same coordinate system as the input. `tolerance` is in input units.

- Input **Dense contours, not sparse polygons**!
- **Axis order doesn't matter** — `(row, col)` from skimage and `(x, y)` from OpenCV both work; output vertices keep the input's order.
- **Closed contours**: pass `is_closed=True`; a duplicated closing point (skimage-style) is detected and stripped automatically.

Notes for the two common contour sources:

| | `skimage.measure.find_contours` | `cv2.findContours` |
|---|---|---|
| axis order | `(row, col)` | `(x, y)` |
| coordinates | float, sub-pixel | integer pixel indices |
| boundary semantics | between pixel centers (half-integers at `level=0.5`) | through the centers of the outermost object pixels — ~0.5 px inside the true region edge |
| array shape | `(N, 2)` | `(N, 1, 2)` accepted directly — cv2's general contour shape |
| density | dense | dense only with `CHAIN_APPROX_NONE` |

- With OpenCV, use `cv2.findContours(..., cv2.CHAIN_APPROX_NONE)`: the common `CHAIN_APPROX_SIMPLE` pre-simplifies collinear runs, starving the least-squares fits of exactly the evidence this algorithm relies on.
- The half-pixel difference in boundary semantics is deliberate, and the fitter does not compensate — vertices come back in the input's own convention. Account for it when comparing results from different contour extractors, or against the original mask.

## Example

Running `example_usage.py` (see [Quick Start](#quick-start)) walks through the following steps:

- The input is a dense bitmask produced by a segmentation model.

![input bitmask](https://raw.githubusercontent.com/AlexanderHaritonov/Mask2PolyMin/main/docs/step1_bitmap.png)
- A contour is extracted from the bitmask using skimage.measure.find_contours

![extracted contour](https://raw.githubusercontent.com/AlexanderHaritonov/Mask2PolyMin/main/docs/step2_contour.png)

- Mask2PolyMin fits a minimal‑segment polyline to this contour

![fitted segments](https://raw.githubusercontent.com/AlexanderHaritonov/Mask2PolyMin/main/docs/step3_fitted_segments.png)

- `fit()` returns `(polygon, segments)`: a closed polygon of float (sub-pixel) vertices, ready for GeoJSON/SVG/COCO export, plus the underlying fitted segments

## Performance
The implementation is optimized, uses NumPy broadcasting.

Benchmarked against RDP (`cv2.approxPolyDP`) on synthetic shapes across noise levels
([performance_test/](performance_test/)): comparable on most fidelity metrics (IoU, RMS, Hausdorff). But Mask2PolyMin avoids corner-cutting bias — see [corner_bias](performance_test/charts/fig6_corner_bias.png), [corner_bias comparison](performance_test/charts/comparison_corner_bias.png) and perimeter shrinkage in [perimeter_ratio](performance_test/charts/fig8_perimeter.png), [perimeter_ratio comparison](performance_test/charts/comparison_perimeter.png). Tradeoff: Mask2PolyMin is far slower — although not dramatically slow in absolute terms: 63 ms per contour on average, even on a weak laptop (Intel i5-12450H, UHD Graphics), single-threaded — see [wall-clock time](performance_test/charts/fig11_walltime.png).

## future work and ideas
- explore line fitting with Theil–Sen and respectively the Median or Mean Absolute Error as stop criterion ?

## Running Tests

```bash
.venv/bin/pytest test/
```

Tests run headless by default (no plot windows). To show plots during a test run:

```bash
SHOW_PLOTS=1 .venv/bin/pytest test/
```

Install dev dependencies first if needed: `pip install -r requirements-dev.txt` — this installs the package itself in editable mode (`-e .`), so no path tricks are needed to import `mask2polymin`.



