# Mask2PolyMin
Package to turn noisy raster segmentation masks into clean polygons with a minimal number of segments.

Useful for post‑processing bitmask segmentation outputs from models such as MaskRCNN or YOLO‑Seg, especially when regular or low‑complexity shapes are required:
- obtaining simple geometric representations
- to reconstruct artificial objects that consist of straight edges, sharp corners, and regular geometric properties.

Unlike common point‑thinning algorithms (Ramer–Douglas–Peucker, Visvalingam–Whyatt, Zhang–Suen), this method:
- minimizes segment count while preserving the raw shape
- does not shrink the area or remove corners
- reconstructs corners with sub-pixel accuracy: vertices are intersections of least-squares fitted lines.

## Algorithm

### Orphaned junction points
A junction point — where one fitted segment ends and the next begins — is often an outlier
to one or both segments, and in a least-squares fit an outlier at the segment's end has
disproportionately large influence. A single misplaced pixel can rotate the fitted line and
drag the reconstructed vertex.
Mask2PolyMin therefore may leave up to `max_orphans_per_junction` (default=2) point(s) at each junction *orphaned* — assigned to no segment: a point is orphaned iff it lies farther than `tolerance` from both adjacent lines, and the orphans' mean then anchors the corner reconstruction.

## Example

python -m venv .venv && source .venv/bin/activate && pip install -r requirements-examples.txt && python example_usage.py

- The input is a dense bitmask produced by a segmentation model.

![input bitmask](step1_bitmap.png)
- A contour is extracted from the bitmask using skimage.measure.find_contours

![extracted contour](step2_contour.png)

- Mask2PolyMin fits a minimal‑segment polyline to this contour

![fitted segments](step3_fitted_segments.png)

- `segments_to_polyline()` converts the fitted segments to a closed polygon of float (sub-pixel) vertices, ready for GeoJSON/SVG/COCO export

## Performance
The implementation is optimized, uses NumPy broadcasting.

## future work
- performance tests and comparison

## Running Tests

```bash
.venv/bin/pytest test/
```

Tests run headless by default (no plot windows). To show plots during a test run:

```bash
SHOW_PLOTS=1 .venv/bin/pytest test/
```

In VSCode, right-click the `test/` folder → **Run Tests**. Plots are shown automatically (configured via `.vscode/settings.json`).

Install dev dependencies first if needed: `pip install -r requirements-dev.txt`



