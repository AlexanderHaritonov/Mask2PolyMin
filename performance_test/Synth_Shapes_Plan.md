# Tier 0 Shape Generator — Plan (`synth_shapes.py`)

Detail plan for step 3 of [Perf_Test_Plan.md](Perf_Test_Plan.md): the synthetic GT shapes,
their storage, and the distortion pipeline. Review gates are marked ⏸.

## Shape catalog — 10 families

| Family | Corners | Reflex | Why it is in the set |
|---|---|---|---|
| `rect` | 4 | – | 90° corners, elongation 1.8:1 (not just a square) |
| `lshape` | 6 | 1 | 90° corners plus one 270° concavity |
| `hexagon` | 6 | – | obtuse 120° corners — hardest to localize precisely |
| `star` (5-point) | 10 | 5 | acute ~45° tips — blur erodes them hardest; inner/outer radius ratio 0.45 |
| `tab` | 8 | 2 | the union-of-two-rectangles shape from [simple_bitmap_example.py](../examples/simple_bitmap_example.py), proportions 50×60 |
| `plane` | 27 | 11 | airliner pictogram (top view), traced from an icon via the fitter itself: engines, nose arc, tailplane. Detail-rich: min spacing 0.077 unit → smallest legal size for the traced geometry is ⌀ 128; the d64 slot uses a coarsened 25-vertex variant (min spacing 0.145 unit) |
| `house` | 9 | 2 | stylized house pictogram: gable roof + door notch in the bottom edge |
| `ship` | ~11 | yes | stylized ship pictogram: hull with pointed bow + stepped superstructure |
| `arrow` | 7 | 2 | map direction-arrow (shaft + triangular head): acute tip, two reflex barbs where head meets shaft; strongly orientation-dependent |
| `car` | 22 | 8 | stylized car rear silhouette: wide flat roof, obtuse shoulders, two door mirrors + two wheel bumps — small sub-features that tempt the fitter to drop them; takes the 64 px fallback (min spacing 0.150 unit) |

Pictogram vertex coordinates are hand-authored constants in the module, defined at unit
scale (circumscribed radius 1), mirror-symmetric where natural. Exact proportions are
judged at the shape-review gate, not in this document.

**Corner-spacing constraint:** `corner_metrics` matches nearest-neighbour, so GT corners
must stay ≥ 2·τ = 4 px apart (enforced at ≥ 4.5 px). At the smallest size (⌀ 48 px,
radius 24) this means ≥ 0.19 unit spacing between any two vertices — pictograms must
stay coarse (no chimneys, masts, or thin funnels). The generator asserts this for every
(shape, size). Fallback if a pictogram cannot meet it at ⌀ 48 without looking wrong:
raise that family's small size to 64 px rather than distort the design.
Outcome of the step-0 review: `car` takes the 64 px fallback; `plane` (traced, detail-rich)
needs ⌀ 128 as its smallest size; the other 8 families pass at ⌀ 48.
Step-1 review amendment: plane's small slot is filled at ⌀ 64 by a coarsened 25-vertex
variant (`SIZE_VARIANTS` in the generator) — engine pods 3 vertices instead of 4, wider
nose flat and tail/wingtip chords; d128/d320 keep the traced geometry. Caveat this
implies: for plane, cross-size comparisons mix a geometry change with the scale change.

## Dataset axes

| Axis | Values | Count |
|---|---|---|
| family | see catalog | 10 |
| size (circumscribed ⌀) | 48, 128, 320 px (car and plane: 64 instead of 48) | 3 |
| rotation | 0°, 10°, 22.5°, 37°, 45° | 5 |
| **base GT shapes** | 30 (family, size) cells × 5 angles | **150** |
| noise level | 0 (clean), 1–4 | 5 |
| seeds per noisy level | 3 (level 0 deterministic → 1) | |
| **distorted contours** | 150 × (1 + 4×3) | **1950** |

0° and 45° are the rasterization special cases (axis-aligned is RDP's best case and must
be included); 10°/22.5°/37° are generic angles where staircase aliasing is worst.
Benchmark volume: 1950 contours × 2 algorithms × 4 tolerances = 15 600 rows.
Per aggregation cell: 150 contours at level 0, 450 at levels 1–4.

## Storage — canonical GT files; the pipeline starts by loading them

The 30 JSON files are the canonical GT source: written once by the generator, approved
at the shape-review gate, and **committed to git** (small, text, diffable — a file in
gitignored `data/` would pin nothing). `dataset()` starts by loading them, applies the
5 rotation angles to `polygon_xy` (about the canvas centre), and rasterizes every mask
in memory (`cv2.fillPoly` with fixed-point `shift`).

The canonical format is the polygon, not the mask: rotation is applied to the polygon
**before** subpixel rasterization — never to a stored mask, which would corrupt the GT
with resampling artifacts. That is why the PNGs are derived artifacts: committed alongside so anyone can review
the shapes straight from the repo (in a PR or file browser) without running any code,
but never read by the pipeline. Each mask sits on a
per-shape square canvas of `size + 2 × 24 px` margin so distortion never touches the
border.

Step 1 writes the 30 PNG+JSON pairs (60 files) to `performance_test/gt_shapes/`
(committed, unlike the regenerable `data/`) — one per family × size, **0° rotation
only** — for close-up review, plus 4 gallery contact sheets to `shape_review/` — one
per size folder, listing exactly the families stored in it — on which all 150 instances
(including the 120 rotated variants, which get no individual files) appear as
thumbnails, each exactly once:

```
performance_test/gt_shapes/
  d048/
    rect_d048_a0.png   # GT mask, 0/255 (for IDE viewability; pipeline masks are {0, 1})
    rect_d048_a0.json  # GT description, schema below
    ...                # 8 pairs: one per family passing at ⌀ 48, 0° only
  d064/ ...            # car and plane (64 px fallback; plane's d64 is the coarse variant)
  d128/ ...            # all 10 families
  d320/ ...            # all 10 families
performance_test/shape_review/
  gallery_d048.png     # contact sheets, one per size folder: its families × 5 angles,
  gallery_d064.png     #   mask + GT polygon (red) + corner dots
  gallery_d128.png
  gallery_d320.png
```

JSON schema — self-contained GT description; the PNG is derived from `polygon_xy`:

```jsonc
{
  "shape_id": "house_d128_a0",         // matches the PNG basename
  "family": "house",
  "size_px": 128,                      // circumscribed diameter (folder axis)
  "angle_deg": 0.0,
  "canvas_hw": [176, 176],             // size + 2 × 24 margin
  "polygon_xy": [[x, y], ...],         // closed (first == last), float canvas pixel coords
  "corners_xy": [[x, y], ...],         // GT corners for corner_metrics (= vertices without closing dup)
  "params": {"margin_px": 24, "rect_aspect": 1.8, "star_inner_ratio": 0.45}
}
```

Deliberately absent: noise parameters (these files are pure GT) and contours (extracted
at benchmark time from distorted masks). Distorted masks are not stored either — they
are regenerated deterministically from per-record seeds (derived from shape index +
noise level + rep), so any benchmark row can be reproduced in isolation.

Regenerating the GT files is an explicit action (`synth_shapes.py --write-gt`), never a
side effect of a benchmark run. Floats are serialized at full precision with stable key
order, so regenerating unchanged shapes is byte-identical; any git diff under
`gt_shapes/` means the GT changed and existing results are stale.

## Distortion pipeline (step 2, after shape review)

Order: elastic jitter → blur + re-threshold → speckle. Parameters per level:

| Level | Blur σ (px) | Jitter amp (px) | Speckle p | Reads as |
|---|---|---|---|---|
| 0 | – | – | – | clean rasterization (staircase only) |
| 1 | 0.5 | 0.5 | – | good segmentation net |
| 2 | 1.0 | 1.0 | – | decent segmentation net |
| 3 | 1.5 | 1.75 | 0.03 | mediocre segmentation net |
| 4 | 2.0 | 2.5 | 0.06 | sloppy segmentation net |

Levels 2 and 4 are the noise-review-gate anchors (formerly levels 1 and 2); levels 1
and 3 interpolate between the reviewed settings.

- *Elastic jitter*: smooth random displacement field (Gaussian-smoothed white noise,
  correlation length ~8 px) applied via `cv2.remap` — correlated boundary wobble, not
  salt-and-pepper.
- *Blur + re-threshold at 0.5* — rounds corners; the effect the benchmark is about.
- *Speckle*: pixel flips restricted to a ±2 px boundary band, then 3×3 close/open to
  make the noise blobby; levels 3–4 only.

Exact numbers are tuned at the noise-review gate. Contours are extracted from the
distorted mask with `cv2.findContours` (`RETR_EXTERNAL`, `CHAIN_APPROX_NONE`, largest
contour by area) — integer pixel coordinates, exactly what a real user of the library
feeds the fitter. Tier 0 keeps cv2 extraction; skimage subpixel contours remain Tier 1.

## Build steps
0. ~~create the 10 main shapes one by one and get it reviewed. store them as files.~~ —
   **done**: all 10 families reviewed and accepted; unit-scale constants in
   `synth_shapes.py` (`FAMILIES`), review renders in `shape_review/`.

1. ~~**GT shapes**: unit-scale family definitions, place (scale/rotate/center),
   rasterize, write the 30 stored pairs and the 4 gallery sheets per the layout
   above.~~ — **done**: canonical pairs in `gt_shapes/`, galleries in `shape_review/`;
   regeneration is byte-identical.
   ✅ **Review gate: shapes** (closed) — proportions, star sharpness, pictogram designs approved.
2. **Distortion + dataset iterator**:
   - ~~`distort(mask, level, rng)`, `extract_contour`, and a `dataset(reps=3)`
     generator (loads the canonical JSONs, applies rotations, rasterizes in memory;
     yields one record per (shape, level, rep)) as the single enumeration point for
     `run_benchmark.py`; render `shape_review/preview_noise.png`.~~ — **done**:
     per-record seeds reproduce contours bit-for-bit in isolation.
     ✅ **Review gate: noise levels** (closed) — the reviewed settings are anchors 2
     and 4 of the 5-level ladder; levels 1 and 3 interpolate them.
   - ~~**The Tier 0 run** (details in [Perf_Test_Plan.md](Perf_Test_Plan.md)):
     `run_benchmark.py` consumes `dataset()`; for each of the 1950 contours it runs
     both algorithms at the 4 tolerance pairs (ε = 0.5/1/2/4, tol = ε/√2) →
     15 600 rows in gitignored `results/raw.csv`, failures logged and skipped.~~ —
     **done** for the earlier 3-level ladder (8 400 rows, 0 failures); rerun at 5
     levels pending. Aggregation (median/p25/p75/p95 → `summary.csv` + printed
     median table) and the Tier 0 figures in `plot_results.py`.
3. ~~Add `performance_test/data/` and `performance_test/results/` to `.gitignore`
   (`performance_test/gt_shapes/` stays committed).~~ — **done**, progress marked in
   [Perf_Test_Plan.md](Perf_Test_Plan.md).
