# Performance benchmark

Mask2PolyMin vs `cv2.approxPolyDP` (RDP) on synthetic GT shapes with simulated segmentation noise.

Design and rationale: [Perf_Test_Plan.md](Perf_Test_Plan.md), [Synth_Shapes_Plan.md](Synth_Shapes_Plan.md). Run all commands from this folder.

## Choosing tolerance for a given noise level

`NOISE_LEVELS` in [synth_shapes.py](synth_shapes.py) defines each level's `jitter_amp` 
— the per-pixel standard deviation of the elastic boundary displacement used to simulate segmentation noise. `run_benchmark.matched_pair()` derives that level's `(rdp_epsilon, mask2polymin_tolerance)` from it:

```
tolerance = max(1.0, jitter_amp)      # the 1.0 floor covers pixel-quantization jitter present even in a clean mask
epsilon   = tolerance · √2
```

| noise level | jitter_amp (px) | tolerance | epsilon |
|---|---|---|---|
| 0 (clean) | 0.0 | 1.0 | 1.41 |
| 1 (good segmentation net) | 0.5 | 1.0 | 1.41 |
| 2 (decent) | 1.0 | 1.0 | 1.41 |
| 3 (mediocre) | 1.75 | 1.75 | 2.47 |
| 4 (sloppy) | 2.5 | 2.5 | 3.54 |

The benchmark runs each contour once, at its own noise level's matched pair.

## 1. GT shapes — committed, regenerate only after a design change

```bash
python synth_shapes.py --write-gt
```

Produces `gt_shapes/dXXX/*.png|json` (30 canonical GT pairs) and 4 gallery sheets in `shape_review/`. 
Optional reviews: `--preview` (per-family renders),
`--preview-noise` (`shape_review/preview_noise.png`, noise levels).

## 2. Benchmark sweep

```bash
python run_benchmark.py            # ~10 min; --limit N for a quick smoke pass
```

Produces `summarized_csvs/raw.csv` (gitignored): 3900 rows = 1950 contours × 2 algorithms, one row per run.

## 3. Aggregate + figures

```bash
python plot_results.py
```

Produces `summarized_csvs/summary.csv` (mean/median per algorithm × noise level × shape class), prints the median table,
and renders `charts/`:
[fig1_segments_vs_rms.png](charts/fig1_segments_vs_rms.png),
[fig2_corner_recall.png](charts/fig2_corner_recall.png),
[fig3_hausdorff.png](charts/fig3_hausdorff.png),
[fig4_rms.png](charts/fig4_rms.png),
[fig5_corner_loc_err.png](charts/fig5_corner_loc_err.png),
[fig6_corner_bias.png](charts/fig6_corner_bias.png),
[fig7_area.png](charts/fig7_area.png),
[fig8_perimeter.png](charts/fig8_perimeter.png),
[fig9_iou.png](charts/fig9_iou.png),
[fig10_corner_angle.png](charts/fig10_corner_angle.png)
-- each split simple vs. complex (car/plane/ship) shapes -- plus
[fig11_walltime.png](charts/fig11_walltime.png) (median wall time vs noise, log scale)
and `summarized_csvs/runtime_summary.csv` (mean/p95 wall time per algorithm, pooled and per image size).

## 4. Per-metric win/loss comparison

```bash
python plot_comparison.py
```

Produces `summarized_csvs/comparison_summary.csv` (one row per metric × noise level × shape class:
`n_pairs`, `stat`, `comparison_center`, `win_count`) and renders `charts/`:
[comparison_segments.png](charts/comparison_segments.png),
[comparison_hausdorff.png](charts/comparison_hausdorff.png),
[comparison_rms.png](charts/comparison_rms.png),
[comparison_corner_recall_precision.png](charts/comparison_corner_recall_precision.png),
[comparison_corner_loc_err.png](charts/comparison_corner_loc_err.png),
[comparison_corner_bias.png](charts/comparison_corner_bias.png),
[comparison_area.png](charts/comparison_area.png),
[comparison_perimeter.png](charts/comparison_perimeter.png),
[comparison_iou.png](charts/comparison_iou.png),
[comparison_corner_angle.png](charts/comparison_corner_angle.png)
-- each split simple vs. complex (car/plane/ship) shapes; `wall_time_ms` excluded (see
`fig11_walltime.png` above for that).

Unlike section 3's aggregate-of-medians view, this pairs rdp and mask2polymin on the
*same* noisy contour (same `contour_id`) and plots the per-contour win/loss directly: a
line for the mean/median comparison value (mask2polymin vs. RDP, positive = mask2polymin
better) and bars for how many contours it won, both against noise level. Design
rationale: [Comparison_Charts_Plan.md](Comparison_Charts_Plan.md).

## Metrics

Full definitions in [metrics.py](metrics.py); `n_input_points`, `n_segments`, `wall_time_ms` are
recorded directly in [run_benchmark.py](run_benchmark.py).

| metric | meaning |
|---|---|
| `n_input_points` | vertex count of the extracted (noisy) input contour |
| `n_segments` | vertex count of the fitted polygon -- lower is a more compact fit |
| `hausdorff` | symmetric max boundary distance to GT, px -- worst-case error, one outlier dominates |
| `hd95` | 95th-percentile symmetric boundary distance, px -- robust companion to `hausdorff` |
| `iou` | intersection-over-union of fitted vs. GT filled area |
| `rms_sym` | symmetric RMS boundary distance to GT, px -- sees both dropped features and invented geometry |
| `rms_dir` | directed RMS, GT → fit, px -- sees dropped features only; compare to `rms_sym` to spot invented geometry |
| `corner_recall` | fraction of GT corners with a fitted vertex within τ=2px |
| `corner_precision` | fraction of fitted vertices within τ of a GT corner -- penalizes spurious vertices |
| `corner_loc_err` | mean GT-corner → nearest-fitted-vertex distance, px, over recalled corners |
| `corner_bias` | signed corner displacement, px -- positive = corner-cutting (inward), negative = overshoot (outward) |
| `corner_angle_err` | mean absolute turning-angle error at matched corners, degrees -- catches wrong local shape even when position looks fine |
| `area_ratio` | fitted / GT area -- <1 corners cut, >1 overshoot; insensitive to spurious co-linear vertices |
| `perimeter_ratio` | fitted / GT perimeter -- same reading as `area_ratio` |
| `wall_time_ms` | fit time per contour, milliseconds |
