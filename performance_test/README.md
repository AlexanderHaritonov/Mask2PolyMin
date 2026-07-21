# Performance benchmark

Mask2PolyMin vs `cv2.approxPolyDP` (RDP) on synthetic GT shapes with simulated
segmentation noise. Design and rationale: [Perf_Test_Plan.md](Perf_Test_Plan.md),
[Synth_Shapes_Plan.md](Synth_Shapes_Plan.md). Run all commands from this folder.

## Choosing tolerance for a given noise level

`NOISE_LEVELS` in [synth_shapes.py](synth_shapes.py) defines each level's `jitter_amp` — the
per-pixel standard deviation of the elastic boundary displacement used to simulate segmentation
noise. `run_benchmark.matched_pair()` derives that level's `(rdp_epsilon, mask2polymin_tolerance)`
from it:

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

Produces `gt_shapes/dXXX/*.png|json` (30 canonical GT pairs) and 4 gallery sheets in
`shape_review/`. Byte-identical when nothing changed; any git diff means the GT moved
and existing results are stale. Optional reviews: `--preview` (per-family renders),
`--preview-noise` (`shape_review/preview_noise.png`, noise levels).

## 2. Benchmark sweep

```bash
python run_benchmark.py            # ~10 min; --limit N for a quick smoke pass
```

Produces `results/raw.csv` (gitignored): 3900 rows = 1950 contours × 2 algorithms, one row
per run, each at its noise level's matched tolerance (see above). Deterministic except
`wall_time_ms`.

## 3. Aggregate + figures

```bash
python plot_results.py
```

Produces `results/summary.csv` (median/p25/p75/p95 per algorithm × noise level), prints the
median table, and renders `results/charts/fig1_segments_vs_rms.png` and
`results/charts/fig2_corner_recall.png` (plus the per-family figures).
