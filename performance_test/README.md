# Performance benchmark

Mask2PolyMin vs `cv2.approxPolyDP` (RDP) on synthetic GT shapes with simulated
segmentation noise. Design and rationale: [Perf_Test_Plan.md](Perf_Test_Plan.md),
[Synth_Shapes_Plan.md](Synth_Shapes_Plan.md). Run all commands from this folder.

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

Produces `results/raw.csv` (gitignored): 15 600 rows = 1950 contours × 2 algorithms ×
4 tolerance pairs, one row per run. Deterministic except `wall_time_ms`.

## 3. Aggregate + figures

```bash
python plot_results.py
```

Produces `results/summary.csv` (median/p25/p75/p95 per algorithm × tolerance × noise
level), prints the median table, and renders `results/fig1_segments_vs_rms.png` and
`results/fig2_corner_recall.png`.
