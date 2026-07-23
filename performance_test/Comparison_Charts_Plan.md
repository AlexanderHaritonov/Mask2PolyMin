# Per-Metric Win/Loss Charts — Plan (`plot_comparison.py`)

Narrow slice of [Diff_Analysis_Plan.md](Diff_Analysis_Plan.md), scoped to one deliverable:
per metric, per shape class (simple/complex), a chart of *how much* mask2polymin wins or
loses against RDP across noise levels, with *how often* it wins on a second y-axis on the
same panel. No Wilcoxon, no per-family breakdown beyond simple/complex, no wall time.

## Sign convention (same as Diff_Analysis_Plan.md, restated for this metric set)

`comparison = ` a per-contour value, positive whenever mask2polymin is the better fit,
derived per metric direction:

| direction | metrics | `comparison` formula |
|---|---|---|
| lower is better | `n_segments`, `hausdorff`, `hd95`, `rms_sym`, `rms_dir`, `corner_loc_err`, `corner_angle_err` | `rdp − m2p` |
| higher is better | `iou`, `corner_recall`, `corner_precision` | `m2p − rdp` |
| closer to 0 (signed, directional) | `corner_bias` | `\|rdp\| − \|m2p\|` |
| closer to 1 (ratio) | `area_ratio`, `perimeter_ratio` | `\|rdp − 1\| − \|m2p − 1\|` |

13 metrics — `wall_time_ms` excluded per scope, `n_input_points` excluded (identical for
both algorithms on a pair, not a fidelity signal). `corner_loc_err`, `corner_bias`,
`corner_angle_err`, `area_ratio`, `perimeter_ratio` are NaN on a given contour when the
row it comes from recalled zero corners; drop that contour from *that metric's* average
and win count (not from other metrics') — same nan-aware discipline `plot_results.py`
already applies elsewhere.

## Aggregation statistic per metric

Mean by default, median for metrics where a single contour's outlier value can swing the
cell's center — resolved per metric, not left as a blanket choice:

```python
AGG_STAT = {  # "mean" | "median" -- median for outlier-prone metrics, mean otherwise
    "hausdorff": "median",
    "corner_loc_err": "median",
    "corner_bias": "median",
    "corner_angle_err": "median",
    # everything else (n_segments, hd95, rms_sym, rms_dir, iou, corner_recall,
    # corner_precision, area_ratio, perimeter_ratio): "mean"
}
```

Rationale for the four `"median"` entries — see the reply in chat for the full reviewable
list; in short: `hausdorff` is a per-contour *max* by construction (the project's own docs
say as much — it's the whole reason `hd95` exists as its robust companion), and
`corner_loc_err`/`corner_bias`/`corner_angle_err` are each a per-contour mean over
*matched* corners, so a contour that only matched one corner at high noise contributes a
much noisier value than a contour that matched all of them — a form of outlier risk
`hausdorff` also has but the others in this metric set don't.

## What each chart shows

One panel = one metric × one shape class (simple or complex) — same 2-column layout
`plot_results.py` already uses everywhere else, so this reads as a natural extension of
the existing figures, not a new visual language. Per panel, x-axis = noise level (0–4):

- **Left y-axis, line** — mean or median `comparison` over the cell's contours, per
  `AGG_STAT` above. A dashed 0-reference line marks the win/loss boundary, same treatment
  `fig6_corner_bias.png` already uses for a signed metric. Color: mask2polymin's existing
  blue (`COLORS["mask2polymin"]`) — it's mask2polymin's result expressed relative to
  RDP's baseline, not a new series needing a new color. The y-axis label states which
  statistic that row is ("median comparison" vs. "mean comparison") — needed because a
  grouped figure can mix the two (`comparison_hausdorff.png` stacks `hausdorff`, using
  median, over `hd95`, using mean, as its two rows), so each row must self-declare
  rather than rely on a single figure-wide label.
- **Right y-axis (`ax.twinx()`), bars** — count of contours where `comparison > 0` (a
  strict win; ties don't count). Neutral gray (`INK_2`), drawn at partial opacity
  (alpha ≈ 0.35) so the comparison line stays legible on top rather than fighting the
  bars for attention — bars are the supporting series here, the line is the headline.
  Each bar is labeled `win_count/n_pairs` (small text, `fontsize=8`, `INK_2`, directly
  above the bar) rather than a bare count — `n_pairs` itself varies by noise level
  within a panel (level 0 is 1 seed, levels 1–4 are 3 seeds each, so roughly 3× the
  contours), so a bare count isn't comparable across the x-axis without its denominator
  sitting right next to it.

**Ties matter and are easy to misread.** For continuous metrics (RMS, Hausdorff) ties are
rare. For `n_segments` (small integers) and `corner_recall`/`corner_precision` (low-
cardinality fractions — both algorithms often recall 100% of corners on a clean, low-noise
shape) exact ties will be common, sometimes the majority. A low win-count bar there means
"mostly tied," not "mask2polymin mostly loses" — the per-bar `win_count/n_pairs` label
above is what keeps that legible instead of looking like a loss.

## Aggregation: `results/comparison_summary.csv`

One row per `(metric, shape_class, noise_level)`:

```
metric, shape_class, noise_level, n_pairs, stat, comparison_center, win_count
```

`n_pairs` = non-NaN contour count for that metric in that cell (the averaging and the win
count share this same denominator/pool). `stat` is `"mean"` or `"median"` per `AGG_STAT`,
so `comparison_center` is self-documenting without needing both columns computed for
every row. Small and committed, same convention as `summary.csv` — figures should be
reproducible from it without rereading `raw.csv`.

## Figures

Same grouping `plot_results.py`'s `FIDELITY_CHARTS` already uses for related metric pairs,
so this doesn't invent a second grouping scheme — plus one new entry for `n_segments`,
which didn't have a "vs. noise level" chart before (`fig1` plots it against RMS as a
scatter, not against noise):

| file | metric(s) |
|---|---|
| `comparison_segments.png` | `n_segments` |
| `comparison_hausdorff.png` | `hausdorff`, `hd95` |
| `comparison_rms.png` | `rms_sym`, `rms_dir` |
| `comparison_corner_recall_precision.png` | `corner_recall`, `corner_precision` |
| `comparison_corner_loc_err.png` | `corner_loc_err` |
| `comparison_corner_bias.png` | `corner_bias` |
| `comparison_area.png` | `area_ratio` |
| `comparison_perimeter.png` | `perimeter_ratio` |
| `comparison_iou.png` | `iou` |
| `comparison_corner_angle.png` | `corner_angle_err` |

Grouped metrics (hausdorff/hd95, rms_sym/rms_dir, recall/precision) get one row per metric
within the figure, same as today's `fig3`/`fig4`/`fig2` — each row still carries its own
twin axis pair, just stacked. Named `comparison_*.png`, not the numbered `fig<n>_*`
scheme `plot_results.py` uses — a deliberate difference, not an inconsistency.

## Implementation

```
plot_comparison.py   # contour_id join -> comparison per contour -> comparison_summary.csv;
                      # comparison_*.png (10 figures); reuses COLORS/INK/INK_2/_style/
                      # _draw_icon_rows from plot_results.py rather than restyling from
                      # scratch
```

1. Inner-join `raw.csv` on `contour_id` (rdp row × mask2polymin row), same mechanism as
   Diff_Analysis_Plan.md — no need to persist the per-contour join to disk for this task,
   it's a transient step feeding straight into the per-cell aggregation.
2. `BETTER_DIRECTION` + `comparison()` (the 4-case formula above) + `AGG_STAT` → per-cell
   `n_pairs`, `stat`, `comparison_center`, win count → `comparison_summary.csv`.
3. One plotting function: dual-axis (line + translucent bars, each bar labeled
   `win_count/n_pairs`), 2-column simple/complex, looped over the 10-row table above.
