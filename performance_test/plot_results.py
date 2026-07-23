"""
Aggregation and figures for the benchmark results, per Perf_Test_Plan.md.

Reads summarized_csvs/raw.csv and aggregates mean / median of every metric per (tier, algorithm,
noise_level, shape_class) cell -> summarized_csvs/summary.csv, prints the per-cell median table,
and renders the Tier 0 figures (plan plots 1-2), each split simple vs. complex
(car/plane/ship) shapes. Each noise level has exactly one (rdp_eps, m2p_tol) pair --
matched to it by run_benchmark.matched_pair, not swept -- so tolerance is carried along as
a per-cell scalar for display, never a grouping axis.

Plots 3-4 are Tier 1 and land with the COCO run.
"""
import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np

from synth_shapes import family_sizes, gt_polygon

RESULTS_DIR = Path(__file__).parent / "summarized_csvs"
CHARTS_DIR = Path(__file__).parent / "charts"
FIG_DPI = 150

# categorical palette slots 1-2 (validated order) + neutral inks, light surface
COLORS = {"mask2polymin": "#2a78d6", "rdp": "#008300"}
SERIES_LABEL = {"rdp": "RDP (approxPolyDP)", "mask2polymin": "Mask2PolyMin"}
INK, INK_2 = "#1a1a19", "#6f6d64"
GRID = "#e7e5df"

COMPLEX_FAMILIES = {"car", "plane", "ship"}
SHAPE_CLASSES = ["simple", "complex"]

# one low-res reference silhouette per family, shown as an icon strip above each
# panel's "simple" / "complex" title so the class split is legible at a glance
ICON_FAMILIES = {"simple": ("rect", "star", "arrow"), "complex": ("plane", "ship", "car")}

METRIC_COLS = ["n_input_points", "n_segments", "hausdorff", "hd95", "iou", "rms_sym",
               "rms_dir", "corner_recall", "corner_precision", "corner_loc_err",
               "corner_bias", "corner_angle_err", "area_ratio", "perimeter_ratio",
               "wall_time_ms"]
STATS = ["mean", "median"]


def _shape_class(contour_id: str) -> str:
    family = contour_id.split("_")[0]
    return "complex" if family in COMPLEX_FAMILIES else "simple"


def read_cells(raw_path: Path) -> dict:
    """Group raw.csv rows into (tier, algorithm, noise_level, shape_class) cells --
    shape_class is "simple" or "complex" (car/plane/ship), the only breakdown axis besides algorithm/noise_level.
    Each cell maps metric name -> np.array of values, plus "size" (parsed from contour_id, kept only for runtime_summary's per-size wall-time breakdown) and "tolerance" (constant within a (tier, algorithm, noise_level) triple)."""
    lists = defaultdict(lambda: defaultdict(list))
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            _family, d_size = row["contour_id"].split("_")[:2]
            key = (int(row["tier"]), row["algorithm"], int(row["noise_level"]),
                   _shape_class(row["contour_id"]))
            lists[key]["size"].append(int(d_size[1:]))
            lists[key]["tolerance"].append(float(row["tolerance"]))
            for m in METRIC_COLS:
                lists[key][m].append(float(row[m]))
    return {key: {m: np.array(v) for m, v in metrics.items()}
            for key, metrics in lists.items()}


def _cell_order(key) -> tuple:
    tier, algorithm, noise_level, shape_class = key
    return (tier, noise_level, SHAPE_CLASSES.index(shape_class), algorithm != "rdp")


def _agg(values: np.ndarray, stat: str) -> float:
    # nan-aware for corner_loc_err / corner_bias / corner_angle_err (NaN when a row
    # recalled no corners); a cell where every row is NaN legitimately summarizes to NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fn = np.nanmean if stat == "mean" else np.nanmedian
        return float(fn(values))


def summarize(cells: dict) -> list[dict]:
    """One summary row per (tier, algorithm, noise_level, shape_class) cell: mean and
    median of every metric."""
    rows = []
    for key in sorted(cells, key=_cell_order):
        tier, algorithm, noise_level, shape_class = key
        tolerance = float(cells[key]["tolerance"][0])
        row = {"tier": tier, "algorithm": algorithm, "tolerance": tolerance,
               "noise_level": noise_level, "shape_class": shape_class,
               "n_rows": len(cells[key]["n_segments"])}
        for m in METRIC_COLS:
            for stat in STATS:
                row[f"{m}_{stat}"] = round(_agg(cells[key][m], stat), 4)
        rows.append(row)
    return rows


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_medians(cells: dict) -> None:
    """Per-cell median table: one row per algorithm per noise level per shape_class,
    each already at its own noise-matched tolerance."""
    header = f"{'tolerance':<11}{'algorithm':<14}{'shape':<9}" + "".join(
        f"{c:>9}" for c in ["segs", "rms_sym", "hd95", "iou", "recall", "precis",
                            "loc_err", "ms"])
    for tier in sorted({k[0] for k in cells}):
        for level in sorted({k[2] for k in cells if k[0] == tier}):
            n = len(next(v for k, v in cells.items()
                         if k[0] == tier and k[2] == level)["n_segments"])
            print(f"\ntier {tier}, noise level {level}  (medians over {n} contours)")
            print(header)
            keys = [k for k in cells if k[0] == tier and k[2] == level]
            for key in sorted(keys, key=_cell_order):
                c = cells[key]
                cols = [_agg(c[m], "median") for m in
                        ["n_segments", "rms_sym", "hd95", "iou", "corner_recall",
                         "corner_precision", "corner_loc_err", "wall_time_ms"]]
                print(f"{c['tolerance'][0]:<11.2f}{key[1]:<14}{key[3]:<9}"
                      f"{cols[0]:>9.0f}{cols[1]:>9.2f}{cols[2]:>9.2f}{cols[3]:>9.4f}"
                      f"{cols[4]:>9.2f}{cols[5]:>9.2f}{cols[6]:>9.2f}{cols[7]:>9.2f}")


def _icon_outline(family: str) -> np.ndarray:
    """The family's GT polygon vertices (open, no repeated closing point) at its
    smallest legal size (d48, or the d64 fallback for car/plane) -- the true vector
    outline, not a raster mask, so the icon stays crisp at contour scale."""
    size_px = family_sizes(family)[0]
    poly, _canvas = gt_polygon(family, size_px)
    return poly[:-1]


def _bare_axes(iax) -> None:
    iax.set_xticks([])
    iax.set_yticks([])
    for side in iax.spines.values():
        side.set_visible(False)


def _icon_layout(x0: float, icon_h: float, shape_class: str) -> tuple[list, float]:
    """One entry per family (family, x, icon_w, px0, px1, py0, py1), plus for "simple"
    a trailing ("dots", x, dots_w) slot, left-aligned from x0 at the given icon height
    -- every width here scales linearly with icon_h, which _fit_icon_h relies on.
    Returns (entries, total width)."""
    gap = icon_h * 0.04
    entries, x = [], x0
    for family in ICON_FAMILIES[shape_class]:
        verts = _icon_outline(family)
        vx, vy = verts[:, 0], verts[:, 1]
        pad = 0.12 * max(vx.max() - vx.min(), vy.max() - vy.min())
        px0, px1 = vx.min() - pad, vx.max() + pad
        py0, py1 = vy.min() - pad, vy.max() + pad
        icon_w = icon_h * (px1 - px0) / (py1 - py0)
        entries.append((family, x, icon_w, px0, px1, py0, py1))
        x += icon_w + gap
    if shape_class == "simple":
        dots_w = icon_h * 0.5
        entries.append(("dots", x, dots_w))
        x += dots_w
    return entries, x - gap


def _fit_icon_h(bbox_width: float, shape_class: str, max_icon_h: float) -> float:
    """The largest icon height that still fits shape_class's row into bbox_width,
    capped at max_icon_h. Widths in _icon_layout scale linearly with icon_h, so one
    probe at icon_h=1.0 gives the width-per-unit-height needed to solve for it."""
    _, unit_w = _icon_layout(0.0, 1.0, shape_class)
    return min(max_icon_h, bbox_width / unit_w * 0.98)


def _draw_icon_row(fig, spec, shape_class: str, icon_h: float) -> None:
    """Render shape_class's reference silhouettes as outlines only (no fill),
    left-aligned, at the given icon height (shared with the other class's row, via
    _fit_icon_h, so "simple" and "complex" icons render at the same size) and a small
    fixed gap, directly above where that column's "simple" / "complex" title will
    sit. "simple" gets a trailing 3-dot icon: only 3 of its 7 families are shown,
    while "complex" shows all 3 of its member families."""
    bbox = spec.get_position(fig)
    ax_y0 = bbox.y0 + (bbox.height - icon_h) / 2
    entries, _ = _icon_layout(bbox.x0, icon_h, shape_class)

    for entry in entries:
        if entry[0] == "dots":
            _, x, dots_w = entry
            dax = fig.add_axes((x, ax_y0, dots_w, icon_h))
            dax.set_xlim(0, 1)
            dax.set_ylim(0, 1)
            dot_y = 0.12 + 2.0 / (icon_h * fig.get_size_inches()[1] * FIG_DPI)
            dax.scatter([0.2, 0.5, 0.8], [dot_y] * 3, s=10, color=INK)
            _bare_axes(dax)
        else:
            family, x, icon_w, px0, px1, py0, py1 = entry
            iax = fig.add_axes((x, ax_y0, icon_w, icon_h))
            iax.add_patch(plt.Polygon(_icon_outline(family), closed=True, fill=False,
                                       edgecolor=INK, linewidth=1.3))
            iax.set_xlim(px0, px1)
            iax.set_ylim(py1, py0)  # inverted: polygon y grows downward (image convention)
            _bare_axes(iax)


def _draw_icon_rows(fig, outer) -> None:
    """Draw both classes' icon rows (outer[0, 0] and outer[0, 1]) at one shared icon
    height, so "simple" (which fits comfortably) doesn't render larger than "complex"
    (whose wider ship/car outlines force a smaller fit)."""
    specs = [outer[0, col] for col in range(2)]
    bboxes = [spec.get_position(fig) for spec in specs]
    max_icon_h = min(bbox.height for bbox in bboxes) * 0.95
    icon_h = min(_fit_icon_h(bbox.width, sc, max_icon_h)
                 for bbox, sc in zip(bboxes, SHAPE_CLASSES)) * 0.65
    for spec, shape_class in zip(specs, SHAPE_CLASSES):
        _draw_icon_row(fig, spec, shape_class, icon_h)


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def fig_segments_vs_rms(cells, out_path: Path, tier: int = 0) -> None:
    """Plan plot 1: median segments vs median symmetric RMS to GT, one panel per shape
    class (simple / complex), one point per (algorithm, noise level) at that level's
    noise-matched tolerance, connected noise level 0 -> 4 per algorithm. Lower-left is
    better."""
    levels = sorted({k[2] for k in cells if k[0] == tier})
    fig = plt.figure(figsize=(9.4, 6.1))
    outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=(1, 5), hspace=0.12)
    # set final margins before reading any cell's get_position() (icons included) --
    # tight_layout doesn't account for the nested icon gridspecs or the fig.text
    # footnotes below the axes, so margins are set by hand instead, up front, so
    # every position query below reflects the real layout, not matplotlib's defaults.
    outer.update(left=0.08, right=0.98, top=0.925, bottom=0.15)
    _draw_icon_rows(fig, outer)
    axes = []
    for col, shape_class in enumerate(SHAPE_CLASSES):
        ax = fig.add_subplot(outer[1, col], sharey=axes[0] if axes else None)
        axes.append(ax)
        for algo in ("rdp", "mask2polymin"):
            keys = [(tier, algo, level, shape_class) for level in levels]
            xs = [_agg(cells[k]["n_segments"], "median") for k in keys]
            ys = [_agg(cells[k]["rms_sym"], "median") for k in keys]
            ax.plot(xs, ys, "-o", color=COLORS[algo], linewidth=2, markersize=6,
                    label=SERIES_LABEL[algo])
            for level, x, y in zip(levels, xs, ys):
                ax.annotate(f"n{level}", (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=8, color=INK_2)
        ax.set_title(shape_class, fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xlabel("median segments", fontsize=9, color=INK_2)
        _style(ax)
    axes[0].set_ylabel("median RMS (px)", fontsize=9, color=INK_2)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    axes[-1].annotate("lower-left is better", (0.97, 0.03), xycoords="axes fraction",
                      fontsize=8, color=INK_2, style="italic", ha="right")
    fig.suptitle("segment count vs fidelity, at each noise level's matched tolerance",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.06,
             "tolerance = max(1.0, jitter_amp), ε = tolerance·√2 (README `Parameters`)",
             ha="center", fontsize=8, color=INK_2)
    fig.text(0.5, 0.018, "labels n0-n4 = noise level; each point is a median",
             ha="center", fontsize=8, color=INK_2)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"figure -> {out_path}")


def fig_metrics_vs_noise(cells, metrics, suptitle: str, out_path: Path,
                          tier: int = 0, share_y_across_rows: bool = False,
                          legend_loc: str = "best", note: str | None = None,
                          log_y: bool = False,
                          algos: tuple = ("rdp", "mask2polymin")) -> None:
    """1 or 2 metrics (one per row) vs noise level, each level at its own noise-matched
    tolerance -- one column per shape class. metrics is a list of 1 or 2
    (metric_col, ylabel, ylim) tuples: ylim is an explicit (lo, hi); "floor0" to
    autoscale the top but pin the bottom at 0; "auto" to autoscale both ends freely
    (for ratio metrics like area_ratio that hover near 1 -- flooring at 0 would flatten
    all their variation into a sliver at the top); or None, which also autoscales both
    ends but adds a dashed 0 reference line (for signed metrics like corner_bias).
    log_y logs every row's y-axis (for metrics like wall_time_ms spanning orders of
    magnitude between algorithms); ylim should be "auto" when log_y is set, since
    "floor0"/None both assume a linear, zero-anchored axis."""
    nrows = len(metrics)
    levels = sorted({k[2] for k in cells if k[0] == tier})
    if nrows == 1:
        fig = plt.figure(figsize=(9.4, 6.1))
        outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=(1, 5), hspace=0.12)
        margins = dict(left=0.08, right=0.98, top=0.925, bottom=0.15)
    else:
        fig = plt.figure(figsize=(9.4, 8.0))
        outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=(1, 9), hspace=0.12)
        margins = dict(left=0.08, right=0.98, top=0.95, bottom=0.07)
    plot_cols = [gridspec.GridSpecFromSubplotSpec(nrows, 1, subplot_spec=outer[1, col],
                                                   hspace=0.12) for col in range(2)]
    # set final margins before reading any cell's get_position() (icons included) --
    # tight_layout doesn't account for the nested icon/plot gridspecs, so margins are
    # set by hand instead, up front, so every position query below reflects the real
    # layout, not matplotlib's defaults.
    outer.update(**margins)
    _draw_icon_rows(fig, outer)
    axes = [[None, None] for _ in range(nrows)]
    x_anchor, row_anchor = None, [None] * nrows
    for row, (metric, ylabel, ylim) in enumerate(metrics):
        row_max = 0.0
        for col, shape_class in enumerate(SHAPE_CLASSES):
            sharey = axes[0][0] if share_y_across_rows else row_anchor[row]
            ax = fig.add_subplot(plot_cols[col][row, 0], sharey=sharey, sharex=x_anchor)
            x_anchor = x_anchor or ax
            row_anchor[row] = row_anchor[row] or ax
            axes[row][col] = ax
            for algo in algos:
                keys = [(tier, algo, level, shape_class) for level in levels]
                med = [_agg(cells[k][metric], "median") for k in keys]
                ax.plot(levels, med, "-o", color=COLORS[algo], linewidth=2, markersize=6,
                        label=SERIES_LABEL[algo])
                row_max = max(row_max, max(med))
            ax.set_xticks(levels)
            if log_y:
                ax.set_yscale("log")
            if ylim not in (None, "floor0", "auto"):
                ax.set_ylim(*ylim)
            elif ylim is None:
                ax.axhline(0, color=INK_2, linewidth=0.8, linestyle="--", zorder=0)
            if row == 0:
                ax.set_title(shape_class, fontsize=10.5, color=INK, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
            if row == nrows - 1:
                ax.set_xlabel("noise level", fontsize=9, color=INK_2)
            _style(ax)
        if ylim == "floor0":
            # sharey + a partial set_ylim(bottom=...) on each axes in turn races with
            # matplotlib's own autoscale (whichever axes sets it last wins, and it may
            # not have seen the other column's data yet) -- set both ends from the
            # actual row data instead, once, after both columns are plotted.
            row_anchor[row].set_ylim(0, row_max * 1.05 if row_max > 0 else 1.0)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK, loc=legend_loc)
    fig.suptitle(suptitle, fontsize=11, color=INK)
    if note is not None:
        fig.text(0.5, 0.018, note, ha="center", fontsize=8, color=INK_2, style="italic")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"figure -> {out_path}")


def fig_corner_recall(cells, out_path: Path, tier: int = 0) -> None:
    """Plan plot 2: median corner recall and precision vs noise level."""
    fig_metrics_vs_noise(
        cells,
        [("corner_recall", "median corner recall", (0, 1.05)),
         ("corner_precision", "median corner precision", (0, 1.05))],
        "corner survival vs noise, each level at its noise-matched tolerance (τ = 2 px)",
        out_path, tier=tier, share_y_across_rows=True, legend_loc="lower left")


# additional fidelity metrics, one or two per figure, in the same simple/complex,
# icon-strip style as fig1/fig2 -- ylim "floor0" autoscales the top but pins the
# bottom at 0; "auto" autoscales both ends (ratio metrics that hover near 1, where a
# 0 floor would flatten all the variation into a sliver at the top); None also
# autoscales both ends but draws a dashed 0 reference line (corner_bias is signed:
# positive = corner-cutting, negative = overshoot)
FIDELITY_CHARTS = [
    ("fig3_hausdorff.png",
     [("hausdorff", "median Hausdorff (px)", "floor0"),
      ("hd95", "median HD95 (px)", "floor0")],
     "boundary distance vs noise, each level at its noise-matched tolerance", None),
    ("fig4_rms.png",
     [("rms_sym", "median symmetric RMS (px)", "floor0"),
      ("rms_dir", "median directed RMS (px)", "floor0")],
     "RMS boundary error vs noise, each level at its noise-matched tolerance", None),
    ("fig5_corner_loc_err.png",
     [("corner_loc_err", "median corner loc. error (px)", "floor0")],
     "corner localization error vs noise, each level at its noise-matched tolerance "
     "(τ = 2 px)", None),
    ("fig6_corner_bias.png",
     [("corner_bias", "median corner bias (px)", None)],
     "corner bias vs noise, each level at its noise-matched tolerance (τ = 2 px)",
     "positive = corner-cutting (shrinks inward), negative = overshoot"),
    ("fig7_area.png",
     [("area_ratio", "median area ratio", "auto")],
     "area ratio vs noise, each level at its noise-matched tolerance", None),
    ("fig8_perimeter.png",
     [("perimeter_ratio", "median perimeter ratio", "auto")],
     "perimeter ratio vs noise, each level at its noise-matched tolerance", None),
    ("fig9_iou.png",
     [("iou", "median IoU", "auto")],
     "IoU vs noise, each level at its noise-matched tolerance", None),
    ("fig10_corner_angle.png",
     [("corner_angle_err", "median corner angle error (deg)", "floor0")],
     "corner turning-angle error vs noise, each level at its noise-matched tolerance "
     "(τ = 2 px)", None),
]


def runtime_summary(cells: dict) -> list[dict]:
    """Average and P95 wall_time_ms per algorithm, globally and per image size class
    (pooled across tier/tolerance/noise_level/shape_class -- this is wall-clock cost, not
    fidelity, so it doesn't need that breakdown)."""
    pooled = defaultdict(lambda: {"wall_time_ms": [], "size": []})
    for (_tier, algorithm, _noise_level, _shape_class), metrics in cells.items():
        pooled[algorithm]["wall_time_ms"].append(metrics["wall_time_ms"])
        pooled[algorithm]["size"].append(metrics["size"])
    pooled = {algo: {k: np.concatenate(v) for k, v in d.items()}
              for algo, d in pooled.items()}

    rows = []
    for algo in sorted(pooled):
        wt, sizes = pooled[algo]["wall_time_ms"], pooled[algo]["size"]
        rows.append({"algorithm": algo, "size": "all", "n_rows": len(wt),
                     "wall_time_ms_avg": round(float(np.mean(wt)), 4),
                     "wall_time_ms_p95": round(float(np.percentile(wt, 95.0)), 4)})
        for s in sorted(np.unique(sizes)):
            sel = sizes == s
            rows.append({"algorithm": algo, "size": int(s), "n_rows": int(sel.sum()),
                         "wall_time_ms_avg": round(float(np.mean(wt[sel])), 4),
                         "wall_time_ms_p95": round(float(np.percentile(wt[sel], 95.0)), 4)})
    return rows


def print_runtime_summary(rows: list[dict]) -> None:
    print(f"\n{'algorithm':<14}{'size':>6}{'n':>7}{'avg_ms':>10}{'p95_ms':>10}")
    for row in rows:
        print(f"{row['algorithm']:<14}{str(row['size']):>6}{row['n_rows']:>7}"
              f"{row['wall_time_ms_avg']:>10.3f}{row['wall_time_ms_p95']:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate raw.csv -> summary.csv + figures")
    parser.add_argument("--raw", type=Path, default=RESULTS_DIR / "raw.csv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "summary.csv")
    args = parser.parse_args()
    cells = read_cells(args.raw)
    rows = summarize(cells)
    write_summary(rows, args.out)
    print_medians(cells)
    print(f"\n{len(rows)} cells -> {args.out}")
    runtime_rows = runtime_summary(cells)
    write_summary(runtime_rows, args.out.parent / "runtime_summary.csv")
    print_runtime_summary(runtime_rows)
    print(f"\n{len(runtime_rows)} rows -> {args.out.parent / 'runtime_summary.csv'}")
    out = CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig_segments_vs_rms(cells, out / "fig1_segments_vs_rms.png")
    fig_corner_recall(cells, out / "fig2_corner_recall.png")
    for filename, metrics, suptitle, note in FIDELITY_CHARTS:
        fig_metrics_vs_noise(cells, metrics, suptitle, out / filename, note=note)
    fig_metrics_vs_noise(
        cells, [("wall_time_ms", "median wall time (ms, log scale)", "auto")],
        "Mask2PolyMin wall-clock time vs noise, each level at its noise-matched tolerance",
        out / "fig11_walltime.png",
        note="Intel i5-12450H, Intel UHD Graphics, single contour, single-threaded CPU fit",
        log_y=True, algos=("mask2polymin",))


if __name__ == "__main__":
    main()
