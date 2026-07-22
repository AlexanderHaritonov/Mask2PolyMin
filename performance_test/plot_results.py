"""
Aggregation and figures for the benchmark results, per Perf_Test_Plan.md.

Reads results/raw.csv and aggregates mean / median of every metric per (tier, algorithm,
noise_level, shape_class) cell -> results/summary.csv, prints the per-cell median table,
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
from matplotlib import pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

# categorical palette slots 1-2 (validated order) + neutral inks, light surface
COLORS = {"mask2polymin": "#2a78d6", "rdp": "#008300"}
SERIES_LABEL = {"rdp": "RDP (approxPolyDP)", "mask2polymin": "Mask2PolyMin"}
INK, INK_2 = "#1a1a19", "#6f6d64"
GRID = "#e7e5df"

COMPLEX_FAMILIES = {"car", "plane", "ship"}
SHAPE_CLASSES = ["simple", "complex"]

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
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), sharey=True)
    for ax, shape_class in zip(axes, SHAPE_CLASSES):
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
    fig.text(0.5, 0.045,
             "tolerance = max(1.0, jitter_amp), ε = tolerance·√2 (README `Parameters`)",
             ha="center", fontsize=8, color=INK_2)
    fig.text(0.5, 0.01, "labels n0-n4 = noise level; each point is a median",
             ha="center", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_path}")


def fig_corner_recall(cells, out_path: Path, tier: int = 0) -> None:
    """Plan plot 2: median corner recall and precision vs noise level, each level at its
    own noise-matched tolerance -- one row per metric, one column per shape class."""
    levels = sorted({k[2] for k in cells if k[0] == tier})
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharey=True, sharex=True)
    for row, (metric, title) in enumerate(
            [("corner_recall", "corner recall"), ("corner_precision", "corner precision")]):
        for col, shape_class in enumerate(SHAPE_CLASSES):
            ax = axes[row][col]
            for algo in ("rdp", "mask2polymin"):
                keys = [(tier, algo, level, shape_class) for level in levels]
                med = [_agg(cells[k][metric], "median") for k in keys]
                ax.plot(levels, med, "-o", color=COLORS[algo], linewidth=2, markersize=6,
                        label=SERIES_LABEL[algo])
            ax.set_xticks(levels)
            ax.set_ylim(0, 1.05)
            if row == 0:
                ax.set_title(shape_class, fontsize=10.5, color=INK, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"median {title}", fontsize=9, color=INK_2)
            if row == 1:
                ax.set_xlabel("noise level", fontsize=9, color=INK_2)
            _style(ax)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.suptitle("corner survival vs noise, each level at its noise-matched tolerance "
                 "(τ = 2 px)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_path}")


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
    out = args.out.parent / "charts"
    out.mkdir(parents=True, exist_ok=True)
    fig_segments_vs_rms(cells, out / "fig1_segments_vs_rms.png")
    fig_corner_recall(cells, out / "fig2_corner_recall.png")


if __name__ == "__main__":
    main()
