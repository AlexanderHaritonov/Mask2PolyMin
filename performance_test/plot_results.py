"""
Aggregation and figures for the benchmark results, per Perf_Test_Plan.md.

Reads results/raw.csv and aggregates median / p25 / p75 / p95 of every metric per
(tier, algorithm, noise_level) cell -> results/summary.csv, prints the per-cell median
table, and renders the Tier 0 figures (plan plots 1-2). Each noise level has exactly one
(rdp_eps, m2p_tol) pair -- matched to it by run_benchmark.matched_pair, not swept -- so
tolerance is carried along as a per-cell scalar for display, never a grouping axis.
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
from matplotlib.lines import Line2D
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

# categorical palette slots 1-2 (validated order) + neutral inks, light surface
COLORS = {"mask2polymin": "#2a78d6", "rdp": "#008300"}
SERIES_LABEL = {"rdp": "RDP (approxPolyDP)", "mask2polymin": "Mask2PolyMin"}
INK, INK_2 = "#1a1a19", "#6f6d64"
GRID = "#e7e5df"

METRIC_COLS = ["n_input_points", "n_segments", "hausdorff", "hd95", "iou", "rms_sym",
               "rms_dir", "corner_recall", "corner_precision", "corner_loc_err",
               "wall_time_ms"]
STATS = [("med", 50.0), ("p25", 25.0), ("p75", 75.0), ("p95", 95.0)]


def read_cells(raw_path: Path) -> dict:
    """Group raw.csv rows into (tier, algorithm, noise_level) cells -- one noise-matched
    tolerance per cell, never swept -- each mapping metric name -> np.array of values
    across the cell's contours, plus "family" / "size" (parsed from contour_id) and
    "tolerance" (the single eps/tol value used, constant within the cell)."""
    lists = defaultdict(lambda: defaultdict(list))
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            key = (int(row["tier"]), row["algorithm"], int(row["noise_level"]))
            family, d_size = row["contour_id"].split("_")[:2]
            lists[key]["family"].append(family)
            lists[key]["size"].append(int(d_size[1:]))
            lists[key]["tolerance"].append(float(row["tolerance"]))
            for m in METRIC_COLS:
                lists[key][m].append(float(row[m]))
    return {key: {m: np.array(v) for m, v in metrics.items()}
            for key, metrics in lists.items()}


def _cell_order(key) -> tuple:
    tier, algorithm, noise_level = key
    return (tier, noise_level, algorithm != "rdp")


def _percentile(values: np.ndarray, q: float) -> float:
    # nan-aware for corner_loc_err (NaN when a row recalled no corners); a cell where
    # every row is NaN legitimately summarizes to NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanpercentile(values, q))


def summarize(cells: dict) -> list[dict]:
    """One summary row per cell: median / p25 / p75 / p95 of every metric."""
    rows = []
    for key in sorted(cells, key=_cell_order):
        tier, algorithm, noise_level = key
        tolerance = float(cells[key]["tolerance"][0])
        row = {"tier": tier, "algorithm": algorithm, "tolerance": tolerance,
               "noise_level": noise_level, "n_rows": len(cells[key]["n_segments"])}
        for m in METRIC_COLS:
            for stat, q in STATS:
                row[f"{m}_{stat}"] = round(_percentile(cells[key][m], q), 4)
        rows.append(row)
    return rows


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_medians(cells: dict) -> None:
    """Per-cell median table: one row per algorithm per noise level (rdp above m2p),
    each already at its own noise-matched tolerance."""
    header = f"{'tolerance':<11}{'algorithm':<14}" + "".join(
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
                cols = [_percentile(c[m], 50.0) for m in
                        ["n_segments", "rms_sym", "hd95", "iou", "corner_recall",
                         "corner_precision", "corner_loc_err", "wall_time_ms"]]
                print(f"{c['tolerance'][0]:<11.2f}{key[1]:<14}"
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
    """Plan plot 1: median segments vs median symmetric RMS to GT, one point per
    (algorithm, noise level) at that level's noise-matched tolerance, connected
    noise level 0 -> 4 per algorithm. Lower-left is better. Noise level is now the only
    swept axis (tolerance no longer is: each level uses the one tolerance a caller would
    actually pick for it), so the path shows how each algorithm's realistic operating
    point moves as noise grows, rather than a tight->loose tolerance curve."""
    levels = sorted({k[2] for k in cells if k[0] == tier})
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for algo in ("rdp", "mask2polymin"):
        keys = [(tier, algo, level) for level in levels]
        xs = [_percentile(cells[k]["n_segments"], 50.0) for k in keys]
        ys = [_percentile(cells[k]["rms_sym"], 50.0) for k in keys]
        ax.plot(xs, ys, "-o", color=COLORS[algo], linewidth=2, markersize=6,
                label=SERIES_LABEL[algo])
        for level, x, y in zip(levels, xs, ys):
            ax.annotate(f"n{level}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=INK_2)
    ax.set_xlabel("median segments", fontsize=9, color=INK_2)
    ax.set_ylabel("median RMS (px)", fontsize=9, color=INK_2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.annotate("lower-left is better", (0.97, 0.03), xycoords="axes fraction",
                fontsize=8, color=INK_2, style="italic", ha="right")
    _style(ax)
    fig.suptitle("segment count vs fidelity, at each noise level's matched tolerance",
                 fontsize=11, color=INK)
    fig.text(0.5, 0.01,
             "each point: median over that noise level's contours, tolerance = "
             "max(1.0, jitter_amp), ε = tolerance·√2 (see README `Parameters`); "
             "labels n0-n4 = noise level",
             ha="center", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_path}")


def fig_corner_recall(cells, out_path: Path, tier: int = 0) -> None:
    """Plan plot 2: corner recall (and precision, same scale) vs noise level, each level
    at its own noise-matched tolerance (not a fixed pair across all levels) — median
    solid, p95 (best-case ceiling) dashed."""
    levels = sorted({k[2] for k in cells if k[0] == tier})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, metric, title in zip(axes, ("corner_recall", "corner_precision"),
                                 ("corner recall", "corner precision")):
        for algo in ("rdp", "mask2polymin"):
            keys = [(tier, algo, level) for level in levels]
            med = [_percentile(cells[k][metric], 50.0) for k in keys]
            p95 = [_percentile(cells[k][metric], 95.0) for k in keys]
            ax.plot(levels, med, "-o", color=COLORS[algo], linewidth=2, markersize=6,
                    label=SERIES_LABEL[algo])
            ax.plot(levels, p95, "--", color=COLORS[algo], linewidth=1.6)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{lv}" + (" (clean)" if lv == 0 else "") for lv in levels])
        ax.set_ylim(0, 1.05)
        ax.set_title(title + " (median solid, p95 dashed)", fontsize=10, color=INK)
        ax.set_xlabel("noise level", fontsize=9, color=INK_2)
        _style(ax)
    axes[0].set_ylabel("fraction of corners", fontsize=9, color=INK_2)
    handles = ([Line2D([], [], color=COLORS[a], linewidth=2, label=SERIES_LABEL[a])
                for a in ("rdp", "mask2polymin")]
               + [Line2D([], [], color=INK_2, linewidth=2, label="median"),
                  Line2D([], [], color=INK_2, linewidth=1.6, linestyle="--", label="p95")])
    axes[0].legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK,
                   loc="lower left")
    fig.suptitle("corner survival vs noise, each level at its noise-matched tolerance "
                 "(τ = 2 px)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_path}")


_METRIC_LABEL = {"corner_recall": "corner recall", "corner_precision": "corner precision",
                 "iou": "IoU"}


def fig_metric_per_family(cells, out_path: Path, metric: str, size: int | None = None,
                          tier: int = 0) -> None:
    """Small-multiples breakdown: median corner recall/precision or IoU vs noise level,
    each level at its own noise-matched tolerance, one panel per shape family — pooled
    over all sizes, or restricted to a single size (48/64/128/320). Splits out what the
    pooled median hides — which shapes degrade first as noise grows. Corner panels share
    the full 0-1 axis; IoU panels auto-scale (its dynamic range is a thin band near 1)."""
    levels = sorted({k[2] for k in cells if k[0] == tier})
    first = next(k for k in sorted(cells, key=_cell_order) if k[0] == tier)
    stored = cells[first]["family"] if size is None \
        else cells[first]["family"][cells[first]["size"] == size]
    families = sorted(np.unique(stored))
    nrows = 1 if len(families) <= 5 else 2
    ncols = (len(families) + nrows - 1) // nrows
    # width floor keeps the suptitle/caption inside the canvas for few-panel sizes (d064)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(2.5 * ncols, 7.4), 2.7 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, family in zip(axes.flat, families):
        for algo in ("rdp", "mask2polymin"):
            med = []
            for level in levels:
                c = cells[(tier, algo, level)]
                sel = c["family"] == family
                if size is not None:
                    sel &= c["size"] == size
                med.append(_percentile(c[metric][sel], 50.0))
            ax.plot(levels, med, "-o", color=COLORS[algo], linewidth=1.8,
                    markersize=4.5, label=SERIES_LABEL[algo])
        ax.set_xticks(levels)
        if metric.startswith("corner_"):
            ax.set_ylim(0, 1.05)
        ax.set_title(family, fontsize=9.5, color=INK)
        _style(ax)
    metric_word = _METRIC_LABEL[metric]
    for ax in axes[-1]:
        ax.set_xlabel("noise level", fontsize=9, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"median {metric_word.removeprefix('corner ')}",
                      fontsize=9, color=INK_2)
    axes[0][0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    where = "" if size is None else f" at d{size:03d}"
    tau_note = " (τ = 2 px)" if metric.startswith("corner_") else ""
    fig.suptitle(f"{metric_word} vs noise per shape family{where}, "
                 f"noise-matched tolerance{tau_note}", fontsize=11, color=INK)
    caption = ("median over 3 sizes × 5 angles per point: 15 contours at level 0, "
               "45 at levels 1–4 (3 reps)" if size is None else
               "median over 5 angles per point: 5 contours at level 0, "
               "15 at levels 1–4 (3 reps)")
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93 if nrows == 1 else 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_path}")


def runtime_summary(cells: dict) -> list[dict]:
    """Average and P95 wall_time_ms per algorithm, globally and per image size class
    (pooled across tier/tolerance/noise_level -- this is wall-clock cost, not fidelity,
    so it doesn't need the tolerance/noise breakdown the fidelity metrics get)."""
    pooled = defaultdict(lambda: {"wall_time_ms": [], "size": []})
    for (_tier, algorithm, _noise_level), metrics in cells.items():
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
    fig_metric_per_family(cells, out / "fig2b_corner_recall_per_family.png",
                          "corner_recall")
    fig_metric_per_family(cells, out / "fig2c_corner_precision_per_family.png",
                          "corner_precision")
    fig_metric_per_family(cells, out / "fig2f_iou_per_family.png", "iou")
    sizes = sorted(np.unique(next(iter(cells.values()))["size"]))
    for s in sizes:
        fig_metric_per_family(cells, out / f"fig2d_corner_recall_per_family_d{s:03d}.png",
                              "corner_recall", size=s)
        fig_metric_per_family(cells, out / f"fig2e_corner_precision_per_family_d{s:03d}.png",
                              "corner_precision", size=s)
        fig_metric_per_family(cells, out / f"fig2g_iou_per_family_d{s:03d}.png",
                              "iou", size=s)


if __name__ == "__main__":
    main()
