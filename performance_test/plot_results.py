"""
Aggregation and figures for the benchmark results, per Perf_Test_Plan.md.

Reads results/raw.csv and aggregates median / p25 / p75 / p95 of every metric per
(tier, algorithm, tolerance, noise_level) cell -> results/summary.csv, prints the
per-cell median table with the tolerances shown as aligned pairs, and renders the
Tier 0 figures (plan plots 1-2). Plots 3-4 are Tier 1 and land with the COCO run.
"""
import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
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

# native tolerance -> index of its aligned (rdp_eps, m2p_tol) pair, for ordering/labels
_PAIR_SEQ = [(0.5, 0.35), (1.0, 0.71), (2.0, 1.41), (4.0, 2.83)]
PAIR_IDX = {t: i for i, pair in enumerate(_PAIR_SEQ) for t in pair}
PAIR_LABEL = [f"e{e}/t{t}" for e, t in _PAIR_SEQ]


def read_cells(raw_path: Path) -> dict:
    """Group raw.csv rows into (tier, algorithm, tolerance, noise_level) cells;
    each cell maps metric name -> np.array of values across the cell's contours."""
    lists = defaultdict(lambda: defaultdict(list))
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            key = (int(row["tier"]), row["algorithm"], float(row["tolerance"]),
                   int(row["noise_level"]))
            for m in METRIC_COLS:
                lists[key][m].append(float(row[m]))
    return {key: {m: np.array(v) for m, v in metrics.items()}
            for key, metrics in lists.items()}


def _cell_order(key) -> tuple:
    tier, algorithm, tolerance, noise_level = key
    return (tier, noise_level, PAIR_IDX[tolerance], algorithm != "rdp")


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
        tier, algorithm, tolerance, noise_level = key
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
    """Per-cell median table, tolerances grouped as aligned pairs, rdp above m2p."""
    header = f"{'pair':<12}{'algorithm':<14}" + "".join(
        f"{c:>9}" for c in ["segs", "rms_sym", "hd95", "iou", "recall", "precis",
                            "loc_err", "ms"])
    for tier in sorted({k[0] for k in cells}):
        for level in sorted({k[3] for k in cells if k[0] == tier}):
            n = len(next(v for k, v in cells.items()
                         if k[0] == tier and k[3] == level)["n_segments"])
            print(f"\ntier {tier}, noise level {level}  (medians over {n} contours)")
            print(header)
            keys = [k for k in cells if k[0] == tier and k[3] == level]
            for key in sorted(keys, key=_cell_order):
                c = cells[key]
                cols = [_percentile(c[m], 50.0) for m in
                        ["n_segments", "rms_sym", "hd95", "iou", "corner_recall",
                         "corner_precision", "corner_loc_err", "wall_time_ms"]]
                print(f"{PAIR_LABEL[PAIR_IDX[key[2]]]:<12}{key[1]:<14}"
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


def _curve(cells, tier, level, algorithm, metric):
    """Median metric value at each tolerance pair, tight -> loose."""
    a = 0 if algorithm == "rdp" else 1
    keys = [(tier, algorithm, pair[a], level) for pair in _PAIR_SEQ]
    return [_percentile(cells[k][metric], 50.0) for k in keys]


def fig_segments_vs_rms(cells, out_path: Path, tier: int = 0) -> None:
    """Plan plot 1: median segments vs median symmetric RMS to GT, one panel per noise
    level, one tolerance curve (tight -> loose) per algorithm. Lower-left is better."""
    levels = sorted({k[3] for k in cells if k[0] == tier})
    fig, axes = plt.subplots(1, len(levels), figsize=(3.7 * len(levels), 3.7), sharey=True)
    for ax, level in zip(axes, levels):
        for algo in ("rdp", "mask2polymin"):
            xs = _curve(cells, tier, level, algo, "n_segments")
            ys = _curve(cells, tier, level, algo, "rms_sym")
            ax.plot(xs, ys, "-o", color=COLORS[algo], linewidth=2, markersize=6,
                    label=SERIES_LABEL[algo])
            # selective direct labels: tolerance identity on the endpoints only
            eps, tol = _PAIR_SEQ[0]
            tight = f"ε{eps}" if algo == "rdp" else f"t{tol}"
            eps, tol = _PAIR_SEQ[-1]
            loose = f"ε{eps:g}" if algo == "rdp" else f"t{tol}"
            ax.annotate(tight, (xs[0], ys[0]), textcoords="offset points",
                        xytext=(6, 5), fontsize=8, color=INK_2)
            loose_dy = (-2, 8) if algo == "rdp" else (-4, -14)
            ax.annotate(loose, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=loose_dy, fontsize=8, color=INK_2)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_title(f"noise level {level}" + (" (clean)" if level == 0 else ""),
                     fontsize=10, color=INK)
        ax.set_xlabel("median segments", fontsize=9, color=INK_2)
        _style(ax)
    axes[0].set_ylabel("median RMS (px)", fontsize=9, color=INK_2)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    axes[-1].annotate("lower-left is better", (0.03, 0.03), xycoords="axes fraction",
                      fontsize=8, color=INK_2, style="italic")
    fig.suptitle("segment count vs fidelity, per tolerance sweep",
                 fontsize=11, color=INK)
    eps_list = "/".join(f"{e:g}" for e, _ in _PAIR_SEQ)
    tol_list = "/".join(f"{t:g}" for _, t in _PAIR_SEQ)
    fig.text(0.5, 0.01,
             f"{len(_PAIR_SEQ)} tolerance settings per curve, tight (right) → loose (left):  "
             f"RDP ε = {eps_list} px (L∞)   ·   Mask2PolyMin t = ε/√2 = {tol_list} px (RMS)",
             ha="center", fontsize=9.5, fontweight="bold", color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_path}")


def fig_corner_recall(cells, out_path: Path, tier: int = 0, pair_idx: int = 2) -> None:
    """Plan plot 2: corner recall (and precision, same scale) vs noise level at the
    canonical tolerance pair, medians with p25-p75 band."""
    levels = sorted({k[3] for k in cells if k[0] == tier})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, metric, title in zip(axes, ("corner_recall", "corner_precision"),
                                 ("corner recall", "corner precision")):
        for algo in ("rdp", "mask2polymin"):
            tol = _PAIR_SEQ[pair_idx][0 if algo == "rdp" else 1]
            keys = [(tier, algo, tol, level) for level in levels]
            med = [_percentile(cells[k][metric], 50.0) for k in keys]
            p25 = [_percentile(cells[k][metric], 25.0) for k in keys]
            p75 = [_percentile(cells[k][metric], 75.0) for k in keys]
            ax.plot(levels, med, "-o", color=COLORS[algo], linewidth=2, markersize=6,
                    label=SERIES_LABEL[algo])
            ax.fill_between(levels, p25, p75, color=COLORS[algo], alpha=0.12,
                            linewidth=0)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{lv}" + (" (clean)" if lv == 0 else "") for lv in levels])
        ax.set_ylim(0, 1.05)
        ax.set_title(title + " (median, p25–p75)", fontsize=10, color=INK)
        ax.set_xlabel("noise level", fontsize=9, color=INK_2)
        _style(ax)
    eps, tol = _PAIR_SEQ[pair_idx]
    axes[0].set_ylabel("fraction of corners", fontsize=9, color=INK_2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    fig.suptitle(f"corner survival vs noise at ε={eps} / tol={tol} "
                 f"(τ = 2 px)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_path}")


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
    fig_segments_vs_rms(cells, args.out.parent / "fig1_segments_vs_rms.png")
    fig_corner_recall(cells, args.out.parent / "fig2_corner_recall.png")


if __name__ == "__main__":
    main()
