"""
Per-metric win/loss charts: for each metric, per shape class, how much mask2polymin
wins or loses against RDP on average across noise levels, and how often -- per
Comparison_Charts_Plan.md.

Reads summarized_csvs/raw.csv, pairs rdp/mask2polymin rows by contour_id (same noisy contour,
same noise-matched tolerance -- see run_benchmark.matched_pair), computes a per-contour
`comparison` value per metric (positive = mask2polymin better, direction-normalized --
see BETTER_DIRECTION), aggregates to summarized_csvs/comparison_summary.csv, and renders
comparison_*.png in charts/.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import gridspec
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
import numpy as np

from plot_results import (
    RESULTS_DIR, CHARTS_DIR, FIG_DPI, COLORS, INK, INK_2, SHAPE_CLASSES,
    _shape_class, _style, _draw_icon_rows,
)

TIER = 0

# direction each metric's raw value must move for mask2polymin to be "better" --
# corner_bias and the ratio metrics encode direction (cut vs. overshoot / shrink vs.
# grow) in their sign, not just magnitude, so a raw m2p-rdp delta there can read
# backwards (see Comparison_Charts_Plan.md, "Sign convention"); "zero"/"one" compare
# distance from the neutral value (0 / 1) instead.
BETTER_DIRECTION = {
    "n_segments": "lower", "hausdorff": "lower", "hd95": "lower",
    "rms_sym": "lower", "rms_dir": "lower", "corner_loc_err": "lower",
    "corner_angle_err": "lower",
    "iou": "higher", "corner_recall": "higher", "corner_precision": "higher",
    "corner_bias": "zero",
    "area_ratio": "one", "perimeter_ratio": "one",
}
METRICS = list(BETTER_DIRECTION)

# median for metrics where a single contour's outlier value can swing the cell's
# center -- hausdorff is a per-contour max by construction; the corner_* metrics are
# each a per-contour mean over *matched* corners only, so a contour that matched few
# corners is a noisier sample than one that matched all of them. Mean everywhere else.
AGG_STAT = {m: ("median" if m in
                 ("hausdorff", "corner_loc_err", "corner_bias", "corner_angle_err")
                 else "mean")
            for m in METRICS}

# NaN on a row that recalled zero corners (Perf_Test_Plan.md) -- a pair is dropped
# from that metric's comparison (not from other metrics') if either side is NaN
NAN_ABLE = {"corner_loc_err", "corner_bias", "corner_angle_err", "area_ratio",
            "perimeter_ratio"}

YLABEL = {
    "n_segments": "segments", "hausdorff": "Hausdorff (px)", "hd95": "HD95 (px)",
    "rms_sym": "symmetric RMS (px)", "rms_dir": "directed RMS (px)",
    "corner_recall": "corner recall", "corner_precision": "corner precision",
    "corner_loc_err": "corner loc. error (px)", "corner_bias": "|corner bias| (px)",
    "corner_angle_err": "corner angle error (deg)",
    "area_ratio": "|area ratio - 1|", "perimeter_ratio": "|perimeter ratio - 1|",
    "iou": "IoU",
}

FIGURES = [
    ("comparison_segments.png", ["n_segments"],
     "segment count: comparison & win count vs. noise level"),
    ("comparison_hausdorff.png", ["hausdorff", "hd95"],
     "Hausdorff / HD95: comparison & win count vs. noise level"),
    ("comparison_rms.png", ["rms_sym", "rms_dir"],
     "RMS symmetric / directed: comparison & win count vs. noise level"),
    ("comparison_corner_recall_precision.png", ["corner_recall", "corner_precision"],
     "corner recall / precision: comparison & win count vs. noise level"),
    ("comparison_corner_loc_err.png", ["corner_loc_err"],
     "corner localization error: comparison & win count vs. noise level"),
    ("comparison_corner_bias.png", ["corner_bias"],
     "corner bias magnitude: comparison & win count vs. noise level"),
    ("comparison_area.png", ["area_ratio"],
     "area ratio (distance from 1): comparison & win count vs. noise level"),
    ("comparison_perimeter.png", ["perimeter_ratio"],
     "perimeter ratio (distance from 1): comparison & win count vs. noise level"),
    ("comparison_iou.png", ["iou"],
     "IoU: comparison & win count vs. noise level"),
    ("comparison_corner_angle.png", ["corner_angle_err"],
     "corner turning-angle error: comparison & win count vs. noise level"),
]


def comparison(direction: str, rdp_v: float, m2p_v: float) -> float:
    """Positive = mask2polymin better, in the metric's own unit."""
    if direction == "lower":
        return rdp_v - m2p_v
    if direction == "higher":
        return m2p_v - rdp_v
    if direction == "zero":
        return abs(rdp_v) - abs(m2p_v)
    return abs(rdp_v - 1) - abs(m2p_v - 1)  # "one"


def read_pairs(raw_path: Path) -> dict:
    """contour_id -> {"tier", "noise_level", "shape_class",
    "rdp": {metric: float}, "mask2polymin": {metric: float}} -- only contour_ids with
    both algorithms present; logs a count of any that are missing one side."""
    pairs = defaultdict(dict)
    with open(raw_path) as f:
        for row in csv.DictReader(f):
            entry = pairs[row["contour_id"]]
            entry["tier"] = int(row["tier"])
            entry["noise_level"] = int(row["noise_level"])
            entry["shape_class"] = _shape_class(row["contour_id"])
            entry[row["algorithm"]] = {m: float(row[m]) for m in METRICS}
    complete = {cid: e for cid, e in pairs.items() if "rdp" in e and "mask2polymin" in e}
    missing = len(pairs) - len(complete)
    if missing:
        print(f"WARN: {missing} contour_id(s) missing one algorithm's row -- dropped")
    return complete


def compute_comparisons(pairs: dict) -> dict:
    """(metric, tier, shape_class, noise_level) -> np.array of per-contour comparison
    values (NaN pairs dropped for NAN_ABLE metrics)."""
    buckets = defaultdict(list)
    for entry in pairs.values():
        key_tail = (entry["tier"], entry["shape_class"], entry["noise_level"])
        for metric in METRICS:
            rdp_v, m2p_v = entry["rdp"][metric], entry["mask2polymin"][metric]
            if metric in NAN_ABLE and (np.isnan(rdp_v) or np.isnan(m2p_v)):
                continue
            buckets[(metric,) + key_tail].append(
                comparison(BETTER_DIRECTION[metric], rdp_v, m2p_v))
    return {key: np.array(v) for key, v in buckets.items()}


def _center(values: np.ndarray, stat: str) -> float:
    return float(np.median(values)) if stat == "median" else float(np.mean(values))


def summarize(buckets: dict) -> list[dict]:
    rows = []
    for (metric, _tier, shape_class, noise_level), values in sorted(
            buckets.items(),
            key=lambda kv: (kv[0][0], kv[0][3], SHAPE_CLASSES.index(kv[0][2]))):
        stat = AGG_STAT[metric]
        rows.append({
            "metric": metric, "shape_class": shape_class, "noise_level": noise_level,
            "n_pairs": len(values), "stat": stat,
            "comparison_center": round(_center(values, stat), 4),
            "win_count": int(np.sum(values > 0)),
        })
    return rows


def write_summary(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _panel(ax, buckets: dict, metric: str, shape_class: str, levels: list) -> None:
    stat = AGG_STAT[metric]
    centers, win_counts, n_pairs = [], [], []
    for level in levels:
        values = buckets.get((metric, TIER, shape_class, level), np.empty(0))
        n_pairs.append(len(values))
        centers.append(_center(values, stat) if len(values) else np.nan)
        win_counts.append(int(np.sum(values > 0)) if len(values) else 0)

    ax.axhline(0, color=INK_2, linewidth=0.8, linestyle="--")
    ax.plot(levels, centers, "-o", color=COLORS["mask2polymin"], linewidth=2, markersize=6)
    ax.set_xticks(levels)
    _style(ax)

    ax2 = ax.twinx()
    ax2.bar(levels, win_counts, color=INK_2, alpha=0.35, width=0.5)
    top = max(n_pairs) if n_pairs else 0
    ax2.set_ylim(0, top * 1.15 if top else 1)
    ax2.set_yticks([])
    for side in ax2.spines.values():
        side.set_visible(False)
    # the comparison line (independent scale, ax) can pass behind any bar's label at
    # any noise level -- no fixed offset dodges every case, so the label itself gets
    # a white halo to stay legible over the line/marker rather than trying to avoid it
    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    for level, wc, n in zip(levels, win_counts, n_pairs):
        if n:
            ax2.annotate(f"{wc}/{n}", (level, wc), textcoords="offset points",
                         xytext=(0, 5), ha="center", fontsize=7, color=INK_2,
                         path_effects=halo)


def fig_comparison(buckets: dict, metrics: list, suptitle: str, out_path: Path) -> None:
    levels = sorted({k[3] for k in buckets if k[0] == metrics[0] and k[1] == TIER})
    nrows = len(metrics)
    if nrows == 1:
        fig = plt.figure(figsize=(9.4, 6.1))
        outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=(1, 5), hspace=0.12)
        margins = dict(left=0.1, right=0.96, top=0.9, bottom=0.17)
    else:
        fig = plt.figure(figsize=(9.4, 8.0))
        outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=(1, 9), hspace=0.12)
        margins = dict(left=0.1, right=0.96, top=0.93, bottom=0.13)
    plot_cols = [gridspec.GridSpecFromSubplotSpec(nrows, 1, subplot_spec=outer[1, col],
                                                   hspace=0.4) for col in range(2)]
    # set final margins before reading any cell's get_position() (icons included) --
    # matches plot_results.py's convention: tight_layout doesn't see the nested
    # gridspecs, so margins are fixed by hand up front instead.
    outer.update(**margins)
    _draw_icon_rows(fig, outer)

    for row, metric in enumerate(metrics):
        row_anchor = None
        for col, shape_class in enumerate(SHAPE_CLASSES):
            ax = fig.add_subplot(plot_cols[col][row, 0], sharey=row_anchor)
            row_anchor = row_anchor or ax
            _panel(ax, buckets, metric, shape_class, levels)
            if row == 0:
                ax.set_title(shape_class, fontsize=10.5, color=INK, fontweight="bold")
            if col == 0:
                stat_word = AGG_STAT[metric]
                ax.set_ylabel(f"{stat_word} comparison\n{YLABEL[metric]}",
                              fontsize=9, color=INK_2)
            if row == nrows - 1:
                ax.set_xlabel("noise level", fontsize=9, color=INK_2)

    fig.suptitle(suptitle, fontsize=11, color=INK)
    footnote_y = 0.02 if nrows == 1 else 0.045
    fig.text(0.5, footnote_y,
             "line = comparison (left axis, mask2polymin vs RDP, positive = mask2polymin "
             "better, 0 dashed)\nbars = win count out of n contours (right axis, "
             "labeled win/n)",
             ha="center", va="bottom", fontsize=9, color=INK_2, style="italic",
             fontweight="bold")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"figure -> {out_path}")


def main() -> None:
    pairs = read_pairs(RESULTS_DIR / "raw.csv")
    buckets = compute_comparisons(pairs)
    rows = summarize(buckets)
    write_summary(rows, RESULTS_DIR / "comparison_summary.csv")
    print(f"{len(rows)} cells -> {RESULTS_DIR / 'comparison_summary.csv'}")

    out = CHARTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    for filename, metrics, suptitle in FIGURES:
        fig_comparison(buckets, metrics, suptitle, out / filename)


if __name__ == "__main__":
    main()
