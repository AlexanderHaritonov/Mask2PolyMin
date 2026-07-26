"""
A/B test for FitterConfig.apply_local_defect_margin: does turning it off actually buy the
wall-time speedup (and cost the complex-shape recall) its docstring in fit_to_points_sequence.py
claims (~20% faster, ~10% recall regression on complex shapes)?

Runs mask2polymin twice per contour -- apply_local_defect_margin=True (current default) vs
False -- at the same noise-matched tolerance run_benchmark.py uses, same dataset. Only the
fit call is timed, matching run_benchmark.measure()'s convention. Failures are logged to
stderr and skipped, not fatal.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from mask2polymin import FitterToPointsSequence, FitterConfig
from metrics import corner_metrics
from run_benchmark import matched_pair
from synth_shapes import dataset

RESULTS_DIR = Path(__file__).parent / "summarized_csvs"

# Mirrors plot_results.COMPLEX_FAMILIES -- inlined to avoid pulling in matplotlib for a
# numeric-only script.
COMPLEX_FAMILIES = {"car", "plane", "ship"}

COLUMNS = ["contour_id", "family", "shape_class", "noise_level", "tolerance",
           "apply_local_defect_margin", "n_segments", "corner_recall",
           "corner_precision", "corner_loc_err", "wall_time_ms"]


def measure(record: dict, tolerance: float, apply_margin: bool) -> dict:
    """Run mask2polymin once on one dataset record with apply_local_defect_margin fixed."""
    contour = record["contour_xy"]
    config = FitterConfig(tolerance=float(tolerance), apply_local_defect_margin=apply_margin)
    t0 = time.perf_counter()
    poly, _ = FitterToPointsSequence(config=config).fit(contour, is_closed=True)
    wall_ms = (time.perf_counter() - t0) * 1e3
    recall, precision, loc_err = corner_metrics(record["gt_corners_xy"], poly)
    return {
        "contour_id": record["contour_id"],
        "family": record["family"],
        "shape_class": "complex" if record["family"] in COMPLEX_FAMILIES else "simple",
        "noise_level": record["noise_level"],
        "tolerance": tolerance,
        "apply_local_defect_margin": apply_margin,
        "n_segments": len(poly) - 1,
        "corner_recall": round(recall, 4),
        "corner_precision": round(precision, 4),
        "corner_loc_err": round(loc_err, 4),
        "wall_time_ms": round(wall_ms, 3),
    }


def summarize(rows: list[dict]) -> list[dict]:
    """Mean recall/precision/wall-time per (shape_class, apply_local_defect_margin)."""
    cells: dict[tuple[str, bool], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["shape_class"], row["apply_local_defect_margin"]), []).append(row)

    summary = []
    for (shape_class, apply_margin), cell_rows in sorted(cells.items(), key=lambda kv: (kv[0][0], not kv[0][1])):
        summary.append({
            "shape_class": shape_class,
            "apply_local_defect_margin": apply_margin,
            "n": len(cell_rows),
            "corner_recall_mean": statistics.fmean(r["corner_recall"] for r in cell_rows),
            "corner_precision_mean": statistics.fmean(r["corner_precision"] for r in cell_rows),
            "wall_time_ms_mean": statistics.fmean(r["wall_time_ms"] for r in cell_rows),
            "wall_time_ms_median": statistics.median(r["wall_time_ms"] for r in cell_rows),
        })
    return summary


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'shape_class':<10} {'margin':<7} {'n':>5} {'recall':>8} {'precision':>10} "
          f"{'time_mean':>10} {'time_med':>9}")
    by_class: dict[str, dict[bool, dict]] = {}
    for row in summary:
        by_class.setdefault(row["shape_class"], {})[row["apply_local_defect_margin"]] = row
        print(f"{row['shape_class']:<10} {str(row['apply_local_defect_margin']):<7} "
              f"{row['n']:>5} {row['corner_recall_mean']:>8.4f} "
              f"{row['corner_precision_mean']:>10.4f} {row['wall_time_ms_mean']:>10.3f} "
              f"{row['wall_time_ms_median']:>9.3f}")

    print("\nTrue -> False deltas (what the docstring claims: faster, lower complex recall):")
    for shape_class, variants in by_class.items():
        if True not in variants or False not in variants:
            continue
        t, f = variants[True], variants[False]
        time_pct = (f["wall_time_ms_mean"] - t["wall_time_ms_mean"]) / t["wall_time_ms_mean"] * 100
        recall_pct = (f["corner_recall_mean"] - t["corner_recall_mean"]) / t["corner_recall_mean"] * 100
        print(f"  {shape_class:<10} wall_time: {time_pct:+6.1f}%   corner_recall: {recall_pct:+6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="apply_local_defect_margin True/False A/B test")
    parser.add_argument("--limit", type=int, default=None,
                         help="stop after N dataset records (smoke/timing run)")
    parser.add_argument("--reps", type=int, default=3,
                         help="reps per (shape, angle, noise>0 level); default matches run_benchmark.py")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "defect_margin_ab_raw.csv")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_fail = n_contours = 0
    t_start = time.perf_counter()
    all_rows = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, record in enumerate(dataset(reps=args.reps)):
            if args.limit is not None and i >= args.limit:
                break
            n_contours = i + 1
            _eps, tol = matched_pair(record["noise_level"])
            for apply_margin in (True, False):
                try:
                    row = measure(record, tol, apply_margin)
                    writer.writerow(row)
                    all_rows.append(row)
                    n_rows += 1
                except Exception as exc:
                    n_fail += 1
                    print(f"FAIL {record['contour_id']} margin={apply_margin} tol={tol}: {exc}",
                          file=sys.stderr)
            if n_contours % 105 == 0:
                print(f"  {n_contours} contours, {n_rows} rows, "
                      f"{time.perf_counter() - t_start:.0f} s", flush=True)
    elapsed = time.perf_counter() - t_start
    print(f"{n_rows} rows ({n_contours} contours, {n_fail} failures) "
          f"in {elapsed:.0f} s -> {args.out}")

    summary = summarize(all_rows)
    summary_path = RESULTS_DIR / "defect_margin_ab_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shape_class", "apply_local_defect_margin", "n",
                                                "corner_recall_mean", "corner_precision_mean",
                                                "wall_time_ms_mean", "wall_time_ms_median"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"summary -> {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
