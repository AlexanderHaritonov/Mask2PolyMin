"""
Tier 0 benchmark runner: sweep tolerances x contours x algorithms -> results/raw.csv.

Consumes `synth_shapes.dataset()` (the single enumeration point, 1050 contours) and runs
both algorithms at each of the 5 tolerance pairs on every contour -> one row per
(contour, algorithm, tolerance), 10 500 rows. All fidelity metrics are computed against
the GT polygon / GT mask, never the distorted input contour (see Perf_Test_Plan.md).
Failures are logged to stderr and skipped, not fatal.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

from baselines import mask2polymin, rdp_opencv
from metrics import corner_metrics, hausdorff, hd95, iou_rasterized, rms_directed, rms_distance
from synth_shapes import dataset

RESULTS_DIR = Path(__file__).parent / "results"

# (rdp_epsilon, m2p_tolerance) pairs: a geometric x2 ladder, tol = epsilon / sqrt(2)
TOLERANCE_PAIRS = [(0.5, 0.35), (1.0, 0.71), (2.0, 1.41), (4.0, 2.83), (8.0, 5.66)]

COLUMNS = ["contour_id", "tier", "n_input_points", "algorithm", "tolerance",
           "noise_level", "n_segments", "hausdorff", "hd95", "iou", "rms_sym",
           "rms_dir", "corner_recall", "corner_precision", "corner_loc_err",
           "wall_time_ms"]


def measure(record: dict, algorithm: str, fit_fn, tolerance: float) -> dict:
    """Run one algorithm at one tolerance on one dataset record; only the fit call is timed."""
    contour = record["contour_xy"]
    t0 = time.perf_counter()
    poly = fit_fn(contour, tolerance)
    wall_ms = (time.perf_counter() - t0) * 1e3
    gt_poly = record["gt_polygon_xy"]
    recall, precision, loc_err = corner_metrics(record["gt_corners_xy"], poly)
    return {
        "contour_id": record["contour_id"],
        "tier": 0,
        "n_input_points": len(contour) - 1,
        "algorithm": algorithm,
        "tolerance": tolerance,
        "noise_level": record["noise_level"],
        "n_segments": len(poly) - 1,
        "hausdorff": round(hausdorff(gt_poly, poly), 4),
        "hd95": round(hd95(gt_poly, poly), 4),
        "iou": round(iou_rasterized(poly, record["gt_mask"]), 5),
        "rms_sym": round(rms_distance(gt_poly, poly), 4),
        "rms_dir": round(rms_directed(gt_poly, poly), 4),
        "corner_recall": round(recall, 4),
        "corner_precision": round(precision, 4),
        "corner_loc_err": round(loc_err, 4),
        "wall_time_ms": round(wall_ms, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 0 benchmark sweep")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N dataset records (smoke/timing run)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "raw.csv")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_fail = n_contours = 0
    t_start = time.perf_counter()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, record in enumerate(dataset()):
            if args.limit is not None and i >= args.limit:
                break
            n_contours = i + 1
            for eps, tol in TOLERANCE_PAIRS:
                for name, fit_fn, t in [("rdp", rdp_opencv, eps),
                                        ("mask2polymin", mask2polymin, tol)]:
                    try:
                        writer.writerow(measure(record, name, fit_fn, t))
                        n_rows += 1
                    except Exception as exc:
                        n_fail += 1
                        print(f"FAIL {record['contour_id']} {name} tol={t}: {exc}",
                              file=sys.stderr)
            if n_contours % 105 == 0:
                print(f"  {n_contours} contours, {n_rows} rows, "
                      f"{time.perf_counter() - t_start:.0f} s", flush=True)
    elapsed = time.perf_counter() - t_start
    print(f"{n_rows} rows ({n_contours} contours, {n_fail} failures) "
          f"in {elapsed:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
