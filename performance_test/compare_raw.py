"""
Compare mask2polymin's rows of two benchmark runs (two raw.csv files), e.g. before and after a fitter change.

Rows are paired by contour_id. A metric that is missing in either run (no matched corner) is left out for that contour.
Prints means as "before → after", per noise level x shape class and per contour size x shape class,
then for the corner metrics per size how many contours got better, stayed the same or got worse.

    python compare_raw.py summarized_csvs/raw_before.csv summarized_csvs/raw.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from plot_results import _shape_class

ALGORITHM = "mask2polymin"
METRICS = ["n_segments", "corner_recall", "corner_precision", "corner_loc_err", "corner_angle_err",
           "hausdorff", "rms_sym", "iou", "wall_time_ms"]
# +1: higher is better, -1: lower is better
CORNER_METRICS = {"corner_recall": +1, "corner_loc_err": -1}


def read_rows(path: Path) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {row["contour_id"]: row for row in csv.DictReader(f) if row["algorithm"] == ALGORITHM}


def value(row: dict, metric: str) -> float:
    text = row[metric]
    return math.nan if text in ("", "nan") else float(text)


def size_of(contour_id: str) -> str:
    return contour_id.split("_")[1]


def pairs(before: dict, after: dict, ids: list[str], metric: str) -> list[tuple[float, float]]:
    values = [(value(before[i], metric), value(after[i], metric)) for i in ids]
    return [(b, a) for b, a in values if not (math.isnan(b) or math.isnan(a))]


def mean_cell(before: dict, after: dict, ids: list[str], metric: str) -> str:
    values = pairs(before, after, ids, metric)
    if not values:
        return "-"
    b = sum(v[0] for v in values) / len(values)
    a = sum(v[1] for v in values) / len(values)
    return f"{b:.3g}→{a:.3g}"


def print_means(title: str, groups: dict[str, list[str]], before: dict, after: dict) -> None:
    print(f"\n{title}")
    print(f"{'group':<18}{'n':>5}" + "".join(f"{m:>18}" for m in METRICS))
    for name, ids in groups.items():
        print(f"{name:<18}{len(ids):>5}" + "".join(f"{mean_cell(before, after, ids, m):>18}" for m in METRICS))


def print_better_worse(groups: dict[str, list[str]], before: dict, after: dict) -> None:
    print("\nper contour: better / same / worse")
    print(f"{'group':<18}" + "".join(f"{m:>22}" for m in CORNER_METRICS))
    for name, ids in groups.items():
        cells = []
        for metric, direction in CORNER_METRICS.items():
            gains = [direction * (a - b) for b, a in pairs(before, after, ids, metric)]
            better = sum(g > 1e-9 for g in gains)
            worse = sum(g < -1e-9 for g in gains)
            cells.append(f"{better} / {len(gains) - better - worse} / {worse}")
        print(f"{name:<18}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="compare mask2polymin rows of two raw.csv files")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    before, after = read_rows(args.before), read_rows(args.after)
    ids = sorted(before.keys() & after.keys())
    if len(ids) != len(before) or len(ids) != len(after):
        print(f"only {len(ids)} contours in both runs ({len(before)} before, {len(after)} after)")

    by_noise, by_size = defaultdict(list), defaultdict(list)
    for i in ids:
        shape_class = _shape_class(i)
        by_noise[f"n{before[i]['noise_level']} {shape_class}"].append(i)
        by_size[f"{size_of(i)} {shape_class}"].append(i)
    by_noise = dict(sorted(by_noise.items()))
    by_size = dict(sorted(by_size.items()))

    print_means("means per noise level", by_noise, before, after)
    print_means("means per size", by_size, before, after)
    print_means("all", {"all": ids}, before, after)
    print_better_worse(by_size, before, after)


if __name__ == "__main__":
    main()
