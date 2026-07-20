"""
Analysis backing Fitter_Improvements_Plan.md item 1: does the convergence angle between
two consecutive fitted segments predict whether segments_to_polyline's line-intersection
corner lands near the true GT corner?

Runs FitterToPointsSequence directly (not the baselines.mask2polymin wrapper, which only
returns the polyline) across the full Tier 0 dataset at all 4 mask2polymin tolerances, so
each junction's two flanking segments are available alongside the vertex _corner() actually
produced. For every junction this computes:

  - angle_deg          convergence angle between the two lines, 0 (parallel) to 90
                        (perpendicular) degrees; derived from the cross product already
                        used by _corner()'s intersection formula (see _junction_angle_deg).
  - used_intersection  whether the CURRENT (pre-angle-gate) _corner() logic picked the
                        line intersection over the anchor.
  - actual_dist        distance from _corner()'s actual returned vertex to the nearest GT
                        corner; hit iff <= TAU (corner_metrics' recall radius).
  - anchor_dist        distance from the junction's anchor (orphan mean / projected
                        midpoint) to that same GT corner -- the counterfactual "what if
                        this junction fell back to the anchor instead" value.

The full sweep (1950 records x 4 tolerances, ~99k junctions) takes ~9 minutes, so results
are cached to results/corner_angle_observations.csv (regenerable, gitignored like raw.csv)
and reused on subsequent runs; pass --refresh to force recomputation.

Prints:
  1. angle_deg percentiles, hit vs. miss, pooled across all junctions (the plan's literal
     ask: "split by hit/miss ... find where the distributions separate").
  2. the same, restricted to used_intersection junctions -- the only ones an angle gate
     can actually change.
  3. a threshold sweep: for candidate angle cutoffs, how many currently-intersection
     junctions would flip to the anchor, and the hit-count delta that flip would cause,
     both pooled and broken out per tolerance (a threshold that helps one tolerance and
     hurts another is not a safe pick).
  4. the same net hit-count delta, broken out per family at a few candidate thresholds --
     the fix is expected to concentrate on families with genuine shallow-angle corners
     (plane, car, ship, ...), not spread evenly.

Run from performance_test/:  python analyze_corner_angles.py [--limit N] [--reps N] [--refresh]
"""
import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mask2polymin import FitterToPointsSequence, FitterConfig
from mask2polymin.polyline import _junction_anchor, _length
from synth_shapes import dataset

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_PATH = RESULTS_DIR / "corner_angle_observations.csv"

TOLERANCES = [0.35, 0.71, 1.41, 2.83]
TAU = 2.0  # corner_metrics' hit/miss radius
CANDIDATE_THRESHOLDS_DEG = [0, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40]
FAMILY_THRESHOLDS_DEG = [15, 18, 20, 25, 30]

CACHE_COLUMNS = ["family", "tolerance", "noise_level", "angle_deg", "used_intersection", "actual_dist", "anchor_dist"]


@dataclass
class Observation:
    family: str
    tolerance: float
    noise_level: int
    angle_deg: float
    used_intersection: bool
    actual_dist: float
    anchor_dist: float

    @property
    def hit(self) -> bool:
        return self.actual_dist <= TAU

    @property
    def anchor_hit(self) -> bool:
        return self.anchor_dist <= TAU


def _junction_angle_deg(seg_a, seg_b) -> float:
    """Convergence angle between two lines, 0-90 degrees. |cross| = sin(theta) where theta
    is the angle between the direction *vectors*; since a line's direction sign is
    arbitrary, theta and 180-theta are the same geometric junction, and sin is identical
    for both, so |cross| alone (no need for the dot product) already gives the correct
    undirected angle between the *lines*. Also well-conditioned exactly where it matters:
    arcsin's derivative is tame near 0, unlike arccos(dot)'s near 1 (small angles)."""
    a, b = seg_a.line_segment_params, seg_b.line_segment_params
    cross = a.direction[0] * b.direction[1] - a.direction[1] * b.direction[0]
    return float(np.degrees(np.arcsin(np.clip(abs(cross), 0.0, 1.0))))


def _observe_junction(seg_a, seg_b, tolerance: float, gt_corners: np.ndarray) -> tuple[float, bool, float, float]:
    """Mirrors _corner()'s current (pre-angle-gate) branch so the "used_intersection" flag
    and the anchor counterfactual are both available; the plan's proposed fix would insert
    the angle gate into exactly this branch, which is why this analysis is being run."""
    a, b = seg_a.line_segment_params, seg_b.line_segment_params
    anchor = _junction_anchor(seg_a, seg_b)
    angle_deg = _junction_angle_deg(seg_a, seg_b)

    cross = a.direction[0] * b.direction[1] - a.direction[1] * b.direction[0]
    used_intersection = False
    actual = anchor
    if abs(cross) > 1e-9:
        dp = b.start_point - a.start_point
        t = (dp[0] * b.direction[1] - dp[1] * b.direction[0]) / cross
        intersection = a.start_point + t * a.direction
        max_offset = max(3.0 * tolerance, min(_length(a), _length(b)))
        if np.linalg.norm(intersection - anchor) <= max_offset:
            used_intersection = True
            actual = intersection

    actual_dist = float(np.linalg.norm(gt_corners - actual, axis=1).min())
    anchor_dist = float(np.linalg.norm(gt_corners - anchor, axis=1).min())
    return angle_deg, used_intersection, actual_dist, anchor_dist


def observations(tolerances=TOLERANCES, reps: int = 3, limit: int | None = None):
    """Yield one Observation per (dataset record, tolerance, corner junction)."""
    for i, record in enumerate(dataset(reps=reps)):
        if limit is not None and i >= limit:
            break
        contour = record["contour_xy"]
        gt_corners = record["gt_corners_xy"]
        for tol in tolerances:
            config = FitterConfig(tolerance=tol)
            try:
                _, segments = FitterToPointsSequence(contour, is_closed=True, config=config).fit()
            except Exception as exc:
                print(f"FAIL {record['contour_id']} tol={tol}: {exc}")
                continue
            n = len(segments)
            if n < 2:
                continue
            for j in range(n):
                angle_deg, used_intersection, actual_dist, anchor_dist = _observe_junction(
                    segments[j], segments[(j + 1) % n], tol, gt_corners)
                yield Observation(
                    family=record["family"], tolerance=tol, noise_level=record["noise_level"],
                    angle_deg=angle_deg, used_intersection=used_intersection,
                    actual_dist=actual_dist, anchor_dist=anchor_dist)


def write_cache(obs: list[Observation], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CACHE_COLUMNS)
        for o in obs:
            writer.writerow([o.family, o.tolerance, o.noise_level, o.angle_deg,
                              int(o.used_intersection), o.actual_dist, o.anchor_dist])


def load_cache(path: Path = CACHE_PATH) -> list[Observation]:
    with open(path, newline="") as f:
        return [Observation(
            family=row["family"], tolerance=float(row["tolerance"]),
            noise_level=int(row["noise_level"]), angle_deg=float(row["angle_deg"]),
            used_intersection=bool(int(row["used_intersection"])),
            actual_dist=float(row["actual_dist"]), anchor_dist=float(row["anchor_dist"]))
            for row in csv.DictReader(f)]


def _percentiles(values: np.ndarray) -> str:
    if len(values) == 0:
        return "n=0"
    pcts = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
    labels = ["min", "p10", "p25", "median", "p75", "p90", "max"]
    body = ", ".join(f"{l}={v:.1f}" for l, v in zip(labels, pcts))
    return f"n={len(values)}: {body}"


def print_angle_distributions(obs: list[Observation]) -> None:
    print("\n=== 1. angle_deg distribution, hit vs miss (all junctions) ===")
    angles = np.array([o.angle_deg for o in obs])
    hits = np.array([o.hit for o in obs])
    print(f"  hit:  {_percentiles(angles[hits])}")
    print(f"  miss: {_percentiles(angles[~hits])}")

    print("\n=== 2. same, restricted to used_intersection junctions ===")
    used = np.array([o.used_intersection for o in obs])
    angles_u, hits_u = angles[used], hits[used]
    print(f"  hit:  {_percentiles(angles_u[hits_u])}")
    print(f"  miss: {_percentiles(angles_u[~hits_u])}")


def _sweep_row(flipped: list[Observation]) -> tuple[int, int, int, int, int]:
    """(n_flip, hit_to_hit, hit_to_miss, miss_to_hit, net_hit_delta) for a set of
    junctions that would flip from the intersection to the anchor."""
    hit_to_hit = sum(1 for o in flipped if o.hit and o.anchor_hit)
    hit_to_miss = sum(1 for o in flipped if o.hit and not o.anchor_hit)
    miss_to_hit = sum(1 for o in flipped if not o.hit and o.anchor_hit)
    return len(flipped), hit_to_hit, hit_to_miss, miss_to_hit, miss_to_hit - hit_to_miss


def _print_sweep_table(used: list[Observation], thresholds) -> None:
    header = f"{'deg':>4} {'n_flip':>7} {'hit->hit':>9} {'hit->miss':>10} {'miss->hit':>10} {'net_hit_delta':>14}"
    print(header)
    for t in thresholds:
        flipped = [o for o in used if o.angle_deg < t]
        n_flip, hit_to_hit, hit_to_miss, miss_to_hit, net = _sweep_row(flipped)
        print(f"{t:>4} {n_flip:>7} {hit_to_hit:>9} {hit_to_miss:>10} {miss_to_hit:>10} {net:>+14}")


def print_threshold_sweep(obs: list[Observation], thresholds=CANDIDATE_THRESHOLDS_DEG) -> None:
    print("\n=== 3. threshold sweep (junctions that would flip intersection -> anchor) ===")
    used = [o for o in obs if o.used_intersection]
    print(f"{len(used)} of {len(obs)} junctions currently use the intersection.\n")
    _print_sweep_table(used, thresholds)

    print("\n--- same sweep, broken out per tolerance ---")
    for tol in sorted({o.tolerance for o in obs}):
        print(f"\ntolerance={tol}")
        _print_sweep_table([o for o in used if o.tolerance == tol], thresholds)


def print_family_breakdown(obs: list[Observation], thresholds=FAMILY_THRESHOLDS_DEG) -> None:
    print("\n=== 4. net_hit_delta per family, at a few candidate thresholds ===")
    used = [o for o in obs if o.used_intersection]
    families = sorted({o.family for o in obs})
    print("family".ljust(10) + "".join(f"{t:>8}" for t in thresholds) + "   n_used")
    for fam in families:
        used_fam = [o for o in used if o.family == fam]
        row = fam.ljust(10)
        for t in thresholds:
            flipped = [o for o in used_fam if o.angle_deg < t]
            *_, net = _sweep_row(flipped)
            row += f"{net:>8}"
        print(row + f"   {len(used_fam)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Corner convergence-angle vs hit/miss analysis")
    parser.add_argument("--limit", type=int, default=None, help="stop after N dataset records")
    parser.add_argument("--reps", type=int, default=3, help="noise reps per (shape, angle, level)")
    parser.add_argument("--refresh", action="store_true", help="recompute even if a cache file exists")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = parser.parse_args()

    if args.cache.exists() and not args.refresh and args.limit is None:
        obs = load_cache(args.cache)
        print(f"loaded {len(obs)} cached junction observations from {args.cache}")
    else:
        t0 = time.perf_counter()
        obs = list(observations(reps=args.reps, limit=args.limit))
        elapsed = time.perf_counter() - t0
        print(f"{len(obs)} junction observations in {elapsed:.0f}s")
        if args.limit is None:
            write_cache(obs, args.cache)
            print(f"cached to {args.cache}")

    print_angle_distributions(obs)
    print_threshold_sweep(obs)
    print_family_breakdown(obs)


if __name__ == "__main__":
    main()
