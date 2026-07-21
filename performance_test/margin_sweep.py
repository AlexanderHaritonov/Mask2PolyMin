"""
Targeted sweep of LOCAL_DEFECT_MARGIN candidates, backing Fitter_Improvements_Plan.md item 2.

Reproduces the plan's two evidence cells across a range of candidate margins (not just the
three values -- plain A / 9.0 / 4.0 -- the original investigation happened to try), to answer
the plan's open question: is there an intermediate value that recovers more of car's recall
without reopening the tol=0.35 quantization-jitter regression as much as margin=4.0 (the
current value) did? Also checks plane specifically, since step 1 (the angle-aware _corner()
fix) already moved plane's recall and this shouldn't be re-eroded by a margin change.

  - Recall slice: tol=1.41, noise=0, every family/size, broken out PER SIZE -- reproduces
    "car @ d064, tol=1.41, noise=0" from the plan. Per-size matters: car/plane's problem
    sizes are their d064 variant specifically (SMALL_SIZE_OVERRIDE) -- d128/d320 already
    recall near-perfectly regardless of margin, so a family-pooled median across sizes
    dilutes the real signal down to noise (confirmed the hard way: pooled car recall read
    1.0 at margin=4.0, while car@d064 alone reads 0.333, matching the plan exactly).
  - Jitter slice: tol=0.35, noise 0-2, every family -- reproduces the plan's other evidence
    row (pinned-at-cap fraction, median precision): the quantization-jitter regression risk.

LOCAL_DEFECT_MARGIN is a module-level constant, not a FitterConfig field, so candidates are
applied by monkeypatching fit_to_points_sequence.LOCAL_DEFECT_MARGIN before each fit() call
(_has_severe_local_defect reads it as a global at call time, not at import time, so a
reassignment here takes effect on the next fit()).

Run from performance_test/:  python margin_sweep.py
"""
from collections import defaultdict

import numpy as np

import mask2polymin.fit_to_points_sequence as fps
from mask2polymin import FitterConfig
from baselines import mask2polymin
from synth_shapes import dataset
from metrics import corner_metrics

CANDIDATES = [1.0, 2.0, 3.0, 4.0, 4.5, 4.6, 5.0, 6.0, 8.0, 10.0, 15.0, 30.0]
RECALL_TOL = 1.41
JITTER_TOL = 0.35
JITTER_NOISE_LEVELS = {0, 1, 2}
MAX_SEGMENTS = FitterConfig().max_segments_count
DIAGNOSTIC_SIZE = 64  # car/plane's SMALL_SIZE_OVERRIDE size -- where their problem concentrates


def main():
    records = list(dataset())
    recall_records = [r for r in records if r["noise_level"] == 0]
    jitter_records = [r for r in records if r["noise_level"] in JITTER_NOISE_LEVELS]

    print(f"{len(recall_records)} contours in the tol={RECALL_TOL} recall slice (noise=0, "
          f"all families/sizes)")
    print(f"{len(jitter_records)} contours in the tol={JITTER_TOL} jitter slice (noise 0-2, "
          f"all families)\n")

    recall_by_margin = {}   # margin -> {(family, size_px): [recall, ...]}
    jitter_by_margin = {}    # margin -> (pinned_at_cap_fraction, median_precision)

    for margin in CANDIDATES:
        fps.LOCAL_DEFECT_MARGIN = margin

        by_family_size = defaultdict(list)
        for r in recall_records:
            poly = mask2polymin(r["contour_xy"], RECALL_TOL)
            rec, _, _ = corner_metrics(r["gt_corners_xy"], poly)
            by_family_size[(r["family"], r["size_px"])].append(rec)
        recall_by_margin[margin] = by_family_size

        pinned = 0
        precisions = []
        for r in jitter_records:
            poly = mask2polymin(r["contour_xy"], JITTER_TOL)
            _, prec, _ = corner_metrics(r["gt_corners_xy"], poly)
            precisions.append(prec)
            if len(poly) - 1 >= MAX_SEGMENTS:
                pinned += 1
        jitter_by_margin[margin] = (pinned / len(jitter_records), float(np.median(precisions)))

    print(f"=== 1. car@d{DIAGNOSTIC_SIZE:03d} / plane@d{DIAGNOSTIC_SIZE:03d} recall "
          f"(tol={RECALL_TOL}, noise=0) vs. jitter slice (tol={JITTER_TOL}, noise 0-2) ===\n")
    print(f"{'margin':>7}  {'car@d064':>9}  {'plane@d064':>11}  {'pinned@cap':>11}  "
          f"{'t035_precision':>15}")
    for margin in CANDIDATES:
        car_med = np.median(recall_by_margin[margin][("car", DIAGNOSTIC_SIZE)])
        plane_med = np.median(recall_by_margin[margin][("plane", DIAGNOSTIC_SIZE)])
        pinned_frac, prec_med = jitter_by_margin[margin]
        print(f"{margin:>7.1f}  {car_med:>9.4f}  {plane_med:>11.4f}  {pinned_frac:>10.1%}  "
              f"{prec_med:>15.4f}")

    print(f"\n=== 2. car / plane recall broken out per size (tol={RECALL_TOL}, noise=0) ===\n")
    sizes = sorted({s for (fam, s) in recall_by_margin[CANDIDATES[0]] if fam in ("car", "plane")})
    print(f"{'family':<8}{'size':<7}" + "".join(f"{m:>7.1f}" for m in CANDIDATES))
    for family in ("car", "plane"):
        for size in sizes:
            row = [np.median(recall_by_margin[m].get((family, size), [float("nan")]))
                   for m in CANDIDATES]
            print(f"{family:<8}{size:<7}" + "".join(f"{v:>7.3f}" for v in row))

    print(f"\n=== 3. per-family recall pooled across sizes (tol={RECALL_TOL}, noise=0) ===\n")
    families = sorted({fam for (fam, s) in recall_by_margin[CANDIDATES[0]]})
    print(f"{'family':<10}" + "".join(f"{m:>7.1f}" for m in CANDIDATES))
    for family in families:
        row = []
        for m in CANDIDATES:
            vals = [v for (fam, s), lst in recall_by_margin[m].items() if fam == family for v in lst]
            row.append(np.median(vals))
        print(f"{family:<10}" + "".join(f"{v:>7.3f}" for v in row))


if __name__ == "__main__":
    main()
