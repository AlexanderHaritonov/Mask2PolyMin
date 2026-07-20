# Mask2PolyMin — Fitter Improvements Plan

Three outstanding items that came out of the Option A/B/margin work on
[fit_to_points_sequence.py](../mask2polymin/fit_to_points_sequence.py)'s termination logic and the
subsequent per-family benchmark investigation. Unlike A/B/margin (already landed and committed),
these three were not implemented yet; going through them one at a time, item 1 is now done (see
its Resolved note below).

## Suggested order

1. **`_corner()` angle-awareness** first — fully isolated (lives in `polyline.py`, downstream of
   all splitting/termination logic), so it can't be affected by or interfere with the other two.
   **Done.**
2. **Re-tune `LOCAL_DEFECT_MARGIN`** after (1) lands — car's recall shortfall is *partly* explained
   by the margin (see evidence below), but not necessarily *entirely*; the corner-reconstruction fix
   may itself move car's numbers and change what "optimal" means. Re-run the car investigation after
   (1), not before.
3. **`max_deviation` split-ranking flag** is independent of the other two and can happen any time —
   it changes which segment gets picked *among several already-eligible ones*, not whether a defect
   is flagged or how a finished segmentation's corners get reconstructed.
4. Promote the created params to config vals
---

## 1. `_corner()`'s plausibility check should be angle-aware, not just distance-aware

**Problem.** `segments_to_polyline`'s `_corner()` ([polyline.py:50-78](../mask2polymin/polyline.py#L50-L78))
reconstructs a vertex as the intersection of two adjacent fitted lines, falling back to an anchor
(orphan-point mean, or projected-endpoint midpoint) only if the intersection lands farther than
`max_offset = max(3·tolerance, min(len(a), len(b)))` from that anchor — a **distance-only** check.

When a short connecting edge between two long edges gets legitimately orphaned (within
`max_orphans_per_junction`), the corner is reconstructed by intersecting the two long lines directly.
If those two lines converge at a shallow angle, the intersection point is numerically unstable — a
small direction error in either TLS fit gets amplified into a large positional error along the lines.
`max_offset` doesn't catch this because it's calibrated by segment *length* (tens of px), which is
far larger than the actual error.

**Evidence** (`plane_d064_a0_n0_r0`, tolerance=1.41 — see conversation for the full investigation):

| GT corners | fitted segments (dir, len, loss) | convergence angle | miss distance |
|---|---|---|---|
| 1, 2 (wingtip, true edge between them ≈4.8px) | `[21..47]` dir −28.3°, len 28.6, loss 2.51 / `[50..76]` dir −14.5°, len 26.0, loss 2.91 | **13.8°** | 18-20 px |
| 5, 6 / 8, 9 (tail-fin, similar short connecting edges) | same orphan-then-intersect pattern at segment boundaries | shallow | 2-6 px |
| 0, 3, 4, 7, 11, 14, 17 (well-recalled) | — | much less shallow | <1 px |

Both flanking segments are *good* fits (mean loss/point ≈0.09-0.11, well under `tolerance_sq`) — this
isn't a fitting-quality problem, it's specifically an intersection-stability problem.

Note: GT polygon interior angle at a corner is **not** a usable proxy for this — it's the angle
between the two *edges meeting at that corner*, not the angle between the two *fitted segment
directions* on either side of an orphan gap (which can span many more points than just the two edges
immediately touching the corner). An early attempt to correlate interior angle with miss distance
was refuted directly: corners with near-identical interior angles (~119°) had wildly different miss
distances (0.29 px vs 19.77 px). Any threshold must be derived from the segments' fitted directions,
not the GT geometry.

**Proposed approach.** Add an angle-based condition to `_corner()`: require `a.direction` and
`b.direction` to diverge by more than some minimum angle before trusting the intersection at all,
independent of the distance-to-anchor check — fall back to the anchor when the lines are too close
to parallel, regardless of how close the intersection lands to the anchor.

**Open question.** What's the right angle threshold? Nothing derived yet — needs a proper sweep:
compute the actual fitted-segment convergence angle (not GT interior angle) at every corner junction
across the Tier 0 dataset, split by hit/miss (τ=2px), and find where the distributions separate.
13.8° is one confirmed bad case; need the good-case distribution too before picking a cutoff.

**Validation.** This touches every `fit()` call (via `segments_to_polyline`), so needs a full
benchmark re-run afterward, not just the plane/car slice — check for regressions on families that
currently recall well. Also worth a dedicated unit test with a synthetic shallow-angle junction
(mirroring how `test_split_segment_clamps_pivot_near_segment_start/_end` were added for the B pivot
clamp), so the behavior is locked in independent of the full benchmark noise floor.

**Resolved.** Added `MIN_CORNER_ANGLE_DEG = 20.0` to `polyline.py`: below this convergence angle,
`_corner()` falls back to the anchor regardless of how close the intersection lands (the distance
check is untouched, now a second independent condition). The threshold came from a full-dataset
hit/miss sweep, not a guess — see `performance_test/analyze_corner_angles.py` (cached to
`results/corner_angle_observations.csv`), which computes the actual convergence angle at every
corner junction across all 1950 Tier 0 contours × 4 tolerances and simulates flipping
intersection → anchor at candidate thresholds. Net corner-recall hits turn positive above ~15°,
peak around 18-20°, and go sharply negative above ~30° (too aggressive).

The per-family breakdown was the key finding: **the gain is almost entirely one family.** At 20°,
plane nets +206 hit junctions; every other family is flat to slightly negative (car -10, hexagon
-13, star -13, house -10, out of ~7-11k intersection junctions each) — matching the framing above:
this item was always diagnosed as a plane problem, car's is the margin below. A full benchmark
re-run (isolated via `git stash` on just `polyline.py`, since the previously-committed baseline
turned out to be stale relative to the current `fit_to_points_sequence.py` for unrelated reasons)
confirmed it end-to-end: pooled across noise levels, plane's median corner recall at tol=1.41 moved
0.333 → 0.389 (precision 0.412 → 0.444); every other family's median recall/precision is
byte-identical, means move by under 0.1pp. Segment/corner counts are unaffected in every cell, as
expected — `_corner()` only places a vertex for an already-finalized segment list, it can't change
how many there are (confirmed empirically, not just by inspection: the same 100-contour sample
with vs. without the change produced byte-identical segment counts in all 400 rows).

---

## 2. Tune `LOCAL_DEFECT_MARGIN` — no single value serves both regimes cleanly

**Problem.** `LOCAL_DEFECT_MARGIN` ([fit_to_points_sequence.py:21-22](../mask2polymin/fit_to_points_sequence.py#L21-L22))
gates whether the no-improvement stop can be overridden by a "severe local defect." It has to
distinguish two things that occupy overlapping magnitude ranges:

- **Pixel-quantization jitter** at tight tolerances (a straight, quantized edge can put a single
  point up to ~2x tolerance over the line with zero real geometry) — must *not* count as severe.
- **Car's genuinely-real but modest corners** — apparently many of them sit in that same
  2-3x-tolerance band, so a margin tight enough to exclude quantization jitter also excludes some
  of car's real geometry.

**Evidence** (car @ d064, tol=1.41, noise=0; tol=0.35 pooled across all families, noise 0-2):

| | plain A (no margin) | margin=9.0 (3x linear) | margin=4.0 (2x linear) |
|---|---|---|---|
| car recall (median) | 0.389 | 0.111 | 0.333 |
| tol=0.35 noise=1 pinned-at-cap | — | 41% | 58% |
| tol=0.35 noise=1 median precision | — | 0.667 | 0.556 |

Plane's recall did **not move at all** between margin=9.0 and margin=4.0 (0.444 both times, identical
segment counts) — its shortfall is the `_corner()` issue above, not this margin. Don't expect
margin-tuning to move plane; only re-evaluate car here.

**Open question.** Is there an intermediate value (6.0-7.0?) that recovers more of car's recall
without reopening tol=0.35 as much as margin=4.0 did? Or is a single global magnitude threshold
structurally unable to separate these two cases, since they overlap in the same relative-to-tolerance
range? If the latter, alternatives worth considering (not yet explored):
- A minimum contiguous run-length of over-threshold points, distinguishing an isolated noisy pixel
  from a real (multi-point) corner region — flagged early in the A/B discussion as more complex and
  of uncertain effectiveness, never built.
- Accepting the trade-off as inherent and picking a value based on which regime matters more for
  real (non-synthetic) input — real segmentation masks likely have a gentler quantization floor than
  this benchmark's worst case, which might make a lower margin safer in practice than the synthetic
  tol=0.35 cells suggest. Untested; would need Tier 1 (COCO) data, which per
  [Perf_Test_Plan.md](Perf_Test_Plan.md) isn't built yet.

**Validation.** A full sweep is expensive (~9-12 min); use a targeted script first (pattern used
during the investigation: filter `synth_shapes.dataset()` to the relevant families/tolerances,
call `mask2polymin()` directly, skip `rdp`/irrelevant tolerances — cuts a full sweep down to under a
minute). Only run the full sweep once a candidate value looks good on the targeted check.

---

## 3. `max_deviation`-based split-ranking, under a config flag

**Problem.** `choose_segment_index_for_split` ([fit_to_points_sequence.py:204-220](../mask2polymin/fit_to_points_sequence.py#L204-L220))
ranks eligible segments by mean loss (`loss / points_count`) to decide which one gets the next split
when the segment budget (`max_segments_count`) is tight and several segments are eligible at once.
Mean loss dilutes a single severe local defect across the whole segment's point count, so a long
segment with one severe outlier can rank *below* a segment with diffuse, uniformly-mediocre error —
even though the outlier is the more urgent, more visually obvious defect.

**Design established in discussion** (see conversation — this corrects an earlier draft):
- Rank by `max_deviation` alone, **not** `max(mean_loss, max_deviation)` — the combination was shown
  to mathematically collapse to `max_deviation` in every case that matters (max ≥ mean always, up to
  a small Bessel/count-1 correction that only flips the inequality in a degenerate near-uniform case
  where the distinction is moot anyway). No need to compute or carry the mean-loss term for ranking.
- `needs_split`'s eligibility test is untouched either way — it has a case the mean check catches
  that max deviation alone would miss (every point sitting right at the tolerance boundary can trip
  the mean check via Bessel inflation without any single point exceeding tolerance). Only the
  **ranking among already-eligible segments** changes.
- Compute the array fresh in the ranking function, matching the "plain version" of
  `_max_error_pivot_index` (Option B) — an earlier "opportunistic reuse with `needs_split`" design
  was considered and explicitly rejected: threading an `Optional[np.ndarray]` through three method
  signatures to save a handful of microseconds, bounded to ≤`max_segments_count` times per fit, wasn't
  worth the permanent complexity.

**Requirement:** gate this behind a new `FitterConfig` flag (e.g. `rank_split_by_max_deviation: bool
= False`), defaulting to current (mean-loss) behavior, since this hasn't been benchmarked yet and
carries a known, distinct risk (below).

**Known risk, not yet observed in practice:** ranking by max deviation amplifies the influence of a
single spurious/isolated noisy point on split priority — under a tight budget, a stray artifact could
steal a split slot from a real corner elsewhere. Mean-loss ranking dilutes a lone point's pull by
`1/points_count`; max-deviation ranking doesn't dilute it at all. Worth watching for in the benchmark
specifically, not assuming away.

**Validation.** Full benchmark with the flag on vs. off; also worth constructing (or finding within
Tier 0) a case with several simultaneously-eligible segments under a tight budget, to directly check
whether ranking changes which one wins — the full sweep's aggregate medians may not surface this if
it only changes outcomes in a minority of budget-constrained cells.
