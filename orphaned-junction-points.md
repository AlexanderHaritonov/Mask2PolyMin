# Orphaned Junction Points ("Option 3")

## Motivation

A junction point — the last point of one segment / first point of the next — has
maximum leverage on a total-least-squares line fit (it rotates the fitted direction),
and it's precisely the point whose segment membership is most ambiguous: it sits on
the corner arc, not cleanly on either line. When contour extraction misassigns it by
even one step (~1–1.4 px on a 1-px marching-squares contour), that single point drags
the fitted line's direction and, downstream, the reconstructed vertex — degrading
corner quality out of proportion to the size of the error.

# Plan:

Fix corner degradation caused by a single misaligned point at segment junctions by
letting such points belong to **no** segment. This replaces the hard invariant
`prev.last_index + 1 == next.first_index (mod N)` with a relaxed one:

```
next.first_index == prev.last_index + 1 + gap   (mod N),   0 <= gap <= MAX_ORPHANS_PER_JUNCTION
```

Points inside a gap are "orphaned": they contribute to no fit, no loss, no gate.
They model the truth — a point on the corner arc lies on neither line.

## Design constants

| Constant | Value | Rationale |
|---|---|---|
| `MAX_ORPHANS_PER_JUNCTION` | `2` | Cap on gap size. On 1-px binary contours only 1 point per junction ever qualifies, so 2 is free headroom; the penalty rule keeps orphaning data-driven. Named module constant, not a config knob. |
| orphan penalty | `config.tolerance_sq` per orphaned point | Orphaning saves `min(d₁², d₂²)` from the objective and costs one penalty ⇒ a point is orphaned **iff** it lies farther than `tolerance` from *both* lines. Same semantics as the split gate; no new knob. |
| `MIN_SEGMENT_POINTS` | `2` | Hard floor for a line fit. The separation search never accepts an orphaning candidate that squeezes either neighbor below it. |

Both constants go at module level in `mask2polymin/fit_to_points_sequence.py`.

---

## Step 1 — Safe prep (correct & harmless while gap==0 everywhere)

### 1a. `sse_per_segment` denominator — `fit_to_points_sequence.py:267`

Current: `sum(loss_s · n_s) / len(whole_sequence)` — point-weights `n_s/N` sum to 1
only when segments partition the contour. With orphans they'd sum to < 1, silently
loosening the global and improvement gates.

Fix: divide by the **assigned** point count:

```python
assigned_count = sum(s.points_count() for s in segments)
sse_per_segment = sum(s.line_segment_params.loss * s.points_count() for s in segments) / assigned_count
```

Orphaned points' errors deliberately vanish from this objective (they're on no line —
that's their definition); the per-orphan penalty inside the separation search is what
prevents "orphan everything" from winning.

The other `sse_per_segment` at `_fit` line 56 is the initial single full-coverage
segment — no orphans possible, unchanged.

### 1b. Merge-gate max-distance check — `fit_to_points_sequence.py:108`

`_merge_collinear_segments` builds the merged segment as
`subsequence(a.first_index, b.last_index)`, which (once gaps exist) silently
**re-adopts the orphans in the gap between a and b**. That is deliberate and correct —
merging asserts the corner between them was spurious, so its orphans belong on the
common line — but the gate must then be able to veto: a true corner orphan sitting
~2 px off the combined line must block the merge even when the *mean* survives.
Mirror `needs_split`:

```python
if (combined_fit.loss / len(combined) <= self.config.tolerance_sq
        and combined_fit.squared_distances_to_line(combined).max() <= self.config.tolerance_sq):
```

Without this, a merge can reconstruct exactly the corner-straddling segment the rest
of the machinery just eliminated. (Outer gaps — before `a`, after `b` — are untouched.)

---

## Step 2 — Rewrite `best_consecutive_segments_separation` (`fit_to_points_sequence.py:283-296`)

### New contract

```python
def best_consecutive_segments_separation(self, segment1, segment2) -> tuple[int, int]:
    """:returns (last index of segment1, first index of segment2); between them
       0..MAX_ORPHANS_PER_JUNCTION points may be left orphaned."""
```

Entry assert relaxes from strict adjacency to bounded gap:

```python
gap = (segment2.first_index - segment1.last_index - 1) % n
assert gap <= MAX_ORPHANS_PER_JUNCTION
```

Note the contested window `subsequence(left_limit, right_limit)` (mid of seg1 → mid
of seg2) already spans any existing gap contiguously — **previously orphaned points
are re-contested on every adjust pass, so orphaning is reversible** (a point exactly
on a refitted line costs 0 < penalty and gets re-adopted).

### Generalized cumsum search

Window coordinates: `W` points, cut `i` means seg1 gets window points `[0..i]`,
orphans get `[i+1..i+g]`, seg2 gets `[i+g+1..W-1]`.

```python
cum1 = squared_errors_seg1.cumsum()                    # cum1[i]  = seg1 cost of [0..i]
tail2 = squared_errors_seg2[::-1].cumsum()[::-1]       # tail2[j] = seg2 cost of [j..W-1]

for g in 0..MAX_ORPHANS_PER_JUNCTION:
    # i ranges 0..W-2-g  (seg2 keeps at least one window point); row empty → stop
    costs_g = cum1[: W-1-g] + tail2[g+1 :] + g * tolerance_sq
```

Shapes check: `i ∈ [0, W-2-g]` is `W-1-g` candidates; `cum1[:W-1-g]` ✓,
`tail2[i+g+1]` for those `i` is `tail2[g+1:]`, also length `W-1-g` ✓.
`g=0` reproduces today's exact computation (`cum1[:-1] + tail2[1:]`), and since
`W >= 2` is already asserted, the `g=0` row is never empty ⇒ the argmin always exists.

**Orphaning semantics check** (why penalty = tolerance_sq is exactly the split-gate rule):
candidate `(i, g=1)` beats `(i, g=0)` iff `err2[i+1] > penalty`, and beats
`(i+1, g=0)` iff `err1[i+1] > penalty` ⇒ point `i+1` is orphaned iff
`min(err1, err2) > tolerance_sq`, i.e. farther than `tolerance` from both lines.

**Tie-breaking:** iterate `g` ascending and accept only strictly better cost ⇒ fewer
orphans preferred on ties; within a row `np.argmin` keeps today's leftmost-cut choice.

### Min-segment guard (only needed for g ≥ 1 rows; g=0 keeps today's behavior)

Points outside the window always stay with their segment:

```python
retained1_outside = points strictly before left_limit in seg1
retained2_outside = points strictly after right_limit in seg2
remaining1(i)    = retained1_outside + i + 1
remaining2(i, g) = retained2_outside + (W - 1 - i - g)
```

For `g >= 1`, mask candidates where either remaining count `< MIN_SEGMENT_POINTS`
(set cost to `inf`; skip the row if fully masked). This guarantees orphaning never
squeezes a segment below the fit minimum — a segment loses points at *both* its
junctions, up to `2·G` total, so this guard is required.

⚠ Degenerate-trap: `points_count(a, a)` returns `N+1` (the else-branch), so compute
`retained*_outside` with an explicit `if limit == end: 0 else: points_count(...) - 1`
guard rather than trusting `points_count` on a possibly-equal pair.

Return:

```python
new_last1  = (left_limit + best_i) % n
new_first2 = (left_limit + best_i + best_g + 1) % n
return new_last1, new_first2
```

Cost: O(window × (G+1)) instead of O(window); three tiny vectorized rows.

---

## Step 3 — Adapt `find_optimal_break_and_adjust` (`fit_to_points_sequence.py:212-225`)

Consume the pair; assign both sides; replace the wrap-around special case with a
circular distance:

```python
new_last, new_first = self.best_consecutive_segments_separation(previous_segment, next_segment)
boundary_shift = max(circ_dist(previous_segment.last_index, new_last),
                     circ_dist(next_segment.first_index, new_first))
if boundary_shift > 0:
    previous_segment.last_index = new_last
    next_segment.first_index = new_first
return boundary_shift
```

where `circ_dist(a, b) = min(|a-b|, N - |a-b|)`.

Why `max` of the two shifts: the old return value ("count of points changing
segment") fed two thresholds in `adjust_segmentation` — `> 0` ⇒ apply + refit,
`> 1` ⇒ counts toward `changes_count` (the ±1-oscillation damper). For gap-0 → gap-0
transitions both boundary shifts equal the old point-transfer count, so `max`
**exactly reproduces today's values** in today's states; for orphaning transitions it
generalizes per-boundary. The old `if optimal_last_index > previous_segment.first_index`
wrap-around branch (lines 214-219) dies entirely — `circ_dist` covers it.

The three call sites in `adjust_segmentation` (forward pass, reverse pass, closed
wrap) keep their structure; only the variable name changes. Refit rules unchanged.

---

## Step 4 — Orphan-aware corner anchor (`mask2polymin/polyline.py:52-74`)

When the gap is non-empty, the orphaned points *are* the corner arc — their mean is a
better anchor than the midpoint-of-projections construction:

```python
n = len(seg_a.whole_sequence)
gap = (seg_b.first_index - seg_a.last_index - 1) % n
if gap > 0:
    idx = [(seg_a.last_index + 1 + k) % n for k in range(gap)]
    anchor = seg_a.whole_sequence[idx].astype(np.float64).mean(axis=0)
else:
    anchor = 0.5 * (...)   # current formula, unchanged
```

Gap of zero keeps the current formula bit-for-bit. `max_offset` plausibility radius
stays — the anchor only gets closer to the true corner. Update the two docstrings
("anchored at the input point where the two segments meet").

Note for open polylines: endpoints (`polyline.py:47-48`) touch only the outermost
indices, which the separation search never orphans — no change needed.

---

## Step 5 — Tests

### Update: `test/test_consecutive_segments_separation.py`
Return type changes `int` → `(int, int)`. Unpack and assert on `last1`; expected
values are unchanged because none of these fixtures contains a point farther than
tolerance from both lines (hand-verified against the new cost rows):

| test | window errs (seg1 / seg2) | expected result |
|---|---|---|
| horizontal_then_vertical | `[0,0,0,1,4]` / `[4,1,0,0,0]` | `(2, 3)`, gap 0 |
| …with_step | `[0,0,0,1,4]` / `[9,4,1,0,0]` | `(3, 4)`, gap 0 |
| diagonal_then_horizontal | `[0,0,0,.5,2]` / `[4,1,0,0,0]` | `(2, 3)`, gap 0 |
| noisy | sub-tolerance noise | `last1 ∈ (2,3,4)`, gap 0 |
| closed_polygon_square | `[0,0,1,4]` / `[1,0,0,0]` | `(1, 2)`, gap 0 |

### Update: `test/test_consecutive_segments_separation_circular.py`
Same unpacking; assert both ints and `gap <= MAX_ORPHANS_PER_JUNCTION`. The
`build_segments` adjacency asserts remain valid (test setup constructs gapless pairs).

### Unchanged: `test/test_split_segment.py`
Splits are still gapless — `split_segment` is untouched. All asserts stay valid.

### New: `test/test_orphaned_junction_points.py`

1. **Unit — separation orphans a corner-straddling point.**
   seg1 = y=0 through x∈0..3 (idx 0-3), corner point `[4.5, 1.5]` (idx 4, distance
   1.5 > tolerance from both lines), seg2 = x=6, `[6,3]..[6,6]` (idx 5-8).
   Hand-computed costs: best g=0 candidate 2.25, best g=1 candidate **1.0**
   (cut i=2), best g=2 candidate 2.0 ⇒ expect `(3, 5)` — point 4 orphaned.
2. **Unit — sub-tolerance corner point is NOT orphaned** (e.g. the marching-squares
   half-pixel chamfer at ~0.5 px): expect gap 0.
3. **Unit — min-segment guard**: a tiny segment (2-3 points) next to a long one with
   a hostile outlier; assert no candidate leaves a neighbor with < 2 points.
4. **End-to-end — square with displaced corner points.** Closed contour of a
   6×6 square sampled at 1-px steps, each true corner point displaced diagonally
   outward by ~1.5 px (> tolerance from both adjacent lines). Assert:
   4 segments; every inter-segment gap ≤ 2; each fitted line matches its side
   (direction and offset within tight eps); each polyline vertex within ~0.3 px
   of the true corner. This is the degradation scenario the whole change targets —
   on current `main` the displaced point rotates a fitted line and drags the vertex.
5. **End-to-end — orphan re-adoption / no orphans on clean input.** Clean axis-aligned
   square (no displaced points): fit yields 4 segments and **zero** gaps (all
   corner points ≤ tolerance from a line ⇒ never orphaned / re-adopted).

### Watch item: `test/test_fit_closed_polygon.py` (tolerance = 0.001)
Penalty = 1e-6 ⇒ orphaning is near-free during intermediate iterations with skewed
fits. Expected to still pass (asserts only `len(segments) == 4`; orphans get
re-adopted once refitted lines are exact since cost 0 < penalty) — but this is the
test most likely to surface a surprise. If it flakes, the diagnosis order is:
re-contest window not spanning the gap, or `changes_count` damping stopping one
iteration too early.

---

## Step 6 — Verification

1. `pytest` — full suite.
2. `python -m performance_test.baselines` — smoke: closed-polyline contract asserts,
   IoU sanity, and eyeball the segment counts / metrics vs. current main (record the
   before/after table in the review notes).
3. `python examples/simple_bitmap_example.py` outputs still render (orphaned points
   show as uncolored contour dots in plots — expected, and a useful visualization).
4. README: check whether the algorithm description mentions the partition invariant;
   update if so (one paragraph on orphaned junction points).

## Step 7 — Unmasked separation: score against core-half fits

The orphaning rule ("farther than tolerance from *both* lines") is evaluated against
lines fitted **with the contested point included** — circular. A displaced junction
point has maximum leverage on its own segment's TLS fit: rotating the line about the
centroid costs the interior points little and buys the outlier much, so the fit tilts
until it covers the point (masking). Displaced-corner probe: line dragged ~8.7°, the
corner ends 0.36 px from its own line vs. 1.06 px from the honest one — never orphaned;
only corners that happen to belong to *neither* neighbor's fit get orphaned.

Fix: in `best_consecutive_segments_separation`, compute `squared_errors_seg1/2` against
lines fitted on each segment's **uncontested core half** — `[first_index..left_limit]`
for seg1, `[right_limit..last_index]` for seg2 — which never contain the contested
window points. The corner then faces the honest line, gets orphaned, and the refit
snaps the segment back to true.

Performance: naively this adds two `fit_line_segment` calls per separation call
(~80–160 per contour) ⇒ measured ≈ 2× total `fit()` time (+47…124%), dominated by
numpy per-call overhead (`np.cov` + `eigh`). Optimization: points are static, so
precompute prefix sums of the moments (Σx, Σy, Σx², Σy², Σxy) once in `__init__`;
the TLS centroid, direction (closed-form 2×2 eigenvector) and loss of any contiguous
range are then O(1) — scoring needs no start/end extremes. Overhead drops to a few
percent, and `refit()` / the merge gate can reuse the same machinery for a net speedup.
Center coordinates by the global centroid at init for numerical stability.

### As built

- Core scoring is active only in fine-tuning mode (the trust gate: both segments'
  mean loss within tolerance). Against coarse intermediate fits — segments still
  spanning corners — half-cores misrepresent the segment and derail the split
  trajectory (octagon test), so coarse mode keeps whole-fit gapless scoring.
- The core's **outer** end may hold the neighboring junction's undetected outlier
  (which poisons a small core worse than a whole fit), so up to
  `max_orphans_per_junction` points are trimmed there while the core keeps
  ≥ MIN_SEGMENT_POINTS; too-small cores fall back to the whole-segment fit.
- `adjust_segmentation` grants one extra pass after the sse gate is satisfied:
  fine-tuning moves only become visible against lines refitted during the pass
  in which the gate first fires.
- Measured: displaced-corner square — all 4 corners orphaned, axis-aligned sides,
  exact vertices (strict test promoted from xfail). Noisy circle tol 0.71:
  IoU 0.9828→0.9875, Hausdorff 1.04→1.09, 19→18 segments; cost ≈ 2× adjust work
  (~2× refits — fine mode keeps finding real improvements) ⇒ 41→128 ms; straight-edge
  contours +2…40%. Follow-up: step 8.

## Step 8 — O(1) moment-based refit

Motivation: `refit()` → `fit_line_segment()` is O(n) per call (slice, mean, `np.cov`,
`eigh`, min/max projections) and dominates runtime — ~850 calls ≈ 50% of `fit()` time
on the noisy-circle benchmark, doubled by step 7's extra fine-tuning passes.

Everything the adjust loop needs about a segment is O(1) from the step-7 prefix
moments: count and range sums (two prefix-row subtractions), centroid, the 2×2
covariance from second moments, direction via the closed-form principal axis
θ = ½·atan2(2·cov_xy, cov_xx − cov_yy), and loss from
λ_min = (cov_xx+cov_yy)/2 − √(((cov_xx−cov_yy)/2)² + cov_xy²).
The only O(n) part of `fit_line_segment` — the extreme projections that become
`start_point`/`end_point` — is needed nowhere in the adjust loop, only at the end
(polyline corners, `_length`). So: refits during adjustment become O(1) moment
lookups; one final exact fit per segment fills in the endpoints.

⚠ Loss convention: `fit_line_segment` computes `loss = N·λ_min` from `np.cov`, whose
default is **ddof=1** (sample covariance, divide by N−1) — so its "loss" is actually
SSE·N/(N−1), not the true sum of squared distances (1.5× for a 3-point segment).
All tolerance gates are calibrated against that convention, so the moment-based loss
must reproduce it — `loss = m²/(m−1)·λ_min_population` — to keep gate decisions
bit-compatible. (Fixing the convention to honest SSE everywhere is a separate,
deliberate change.)

Expected gain: reclaims most of step 7's 2× on noisy inputs and likely beats the
pre-orphan baseline outright; the merge gate's `fit_line_segment(combined)` can
reuse the same machinery.

### Unifying the two fitting paths

Since step 7 there are two TLS implementations: `fit_line_segment(points)`
(O(n): `np.cov` + `eigh`, loss, start/end extremes) and the moment-based
`_range_line_fit` (O(1): centroid + direction only, used for core-line scoring).
They agree on the direction by construction (the eigenvector is invariant to the
N vs N−1 covariance scaling), but the duplicated math can drift. Step 8 ends with
one authoritative path:

- `fit_range(first, last) -> LineSegmentParams` on the fitter — moment-backed,
  ddof-compatible loss, endpoints computed on demand — used by everything that
  works on index ranges (refit, core lines, merge's combined fit);
  `_range_line_fit` folds into it.
- `fit_line_segment(points)` stays as the utility for raw point arrays
  (initial whole-contour fit, external callers, tests).
- both share one closed-form `principal_axis(cov_xx, cov_yy, cov_xy)` helper in
  `fit_line_segment.py`, so the eigen-math exists exactly once.
- cross-check test: random ranges, assert both paths agree on direction and loss.

## Sequencing for review

| Step | Contents | Risk |
|---|---|---|
| 1 | sse denominator + merge max-gate (behavior-neutral today) | none |
| 2+3 | separation rewrite + caller (the real change; atomic, must land together) | the change |
| 4 | corner anchor from orphan mean | low |
| 5 | test updates + new tests | — |
| 6 | full verification + README | — |
| 7 | core-half fits in separation + O(1) prefix-moment range fits | separate change |
| 8 | O(1) moment-based refit; endpoints via one final exact fit | perf only, gates must stay bit-compatible |

