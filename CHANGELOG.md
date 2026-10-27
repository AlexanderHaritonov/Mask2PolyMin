# Changelog

All notable changes to this project are documented here.

## [0.3.0] - 2026-10-28

### Changed
- **Breaking:** `FitterConfig.apply_local_defect_margin` removed.
- **Breaking:** `fit()` returns a `FitResult`, which unpacks as `polygon, segments, energy`. Change `polygon, segments = fitter.fit(...)` to `polygon, segments, energy = fitter.fit(...)`.
- `fit()` results change: a split, merge or junction move is kept only if it lowers the Energy (see `segment_cost` in the README).
  On the benchmark (1950 contours): corner recall 0.522 → 0.551, Hausdorff 4.57 → 4.30, RMS 1.35 → 1.30, wall time 63 → 12 ms per contour.
- `loss` of a fitted line is the plain sum of squared distances; it was scaled by n/(n−1).
- **Breaking:** `LineSegmentParams`: `loss` and `straightness` are keyword-only, and `loss` defaults to `0.0`.

### Added
- `FitterConfig.segment_cost` (default `2`): the cost of one segment in the Energy, in units of `tolerance²`.
- `FitterConfig.energy`: your own Energy, `energy(segments) -> float`, instead of the built-in one. `mask2polymin.energy` has `segment_sse(segment)` and `orphans_count(segments)` to build it.
- `refit()`: fit again, starting from a previous `FitResult`, with segments pinned to given lines. See the README.
- `on_step` keyword of `fit()` and `refit()`: an observer, called after each step of the fit, e.g. to draw it.
- `FitResult`, returned by `fit()` and `refit()`: unpacks as `polygon, segments, energy`, and has `sse()` and `orphans_count()`.
- `SequenceSegment.frozen` and `SequenceSegment.freeze(point, direction, key=None)`. `refit()` keeps the key, and `FitResult.frozen[key]` finds the frozen segment in its result.
- `SequenceSegment.points()`: the segment's points. `fit_line(points)`: the line through any points, fitted same way the fitter fits a segment.
- `examples/brep/stitch_faces.py`: stitches two neighboring faces to share an edge. Its demo GIF is in the README.

## [0.2.0] - 2026-07-26

### Changed
- **Breaking:** `points_sequence` and `is_closed` moved from `FitterToPointsSequence.__init__` to `fit()`. Construct the fitter with just a `FitterConfig`, then call `fitter.fit(points_sequence, is_closed=...)`.

### Added
- Video walkthrough of `examples/example_house/example_house.py`, linked from the README.

### Docs
- README polish: quick-start `pip install` snippet, section reordering, sized image embeds for the walkthrough screenshots.

## [0.1.1] - 2026-07-24
- README/badge polish: added PyPI badge and title heading, tightened the `tolerance` rule-of-thumb wording, shrunk the logo to badge scale.

## [0.1.0] - 2026-07-23
- Initial PyPI release.