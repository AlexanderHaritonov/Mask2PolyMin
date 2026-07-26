# Changelog

All notable changes to this project are documented here.

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