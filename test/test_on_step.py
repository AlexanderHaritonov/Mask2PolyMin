"""on_step of fit() and refit(): reports each step of the fit."""
import numpy as np
import pytest

from mask2polymin import FitResult
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig
from mask2polymin.sequence_moments import SequenceMoments, fit_range
from mask2polymin.sequence_segment import SequenceSegment


def square_contour(side):
    """closed 1-px-step square contour; corner k sits at index k * side"""
    pts = []
    for x in range(0, side): pts.append([x, 0])
    for y in range(0, side): pts.append([side, y])
    for x in range(side, 0, -1): pts.append([x, side])
    for y in range(side, 0, -1): pts.append([0, y])
    return np.array(pts, dtype=np.float64)


def noisy_rectangle_contour(rng, sigma):
    """closed 1-px-step rectangle contour of random size and rotation, with gaussian jitter"""
    width, height = int(rng.integers(8, 25)), int(rng.integers(8, 25))
    pts = []
    for x in range(0, width): pts.append([x, 0])
    for y in range(0, height): pts.append([width, y])
    for x in range(width, 0, -1): pts.append([x, height])
    for y in range(height, 0, -1): pts.append([0, y])
    pts = np.array(pts, dtype=np.float64)
    angle = rng.uniform(0, np.pi / 2)
    c, s = np.cos(angle), np.sin(angle)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    return pts + rng.normal(0, sigma, pts.shape)


def segments_from_ranges(points, ranges):
    moments = SequenceMoments(points)
    return [SequenceSegment(points, first, last, fit_range(moments, first, last, with_endpoints=True)) for first, last in ranges]


def index_ranges(segments):
    return [(s.first_index, s.last_index) for s in segments]


def recorder(steps):
    """on_step that records each step as (kind, index ranges of the segments, index)"""
    def on_step(kind, segments, index):
        steps.append((kind, index_ranges(segments), index))
    return on_step


def freeze_every_other_shifted(result, shift):
    """freeze every other segment to its line, moved sideways by shift"""
    for segment in result.segments[::2]:
        params = segment.line_segment_params
        normal = np.array([-params.direction[1], params.direction[0]])
        segment.freeze(params.start_point + shift * normal, params.direction)


def assert_each_change_is_reported(steps, n):
    """Each step changes the index ranges of the step before it as its kind says: no change goes unreported."""
    before_split = None
    for (_, previous, _), (kind, ranges, index) in zip(steps, steps[1:]):
        if kind == "split":
            before_split = previous
            first, second = ranges[index], ranges[index + 1]
            assert ranges[:index] + ranges[index + 2:] == previous[:index] + previous[index + 1:]
            assert (first[0], second[1]) == previous[index]
            assert (first[1] + 1) % n == second[0]
        elif kind == "junctions":
            assert len(ranges) == len(previous) and ranges != previous
        elif kind == "split undone":
            assert ranges == before_split
        elif kind == "merge":
            in_place = previous[:index] + [(previous[index][0], previous[index + 1][1])] + previous[index + 2:]
            around = previous[1:-1] + [(previous[-1][0], previous[0][1])]  # the last and the first segment of a closed contour
            assert ranges == in_place or (ranges == around and index == len(ranges) - 1)
        else:
            pytest.fail(f"unexpected kind {kind!r}")


@pytest.mark.parametrize("is_closed", [True, False])
@pytest.mark.parametrize("seed", range(20))
def test_fit_reports_each_step(seed, is_closed):
    # Over these seeds: every kind, merges included.
    points = noisy_rectangle_contour(np.random.default_rng(seed), sigma=0.7)
    steps = []
    result = FitterToPointsSequence().fit(points, is_closed=is_closed, on_step=recorder(steps))

    assert steps[0] == ("start", [(0, len(points) - 1)], None)
    assert_each_change_is_reported(steps, len(points))
    assert steps[-1][1] == index_ranges(result.segments)


@pytest.mark.parametrize("is_closed", [True, False])
@pytest.mark.parametrize("seed", range(20))
def test_refit_reports_each_step(seed, is_closed):
    # Without the guard, the last step is the result.
    points = noisy_rectangle_contour(np.random.default_rng(seed), sigma=0.7)
    fitter = FitterToPointsSequence()
    previous = fitter.fit(points, is_closed=is_closed)
    freeze_every_other_shifted(previous, 0.5)
    steps = []
    result = fitter.refit(points, previous, is_closed=is_closed, guard=False, on_step=recorder(steps))

    assert steps[0] == ("start", index_ranges(previous.segments), None)
    assert_each_change_is_reported(steps, len(points))
    assert steps[-1][1] == index_ranges(result.segments)


def test_split_undone_reports_the_segmentation_before_the_split():
    # With so high a segment cost, no split lowers the Energy.
    points = square_contour(10)
    steps = []
    result = FitterToPointsSequence(FitterConfig(segment_cost=1000)).fit(points, is_closed=True, on_step=recorder(steps))

    assert [kind for kind, _, _ in steps if kind != "junctions"] == ["start", "split", "split undone"]
    assert steps[-1][1] == [(0, 39)] == index_ranges(result.segments)


@pytest.mark.parametrize("ranges, merged, index", [
    ([(0, 4), (5, 9), (10, 19), (20, 29), (30, 39)], [(0, 9), (10, 19), (20, 29), (30, 39)], 0),   # side 0 cut in two
    ([(0, 9), (10, 19), (20, 29), (30, 34), (35, 39)], [(0, 9), (10, 19), (20, 29), (30, 39)], 3),  # side 3 cut in two
    ([(5, 9), (10, 19), (20, 29), (30, 39), (0, 4)], [(10, 19), (20, 29), (30, 39), (0, 9)], 3),   # the last and the first segment
])
def test_merge_reports_the_merged_segment(ranges, merged, index):
    # The junctions sit on the corners, so they don't move.
    points = square_contour(10)
    steps = []
    FitterToPointsSequence().refit(points, FitResult(np.empty((0, 2)), segments_from_ranges(points, ranges)), is_closed=True,
                                   on_step=recorder(steps))

    assert steps == [("start", ranges, None), ("merge", merged, index)]


@pytest.mark.parametrize("is_closed", [True, False])
@pytest.mark.parametrize("seed", range(10))
def test_on_step_does_not_change_the_result(seed, is_closed):
    points = noisy_rectangle_contour(np.random.default_rng(seed), sigma=0.7)
    fitter = FitterToPointsSequence()
    on_step = lambda kind, segments, index: None

    fitted = fitter.fit(points, is_closed=is_closed)
    fitted_observed = fitter.fit(points, is_closed=is_closed, on_step=on_step)
    assert index_ranges(fitted_observed.segments) == index_ranges(fitted.segments)
    assert np.array_equal(fitted_observed.polygon, fitted.polygon)

    freeze_every_other_shifted(fitted, 0.5)
    refitted = fitter.refit(points, fitted, is_closed=is_closed)
    refitted_observed = fitter.refit(points, fitted, is_closed=is_closed, on_step=on_step)
    assert index_ranges(refitted_observed.segments) == index_ranges(refitted.segments)
    assert np.array_equal(refitted_observed.polygon, refitted.polygon)
