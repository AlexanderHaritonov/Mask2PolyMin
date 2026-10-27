"""FitterConfig.energy: the caller's own Energy, for fit() and refit()."""
import numpy as np
import pytest

from mask2polymin.energy import energy, orphans_count, segment_sse
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


def misplaced_junctions(points, side, shift):
    """the 4 sides of square_contour(side) as fitted segments, every junction `shift` points past its corner"""
    n = len(points)
    moments = SequenceMoments(points)
    segments = []
    for k in range(4):
        first, last = (k * side + shift) % n, ((k + 1) * side + shift - 1) % n
        segments.append(SequenceSegment(points, first, last, fit_range(moments, first, last, with_endpoints=True)))
    return segments


def freeze_shifted(segment, shift):
    """freeze the segment to its current line, moved sideways by `shift`"""
    line = segment.line_segment_params
    segment.freeze(line.start_point + shift * np.array([-line.direction[1], line.direction[0]]), line.direction)


def index_ranges(segments):
    return [(s.first_index, s.last_index) for s in segments]


def sse(segments):
    return sum(segment_sse(segment) for segment in segments)


def built_in_energy(config):
    """the built-in Energy as an own energy"""
    return lambda segments: energy(sse(segments), orphans_count(segments), len(segments), config.tolerance, config.segment_cost)


@pytest.mark.parametrize("frozen_shift", [None, 0.5])
@pytest.mark.parametrize("seed", range(20))
def test_the_built_in_energy_as_own_energy_gives_the_same_result(seed, frozen_shift):
    points = noisy_rectangle_contour(np.random.default_rng(seed), sigma=0.7)
    config = FitterConfig(tolerance=1.0)
    built_in = FitterToPointsSequence(config)
    own = FitterToPointsSequence(FitterConfig(tolerance=1.0, energy=built_in_energy(config)))

    expected = built_in.fit(points, is_closed=True)
    result = own.fit(points, is_closed=True)
    assert index_ranges(result.segments) == index_ranges(expected.segments)
    np.testing.assert_allclose(result.polygon, expected.polygon)

    if frozen_shift is not None:
        for previous in (expected, result):
            for segment in previous.segments[::2]:
                freeze_shifted(segment, frozen_shift)
    expected = built_in.refit(points, expected, is_closed=True)
    result = own.refit(points, result, is_closed=True)
    assert index_ranges(result.segments) == index_ranges(expected.segments)
    np.testing.assert_allclose(result.polygon, expected.polygon)


def test_own_energy_decides_the_splits():
    # a segment costs more than any SSE a split saves
    config = FitterConfig(energy=lambda segments: sse(segments) + 1e6 * len(segments))
    result = FitterToPointsSequence(config).fit(square_contour(10), is_closed=True)
    assert len(result.segments) == 1


def test_result_energy_uses_own_energy_with_the_loss_of_the_current_lines():
    config = FitterConfig()
    result = FitterToPointsSequence(config).fit(square_contour(10), is_closed=True)
    freeze_shifted(result.segments[0], 0.5)  # its loss is stale now
    assert result.energy(FitterConfig(energy=built_in_energy(config))) == pytest.approx(result.energy(config))


@pytest.mark.parametrize("sign", [1, -1])
def test_guard_compares_own_energy(monkeypatch, sign):
    # _fit() returns the 4 sides with every junction 3 points past its corner: a higher SSE than the start.
    # sign -1: an energy that prefers it.
    points = square_contour(10)
    fitted = FitterToPointsSequence().fit(points, is_closed=True)
    fitter = FitterToPointsSequence(FitterConfig(energy=lambda segments: sign * sse(segments)))
    monkeypatch.setattr(fitter, "_fit", lambda segments: misplaced_junctions(points, side=10, shift=3))

    result = fitter.refit(points, fitted, is_closed=True)
    expected = fitted.segments if sign == 1 else misplaced_junctions(points, side=10, shift=3)
    assert index_ranges(result.segments) == index_ranges(expected)
