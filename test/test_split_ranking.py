import numpy as np
from mask2polymin.sequence_segment import SequenceSegment
from fit_line_segment_reference import fit_line_segment
from mask2polymin.fit_to_points_sequence import FitterToPointsSequence, FitterConfig


def make_segment(seq, first, last):
    if last > first:
        params = fit_line_segment(seq[first:last + 1])
    else:
        params = fit_line_segment(seq[first:] + seq[:last + 1])
    return SequenceSegment(seq, first, last, params)


def test_rank_split_by_max_deviation_flips_which_segment_wins():
    # Segment A (indices 0-9): diffuse error -- alternating +-1.3 perpendicular offset on
    # every point. High mean loss (~3.6/pt), modest max single-point deviation (~5.35).
    n_a = 10
    xs_a = np.arange(n_a, dtype=float)
    seq_a = np.stack([xs_a, xs_a], axis=1)
    for i in range(n_a):
        d = 1.3 if i % 2 == 0 else -1.3
        seq_a[i] += [-d, d]

    # Segment B (indices 10-39): one severe outlier among 29 points sitting exactly on the
    # line. Diluted mean loss (~1.07/pt, well under A's), but a huge max single-point
    # deviation (~29.9) -- an unresolved corner-like feature that mean-loss ranking hides.
    n_b = 30
    xs_b = np.arange(n_b, dtype=float)
    seq_b = np.stack([xs_b, xs_b], axis=1).astype(float)
    seq_b[n_b // 2] += [-4.0, 4.0]

    whole_sequence = np.vstack([seq_a, seq_b])
    segment_a = make_segment(whole_sequence, 0, n_a - 1)
    segment_b = make_segment(whole_sequence, n_a, n_a + n_b - 1)
    segments = [segment_a, segment_b]

    # Sanity-check the construction itself, independent of which fitter config is used.
    mean_loss_a = segment_a.line_segment_params.loss / segment_a.points_count()
    mean_loss_b = segment_b.line_segment_params.loss / segment_b.points_count()
    max_dev_a = segment_a.line_segment_params.squared_distances_to_line(seq_a).max()
    max_dev_b = segment_b.line_segment_params.squared_distances_to_line(seq_b).max()
    assert mean_loss_a > mean_loss_b
    assert max_dev_b > max_dev_a

    default_fitter = FitterToPointsSequence()
    default_fitter._prepare(whole_sequence)
    assert default_fitter._choose_segment_index_for_split(segments) == 0  # mean loss: A wins

    max_dev_fitter = FitterToPointsSequence(config=FitterConfig(rank_split_by_max_deviation=True))
    max_dev_fitter._prepare(whole_sequence)
    assert max_dev_fitter._choose_segment_index_for_split(segments) == 1  # max deviation: B wins
