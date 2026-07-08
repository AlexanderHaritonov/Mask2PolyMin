import numpy as np
from dataclasses import dataclass

from mask2polymin.fit_line_segment import fit_line_segment
from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.sequence_segment import SequenceSegment, subsequence

PLOT_SEGMENTS = False

@dataclass
class FitterConfig:
    max_segments_count: int = 30
    max_adjust_iterations: int = 20
    # Max allowed deviation in pixels (perpendicular distance to the fitted line).
    # tolerance_sq gates everything:
    #   - split eligibility and collinear merge: a segment's mean and max
    #     squared point distance must stay within it;
    #   - global stop: the average SSE per segment must stay within it,
    #     i.e. each segment gets a budget of ~one tolerance-sized outlier;
    #   - improvement: each split must reduce that SSE by at least one
    #     outlier's worth, else the fitter gives up.
    tolerance: float = 1.0
    verbose: bool = False

    def __post_init__(self):
        # squared form used internally for L2 comparisons
        self.tolerance_sq = self.tolerance ** 2

class FitterToPointsSequence:
    def __init__(self,
                 points_sequence: np.ndarray,
                 is_closed: bool = False,
                 config: FitterConfig = None):

        # If closed contour and last point equals first point, remove the duplicate
        if is_closed and np.array_equal(points_sequence[0], points_sequence[-1]):
            self.whole_sequence = points_sequence[:-1]
        else:
            self.whole_sequence = points_sequence
        self.is_closed = is_closed
        self.config = config or FitterConfig()

    def fit(self) -> list[SequenceSegment]:
        segments_sequence = self._fit()
        return self._merge_collinear_segments(segments_sequence)

    def _fit(self) -> list[SequenceSegment]:
        segment_params: LineSegmentParams = fit_line_segment(self.whole_sequence)
        initial_segment = SequenceSegment(
            whole_sequence=self.whole_sequence,
            first_index=0,
            last_index=len(self.whole_sequence)-1,
            line_segment_params=segment_params)
        segments = [initial_segment]

        sse_per_segment = segment_params.loss  # single segment: total SSE == SSE per segment
        if sse_per_segment < self.config.tolerance_sq:
            return segments

        while True:
            if len(segments) >= self.config.max_segments_count:
                if self.config.verbose: print("Max segments count reached. Exiting.")
                return segments

            index_of_segment_to_split = self.choose_segment_index_for_split(segments)
            if index_of_segment_to_split is None:
                if self.config.verbose: print("No segments to split. Breaking up at", len(segments), "segments.")
                return segments
            if self.config.verbose:
                print(f"splitting at {index_of_segment_to_split}")

            segmentation_after_split = self.split_segment(segments, index_of_segment_to_split)

            new_sse_per_segment = self.adjust_segmentation(segmentation_after_split, index_of_segment_to_split)
            if self.config.verbose: print(f"sse per segment: {new_sse_per_segment}")
            if self.config.verbose and PLOT_SEGMENTS:
                from mask2polymin.plotting import plot_segments
                plot_segments(segmentation_after_split)

            if new_sse_per_segment < self.config.tolerance_sq:
                if self.config.verbose: print(f"Breaking up at {len(segmentation_after_split)} because sse per segment is within tolerance.")
                return segmentation_after_split

            if new_sse_per_segment > sse_per_segment - self.config.tolerance_sq:
                if self.config.verbose: print("Breaking up because of no improvement at", len(segments), "segments.")
                return segments

            segments = segmentation_after_split
            sse_per_segment = new_sse_per_segment

    def _merge_collinear_segments(self, segments: list[SequenceSegment]) -> list[SequenceSegment]:
        # single-pass index walk,
        # in case of merge index is set back to try the merge with neighbout segment
        i = 0 
        while True:
            n = len(segments)
            limit = n if self.is_closed else n - 1
            if i >= limit or n < 2:
                break
            a = segments[i]
            b = segments[(i + 1) % n]
            # cheap pre-check: skip pairs whose directions diverge by more than ~11°
            if abs(float(np.dot(a.line_segment_params.direction, b.line_segment_params.direction))) < 0.98:
                i += 1
                continue
            combined = subsequence(self.whole_sequence, a.first_index, b.last_index)
            combined_fit = fit_line_segment(combined)
            if combined_fit.loss / len(combined) <= self.config.tolerance_sq:
                # do the merge
                merged = SequenceSegment(
                    whole_sequence=self.whole_sequence,
                    first_index=a.first_index,
                    last_index=b.last_index,
                    line_segment_params=combined_fit)
                if self.is_closed and i == n - 1:
                    segments = [merged] + segments[1:-1]
                else:
                    segments = segments[:i] + [merged] + segments[i+2:]
                if self.config.verbose:
                    print(f"Merged segments {i} and {(i+1) % n}")
                i = max(0, i - 1)
            else:
                i += 1
        return segments

    def choose_segment_index_for_split(self, segments: list[SequenceSegment]) -> int | None:
        def needs_split(segment: SequenceSegment) -> bool:
            if segment.points_count() <= 3:
                return False
            if segment.line_segment_params.loss / segment.points_count() > self.config.tolerance_sq:
                return True
            # The mean-based check alone lets a long segment absorb a few far-off
            # points around a corner, so also split when any single point is
            # farther than tolerance from the fitted line.
            points = self.subsequence(segment.first_index, segment.last_index)
            return segment.line_segment_params.squared_distances_to_line(points).max() > self.config.tolerance_sq

        eligible = [i for i in range(len(segments)) if needs_split(segments[i])]
        if eligible:
            return max(eligible, key=lambda i: segments[i].line_segment_params.loss / segments[i].points_count())
        else:
            return None

    ''' computes middle index / pivot point, while respecting the possibility for right_index<left_index because of closed sequence / circularity '''
    def lower_mid_index(self, left_index, right_index) -> int:
        if left_index <= right_index:
            return (left_index + right_index) // 2
        else:
            l = len(self.whole_sequence)
            mid_index = (left_index + right_index + l) // 2
            if mid_index < l:
                return mid_index
            else:
                return mid_index - l

    def subsequence(self, left, right) -> np.ndarray:
        return subsequence(self.whole_sequence, left, right)

    def points_count(self, first_index, last_index) -> int:
        if last_index > first_index:
            return last_index - first_index + 1
        else:
            return len(self.whole_sequence) - first_index + last_index + 1 # for closed polygon / circular case

    def split_segment(self, segments: list[SequenceSegment], segment_to_split_index: int) -> list[SequenceSegment]:
        segment = segments[segment_to_split_index]
        if segment.first_index < segment.last_index:
            pivot_point_index = (segment.first_index + segment.last_index) // 2
            part1 = self.whole_sequence[segment.first_index : pivot_point_index + 1]
            part2 = self.whole_sequence[pivot_point_index + 1 : segment.last_index + 1]
        else: # can happen if is_closed
            pivot_point_index = (segment.first_index + segment.last_index + len(self.whole_sequence)) // 2
            if pivot_point_index < len(self.whole_sequence):
                part1 = self.whole_sequence[segment.first_index : pivot_point_index + 1]
                part2 = np.vstack([self.whole_sequence[pivot_point_index + 1 :], self.whole_sequence[: segment.last_index + 1]])
            else:
                pivot_point_index -= len(self.whole_sequence)
                part1 = np.vstack([self.whole_sequence[segment.first_index:], self.whole_sequence[:pivot_point_index + 1]])
                part2 = self.whole_sequence[pivot_point_index + 1 : segment.last_index + 1]

        part1_fit: LineSegmentParams = fit_line_segment(part1)
        segment1 = SequenceSegment(
            whole_sequence= self.whole_sequence,
            first_index= segment.first_index,
            last_index= pivot_point_index,
            line_segment_params=part1_fit)
        part2_fit: LineSegmentParams = fit_line_segment(part2)
        seg2_first_index = pivot_point_index + 1
        if seg2_first_index >= len(self.whole_sequence):
            seg2_first_index -= len(self.whole_sequence)
        segment2 = SequenceSegment(
            whole_sequence= self.whole_sequence,
            first_index= seg2_first_index,
            last_index= segment.last_index,
            line_segment_params=part2_fit)

        # Clone segments before and after the split point
        cloned_before = [seg.clone() for seg in segments[:segment_to_split_index]]
        cloned_after = [seg.clone() for seg in segments[segment_to_split_index+1:]]
        return cloned_before + [segment1, segment2] + cloned_after

    def adjust_segmentation(self, segments: list[SequenceSegment], start_segment_index:int) -> float:
        sse_per_segment = 0
        for iterations_count in range(self.config.max_adjust_iterations):
            # The start segment is typically the 1st segment of the pair resulting from the split.
            # Then we do iterations of following (at most MAX_ITERATIONS)
            # we go in the direction of increasing index and for each segment:
            #   1. find out where the border to the next consecutive segment should be
            #   2. adjust where the previous segment ends and the next segment starts
            #   3. re-fit the current segment.

            def find_optimal_break_and_adjust(previous_segment, next_segment)->int:
                optimal_last_index = self.best_consecutive_segments_separation(previous_segment, next_segment)
                if optimal_last_index > previous_segment.first_index:
                    count_of_points_changing_segment = abs(optimal_last_index - previous_segment.last_index)
                else: # this might happen with closed polygons
                    old_points_count = previous_segment.points_count()
                    new_points_count = self.points_count(previous_segment.first_index, optimal_last_index)
                    count_of_points_changing_segment = abs(new_points_count - old_points_count)

                if count_of_points_changing_segment > 0:
                    previous_segment.last_index = optimal_last_index
                    next_segment.first_index = (optimal_last_index + 1) % len(self.whole_sequence)

                return count_of_points_changing_segment

            changes_count = 0

            start_segment_dirty = False
            for i in range(start_segment_index, len(segments) - 1):
                count_of_points_changing_segment = find_optimal_break_and_adjust(segments[i], segments[i+1])
                if count_of_points_changing_segment > 1:
                    changes_count += 1
                if count_of_points_changing_segment > 0:
                    if i == start_segment_index:
                        start_segment_dirty = True
                    segments[i+1].refit()
            if start_segment_dirty:
                segments[start_segment_index].refit()

            reverse_run_start = start_segment_index - 1 if start_segment_index > 0 else len(segments) - 2
            segment0_dirty = False
            for i in range(reverse_run_start, -1, -1):
                count_of_points_changing_segment = find_optimal_break_and_adjust(segments[i], segments[i+1])
                if count_of_points_changing_segment > 1:
                    changes_count += 1
                if count_of_points_changing_segment > 0:
                    if i == 0:
                        segment0_dirty = True
                    segments[i+1].refit()
            if segment0_dirty:
                segments[0].refit()

            if self.is_closed:
                count_of_points_changing_segment = find_optimal_break_and_adjust(segments[-1], segments[0])
                if count_of_points_changing_segment > 1:
                    changes_count += 1
                if count_of_points_changing_segment > 0:
                    segments[-1].refit()
                    segments[0].refit()

            # Point-weighted mean of per-segment SSE (loss is a sum of squared
            # distances): ~ SSE_total / len(segments) for evenly sized segments.
            # Scales with sampling density: on denser contours the early-stop
            # and no-improvement gates in _fit fire later, so splitting runs
            # closer to what the per-point eligibility criterion demands.
            sse_per_segment = sum(s.line_segment_params.loss * s.points_count() for s in segments) / len(self.whole_sequence)
            if sse_per_segment < self.config.tolerance_sq:
                if self.config.verbose:
                    print(f"at {len(segments)} segments, {iterations_count} iterations sse per segment within tolerance. Breaking up")
                return sse_per_segment

            if changes_count == 0:
                if self.config.verbose:
                    print(f"at {len(segments)} segments, {iterations_count} iterations were no changes. Breaking up")
                return sse_per_segment

        if self.config.verbose:
            print(f"at {len(segments)} segments, adjust_segmentation reached {self.config.max_adjust_iterations} iterations.")
        return sse_per_segment


    ''' :returns index where segment1 should end'''
    def best_consecutive_segments_separation(self, segment1: SequenceSegment, segment2: SequenceSegment) -> int:
        assert (segment1.last_index == segment2.first_index - 1) or (segment2.first_index==0 and segment1.last_index == len(self.whole_sequence) - 1)
        left_limit = self.lower_mid_index(segment1.first_index, segment1.last_index)
        right_limit = self.lower_mid_index(segment2.first_index, segment2.last_index)
        relevant_points = self.subsequence(left_limit, right_limit)
        assert relevant_points is not None and relevant_points.shape[0] >= 2
        squared_errors_seg1 = segment1.line_segment_params.squared_distances_to_line(relevant_points)
        squared_errors_seg2 = segment2.line_segment_params.squared_distances_to_line(relevant_points)
        errors_cumsum1 = squared_errors_seg1.cumsum()
        errors_cumsum2 = squared_errors_seg2[::-1].cumsum()
        compound_error_sums = errors_cumsum1[:-1] + errors_cumsum2[-2::-1]
        optimal_last_index = np.argmin(compound_error_sums) # counting from left_limit as 0
        return (int(optimal_last_index) + left_limit) % len(self.whole_sequence)











