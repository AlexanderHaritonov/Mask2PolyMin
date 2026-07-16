import numpy as np
from dataclasses import dataclass

from mask2polymin.sequence_moments import subsequence, fit_range, refit_segment, SequenceMoments, DEGENERATE_STRAIGHTNESS
from mask2polymin.line_segment_params import LineSegmentParams
from mask2polymin.polyline import segments_to_polyline
from mask2polymin.sequence_segment import SequenceSegment, points_count

PLOT_SEGMENTS = False

# Hard floor for a line fit; orphaning never squeezes a segment below it.
MIN_SEGMENT_POINTS = 2

# Below this many points a fitted direction is noise-dominated,
# so direction-based shortcuts must not trust it.
MIN_POINTS_FOR_DIRECTION = 5

# cos(~11°): merge candidates whose direction dot product falls below this diverge too much.
COLLINEAR_DIRECTIONS_MIN_DOT = 0.98

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
    # Max points per junction that may be left orphaned (assigned to no segment);
    # orphaning is data-driven: a point is orphaned iff farther than tolerance from both adjacent lines.
    max_orphans_per_junction: int = 2
    verbose: bool = False

    def __post_init__(self):
        if self.max_segments_count < 1:
            raise ValueError(f"max_segments_count must be >= 1, got {self.max_segments_count}")
        if self.max_adjust_iterations < 1:
            raise ValueError(f"max_adjust_iterations must be >= 1, got {self.max_adjust_iterations}")
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {self.tolerance}")
        if self.max_orphans_per_junction < 0:
            raise ValueError(f"max_orphans_per_junction must be >= 0, got {self.max_orphans_per_junction}")

    @property
    def tolerance_sq(self) -> float:
        # squared form used internally for L2 comparisons; derived on access so it
        # can never desync from a later change to tolerance
        return self.tolerance ** 2

class FitterToPointsSequence:
    def __init__(self,
                 points_sequence: np.ndarray,
                 is_closed: bool = False,
                 config: FitterConfig = None):

        # no-op for float64 (N, 2) ndarray input; coerces lists and integer contours once, up front
        points_sequence = np.asarray(points_sequence, dtype=np.float64)
        if points_sequence.ndim != 2 or points_sequence.shape[1] != 2:
            raise ValueError(f"points_sequence must have shape (N, 2), got {points_sequence.shape}")

        # If closed contour and last point equals first point, remove the duplicate
        if is_closed and len(points_sequence) > 0 and np.array_equal(points_sequence[0], points_sequence[-1]):
            self.whole_sequence = points_sequence[:-1]
        else:
            self.whole_sequence = points_sequence
        if len(self.whole_sequence) < 2:
            raise ValueError(f"points_sequence must contain at least 2 points (excluding the closing duplicate), got {len(self.whole_sequence)}")
        self.is_closed = is_closed
        self.config = config or FitterConfig()

        self._moments = SequenceMoments(self.whole_sequence)

    def fit(self) -> tuple[np.ndarray, list[SequenceSegment]]:
        segments_sequence = self._fit()
        segments = self._merge_collinear_segments(segments_sequence)
        # all fits so far skipped the O(n) endpoint-extents pass (start/end points sit at the centroid);
        # one exact refit per final segment fills in the real extents for the polyline and for callers
        for segment in segments:
            refit_segment(self._moments, segment, with_endpoints=True)
        polygon = segments_to_polyline(segments, is_closed=self.is_closed, tolerance=self.config.tolerance)
        return polygon, segments

    def _fit(self) -> list[SequenceSegment]:
        segment_params: LineSegmentParams = fit_range(self._moments, 0, len(self.whole_sequence) - 1, with_endpoints=False)
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
        # in case of merge index is set back to try the merge with neighbour segment
        i = 0 
        while True:
            n = len(segments)
            limit = n if self.is_closed else n - 1
            if i >= limit or n < 2:
                break
            a = segments[i]
            b = segments[(i + 1) % n]
            
            # cheap pre-check: skip pairs whose directions diverge too much;
            # bypassed when either segment is too short for a trustworthy direction estimate
            directions_trustworthy = (a.points_count() >= MIN_POINTS_FOR_DIRECTION
                                      and b.points_count() >= MIN_POINTS_FOR_DIRECTION)
            if directions_trustworthy:
                directions_diverge = (COLLINEAR_DIRECTIONS_MIN_DOT > abs(float(np.dot(
                                        a.line_segment_params.direction, b.line_segment_params.direction))))
                if directions_diverge:
                    i += 1
                    continue
            combined = subsequence(self.whole_sequence, a.first_index, b.last_index)
            combined_fit = fit_range(self._moments, a.first_index, b.last_index, with_endpoints=False)
            both_sides_collinear_on_average = combined_fit.loss / len(combined) <= self.config.tolerance_sq
            no_point_off = combined_fit.squared_distances_to_line(combined).max() <= self.config.tolerance_sq
            if both_sides_collinear_on_average and no_point_off:
                # do the merge
                merged = SequenceSegment(
                    whole_sequence=self.whole_sequence,
                    first_index=a.first_index,
                    last_index=b.last_index,
                    line_segment_params=combined_fit)
                if self.config.verbose:
                    print(f"Merged segments {i} and {(i+1) % n}")
                if self.is_closed and i == n - 1:
                    # the wrap merge consumed the first and last elements; the list is a ring, so append merged at the END and resume just before it
                    # the walk then checks exactly the two pairs involving 'merged'.
                    # All earlier pairs are unchanged and were already checked, so nothing is re-walked.
                    segments = segments[1:-1] + [merged]
                    i = max(0, len(segments) - 2)
                else:
                    segments = segments[:i] + [merged] + segments[i+2:]
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
            points = subsequence(self.whole_sequence, segment.first_index, segment.last_index)
            return segment.line_segment_params.squared_distances_to_line(points).max() > self.config.tolerance_sq

        eligible = [i for i in range(len(segments)) if needs_split(segments[i])]
        if eligible:
            return max(eligible, key=lambda i: segments[i].line_segment_params.loss / segments[i].points_count())
        else:
            return None

    def _lower_mid_index(self, left_index, right_index) -> int:
        """Computes middle index / pivot point, while respecting the possibility for
        right_index < left_index because of closed sequence / circularity."""
        if left_index <= right_index:
            return (left_index + right_index) // 2
        else:
            l = len(self.whole_sequence)
            mid_index = (left_index + right_index + l) // 2
            if mid_index < l:
                return mid_index
            else:
                return mid_index - l

    def _squared_errors_to_core_line(self, core_first, core_last, fallback: LineSegmentParams, points) -> np.ndarray:
        """Squared distances of `points` to the TLS line of a segment's uncontested core [core_first..core_last].
          Scoring against the core — never against a line fitted with the contested points themselves — prevents a junction outlier from masking itself by dragging its own segment's fit.
          Falls back to the segment's current line when the core is too small to define one."""
        if core_first == core_last:  # a single point cannot define a line
            return fallback.squared_distances_to_line(points)
        core_line = fit_range(self._moments, core_first, core_last, with_endpoints=False)
        if core_line.straightness == DEGENERATE_STRAIGHTNESS:
            return fallback.squared_distances_to_line(points)
        return core_line.squared_distances_to_line(points)

    def split_segment(self, segments: list[SequenceSegment], segment_to_split_index: int) -> list[SequenceSegment]:
        segment = segments[segment_to_split_index]
        pivot_point_index = self._lower_mid_index(segment.first_index, segment.last_index)
        seg2_first_index = (pivot_point_index + 1) % len(self.whole_sequence)
        segment1 = SequenceSegment(
            whole_sequence=self.whole_sequence,
            first_index=segment.first_index,
            last_index=pivot_point_index,
            line_segment_params=fit_range(self._moments, segment.first_index, pivot_point_index, with_endpoints=False))
        segment2 = SequenceSegment(
            whole_sequence=self.whole_sequence,
            first_index=seg2_first_index,
            last_index=segment.last_index,
            line_segment_params=fit_range(self._moments, seg2_first_index, segment.last_index, with_endpoints=False))

        # Clone segments before and after the split point
        cloned_before = [seg.clone() for seg in segments[:segment_to_split_index]]
        cloned_after = [seg.clone() for seg in segments[segment_to_split_index+1:]]
        return cloned_before + [segment1, segment2] + cloned_after

    def adjust_segmentation(self, segments: list[SequenceSegment], start_segment_index:int) -> float:
        sse_per_segment = 0

        # The separation search is a pure function of the two segments' index ranges.
        # So if a junction's state is unchanged,that scoring necessarily returned "no move" and would do so again.
        # Remember the state and skip the re-scoring.
        junction_last_scored_state = [None] * len(segments)

        def find_optimal_break_and_adjust(junction_index, previous_segment, next_segment) -> int:
            state = (previous_segment.first_index, previous_segment.last_index,
                     next_segment.first_index, next_segment.last_index)
            if junction_last_scored_state[junction_index] == state:
                return 0
            junction_last_scored_state[junction_index] = state
            new_last, new_first, boundary_shift = self.best_consecutive_segments_separation(previous_segment, next_segment)
            if boundary_shift > 0:
                previous_segment.last_index = new_last
                next_segment.first_index = new_first
                refit_segment(self._moments, previous_segment, with_endpoints=False)
                refit_segment(self._moments, next_segment, with_endpoints=False)

            return boundary_shift

        for iterations_count in range(self.config.max_adjust_iterations):
            # The start segment is typically the 1st segment of the pair resulting from the split.
            # Then we do iterations of following (at most MAX_ITERATIONS)
            # we go in the direction of increasing index and for each segment:
            #   1. find out where the border to the next consecutive segment should be
            #   2. adjust where the previous segment ends and the next segment starts
            #   3. re-fit both segments.

            changes_count = 0

            for i in range(start_segment_index, len(segments) - 1):
                boundary_shift = find_optimal_break_and_adjust(i, segments[i], segments[i+1])
                if boundary_shift > 1:
                    changes_count += 1

            reverse_run_start = start_segment_index - 1
            for i in range(reverse_run_start, -1, -1):
                boundary_shift = find_optimal_break_and_adjust(i, segments[i], segments[i+1])
                if boundary_shift > 1:
                    changes_count += 1

            if self.is_closed:
                boundary_shift = find_optimal_break_and_adjust(len(segments) - 1, segments[-1], segments[0])
                if boundary_shift > 1:
                    changes_count += 1

            # Point-weighted mean of per-segment SSE for evenly sized segments. Plus "penalty":
            # Each orphan is charged one tolerance-sized outlier, spread over the segments, so orphaning is never free in the stop/improvement gates.
            assigned_count = sum(s.points_count() for s in segments)
            orphans_count = len(self.whole_sequence) - assigned_count
            orphans_penalty = orphans_count * self.config.tolerance_sq / len(segments)
            sse_per_segment = (sum(s.line_segment_params.loss * s.points_count() for s in segments) / assigned_count
                               + orphans_penalty)
            # In a "good enough" state improving moves may still be pending: they only
            # become visible against the lines refitted during this pass. Grant one
            # extra pass after the gate is satisfied before returning.
            if sse_per_segment < self.config.tolerance_sq and iterations_count > 0:
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


    def best_consecutive_segments_separation(self, segment1: SequenceSegment, segment2: SequenceSegment) -> tuple[int, int, int]:
        """:returns (last index of segment1, first index of segment2, boundary shift),
        between the two indices 0..config.max_orphans_per_junction points may be left orphaned;
        the shift is how many positions the junction moved (max over its two ends).
        The contested window (that is range of points that can change segment they belong to) is scored against each segment's uncontested core-half line,
        so a junction outlier cannot vouch for itself through its own segment's fit.
        Orphaning a point costs tolerance_sq, so a point is orphaned iff it lies farther than tolerance from both core lines."""

        n = len(self.whole_sequence)
        assert (segment2.first_index - segment1.last_index - 1) % n <= self.config.max_orphans_per_junction
        left_limit = self._lower_mid_index(segment1.first_index, segment1.last_index)
        right_limit = self._lower_mid_index(segment2.first_index, segment2.last_index)
        relevant_points = subsequence(self.whole_sequence, left_limit, right_limit)
        assert relevant_points.shape[0] >= 2

        # current cut in window coordinates: both ends lie inside the window, so plain
        # differences against them give the true shift with no circular-distance handling
        old_i = (segment1.last_index - left_limit) % n
        old_j = (segment2.first_index - left_limit) % n

        # Fine-tuning is only sound when both lines already fit their points well.
        both_fits_within_tolerance = (
            segment1.line_segment_params.loss / segment1.points_count() <= self.config.tolerance_sq
            and segment2.line_segment_params.loss / segment2.points_count() <= self.config.tolerance_sq)
        if both_fits_within_tolerance:
            # Score against the uncontested core halves: a junction outlier cannot vouch for itself through the fit it has dragged.
            # The core's outer end may itself hold an undetected outlier of the neighboring junction, so trim up to max_orphans_per_junction points there.
            trim1 = max(0, min(self.config.max_orphans_per_junction,
                               points_count(n, segment1.first_index, left_limit) - MIN_SEGMENT_POINTS))
            trim2 = max(0, min(self.config.max_orphans_per_junction,
                               points_count(n, right_limit, segment2.last_index) - MIN_SEGMENT_POINTS))
            squared_errors_seg1 = self._squared_errors_to_core_line(
                (segment1.first_index + trim1) % n, left_limit, segment1.line_segment_params, relevant_points)
            squared_errors_seg2 = self._squared_errors_to_core_line(
                right_limit, (segment2.last_index - trim2) % n, segment2.line_segment_params, relevant_points)
        else:
            squared_errors_seg1 = segment1.line_segment_params.squared_distances_to_line(relevant_points)
            squared_errors_seg2 = segment2.line_segment_params.squared_distances_to_line(relevant_points)
        head_cum = squared_errors_seg1.cumsum()              # head_cum[i] = seg1 cost of window points [0..i]
        tail_cum = squared_errors_seg2[::-1].cumsum()[::-1]  # tail_cum[j] = seg2 cost of window points [j..w-1]

        # points outside the contested window always stay with their segment
        retained1_outside = (0 if left_limit == segment1.first_index
                             else points_count(n, segment1.first_index, left_limit) - 1)
        retained2_outside = (0 if right_limit == segment2.last_index
                             else points_count(n, right_limit, segment2.last_index) - 1)

        # refuse to starve either segment below two points, the minimum needed to fit a line
        w = relevant_points.shape[0]
        i_range = np.arange(w - 1)
        remaining1 = retained1_outside + i_range + 1
        remaining2 = retained2_outside + (w - 1 - i_range)
        
        # masking: infinite cost for cuts that squeeze a segment below the fit minimum
        costs = np.where((remaining1 < MIN_SEGMENT_POINTS) | (remaining2 < MIN_SEGMENT_POINTS),
                         np.inf, head_cum[:-1] + tail_cum[1:])
        best_i = int(np.argmin(costs))
        best_cost = costs[best_i]
        if not np.isfinite(best_cost):  # no valid cut: keep the current boundary
            return segment1.last_index, segment2.first_index, 0

        if both_fits_within_tolerance:
            best_i, best_gap = self._best_cut_with_orphans(head_cum, tail_cum, retained1_outside, retained2_outside,
                                                           gapless_cost=best_cost, gapless_i=best_i)
        else:
            best_gap = 0
        new_j = best_i + best_gap + 1
        boundary_shift = max(abs(best_i - old_i), abs(new_j - old_j))
        return (left_limit + best_i) % n, (left_limit + new_j) % n, boundary_shift

    def _best_cut_with_orphans(self, head_cum, tail_cum, retained1_outside, retained2_outside,
                               gapless_cost, gapless_i) -> tuple[int, int]:
        """Search cuts leaving 1..max_orphans_per_junction window points orphaned;
        :returns (cut index i, gap) in window coordinates. Fewer orphans win ties."""
        w = len(head_cum)
        best_cost, best_i, best_gap = gapless_cost, gapless_i, 0
        max_gap = min(self.config.max_orphans_per_junction, w - 2)  # each segment keeps >= 1 window point
        for gap in range(1, max_gap + 1):
            # cut i: seg1 gets window points [0..i], orphans [i+1..i+gap], seg2 the rest
            candidates_count = w - 1 - gap
            costs = head_cum[:candidates_count] + tail_cum[gap + 1:] + gap * self.config.tolerance_sq
            i_range = np.arange(candidates_count)
            remaining1 = retained1_outside + i_range + 1
            remaining2 = retained2_outside + (candidates_count - i_range)

            # masking: infinite cost for cuts that squeeze a segment below the fit minimum
            costs = np.where((remaining1 < MIN_SEGMENT_POINTS) | (remaining2 < MIN_SEGMENT_POINTS), np.inf, costs)
            i = int(np.argmin(costs))
            if costs[i] < best_cost:  # strictly better only: fewer orphans win ties
                best_cost, best_i, best_gap = costs[i], i, gap
        return best_i, best_gap











