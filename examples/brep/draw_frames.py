"""
Draws the video's frames, from 01 on, into frames/: runs stitch_faces() with a fitter that draws

1. each step of fit() and refit(), via their on_step observer, and each result;
2. per round of the outer loop: the shared line, then the frozen segment at the start of each face's refit();
3. the final polygons.

"""
from functools import partial
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np
import matplotlib.pyplot as plt

from mask2polymin import FitterToPointsSequence as Fitter, FitResult, LineSegmentParams, SequenceSegment
from stitch_faces import FACES, load_contours, stitch_faces

FACE_COLORS = {"left": "tab:red", "right": "tab:blue"}  # as in generation/extract_contours.py
SEGMENT_COLORS = ("tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan")  # no red or blue: the faces have them
FROZEN_COLOR = "black"
ORPHAN_COLOR = "white"  # contour points that belong to no segment
BACKGROUND = (0.7, 0.7, 0.7)  # the gray of frame 00
FRAMES_DIR = HERE / "frames"
FIRST_FRAME = "00_contours.png"  # from generation/extract_contours.py
VIEW_MARGIN = 3  # px around the contours

STEP_TITLES = {
    "start": "fit() starts with one segment",
    "split": "split segment {index}",
    "junctions": "junctions moved",
    "split undone": "split undone",
    "merge": "merged into segment {index}",
}


class DrawingFitter(Fitter):
    """Draws a frame for each step of fit() and refit(), and one for each result.
    Each refit() of the first face starts a round of the outer loop: it first draws the shared line, which both faces' frozen segments are on by then."""

    def __init__(self):
        super().__init__()
        self.contours = load_contours()
        self.results = {}  # each face's latest result
        self.colored = {}  # each face's segments in its latest frame, as (point indices, color)
        self.round = 0  # 0 during fit()
        self.frames_count = 0

    def fit(self, points_sequence, is_closed=False):
        face = self.face_of(points_sequence)
        result = super().fit(points_sequence, is_closed, on_step=partial(self.draw_step, face))
        self.draw_result(face, result, "fit()")
        return result

    def refit(self, points_sequence, previous, is_closed=False):
        face = self.face_of(points_sequence)
        if face == FACES[0]:
            self.round += 1
            self.draw_shared_line()
        result = super().refit(points_sequence, previous, is_closed, on_step=partial(self.draw_step, face))
        self.draw_result(face, result, "refit()")
        return result

    def draw_step(self, face, kind, segments, index):
        """on_step of fit() and refit(). The start of refit() is the segmentation with the newly frozen segment."""
        if kind == "start" and self.round > 0:
            frozen_index = next(i for i, segment in enumerate(segments) if segment.frozen)
            title = f"Round {self.round}: freezing segment {frozen_index} on the {face} face"
        else:
            title = f"{self.prefix(face)}: {STEP_TITLES[kind].format(index=index)}"
        fig, ax = new_frame(title, self.contours)
        self.draw_other_face(ax, face)
        draw_segments(ax, self.contours[face], segments, self.segment_colors(face, segments), with_lines=True)
        self.save(fig, f"{face}_{kind.replace(' ', '_')}")

    def draw_result(self, face, result: FitResult, method: str):
        self.results[face] = result
        fig, ax = new_frame(f"{self.prefix(face)}: the polygon from {method}", self.contours)
        self.draw_other_face(ax, face)
        draw_segments(ax, self.contours[face], result.segments, self.segment_colors(face, result.segments), with_lines=False)
        draw_polygon(ax, face, result.polygon)
        self.save(fig, f"{face}_result")

    def draw_shared_line(self):
        """Both faces with their latest polygons, and the shared line through the points of their frozen segments."""
        fig, ax = new_frame(f"Round {self.round}: one line through both faces' points", self.contours)
        draw_stitched(ax, self.contours, self.results)
        self.save(fig, "shared_line")

    def draw_final(self, results: dict[str, FitResult]):
        fig, ax = new_frame(f"Final polygons, after {self.round} rounds", self.contours)
        draw_stitched(ax, self.contours, results)
        self.save(fig, "final")

    def draw_other_face(self, ax, face):
        """The other face as it stands: its contour points, and its latest polygon once it has one."""
        other = next(f for f in FACES if f != face)
        result = self.results.get(other)
        draw_face_points(ax, other, self.contours[other], alpha=0.4)
        if result is not None:
            draw_polygon(ax, other, result.polygon)

    def segment_colors(self, face, segments: list[SequenceSegment]) -> list[str]:
        """Colors that stay with the points: each segment keeps the color of the segment in the face's latest frame it shares the most points with,
        e.g. the larger half of a split. The others get a color no segment of the face has."""
        # before the face's first frame, all its points have the face's color, as in frame 00
        previous = self.colored.get(face, [(set(range(len(self.contours[face]))), FACE_COLORS[face])])
        indices = [point_indices(segment) for segment in segments]
        overlaps = sorted(((len(new & old), i, color) for i, new in enumerate(indices) for old, color in previous), reverse=True)
        colors = [None] * len(segments)
        for overlap, i, color in overlaps:
            if overlap > 0 and colors[i] is None and color not in colors:
                colors[i] = color
        for i in range(len(colors)):
            if colors[i] is None:
                colors[i] = next(color for color in SEGMENT_COLORS if color not in colors)
        self.colored[face] = list(zip(indices, colors))
        return colors

    def prefix(self, face) -> str:
        return f"{face.capitalize()} face" if self.round == 0 else f"Round {self.round}, {face} face"

    def face_of(self, points_sequence) -> str:
        return next(face for face in FACES if np.array_equal(self.contours[face], points_sequence))

    def save(self, fig, name):
        self.frames_count += 1
        fig.savefig(FRAMES_DIR / f"{self.frames_count:02d}_{name}.png", dpi=100)  # 600x600, as frame 00
        plt.close(fig)


def draw_frames():
    remove_old_frames()
    fitter = DrawingFitter()
    left, right = stitch_faces(fitter)
    fitter.draw_final({"left": left, "right": right})
    print(f"Saved {fitter.frames_count} frames to {FRAMES_DIR}")


def remove_old_frames() -> None:
    """The frames of an earlier run, which may have had more of them. Frame 00 stays."""
    FRAMES_DIR.mkdir(exist_ok=True)
    for path in FRAMES_DIR.glob("*.png"):
        if path.name != FIRST_FRAME:
            path.unlink()


def new_frame(title: str, contours: dict):
    """A frame laid out like frame 00: title, margins, view; the gray as background, since there are no masks."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor(BACKGROUND)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal')
    set_view(ax, contours)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.02)
    return fig, ax


def draw_face_points(ax, face, contour: np.ndarray, alpha=1.0) -> None:
    ax.plot(contour[:, 1], contour[:, 0], 'o', color=FACE_COLORS[face], markersize=6, alpha=alpha)


def draw_polygon(ax, face, polygon: np.ndarray) -> None:
    ax.plot(polygon[:, 1], polygon[:, 0], '-o', color=FACE_COLORS[face], linewidth=2, markersize=4, zorder=3)


def draw_segments(ax, contour: np.ndarray, segments: list[SequenceSegment], colors: list[str], with_lines: bool) -> None:
    """Each segment's points in its color; the frozen segment's line in black, under them.
    with_lines: each free segment's line too, in its color.
    The lines have no start and end points during the steps."""
    ax.plot(contour[:, 1], contour[:, 0], 'o', color=ORPHAN_COLOR, markersize=6)
    for segment, color in zip(segments, colors):
        points = segment.points()
        ax.plot(points[:, 1], points[:, 0], 'o', color=color, markersize=6)
        if segment.frozen:
            draw_line(ax, segment.line_segment_params, points, color=FROZEN_COLOR, linewidth=5, zorder=1)
        elif with_lines:
            draw_line(ax, segment.line_segment_params, points, color=color, linewidth=2)


def draw_stitched(ax, contours: dict, results: dict[str, FitResult]) -> None:
    """Both faces with their polygons, and the shared line in black, along the points of both frozen segments."""
    frozen = [segment for face in FACES for segment in results[face].segments if segment.frozen]
    for face in FACES:
        draw_face_points(ax, face, contours[face])
        draw_polygon(ax, face, results[face].polygon)
    points = np.concatenate([segment.points() for segment in frozen])
    draw_line(ax, frozen[0].line_segment_params, points, color=FROZEN_COLOR, linewidth=5, zorder=1)


def point_indices(segment: SequenceSegment) -> set[int]:
    """The indices of the segment's points in the contour; a range may wrap past the end."""
    count = len(segment.whole_sequence)
    return {(segment.first_index + k) % count for k in range(segment.points_count())}


def draw_line(ax, line: LineSegmentParams, points: np.ndarray, **style) -> None:
    """The line, between the projections of the outermost of the points onto it."""
    along = (points - line.start_point) @ line.direction
    ends = line.start_point + np.outer([along.min(), along.max()], line.direction)
    ax.plot(ends[:, 1], ends[:, 0], '-', **style)


def set_view(ax, contours: dict) -> None:
    """Zoom to the contours, VIEW_MARGIN px around them."""
    points = np.concatenate(list(contours.values()))
    row_min, col_min = points.min(axis=0) - VIEW_MARGIN
    row_max, col_max = points.max(axis=0) + VIEW_MARGIN
    ax.set_xlim(col_min, col_max)
    ax.set_ylim(row_max, row_min)


if __name__ == "__main__":
    draw_frames()
