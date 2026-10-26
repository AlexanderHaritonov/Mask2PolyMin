"""
Draws the video's frames, from 01 on, into frames/: runs stitch_faces() with a fitter that draws

1. each step of fit() and refit(), via their on_step observer, and each result;
   a junctions step also point by point: one frame per point that changes segments;
   a split also in steps: the old segment blinks, a cut appears, the split off half takes its color and blinks;
2. per round of the outer loop: the shared line, then the frozen segment at the start of each face's refit();
3. the final polygons.

Below the title, every frame shows the Energy of each face as the frame shows it, and their sum.
"""
from functools import partial
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np
from PIL import Image
import matplotlib.colors
import matplotlib.pyplot as plt

from mask2polymin import FitterToPointsSequence as Fitter, FitResult, LineSegmentParams, SequenceSegment, fit_line
from mask2polymin.energy import energy
from stitch_faces import FACES, load_contours, stitch_faces

FACE_COLORS = {"left": "tab:red", "right": "tab:blue"}  # as in generation/extract_contours.py
SEGMENT_COLORS = ("tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan")  # no red or blue: the faces have them
SHARED_LINE_COLOR = "black"  # only in the matching edge's frames
FROZEN_RING_COLOR = "white"  # rings around the points of the segment being frozen, only in its freezing frame
ORPHAN_COLOR = "white"  # contour points that belong to no segment
CUT_COLOR = "red"  # the cut across the contour at a split place, during the split's animation
CUT_LENGTH = 2.5  # px
BLINKS = 2  # off and on again, per blink
MASKS_DIR = HERE / "generation"  # left_mask.bmp and right_mask.bmp: the background of every frame, as in frame 00
FRAMES_DIR = HERE / "frames"
FIRST_FRAME = "00_contours.png"  # from generation/extract_contours.py
VIEW_MARGIN = 3  # px around the contours

STEP_TITLES = {  # split and merge name the segments by their colors: see draw_step()
    "start": "start with one segment",
    "junctions": "junctions moved",
    "split undone": "split undone",
}


class DrawingFitter(Fitter):
    """Draws a frame for each step of fit() and refit(), and one for each result.
    Each refit() of the first face starts a round of the outer loop: it first draws the shared line, which both faces' frozen segments are on by then."""

    def __init__(self):
        super().__init__()
        self.contours = {face: without_closing_duplicate(contour) for face, contour in load_contours().items()}  # point indices refer to these
        self.background = masks_image()
        self.results = {}  # each face's latest result
        self.colored = {}  # each face's segments in its latest frame, as (point indices, color): the face stays so in the other face's frames
        self.energies = {}  # each face's Energy in its latest frame
        self.new_colors = iter(SEGMENT_COLORS)  # each new segment takes the next one, so no two segments share a color, across both faces
        self.round = 0  # 0 during fit()
        self.frames_count = 0
        self.animation_frames_count = 0

    def fit(self, points_sequence, is_closed=False):
        face = self.face_of(points_sequence)
        result = super().fit(points_sequence, is_closed, on_step=partial(self.draw_step, face))
        self.draw_result(face, result)
        return result

    def refit(self, points_sequence, previous, is_closed=False):
        face = self.face_of(points_sequence)
        if face == FACES[0]:
            self.round += 1
            self.draw_polygons(f"Round {self.round}", self.results, "round")
            self.draw_shared_line()
        result = super().refit(points_sequence, previous, is_closed, on_step=partial(self.draw_step, face))
        self.draw_result(face, result)
        return result

    def draw_step(self, face, kind, segments, index):
        """on_step of fit() and refit(). The start of refit() is the segmentation with the newly frozen segment."""
        previous = self.colored.get(face, [])
        previous_colors = {color for _, color in previous}
        colors = self.segment_colors(face, segments)
        frozen_points = None
        if kind == "start" and self.round > 0:
            frozen = next(i for i, segment in enumerate(segments) if segment.frozen)
            step = f"freezing the {color_name(colors[frozen])} segment"
            frozen_points = segments[frozen].points()
        elif kind == "split":  # the half with a color from the previous frame is the old segment
            old, new = sorted(colors[index:index + 2], key=lambda color: color not in previous_colors)
            step = f"{color_name(new)} split off the {color_name(old)} segment"
        elif kind == "merge":
            step = f"merged into the {color_name(colors[index])} segment"
        else:
            step = STEP_TITLES[kind]
        title = f"{face.capitalize()}: {step}"
        if kind == "split":
            self.draw_splitting(face, previous, segments, colors, index, old, title)
        elif kind == "junctions":
            self.draw_moves(face, previous, segments, colors)
        lines = [(segment.line_segment_params, segment.points(), color) for segment, color in zip(segments, colors)]
        self.energies[face] = self.face_energy(face, self.colored[face], lines)
        fig, ax = self.new_frame(title)
        self.draw_other_face(ax, face)
        draw_points(ax, self.contours[face], self.colored[face])
        if frozen_points is not None:
            draw_rings(ax, frozen_points)
        for line, points, color in lines:  # the lines have no start and end points during the steps
            draw_line(ax, line, points, color=color, linewidth=2)
        self.save(fig, f"{face}_{kind.replace(' ', '_')}")

    def draw_moves(self, face, previous: list[tuple[set[int], str]], segments: list[SequenceSegment], colors: list[str]):
        """A junctions step point by point: one frame per point that changes segments, except the last, which the step's own frame shows.
        The point takes its new segment's color, or white as an orphan, and each line is fitted to its segment's points again; a frozen segment keeps its line.
        The face's Energy stays at the step's start until the step's own frame: the moves are a reconstruction, not the fitter's path, and on the way it may rise."""
        contour = self.contours[face]
        frozen_lines = {color: segment.line_segment_params for segment, color in zip(segments, colors) if segment.frozen}
        moves = junction_moves(owner_colors(previous), owner_colors(self.colored[face]), len(contour))
        for move, owners in enumerate(moves[:-1], start=1):
            colored = [({i for i, owner in owners.items() if owner == color}, color) for color in dict.fromkeys(owners.values())]
            lines = []
            for indices, color in colored:
                if len(indices) >= 2:
                    points = contour[sorted(indices)]
                    lines.append((frozen_lines[color] if color in frozen_lines else fit_line(points), points, color))
            self.draw_animation_frame(face, f"{face.capitalize()}: moving junctions", colored, lines, "moving", move)

    def draw_splitting(self, face, previous: list[tuple[set[int], str]], segments: list[SequenceSegment], colors: list[str], index, old, title):
        """A split in steps, before the split's own frame: the old segment's points blink, a cut appears at the split place,
        the split off half takes its new color and its own line, and blinks. The split's own frame is without the cut."""
        first, second = segments[index], segments[index + 1]
        old_points = np.concatenate([first.points(), second.points()])
        others = [(segment.line_segment_params, segment.points(), color) for k, (segment, color) in enumerate(zip(segments, colors)) if k not in (index, index + 1)]
        lines_before = others + [(fit_line(old_points), old_points, old)]
        lines_after = others + [(first.line_segment_params, first.points(), colors[index]), (second.line_segment_params, second.points(), colors[index + 1])]
        split_off = point_indices(second if colors[index] == old else first)
        cut = cut_line(first.points()[-1], second.points()[0])  # after the pivot, the point farthest from the old line
        splitting = f"{face.capitalize()}: splitting the {color_name(old)} segment"
        frames = ([(splitting, previous, lines_before, hidden, None) for hidden in blinking(point_indices(first) | point_indices(second))]
                  + [(splitting, previous, lines_before, set(), cut)]
                  + [(title, self.colored[face], lines_after, hidden, cut) for hidden in [set()] + blinking(split_off)])
        for number, (frame_title, colored, lines, hidden, frame_cut) in enumerate(frames, start=1):
            energies = self.energies | {face: self.face_energy(face, colored, lines)}  # the old segmentation's, then the split's
            self.draw_animation_frame(face, frame_title, colored, lines, "splitting", number, hidden, frame_cut, energies)

    def draw_animation_frame(self, face, title, colored: list[tuple[set[int], str]], lines: list[tuple[LineSegmentParams, np.ndarray, str]],
                             name, number, hidden: set[int] = frozenset(), cut: np.ndarray | None = None, energies: dict[str, float] | None = None):
        """A frame of the animation before a step's frame: the face's points but the hidden ones, its lines as (line, points, color), and the cut, if any.
        energies: as in new_frame()."""
        fig, ax = self.new_frame(title, energies)
        self.draw_other_face(ax, face)
        draw_points(ax, self.contours[face], colored, hidden)
        for line, points, color in lines:
            draw_line(ax, line, points, color=color, linewidth=2)
        if cut is not None:
            ax.plot(cut[:, 1], cut[:, 0], '-', color=CUT_COLOR, linewidth=5, zorder=4)
        self.save(fig, f"{face}_{name}", number)

    def draw_result(self, face, result: FitResult):
        self.results[face] = result
        self.energies[face] = result.energy(self.config.tolerance, self.config.segment_cost)
        self.segment_colors(face, result.segments)
        fig, ax = self.new_frame(f"{face.capitalize()}: resulting polygon")
        self.draw_other_face(ax, face)
        draw_points(ax, self.contours[face], self.colored[face])
        draw_polygon(ax, face, result.polygon)
        self.save(fig, f"{face}_result")

    def draw_shared_line(self):
        """Both faces with their latest polygons, and the shared line through the points of their frozen segments."""
        fig, ax = self.new_frame(f"Round {self.round}: matching edge")
        for face in FACES:
            draw_points(ax, self.contours[face], self.colored[face])
            draw_polygon(ax, face, self.results[face].polygon)
        frozen = [segment for face in FACES for segment in self.results[face].segments if segment.frozen]
        points = np.concatenate([segment.points() for segment in frozen])
        draw_line(ax, frozen[0].line_segment_params, points, color=SHARED_LINE_COLOR, linewidth=5, zorder=1)
        self.save(fig, "shared_line")

    def draw_polygons(self, title: str, results: dict[str, FitResult], name: str):
        """Only the polygons' lines, thicker: no points, no vertices. At the start of each round, and at the end."""
        fig, ax = self.new_frame(title)
        for face in FACES:
            draw_polygon(ax, face, results[face].polygon, with_vertices=False, linewidth=4)
        self.save(fig, name)

    def draw_other_face(self, ax, face):
        """The other face's points as in its latest frame, and its latest polygon once it has one.
        Before its first frame, its points in the face's color, as in frame 00."""
        other = next(f for f in FACES if f != face)
        if other in self.colored:
            draw_points(ax, self.contours[other], self.colored[other])
        else:
            draw_face_points(ax, other, self.contours[other])
        if other in self.results:
            draw_polygon(ax, other, self.results[other].polygon)

    def segment_colors(self, face, segments: list[SequenceSegment]) -> list[str]:
        """Colors that stay with the points: each segment keeps the color of the segment in the face's latest frame it shares the most points with,
        e.g. the larger half of a split. The others get a new color, which no segment of either face had before.
        Kept in self.colored[face]."""
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
                colors[i] = next(self.new_colors)
        self.colored[face] = list(zip(indices, colors))
        return colors

    def face_energy(self, face, colored: list[tuple[set[int], str]], lines: list[tuple[LineSegmentParams, np.ndarray, str]]) -> float:
        """As FitResult.energy(): the squared distances of each line's points to it, the face's orphans, its segments in colored."""
        sse = sum(float(line.squared_distances_to_line(points).sum()) for line, points, _ in lines)
        orphans_count = len(self.contours[face]) - sum(len(indices) for indices, _ in colored)
        return energy(sse, orphans_count, len(colored), self.config.tolerance, self.config.segment_cost)

    def face_of(self, points_sequence) -> str:
        return next(face for face in FACES if np.array_equal(self.contours[face], without_closing_duplicate(points_sequence)))

    def new_frame(self, title: str, energies: dict[str, float] | None = None):
        """A frame like frame 00: the masks as background, title, margins, view.
        Below the title, the faces' energies: by default as in their latest frames."""
        fig, ax = plt.subplots(figsize=(6, 6))
        height, width = self.background.shape[:2]
        ax.imshow(self.background, extent=(-0.5, width - 0.5, height - 0.5, -0.5))  # pixel centers at integer coordinates, as find_contours assumes
        set_view(ax, self.contours)
        fig.suptitle(title, fontsize=12, fontweight='bold')
        energies = self.energies if energies is None else energies
        fig.text(0.5, 0.935, energy_line(energies), ha='center', va='center', fontsize=10)  # between the title and the view, which stays where frame 00 has it
        ax.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.02)
        return fig, ax

    def save(self, fig, name, number=None):
        """Frame NN_name.png. The animation before a step's frame NN comes as NN_01_name.png, NN_02_name.png, ...: a digit sorts before a letter."""
        if number is None:
            self.frames_count += 1
            stem = f"{self.frames_count:02d}_{name}"
        else:
            self.animation_frames_count += 1
            stem = f"{self.frames_count + 1:02d}_{number:02d}_{name}"
        fig.savefig(FRAMES_DIR / f"{stem}.png", dpi=100)  # 600x600, as frame 00
        plt.close(fig)


def draw_frames():
    remove_old_frames()
    fitter = DrawingFitter()
    left, right = stitch_faces(fitter)
    fitter.draw_polygons("Final polygons", {"left": left, "right": right}, "final")
    print(f"Saved {fitter.frames_count} frames and {fitter.animation_frames_count} animation frames to {FRAMES_DIR}")


def remove_old_frames() -> None:
    """The frames of an earlier run, which may have had more of them. Frame 00 stays."""
    FRAMES_DIR.mkdir(exist_ok=True)
    for path in FRAMES_DIR.glob("*.png"):
        if path.name != FIRST_FRAME:
            path.unlink()


def without_closing_duplicate(contour: np.ndarray) -> np.ndarray:
    """The contour without its last point if that repeats the first, as the fitter takes a closed contour."""
    return contour[:-1] if np.array_equal(contour[0], contour[-1]) else contour


def masks_image() -> np.ndarray:
    """Each face's mask tinted in its face's color, blended where they overlap, on gray: as in frame 00."""
    masks = {face: load_mask(MASKS_DIR / f"{face}_mask.bmp") for face in FACES}
    coverage = sum(masks[face].astype(float) for face in FACES)
    tints = sum(masks[face][..., None] * light(FACE_COLORS[face]) for face in FACES)
    image = np.full(coverage.shape + (3,), 0.7)  # gray background, as in example_house
    covered = coverage > 0
    image[covered] = tints[covered] / coverage[covered][:, None]
    return image


def load_mask(path) -> np.ndarray:
    """Load mask as a boolean (H, W) array."""
    return np.asarray(Image.open(path).convert("L")) > 0


def light(color) -> np.ndarray:
    """The color blended halfway with white."""
    return (np.array(matplotlib.colors.to_rgb(color)) + 1) / 2


def draw_face_points(ax, face, contour: np.ndarray) -> None:
    ax.plot(contour[:, 1], contour[:, 0], 'o', color=FACE_COLORS[face], markersize=6)


def draw_polygon(ax, face, polygon: np.ndarray, with_vertices=True, linewidth=2) -> None:
    ax.plot(polygon[:, 1], polygon[:, 0], '-o' if with_vertices else '-', color=FACE_COLORS[face], linewidth=linewidth, markersize=4, zorder=3)


def draw_points(ax, contour: np.ndarray, colored: list[tuple[set[int], str]], hidden: set[int] = frozenset()) -> None:
    """A face's points: each segment's in its color; the points of no segment white. The hidden points not at all: they blink."""
    shown = [i for i in range(len(contour)) if i not in hidden]
    ax.plot(contour[shown, 1], contour[shown, 0], 'o', color=ORPHAN_COLOR, markersize=6)
    for indices, color in colored:
        points = contour[sorted(indices - hidden)]
        ax.plot(points[:, 1], points[:, 0], 'o', color=color, markersize=6)


def draw_rings(ax, points: np.ndarray) -> None:
    ax.plot(points[:, 1], points[:, 0], 'o', markerfacecolor='none', markeredgecolor=FROZEN_RING_COLOR, markersize=11, markeredgewidth=1.5)


def color_name(color: str) -> str:
    return color.removeprefix("tab:")


def energy_line(energies: dict[str, float]) -> str:
    """E_Left, E_Right and their sum; a dash for a face before its first frame, and for the sum then."""
    def value(e):
        return "–" if e is None else f"{e:.2f}"
    total = sum(energies.values()) if len(energies) == len(FACES) else None
    parts = [rf"$E_\mathrm{{{face.capitalize()}}}$: {value(energies.get(face))}" for face in FACES] + [rf"$\Sigma E$: {value(total)}"]
    return "      ".join(parts)


def blinking(indices: set[int]) -> list[set[int]]:
    """The hidden points of a blink's frames: off, on, off, on, ..."""
    return [indices, set()] * BLINKS


def cut_line(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """A short line across the contour, between its neighboring points p and q: its two ends, (2, 2)."""
    along = (q - p) / np.linalg.norm(q - p)
    across = np.array([-along[1], along[0]])
    return (p + q) / 2 + np.outer([-CUT_LENGTH / 2, CUT_LENGTH / 2], across)


def point_indices(segment: SequenceSegment) -> set[int]:
    """The indices of the segment's points in the contour; a range may wrap past the end."""
    count = len(segment.whole_sequence)
    return {(segment.first_index + k) % count for k in range(segment.points_count())}


def owner_colors(colored: list[tuple[set[int], str]]) -> dict[int, str]:
    """Each point's segment, by its color. Orphans are left out."""
    return {i: color for indices, color in colored for i in indices}


def junction_moves(old: dict[int, str], new: dict[int, str], count: int) -> list[dict[int, str]]:
    """From old to new owner_colors(), one point at a time: the owners after each move.
    The next point to move is one beside its new segment, or else at the end of its old one, preferably beside the point moved last:
    so each segment stays in one piece, and the junctions move one at a time, point by point."""
    owners = dict(old)
    remaining = [i for i in range(count) if old.get(i) != new.get(i)]
    last = None

    def rank(i):
        neighbors = (owners.get((i - 1) % count), owners.get((i + 1) % count))
        beside_new = new.get(i) in neighbors
        beside_last = last is not None and (i - last) % count in (1, count - 1)
        at_end = any(neighbor != owners.get(i) for neighbor in neighbors)
        return not beside_new, not beside_last, not at_end

    moves = []
    while remaining:
        last = min(remaining, key=rank)  # of equals, the lowest index
        remaining.remove(last)
        if last in new:
            owners[last] = new[last]
        else:
            del owners[last]  # an orphan
        moves.append(dict(owners))
    return moves


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
