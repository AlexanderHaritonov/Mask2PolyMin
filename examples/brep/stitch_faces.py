"""
B-rep demo: stitches the polygons of two neighboring faces, so they share the edge between them.

Input: generation/contours.npz.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np

from mask2polymin import FitterToPointsSequence as Fitter, LineSegmentParams, SequenceSegment, fit_line

FACES = ("left", "right")
CONTOURS_PATH = HERE / "generation" / "contours.npz"
EPSILON = 0.01  # the outer loop stops when a round lowers E by no more than this, in px² (tolerance 1)

def stitch_faces():
    contours = load_contours()
    fitter = Fitter()

    # 1. fit() each face
    left = fitter.fit(contours["left"], is_closed=True)
    right = fitter.fit(contours["right"], is_closed=True)

    # the one edge the two faces share.
    segment_on_left = left.segments[3]
    segment_on_right = right.segments[1]

    config = fitter.config
    energy = np.inf
    drop = np.inf
    while drop > EPSILON:
        # 2. fit the shared line to the points of both faces
        shared_line = fit_shared_line(segment_on_left, segment_on_right)

        # 3. freeze() both segments on the shared line, then refit() each face
        segment_on_left.freeze(shared_line.start_point, shared_line.direction, key="shared")
        segment_on_right.freeze(shared_line.start_point, shared_line.direction, key="shared")
        left = fitter.refit(contours["left"], left, is_closed=True)
        right = fitter.refit(contours["right"], right, is_closed=True)
        segment_on_left = left.frozen["shared"]
        segment_on_right = right.frozen["shared"]

        new_energy = left.energy(config) + right.energy(config)
        drop = energy - new_energy
        energy = new_energy

def fit_shared_line(segment_a: SequenceSegment, segment_b: SequenceSegment) -> LineSegmentParams:
    """The total least squares line through the points of both segments,
       fitted same way the fitter fits a segment"""
    points = np.concatenate([segment_a.points(), segment_b.points()])
    return fit_line(points)

def load_contours() -> dict:
    """Each face's contour, (N, 2) in (row, col)."""
    with np.load(CONTOURS_PATH) as npz:
        return {face: npz[face] for face in FACES}

if __name__ == "__main__":
    stitch_faces()
