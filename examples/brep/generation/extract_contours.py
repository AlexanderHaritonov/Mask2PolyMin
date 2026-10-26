"""
Extracts the demo's input from the two noisy face masks written by generate_masks.py:

1. Extract each face's contour with skimage.measure.find_contours.
2. Save them as contours.npz next to this script, arrays `left` and `right`, (N, 2) in (row, col): the only input of the demo.
3. Draw the masks and their contours: the first frame of the video, examples/brep/video_making/frames/00_contours.png.

Run once, after generate_masks.py; contours.npz is committed.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np
from PIL import Image
from skimage import measure
import matplotlib.colors
import matplotlib.pyplot as plt

FACES = ("left", "right")
FACE_COLORS = {"left": "tab:red", "right": "tab:blue"}
FRAME_PATH = HERE.parent / "video_making" / "frames" / "00_contours.png"
VIEW_MARGIN = 3  # px around the contours


def extract_contours():
    masks = {face: load_mask(HERE / f"{face}_mask.bmp") for face in FACES}
    contours = {face: max(measure.find_contours(mask.astype(np.uint8), level=0.5), key=len) for face, mask in masks.items()}
    np.savez(HERE / "contours.npz", **contours)
    print(f"Saved {HERE / 'contours.npz'}")
    plot_masks_and_contours(masks, contours, FRAME_PATH)


def load_mask(path) -> np.ndarray:
    """Load mask as a boolean (H, W) array."""
    return np.asarray(Image.open(path).convert("L")) > 0


def plot_masks_and_contours(masks: dict, contours: dict, path: Path) -> None:
    """Each mask tinted in its face's color, blended where they overlap, with its contour points on top."""
    coverage = sum(masks[face].astype(float) for face in FACES)
    tints = sum(masks[face][..., None] * light(FACE_COLORS[face]) for face in FACES)
    image = np.full(coverage.shape + (3,), 0.7)  # gray background, as in example_house
    covered = coverage > 0
    image[covered] = tints[covered] / coverage[covered][:, None]

    fig, ax = plt.subplots(figsize=(6, 6))
    height, width = coverage.shape
    ax.imshow(image, extent=(-0.5, width - 0.5, height - 0.5, -0.5))  # pixel centers at integer coordinates, as find_contours assumes
    for face in FACES:
        contour = contours[face]
        ax.plot(contour[:, 1], contour[:, 0], 'o', color=FACE_COLORS[face], markersize=6)
    set_view(ax, contours)

    fig.suptitle('Noisy masks of two neighboring faces and their contours', fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.02)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=100)  # 600x600; no bbox_inches='tight': every video frame needs the same size
    print(f"Saved {path}")


def light(color) -> np.ndarray:
    """The color blended halfway with white."""
    return (np.array(matplotlib.colors.to_rgb(color)) + 1) / 2


def set_view(ax, contours: dict) -> None:
    """Zoom to the contours, VIEW_MARGIN px around them."""
    points = np.concatenate(list(contours.values()))
    row_min, col_min = points.min(axis=0) - VIEW_MARGIN
    row_max, col_max = points.max(axis=0) + VIEW_MARGIN
    ax.set_xlim(col_min, col_max)
    ax.set_ylim(row_max, row_min)


if __name__ == "__main__":
    extract_contours()
