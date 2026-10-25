"""
Generates the demo's input from the house in examples/example_house/mask.bmp:

1. Copy the house's middle column, so its width is even and the split lies on a pixel edge in the middle.
2. Split it by a vertical line through the apex into two faces with 4 corners each.
3. Distort each face's mask separately, like instance segmentation: gaps and overlaps along the shared edge.
4. Save them as left_mask.bmp and right_mask.bmp next to this script.

Run once, then extract_contours.py; the masks are committed. Uses distort() from performance_test/synth_shapes.py, which needs cv2 (requirements-dev.txt).
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "performance_test"))

import numpy as np
from PIL import Image

from synth_shapes import distort

NOISE_LEVEL = 1  # good segmentation net; level 2 shears the 8 px wide faces too much
SEED = 11


def generate_masks():
    house = load_mask(REPO / "examples" / "example_house" / "mask.bmp")
    house, split_column = widen_to_even(house)
    left, right = split_at(house, split_column)

    rng = np.random.default_rng(SEED)
    for face, filename in ((left, "left_mask.bmp"), (right, "right_mask.bmp")):
        save_mask(distort(face, NOISE_LEVEL, rng), HERE / filename)


def load_mask(path) -> np.ndarray:
    """Load mask as a {0, 1} uint8 (H, W) array."""
    mask = (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)
    print(f"Loaded mask {path} with shape {mask.shape}")
    return mask


def widen_to_even(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Copy the middle column of the shape, which has an odd width (the house: 15 px).
    Returns the widened mask, same size (the last column, empty, is dropped), and the first column of its right half."""
    columns = np.flatnonzero(mask.any(axis=0))
    middle = (columns[0] + columns[-1]) // 2
    widened = np.insert(mask, middle, mask[:, middle], axis=1)[:, :-1]
    return widened, middle + 1


def split_at(mask: np.ndarray, column: int) -> tuple[np.ndarray, np.ndarray]:
    """Split mask into the part left of column and the rest."""
    left, right = mask.copy(), mask.copy()
    left[:, column:] = 0
    right[:, :column] = 0
    return left, right


def save_mask(mask: np.ndarray, path) -> None:
    Image.fromarray(mask * 255).save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    generate_masks()
