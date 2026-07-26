"""
End-to-end mask2polymin walkthrough on a small house mask (mask.bmp):

1. Extract its contour with skimage.measure.find_contours and save it as a bitmap.
2. Approximate the contour with mask2polymin and visualize the fitted segments in color.
3. Enlarge the colored-segments view 8x and draw the resulting polygon on top.

Run from anywhere; all outputs are written next to this script.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt

from mask2polymin import FitterToPointsSequence, FitterConfig
from mask2polymin.sequence_moments import subsequence

HERE = Path(__file__).resolve().parent


def load_mask() -> np.ndarray:
    """Load mask.bmp as a boolean (H, W) array."""
    mask = np.asarray(Image.open(HERE / "mask.bmp").convert("L")) > 0
    print(f"Loaded mask 'mask.bmp' with shape {mask.shape}")
    return mask


def extract_contour(mask: np.ndarray) -> np.ndarray:
    """Extract the outer contour of the mask using skimage.measure.find_contours."""
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    contour = max(contours, key=len)
    print(f"Extracted contour with {len(contour)} points")
    return contour


def save_contour_bmp(mask: np.ndarray, contour: np.ndarray, filename: str) -> None:
    """Rasterize the contour points onto a blank canvas and save as a bitmap."""
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    rows = np.clip(np.round(contour[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.round(contour[:, 1]).astype(int), 0, mask.shape[1] - 1)
    canvas[rows, cols] = 255
    Image.fromarray(canvas, mode="L").save(HERE / filename)
    print(f"Contour saved to '{filename}'")


def fit_polygon(contour: np.ndarray):
    """Approximate the contour with mask2polymin, returning (polygon, segments)."""
    fitter = FitterToPointsSequence(
        points_sequence=contour,
        is_closed=True,
        config=FitterConfig(tolerance=1.0, verbose=False),
    )
    polygon, segments = fitter.fit()
    print(f"Fitted polygon with {len(polygon) - 1} vertices from {len(segments)} segments")
    return polygon, segments


def segment_colors(n: int) -> np.ndarray:
    """Per-segment colors shared by save_segments_colored and save_enlarged_with_polygon,
    so segment i is always the same color across both plots. Red is reserved for the polygon."""
    return plt.cm.viridis(np.linspace(0, 1, n))


def save_segments_colored(mask: np.ndarray, contour: np.ndarray, segments, filename: str) -> None:
    """Visualize the fitted segments, each in its own (red-free) color."""
    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap='gray', alpha=0.3)
    plt.plot(contour[:, 1], contour[:, 0], 'k.', alpha=0.2, markersize=3,
              label='Original contour points')

    colors = segment_colors(len(segments))
    for i, segment in enumerate(segments):
        segment_points = subsequence(segment.whole_sequence, segment.first_index, segment.last_index)
        plt.plot(segment_points[:, 1], segment_points[:, 0], 'o-',
                  color=colors[i], linewidth=2.5, markersize=5, label=f'Seg {i} ({segment.points_count()} pts)')

    plt.title(f'({len(segments)} segments)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(HERE / filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{filename}'")
    plt.show()


def save_enlarged_with_polygon(mask: np.ndarray, contour: np.ndarray, segments, polygon: np.ndarray,
                                scale: int, filename: str) -> None:
    """Enlarge the segments_colored view `scale`x and draw the fitted polygon (scaled to match) on top."""
    enlarged = np.kron(mask, np.ones((scale, scale), dtype=mask.dtype))
    # shift by half an enlarged pixel (right, down) so points land on pixel centers instead of corners
    offset = scale / 2
    scaled_contour = contour * scale + offset
    scaled_polygon = polygon * scale + offset

    plt.figure(figsize=(10, 8))
    plt.imshow(enlarged, cmap='gray', alpha=0.3)
    plt.plot(scaled_contour[:, 1], scaled_contour[:, 0], 'k.', alpha=0.2, markersize=6,
              label='Original contour points')

    colors = segment_colors(len(segments))
    for i, segment in enumerate(segments):
        segment_points = subsequence(segment.whole_sequence, segment.first_index, segment.last_index) * scale + offset
        plt.plot(segment_points[:, 1], segment_points[:, 0], 'o-',
                  color=colors[i], linewidth=2.5, markersize=8, label=f'Seg {i} ({segment.points_count()} pts)')

    plt.plot(scaled_polygon[:, 1], scaled_polygon[:, 0], 'r-o',
              linewidth=1, markersize=5, label='Fitted polygon')

    plt.title(f'Colored segments enlarged {scale}x with fitted polygon', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(HERE / filename, dpi=150, bbox_inches='tight')
    print(f"Enlarged result saved to '{filename}'")
    plt.show()


if __name__ == "__main__":
    mask = load_mask()

    contour = extract_contour(mask)
    save_contour_bmp(mask, contour, "contour.bmp")

    polygon, segments = fit_polygon(contour)
    save_segments_colored(mask, contour, segments, filename="segments_colored.png")

    save_enlarged_with_polygon(mask, contour, segments, polygon, scale=8,
                                filename="enlarged_with_polygon.png")
