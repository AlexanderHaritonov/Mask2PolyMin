"""
End-to-end mask2polymin walkthrough on a small house mask (mask.bmp):

1. Extract its contour with skimage.measure.find_contours and save it as a bitmap.
2. Approximate the contour with mask2polymin and visualize the fitted segments in color.
3. Enlarge the colored-segments view 8x and draw the resulting polygon on top.

Run from anywhere; all outputs are written next to this script.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent

import numpy as np
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt

from mask2polymin import FitterToPointsSequence as Fitter, FitterConfig

def run_example():
    mask = load_mask(HERE / "mask.bmp")

    # extract contour using skimage
    contour = max(measure.find_contours(mask.astype(np.uint8), level=0.5), key=len)

    plot_rasterized_contour(mask, contour)

    # approximate the contour with mask2polymin
    fitter = Fitter(FitterConfig(plot_intermediate_steps=True))
    polygon, segments = fitter.fit(points_sequence=contour, is_closed=True)
    plot_segments_colored(mask, contour, segments)

    plot_enlarged_with_polygon(mask, contour, segments, polygon, scale=8)

def load_mask(path) -> np.ndarray:
    """Load mask as a boolean (H, W) array."""
    mask = np.asarray(Image.open(path).convert("L")) > 0
    print(f"Loaded mask {path} with shape {mask.shape}")
    return mask


def rasterize_contour(contour: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize contour points onto a blank (height, width) canvas."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    rows = np.clip(np.round(contour[:, 0]).astype(int), 0, height - 1)
    cols = np.clip(np.round(contour[:, 1]).astype(int), 0, width - 1)
    canvas[rows, cols] = 255
    return canvas

def plot_grayscale(canvas: np.ndarray, title: str) -> None:
    plt.figure()
    plt.imshow(canvas, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show(block=True)

def plot_rasterized_contour(mask: np.ndarray, contour: np.ndarray, filename: str = None) -> None:
    """Rasterize the contour onto a canvas, plot it, and save it as filename."""
    canvas = rasterize_contour(contour, width=mask.shape[0], height=mask.shape[1])
    plot_grayscale(canvas, title='Rasterized contour')
    if filename is not None:
        Image.fromarray(canvas, mode="L").save(HERE / filename)
        print(f"Contour saved to '{filename}'")


def segment_colors(n: int) -> np.ndarray:
    """Per-segment colors shared by save_segments_colored and save_enlarged_with_polygon,
    so segment i is always the same color across both plots. Red is reserved for the polygon."""
    return plt.cm.viridis(np.linspace(0, 1, n))


def plot_segments_colored(mask: np.ndarray, contour: np.ndarray, segments, filename: str = None) -> None:
    """Visualize the fitted segments, each in its own (red-free) color."""

    from mask2polymin.sequence_moments import subsequence

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
    if filename is not None:
        plt.savefig(HERE / filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{filename}'")
    plt.show(block=True)


def plot_enlarged_with_polygon(mask: np.ndarray, contour: np.ndarray, segments, polygon: np.ndarray,
                                scale: int, filename: str = None) -> None:
    """Enlarge the segments_colored view `scale`x and draw the fitted polygon (scaled to match) on top."""

    from mask2polymin.sequence_moments import subsequence

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
    if filename is not None:
        plt.savefig(HERE / filename, dpi=150, bbox_inches='tight')
        print(f"Enlarged result saved to '{filename}'")
    plt.show(block=True)


if __name__ == "__main__":
    run_example()
