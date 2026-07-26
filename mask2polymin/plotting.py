import numpy as np
from matplotlib import pyplot as plt

from mask2polymin.sequence_moments import subsequence
from mask2polymin.sequence_segment import SequenceSegment

def show_fitted_polygon(bitmap, contour, segments, filename=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(bitmap, cmap='gray', alpha=0.3)
    plt.plot(contour[:, 1], contour[:, 0], 'k.', alpha=0.2, markersize=3,
             label='Original contour points')

    _draw_segments(segments)

    plt.title(f'({len(segments)} segments)',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{filename}'")
    plt.show(block=True)

def plot_segments(segments: list[SequenceSegment], filename: str | None = None) -> None:
    """Display only the fitted line segments without bitmap or contour"""
    plt.figure(figsize=(10, 8))

    _draw_segments(segments)

    plt.title(f'Fitted Line Segments ({len(segments)} segments)',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('equal')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Segments-only plot saved to '{filename}'")
    plt.show(block=True)

def _draw_segments(segments: list[SequenceSegment]) -> None:
    """Plot each segment as a distinctly colored line with a legend label."""
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        segment_points = subsequence(segment.whole_sequence, segment.first_index, segment.last_index)
        plt.plot(segment_points[:, 1], segment_points[:, 0], 'o-',
                 color=colors[i], linewidth=2.5, markersize=5,
                 label=f'Seg {i} ({segment.points_count()} pts)')