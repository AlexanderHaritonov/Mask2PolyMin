import numpy as np
from matplotlib import pyplot as plt

from mask2polymin.sequence_segment import SequenceSegment

def show_fitted_polygon(bitmap, contour, segments, filename=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(bitmap, cmap='gray', alpha=0.3)
    plt.plot(contour[:, 1], contour[:, 0], 'k.', alpha=0.2, markersize=3,
             label='Original contour points')

    # Draw each segment
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        # Get the points for this segment
        if segment.first_index <= segment.last_index:
            segment_points = segment.whole_sequence[segment.first_index:segment.last_index + 1]
        else:  # Handle circular/closed case
            segment_points = np.vstack([
                segment.whole_sequence[segment.first_index:],
                segment.whole_sequence[:segment.last_index + 1]
            ])

        # Plot segment points and line
        plt.plot(segment_points[:, 1], segment_points[:, 0], 'o-',
                 color=colors[i], linewidth=2.5, markersize=5,
                 label=f'Seg {i} ({segment.points_count()} pts)')

    plt.title(f'({len(segments)} segments)',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '${filename}'")
    plt.show()

def plot_segments(segments: list[SequenceSegment], filename: str = None) -> None:
    """Display only the fitted line segments without bitmap or contour"""
    plt.figure(figsize=(10, 8))

    # Draw each segment
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    for i, segment in enumerate(segments):
        # Get the points for this segment
        if segment.first_index <= segment.last_index:
            segment_points = segment.whole_sequence[segment.first_index:segment.last_index + 1]
        else:  # Handle circular/closed case
            segment_points = np.vstack([
                segment.whole_sequence[segment.first_index:],
                segment.whole_sequence[:segment.last_index + 1]
            ])

        # Plot segment points and line
        plt.plot(segment_points[:, 1], segment_points[:, 0], 'o-',
                 color=colors[i], linewidth=2.5, markersize=5,
                 label=f'Seg {i} ({segment.points_count()} pts)')

    plt.title(f'Fitted Line Segments ({len(segments)} segments)',
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.axis('equal')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Segments-only plot saved to '${filename}'")
    plt.show()