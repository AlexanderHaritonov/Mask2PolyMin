import numpy as np
from skimage import measure

from mask2polymin.fit_to_points_sequence import FitterToPointsSequence
from mask2polymin.plotting import show_fitted_polygon
from mask2polymin.polyline import segments_to_polyline

"""Create a small bitmap with a simple shape"""
bitmap = np.zeros((100, 100), dtype=np.uint8)
bitmap[20:80, 30:70] = 1
bitmap[30:50, 50:80] = 1

"""extract contour - a dense points sequence"""
contour = measure.find_contours(bitmap, level=0.5)[0]

""" fit a polygon """
segments = FitterToPointsSequence(contour, is_closed=True).fit()

""" convert segments to a closed polygon: (M, 2) float vertices, first equals last """
polygon = segments_to_polyline(segments, is_closed=True)
print(f"polygon with {len(polygon) - 1} vertices:")
print(np.round(polygon, 2))

""" plot """
show_fitted_polygon(bitmap, contour, segments, filename='fitted_segments.png')