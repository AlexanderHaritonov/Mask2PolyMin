import numpy as np
import pytest

from fit_line_segment_reference import fit_line_segment, visualize_edge_with_fit, TEST_EDGE1, TEST_EDGE2


def test_fit_real_mask_edges():
    for edge, title in [(TEST_EDGE1, "Test Edge 1 with Fitted Line"), (TEST_EDGE2, "Test Edge 2 with Fitted Line")]:
        points = np.array(edge, dtype=np.float64)
        params = fit_line_segment(points)

        assert np.linalg.norm(params.direction) == pytest.approx(1.0)
        assert params.loss > 0
        # endpoints are projections of the extreme points; they stay near the data
        lo = points.min(axis=0) - 20
        hi = points.max(axis=0) + 20
        for p in (params.start_point, params.end_point):
            assert np.all(p >= lo) and np.all(p <= hi)

        visualize_edge_with_fit(points, params, title)  # no-op headless (Agg backend)


def test_edge1_is_roughly_vertical():
    params = fit_line_segment(np.array(TEST_EDGE1, dtype=np.float64))
    assert abs(params.direction[1]) > 0.9
