import numpy as np
import pytest
from ensemble_boxes import weighted_boxes_fusion

from cuda_wbf import fuse


def test_scalar_matches_oracle():
    image = [
        (np.array([[0.2, 0.2, 0.6, 0.6]]), np.array([0.9]), np.array([1])),
        (np.array([[0.25, 0.25, 0.65, 0.65]]), np.array([0.6]), np.array([1])),
    ]
    actual = fuse([image], [1, 1], backend="cpp", iou_thr=0.55)[0]
    expected = weighted_boxes_fusion(*zip(*image), weights=[1, 1], iou_thr=0.55, skip_box_thr=0.001)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(actual[0] * 100, [[22, 22, 62, 62]], atol=1e-5)


def test_empty_batch():
    assert fuse([], [1], backend="cpp") == []


def test_float32_score_floor_uses_upstream_comparison():
    image = [(np.array([[0, 0, 1, 1]], np.float32), np.array([0.5], np.float32), np.array([1]))]
    floor = 0.5000000001
    expected = weighted_boxes_fusion(*zip(*image), weights=[1.0], skip_box_thr=floor)
    actual = fuse([image], [1.0], backend="cpp", skip_box_thr=floor)[0]
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_allclose(actual[1], expected[1])


@pytest.mark.parametrize("backend", ["cuda_groups", "cuda_coop"])
def test_cuda_backend_error(backend):
    with pytest.raises(RuntimeError, match="Use backend='cpp'"):
        fuse([], [1], backend=backend)
