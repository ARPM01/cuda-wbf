import os

import numpy as np
import pytest

from cuda_wbf import _native
from cuda_wbf.verification import check_case, fixture_cases, randomized_cases

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    os.getenv("WBF_TEST_CUDA") != "1", reason="set WBF_TEST_CUDA=1 with one GPU selected")]


@pytest.mark.parametrize("case", fixture_cases() + list(randomized_cases()), ids=lambda case: case.name)
def test_cooperative_kernel_parity(case):
    check_case(case, backend="cuda_coop")


def test_match_beyond_128_clusters():
    rows = np.array([[0.9, (i % 16) / 16, (i // 16) / 16,
                      (i % 16) / 16 + 0.02, (i // 16) / 16 + 0.02] for i in range(256)])
    candidate = rows[200].copy()
    candidate[0] = 0.5
    rows = np.vstack([rows, candidate])
    slices = np.array([0, len(rows)], np.int64)
    expected = _native.fuse_groups(rows, slices, 1, 1., 0.7)
    actual = _native.CudaWorkspace().fuse(rows, slices, 1, 1., 0.7, "cuda_coop")
    assert actual[0].shape == (256, 5)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_equal_iou_across_reduction_lanes_chooses_earliest_cluster():
    rows = [[0.95, (i % 16) / 16, 0.5 + (i // 16) / 32,
             (i % 16) / 16 + 0.02, 0.52 + (i // 16) / 32] for i in range(127)]
    # Slots 127 (lane 127) and 128 (lane 0) have equal IoU with the last box.
    rows += [[0.85, 0, 0, 0.375, 0.25], [0.8, 0.625, 0, 1, 0.25], [0.6, 0.25, 0, 0.75, 0.25]]
    rows = np.array(rows)
    slices = np.array([0, len(rows)], np.int64)
    actual = _native.CudaWorkspace().fuse(rows, slices, 1, 1., 0.1, "cuda_coop")
    expected = _native.fuse_groups(rows, slices, 1, 1., 0.1)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[0][128, 1:], [0.625, 0, 1, 0.25])
